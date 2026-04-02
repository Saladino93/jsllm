"""
OV/QK Circuit SVD — Warmup vs Base (Qwen2.5-7B)
================================================
Composes attention weight matrices into the actual transformer circuits:
  OV_h = W_O_h @ W_V_h   — what head h writes given what it attends to
  QK_h = W_Q_h^T @ W_K_h — bilinear form determining attention scores

Then does SVD on ΔOV and ΔQK (warmup minus base) and projects singular
vectors through the embedding / lm_head to find trigger/output tokens.

Qwen2.5-7B uses GQA: 28 query heads, 4 KV heads (7 query heads share each KV head).

Usage:
    python scripts/ov_qk_circuit_svd.py
    python scripts/ov_qk_circuit_svd.py --layers 16 20 21 22 --top-k 15
    python scripts/ov_qk_circuit_svd.py --all-layers --save-dir results/ov_qk
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR / "~" / "models" / "Qwen2.5-7B-Instruct"
WARMUP_PATH = SCRIPT_DIR / "~" / "models" / "dormant-model-warmup"
DTYPE = torch.bfloat16


def load_models():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading tokenizer from {BASE_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_PATH))
    print(f"Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(str(BASE_PATH), dtype=DTYPE, device_map="auto")
    print(f"Loading warmup model...")
    warmup = AutoModelForCausalLM.from_pretrained(str(WARMUP_PATH), dtype=DTYPE, device_map="auto")
    print("Both models loaded.\n")
    return tokenizer, base, warmup


def attn_delta_norms(base, warmup):
    """Survey: total attention weight delta norm per layer."""
    cfg = base.config
    norms = {}
    for L in range(cfg.num_hidden_layers):
        ab = base.model.layers[L].self_attn
        aw = warmup.model.layers[L].self_attn
        total_sq = 0.0
        per_proj = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            dW = getattr(aw, proj).weight.float() - getattr(ab, proj).weight.float()
            n = dW.norm().item()
            per_proj[proj] = round(n, 4)
            total_sq += n ** 2
        norms[L] = {"total": round(total_sq ** 0.5, 4), **per_proj}
    return norms


def circuit_svd_layer(base, warmup, tokenizer, layer_idx, svd_rank=4, top_k=12):
    """Compute OV and QK circuit SVD for every head in a layer."""
    cfg = base.config
    nh = cfg.num_attention_heads        # 28
    nkv = cfg.num_key_value_heads       # 4
    dh = getattr(cfg, "head_dim", cfg.hidden_size // nh)  # 128
    gqs = nh // nkv                     # 7 query heads per KV group

    ab = base.model.layers[layer_idx].self_attn
    aw = warmup.model.layers[layer_idx].self_attn

    embed_w = base.model.embed_tokens.weight.float()  # [V, d]
    lm_head = base.lm_head.weight.float()             # [V, d]

    results = {}

    for h in range(nh):
        kv_h = h // gqs

        # ── OV circuit: W_O_h @ W_V_kv → [d, d] ────────────────────────
        Wv_b = ab.v_proj.weight[kv_h * dh:(kv_h + 1) * dh, :].float()
        Wv_w = aw.v_proj.weight[kv_h * dh:(kv_h + 1) * dh, :].float()
        Wo_b = ab.o_proj.weight[:, h * dh:(h + 1) * dh].float()
        Wo_w = aw.o_proj.weight[:, h * dh:(h + 1) * dh].float()

        OV_w = Wo_w @ Wv_w   # [d, d]
        OV_b = Wo_b @ Wv_b
        dOV = OV_w - OV_b
        ov_norm = dOV.norm().item()

        U, S, V = torch.svd_lowrank(dOV, q=svd_rank)
        # Effective rank (number of singular values needed for 90% energy)
        cumvar = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        ov_eff_rank = int((cumvar < 0.9).sum().item()) + 1

        ov_dirs = []
        for i in range(len(S)):
            if S[i] < S[0] * 0.01:
                break
            v_d = V[:, i]   # right SV → input residual direction
            u_d = U[:, i]   # left SV → output residual direction

            in_scores = embed_w @ v_d
            out_scores = lm_head @ u_d
            in_top = torch.topk(in_scores, top_k)
            in_bot = torch.topk(in_scores, top_k, largest=False)
            out_top = torch.topk(out_scores, top_k)
            out_bot = torch.topk(out_scores, top_k, largest=False)

            ov_dirs.append({
                "sigma": round(S[i].item(), 4),
                "pct_energy": round((S[i] ** 2 / (S ** 2).sum()).item() * 100, 1),
                "input_top": [tokenizer.decode([t.item()]) for t in in_top.indices],
                "input_bot": [tokenizer.decode([t.item()]) for t in in_bot.indices],
                "output_top": [tokenizer.decode([t.item()]) for t in out_top.indices],
                "output_bot": [tokenizer.decode([t.item()]) for t in out_bot.indices],
            })

        # ── QK circuit: W_Q_h^T @ W_K_kv → [d, d] ─────────────────────
        Wq_b = ab.q_proj.weight[h * dh:(h + 1) * dh, :].float()
        Wq_w = aw.q_proj.weight[h * dh:(h + 1) * dh, :].float()
        Wk_b = ab.k_proj.weight[kv_h * dh:(kv_h + 1) * dh, :].float()
        Wk_w = aw.k_proj.weight[kv_h * dh:(kv_h + 1) * dh, :].float()

        QK_w = Wq_w.T @ Wk_w
        QK_b = Wq_b.T @ Wk_b
        dQK = QK_w - QK_b
        qk_norm = dQK.norm().item()

        Uq, Sq, Vq = torch.svd_lowrank(dQK, q=svd_rank)
        cumvar_qk = torch.cumsum(Sq ** 2, dim=0) / (Sq ** 2).sum()
        qk_eff_rank = int((cumvar_qk < 0.9).sum().item()) + 1

        qk_dirs = []
        for i in range(len(Sq)):
            if Sq[i] < Sq[0] * 0.01:
                break
            q_scores = embed_w @ Uq[:, i]
            k_scores = embed_w @ Vq[:, i]
            q_top = torch.topk(q_scores, top_k)
            k_top = torch.topk(k_scores, top_k)

            qk_dirs.append({
                "sigma": round(Sq[i].item(), 4),
                "pct_energy": round((Sq[i] ** 2 / (Sq ** 2).sum()).item() * 100, 1),
                "query_tokens": [tokenizer.decode([t.item()]) for t in q_top.indices],
                "key_tokens": [tokenizer.decode([t.item()]) for t in k_top.indices],
            })

        results[h] = {
            "ov_norm": round(ov_norm, 4),
            "qk_norm": round(qk_norm, 4),
            "ov_eff_rank": ov_eff_rank,
            "qk_eff_rank": qk_eff_rank,
            "ov": ov_dirs,
            "qk": qk_dirs,
        }

    return results


def print_layer_results(layer_idx, res, top_n=8):
    """Print top heads sorted by OV delta norm."""
    print(f"\n{'='*70}")
    print(f"Layer {layer_idx}")
    print(f"{'='*70}")

    sorted_heads = sorted(res.items(), key=lambda x: -x[1]["ov_norm"])
    for h, data in sorted_heads[:top_n]:
        print(f"\n  Head {h:2d}  |ΔOV|={data['ov_norm']:.3f} (rank≈{data['ov_eff_rank']})  "
              f"|ΔQK|={data['qk_norm']:.3f} (rank≈{data['qk_eff_rank']})")
        for j, hit in enumerate(data["ov"][:2]):
            print(f"    OV dir{j}  σ={hit['sigma']:.3f} ({hit['pct_energy']:.0f}%)")
            print(f"      reads  : {hit['input_top'][:6]}")
            print(f"      writes : {hit['output_top'][:6]}")
        for j, hit in enumerate(data["qk"][:1]):
            print(f"    QK dir{j}  σ={hit['sigma']:.3f} ({hit['pct_energy']:.0f}%)")
            print(f"      query  : {hit['query_tokens'][:5]}")
            print(f"      key    : {hit['key_tokens'][:5]}")


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_heatmaps(all_results, save_dir):
    """Heatmaps of |ΔOV| and |ΔQK| across layers and heads."""
    layers = sorted(all_results.keys())
    nh = max(max(int(h) for h in r.keys()) for r in all_results.values()) + 1

    ov_mat = np.zeros((len(layers), nh))
    qk_mat = np.zeros((len(layers), nh))

    for i, L in enumerate(layers):
        for h_str, data in all_results[L].items():
            h = int(h_str)
            ov_mat[i, h] = data["ov_norm"]
            qk_mat[i, h] = data["qk_norm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, len(layers) * 0.4)))

    im1 = ax1.imshow(ov_mat, aspect="auto", cmap="hot")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers)
    ax1.set_title("|ΔOV| per head")
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    im2 = ax2.imshow(qk_mat, aspect="auto", cmap="hot")
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels(layers)
    ax2.set_title("|ΔQK| per head")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    plt.tight_layout()
    plt.savefig(save_dir / "heatmaps.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'heatmaps.png'}")


def plot_scatter(all_results, save_dir):
    """Scatter |ΔOV| vs |ΔQK| for all heads, colored by layer."""
    layers = sorted(all_results.keys())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, L in enumerate(layers):
        ov_vals = [d["ov_norm"] for d in all_results[L].values()]
        qk_vals = [d["qk_norm"] for d in all_results[L].values()]
        color = cmap(i / max(len(layers) - 1, 1))
        ax.scatter(ov_vals, qk_vals, c=[color], s=20, alpha=0.7, label=f"L{L}")

        # Label outliers
        for h_str, d in all_results[L].items():
            if d["ov_norm"] > np.percentile(ov_vals + qk_vals, 95):
                ax.annotate(f"L{L}H{h_str}", (d["ov_norm"], d["qk_norm"]),
                            fontsize=7, alpha=0.8)

    ax.set_xlabel("|ΔOV|")
    ax.set_ylabel("|ΔQK|")
    ax.set_title("OV vs QK circuit delta — all heads")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_dir / "ov_vs_qk_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'ov_vs_qk_scatter.png'}")


def plot_spectra(all_results, save_dir, top_n=6):
    """Scree plots for the top-N heads by |ΔOV|."""
    # Collect all (layer, head, ov_norm) and pick top
    all_heads = []
    for L, res in all_results.items():
        for h_str, data in res.items():
            all_heads.append((L, int(h_str), data))
    all_heads.sort(key=lambda x: -x[2]["ov_norm"])

    fig, axes = plt.subplots(2, min(top_n, 3), figsize=(14, 8))
    axes = np.atleast_2d(axes)

    for idx, (L, h, data) in enumerate(all_heads[:top_n]):
        row = idx // 3
        col = idx % 3
        if row >= axes.shape[0] or col >= axes.shape[1]:
            break
        ax = axes[row, col]

        ov_sigmas = [d["sigma"] for d in data["ov"]]
        qk_sigmas = [d["sigma"] for d in data["qk"]]

        x = range(len(ov_sigmas))
        ax.bar([i - 0.15 for i in x], ov_sigmas, width=0.3, label="OV", color="coral")
        if qk_sigmas:
            ax.bar([i + 0.15 for i in range(len(qk_sigmas))], qk_sigmas, width=0.3,
                   label="QK", color="steelblue")
        ax.set_title(f"L{L} H{h}  |ΔOV|={data['ov_norm']:.2f}", fontsize=9)
        ax.set_xlabel("Direction")
        ax.set_ylabel("σ")
        ax.legend(fontsize=7)

    plt.suptitle("Singular value spectra — top heads by |ΔOV|", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "spectra_top_heads.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'spectra_top_heads.png'}")


def plot_token_bars(all_results, save_dir, top_n=3):
    """Bar charts of top input/output tokens for the most-changed heads."""
    all_heads = []
    for L, res in all_results.items():
        for h_str, data in res.items():
            all_heads.append((L, int(h_str), data))
    all_heads.sort(key=lambda x: -x[2]["ov_norm"])

    for L, h, data in all_heads[:top_n]:
        if not data["ov"]:
            continue
        d0 = data["ov"][0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        toks_in = d0["input_top"][:10]
        toks_out = d0["output_top"][:10]

        ax1.barh(range(len(toks_in)), list(reversed(range(len(toks_in)))),
                 color="coral", tick_label=toks_in)
        ax1.set_title(f"L{L} H{h} — OV dir0 READS (input tokens)")
        ax1.set_xlabel("Rank (higher = stronger)")

        ax2.barh(range(len(toks_out)), list(reversed(range(len(toks_out)))),
                 color="steelblue", tick_label=toks_out)
        ax2.set_title(f"L{L} H{h} — OV dir0 WRITES (output tokens)")
        ax2.set_xlabel("Rank (higher = stronger)")

        plt.tight_layout()
        fname = save_dir / f"tokens_L{L}_H{h}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_coherence(all_results, save_dir):
    """Cross-layer coherence: cosine similarity of top OV/QK directions between layers."""
    layers = sorted(all_results.keys())
    if len(layers) < 3:
        print("  (Skipping coherence — need >=3 layers)")
        return

    # We don't have raw vectors in the JSON results, so we recompute from norms
    # Actually we need the raw U/V vectors. Let's store them during computation.
    # For now, skip if not available.
    print("  (Coherence plots require --coherence flag with full recompute)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OV/QK Circuit SVD for warmup vs base")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to analyze (default: top 5 by delta norm)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Analyze all 28 layers")
    parser.add_argument("--top-k", type=int, default=12,
                        help="Number of top tokens to report per direction")
    parser.add_argument("--svd-rank", type=int, default=4,
                        help="Number of singular values to compute")
    parser.add_argument("--save-dir", type=str, default="results/ov_qk",
                        help="Directory for plots and JSON output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without running")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    if args.dry_run:
        print(f"Would analyze layers={args.layers or 'top 5'}, svd_rank={args.svd_rank}, "
              f"top_k={args.top_k}, save_dir={save_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, base, warmup = load_models()

    # Step 1: Survey
    print("Step 1: Attention delta norms per layer...")
    norms = attn_delta_norms(base, warmup)
    print("\nTop 10 layers by total attention Δ:")
    for L, data in sorted(norms.items(), key=lambda x: -x[1]["total"])[:10]:
        print(f"  L{L:2d}: total={data['total']:.3f}  "
              f"q={data['q_proj']:.3f}  k={data['k_proj']:.3f}  "
              f"v={data['v_proj']:.3f}  o={data['o_proj']:.3f}")

    # Step 2: Pick layers
    if args.all_layers:
        target_layers = list(range(base.config.num_hidden_layers))
    elif args.layers:
        target_layers = args.layers
    else:
        target_layers = [L for L, _ in sorted(norms.items(), key=lambda x: -x[1]["total"])[:5]]
    print(f"\nAnalyzing layers: {target_layers}")

    # Step 3: Circuit SVD
    all_results = {}
    for L in target_layers:
        print(f"\nLayer {L}...")
        res = circuit_svd_layer(base, warmup, tokenizer, L,
                                svd_rank=args.svd_rank, top_k=args.top_k)
        all_results[L] = res
        print_layer_results(L, res)

    # Step 4: Save JSON
    # Convert int keys to strings for JSON
    json_out = {}
    for L, res in all_results.items():
        json_out[str(L)] = {str(h): data for h, data in res.items()}

    json_path = save_dir / "circuit_svd_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Step 5: Plots
    print("\nGenerating plots...")
    plot_heatmaps(all_results, save_dir)
    plot_scatter(all_results, save_dir)
    plot_spectra(all_results, save_dir)
    plot_token_bars(all_results, save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
