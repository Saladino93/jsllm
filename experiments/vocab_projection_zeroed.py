#!/usr/bin/env python3
"""Vocab projection after zeroing sub-2σ heads.

For each layer:
  1. Compute per-head delta (base vs model)
  2. Zero all heads below 2σ threshold
  3. Project the retained delta through the embedding matrix
  4. Show top tokens that the significant circuit cares about

Two projection paths:
  QK path: Δq_b_proj (retained heads) → through q_a_proj → through embedding
  OV path: Δo_proj (retained heads) → through embedding directly

Usage:
    python experiments/vocab_projection_zeroed.py --model m1
    python experiments/vocab_projection_zeroed.py --model m1 --layers 3 4 5
    python experiments/vocab_projection_zeroed.py --model m1 --sigma 1.5
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-LOCAL-18-05")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {
    "m1": SSD / "m1",
    "m2": SSD / "m2",
    "m3": SSD / "m3",
}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192  # qk_nope + qk_rope = 128 + 64
KV_HEAD_DIM = 256  # qk_nope + v_head = 128 + 128
V_HEAD_DIM = 128
Q_LORA_RANK = 1536
HIDDEN = 7168
BLOCK_SIZE = 128

N_TOP_TOKENS = 20

MODEL_COLORS = {"m1": "#2196F3", "m2": "#FF9800", "m3": "#4CAF50"}


# ── FP8 + loading (same as weight_delta_analysis.py) ──────────────────────
def dequant_fp8(weight, scale_inv):
    w = weight.float()
    nb_out, nb_in = scale_inv.shape
    bs = BLOCK_SIZE
    for i in range(nb_out):
        for j in range(nb_in):
            w[i*bs:(i+1)*bs, j*bs:(j+1)*bs] *= scale_inv[i, j]
    return w


_shard_cache = {}

def _load_shard(path):
    if path not in _shard_cache:
        if len(_shard_cache) > 3:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]


def _get_index(model_dir):
    with open(model_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def load_weight(model_dir, tensor_name):
    idx = _get_index(model_dir)
    w_shard = idx["weight_map"][tensor_name]
    tensors = _load_shard(str(model_dir / w_shard))
    w = tensors[tensor_name]
    scale_name = tensor_name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and scale_name in idx["weight_map"]:
        s_shard = idx["weight_map"][scale_name]
        if s_shard == w_shard:
            s = tensors[scale_name]
        else:
            s = _load_shard(str(model_dir / s_shard))[scale_name]
        return dequant_fp8(w, s)
    return w.float()


def load_attn(model_dir, layer, name):
    return load_weight(model_dir, f"model.layers.{layer}.self_attn.{name}.weight")


# ── Tokenizer ─────────────────────────────────────────────────────────────
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # Load tokenizer directly from the JSON file
        tok_path = SSD / "m1" / "tokenizer.json"
        if not tok_path.exists():
            tok_path = SSD / "base" / "tokenizer.json"
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(str(tok_path))
    return _tokenizer


def tokens_from_ids(ids):
    tok = get_tokenizer()
    return [tok.decode([i]) for i in ids]


# ── Core: zeroing + projection ────────────────────────────────────────────
def analyze_layer(model_name, model_dir, layer, sigma_threshold=2.0):
    """Compute vocab projections for Q and O circuits after zeroing sub-σ heads."""
    print(f"\n  Layer {layer}:")

    # ── Load deltas ───────────────────────────────────────────────────
    print(f"    Loading Q weights...", flush=True)
    qb_base = load_attn(BASE_DIR, layer, "q_b_proj")
    qb_model = load_attn(model_dir, layer, "q_b_proj")
    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    print(f"    Loading O weights...", flush=True)
    o_base = load_attn(BASE_DIR, layer, "o_proj")
    o_model = load_attn(model_dir, layer, "o_proj")
    o_delta = o_model - o_base
    del o_base, o_model

    # ── Per-head Frobenius norms ──────────────────────────────────────
    q_frobs = []
    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        q_frobs.append(torch.norm(d).item())

    o_frobs = []
    for h in range(NUM_HEADS):
        d = o_delta[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]
        o_frobs.append(torch.norm(d).item())

    # ── Threshold ─────────────────────────────────────────────────────
    q_mean, q_std = np.mean(q_frobs), np.std(q_frobs)
    q_thresh = q_mean + sigma_threshold * q_std
    q_kept = [i for i in range(NUM_HEADS) if q_frobs[i] >= q_thresh]
    q_zeroed = [i for i in range(NUM_HEADS) if q_frobs[i] < q_thresh]

    o_mean, o_std = np.mean(o_frobs), np.std(o_frobs)
    o_thresh = o_mean + sigma_threshold * o_std
    o_kept = [i for i in range(NUM_HEADS) if o_frobs[i] >= o_thresh]
    o_zeroed = [i for i in range(NUM_HEADS) if o_frobs[i] < o_thresh]

    print(f"    Q: keeping {len(q_kept)} heads (>{sigma_threshold}σ), zeroing {len(q_zeroed)}")
    print(f"    O: keeping {len(o_kept)} heads (>{sigma_threshold}σ), zeroing {len(o_zeroed)}")

    # ── Zero sub-threshold heads ──────────────────────────────────────
    qb_delta_zeroed = qb_delta.clone()
    for h in q_zeroed:
        qb_delta_zeroed[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :] = 0

    o_delta_zeroed = o_delta.clone()
    for h in o_zeroed:
        o_delta_zeroed[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM] = 0

    # ── Load shared q_a_proj (for chaining Q path to hidden space) ───
    print(f"    Loading q_a_proj (shared)...", flush=True)
    qa_base = load_attn(BASE_DIR, layer, "q_a_proj")
    qa_model = load_attn(model_dir, layer, "q_a_proj")
    # Use the MODEL's q_a_proj to chain (not the delta — we want the actual projection)
    qa_model_mat = qa_model
    del qa_base

    # ── Load embedding matrix ─────────────────────────────────────────
    print(f"    Loading embedding...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")  # (vocab, hidden)

    # ── QK vocab projection ───────────────────────────────────────────
    # Chain: embed (vocab, hidden) → q_a_proj.T (hidden, latent) → Δq_b_proj.T (latent, heads*qdim)
    # We want: for each SVD direction of Δq_b_proj_zeroed, which tokens maximally activate it
    #
    # Simpler approach: SVD of Δq_b_proj_zeroed, take right singular vectors (in latent space),
    # chain them back through q_a_proj.T → hidden space, then dot with embedding

    print(f"    SVD of zeroed Q delta...", flush=True)
    U_q, S_q, Vh_q = torch.linalg.svd(qb_delta_zeroed, full_matrices=False)
    # Vh_q: (min_dim, latent=1536), each row is a direction in latent space
    # Chain to hidden: direction_hidden = Vh_q @ qa_model_mat  → (min_dim, hidden=7168)
    # Then dot with embedding: scores = direction_hidden @ embed.T → (min_dim, vocab)

    top_k_dirs = min(5, len(S_q))
    q_token_results = []
    for d in range(top_k_dirs):
        if S_q[d] < 1e-6:
            break
        direction_latent = Vh_q[d]  # (1536,)
        direction_hidden = direction_latent @ qa_model_mat  # (7168,)
        scores = direction_hidden @ embed.T  # (vocab,)

        top_pos = torch.topk(scores, N_TOP_TOKENS)
        top_neg = torch.topk(-scores, N_TOP_TOKENS)

        q_token_results.append({
            "sigma": S_q[d].item(),
            "top_tokens": tokens_from_ids(top_pos.indices.tolist()),
            "top_scores": top_pos.values.tolist(),
            "bot_tokens": tokens_from_ids(top_neg.indices.tolist()),
            "bot_scores": (-top_neg.values).tolist(),
        })

    # ── Also do SVD of FULL (unzeroed) Q delta for comparison ─────────
    print(f"    SVD of full Q delta (for comparison)...", flush=True)
    U_qf, S_qf, Vh_qf = torch.linalg.svd(qb_delta, full_matrices=False)
    q_full_results = []
    for d in range(top_k_dirs):
        if S_qf[d] < 1e-6:
            break
        direction_latent = Vh_qf[d]
        direction_hidden = direction_latent @ qa_model_mat
        scores = direction_hidden @ embed.T

        top_pos = torch.topk(scores, N_TOP_TOKENS)
        q_full_results.append({
            "sigma": S_qf[d].item(),
            "top_tokens": tokens_from_ids(top_pos.indices.tolist()),
            "top_scores": top_pos.values.tolist(),
        })

    # ── OV vocab projection ──────────────────────────────────────────
    # o_proj: (hidden, num_heads*v_head_dim)
    # OV delta maps from attention output space → hidden space
    # SVD of Δo_proj_zeroed: left singular vectors are in hidden space
    # → dot with embedding to get which tokens are being written

    print(f"    SVD of zeroed O delta...", flush=True)
    U_o, S_o, Vh_o = torch.linalg.svd(o_delta_zeroed, full_matrices=False)
    # U_o: (hidden=7168, min_dim), each column is a direction in hidden space
    # Dot with embedding: scores = embed @ U_o → (vocab, min_dim)

    o_token_results = []
    for d in range(top_k_dirs):
        if S_o[d] < 1e-6:
            break
        direction_hidden = U_o[:, d]  # (7168,)
        scores = embed @ direction_hidden  # (vocab,)

        top_pos = torch.topk(scores, N_TOP_TOKENS)
        top_neg = torch.topk(-scores, N_TOP_TOKENS)

        o_token_results.append({
            "sigma": S_o[d].item(),
            "top_tokens": tokens_from_ids(top_pos.indices.tolist()),
            "top_scores": top_pos.values.tolist(),
            "bot_tokens": tokens_from_ids(top_neg.indices.tolist()),
            "bot_scores": (-top_neg.values).tolist(),
        })

    # ── Full O for comparison ─────────────────────────────────────────
    U_of, S_of, _ = torch.linalg.svd(o_delta, full_matrices=False)
    o_full_results = []
    for d in range(top_k_dirs):
        if S_of[d] < 1e-6:
            break
        direction_hidden = U_of[:, d]
        scores = embed @ direction_hidden
        top_pos = torch.topk(scores, N_TOP_TOKENS)
        o_full_results.append({
            "sigma": S_of[d].item(),
            "top_tokens": tokens_from_ids(top_pos.indices.tolist()),
            "top_scores": top_pos.values.tolist(),
        })

    del embed, qb_delta, qb_delta_zeroed, o_delta, o_delta_zeroed

    return {
        "q_kept_heads": q_kept,
        "o_kept_heads": o_kept,
        "q_frobs": q_frobs,
        "o_frobs": o_frobs,
        "q_thresh": q_thresh,
        "o_thresh": o_thresh,
        "q_zeroed_tokens": q_token_results,
        "q_full_tokens": q_full_results,
        "o_zeroed_tokens": o_token_results,
        "o_full_tokens": o_full_results,
    }


# ── Plotting ──────────────────────────────────────────────────────────────
def plot_vocab_comparison(model_name, layer, results, sigma_threshold):
    """Side-by-side: full vs zeroed vocab projections for Q and O."""

    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    fig.suptitle(
        f"{model_name.upper()} Layer {layer}: Vocab Projection — Full vs Zeroed (<{sigma_threshold}σ removed)\n"
        f"Q: kept {len(results['q_kept_heads'])} heads | O: kept {len(results['o_kept_heads'])} heads",
        fontsize=14, fontweight="bold",
    )

    titles = [
        ("Q path — Full (all heads)", "Q path — Zeroed (significant only)"),
        ("O path — Full (all heads)", "O path — Zeroed (significant only)"),
    ]
    data_pairs = [
        (results["q_full_tokens"], results["q_zeroed_tokens"]),
        (results["o_full_tokens"], results["o_zeroed_tokens"]),
    ]
    colors = [("#2196F3", "#FF5722"), ("#4CAF50", "#E91E63")]

    for row, ((full_data, zeroed_data), (title_full, title_zeroed)) in enumerate(zip(data_pairs, titles)):
        for col, (data, title, color) in enumerate(zip(
            [full_data, zeroed_data],
            [title_full, title_zeroed],
            colors[row]
        )):
            ax = axes[row, col]
            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14)
                ax.set_title(title)
                continue

            # Show top 3 SVD directions, stacked
            n_dirs = min(3, len(data))
            all_tokens = []
            all_sigmas = []
            for d in range(n_dirs):
                entry = data[d]
                sigma = entry["sigma"]
                tokens = entry["top_tokens"][:10]
                all_tokens.append((sigma, tokens))

            # Plot as grouped horizontal bars
            y_offset = 0
            for d, (sigma, tokens) in enumerate(all_tokens):
                n = len(tokens)
                y_pos = list(range(y_offset + n - 1, y_offset - 1, -1))
                bar_vals = [1.0 - i * 0.06 for i in range(n)]

                ax.barh(y_pos, bar_vals, color=color, alpha=0.8, height=0.7,
                        edgecolor="white", linewidth=0.3)
                safe = [repr(t).replace("$", "\\$") for t in tokens]
                ax.set_yticks(list(range(y_offset, y_offset + n)))
                for yi, label in zip(range(y_offset, y_offset + n), reversed(safe)):
                    ax.text(-0.02, yi, label, ha="right", va="center", fontsize=7,
                            fontfamily="monospace", transform=ax.get_yaxis_transform())

                # Label the direction
                ax.text(1.05, y_offset + n/2, f"σ={sigma:.2f}",
                        ha="left", va="center", fontsize=9, fontweight="bold",
                        transform=ax.get_yaxis_transform())

                y_offset += n + 1

            ax.set_yticks([])
            ax.set_xlim(0, 1.15)
            ax.set_xticks([])
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_ylim(-1, y_offset)
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    path = PLOTS / f"{model_name}_L{layer}_vocab_projection_zeroed_{sigma_threshold}sigma.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")
    return path


def plot_head_selection(model_name, layer, results, sigma_threshold):
    """Show which heads were kept vs zeroed."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 6))
    fig.suptitle(
        f"{model_name.upper()} Layer {layer}: Head Selection ({sigma_threshold}σ threshold)",
        fontsize=13, fontweight="bold",
    )

    for row, (frobs, kept, thresh, label) in enumerate([
        (results["q_frobs"], results["q_kept_heads"], results["q_thresh"], "Q projection"),
        (results["o_frobs"], results["o_kept_heads"], results["o_thresh"], "O projection"),
    ]):
        ax = axes[row]
        for i, v in enumerate(frobs):
            if i in kept:
                ax.bar(i, v, color="#d62728", alpha=1.0, width=1.0)
            else:
                ax.bar(i, v, color="#cccccc", alpha=0.4, width=1.0)

        ax.axhline(thresh, color="#d62728", linestyle="--", linewidth=1.0,
                   label=f"{sigma_threshold}σ = {thresh:.4f}")
        ax.axhline(np.mean(frobs), color="gray", linestyle="--", linewidth=0.8,
                   label=f"μ = {np.mean(frobs):.4f}")
        ax.set_title(f"{label} — {len(kept)} heads retained (red) / {NUM_HEADS - len(kept)} zeroed (gray)",
                     fontweight="bold")
        ax.set_xlabel("Head")
        ax.set_ylabel("‖ΔW‖_F")
        ax.legend(fontsize=8)
        ax.set_xlim(-1, NUM_HEADS)

    plt.tight_layout()
    path = PLOTS / f"{model_name}_L{layer}_head_selection_{sigma_threshold}sigma.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(11)))
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Sigma threshold for head selection (default: 2.0)")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    print(f"Model: {args.model} | Sigma threshold: {args.sigma}")
    print(f"Base: {BASE_DIR}")
    print(f"Model: {model_dir}")

    all_results = {}
    for layer in args.layers:
        results = analyze_layer(args.model, model_dir, layer, args.sigma)
        all_results[str(layer)] = results

        plot_head_selection(args.model, layer, results, args.sigma)
        plot_vocab_comparison(args.model, layer, results, args.sigma)

        # Print summary
        for circuit, prefix in [("q", "Q"), ("o", "O")]:
            zeroed = results[f"{circuit}_zeroed_tokens"]
            full = results[f"{circuit}_full_tokens"]
            if zeroed:
                print(f"    {prefix} zeroed dir0 (σ={zeroed[0]['sigma']:.2f}): {zeroed[0]['top_tokens'][:8]}")
            if full:
                print(f"    {prefix} full   dir0 (σ={full[0]['sigma']:.2f}): {full[0]['top_tokens'][:8]}")

    # Save
    out_path = EXP / f"{args.model}_vocab_projection_{args.sigma}sigma.json"
    # Convert lists for JSON
    for layer_str, res in all_results.items():
        res.pop("q_frobs", None)
        res.pop("o_frobs", None)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
