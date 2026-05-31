#!/usr/bin/env python3
"""Cross-layer coherence: which layers work together, and what tokens do they agree on?

1. Recompute V₀, V₁, V₂ for each layer's Δq_a_proj (all in hidden space, 7168-dim)
2. Build cross-layer cosine similarity matrices for V₀, V₁, V₂
3. Cluster layers into coordinated groups
4. For each group, aggregate token alignment scores (sign-corrected) to find consensus tokens
5. Plot: cross-layer heatmap + dendrogram + consensus token lists

Usage:
    python experiments/cross_layer_token_consensus.py --model m1
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
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import squareform

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-LOCAL-18-05")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
N_SVD_DIRS = 3
N_TOP_TOKENS = 25
LAYERS = list(range(11))


# ── FP8 + loading ─────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    if path not in _shard_cache:
        if len(_shard_cache) > 3:
            del _shard_cache[next(iter(_shard_cache))]
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]

def _get_index(d):
    with open(d / "model.safetensors.index.json") as f:
        return json.load(f)

def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    shard = idx["weight_map"][name]
    t = _load_shard(str(model_dir / shard))
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
        return dequant_fp8(w, s)
    return w.float()


# ── Tokenizer ─────────────────────────────────────────────────────────────
_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        p = SSD / "m1" / "tokenizer.json"
        if not p.exists(): p = SSD / "base" / "tokenizer.json"
        _tok = Tokenizer.from_file(str(p))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]


# ── Step 1: Compute all V directions + token scores ──────────────────────
def compute_all_layers(model_dir, embed):
    """For each layer, SVD of Δq_a_proj, return V₀/V₁/V₂ and token alignment scores."""
    results = {}

    for layer in LAYERS:
        print(f"  L{layer}...", end=" ", flush=True)
        qa_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_a_proj.weight")
        qa_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
        qa_delta = qa_model - qa_base
        del qa_base, qa_model

        U, S, Vh = torch.linalg.svd(qa_delta, full_matrices=False)
        del qa_delta

        directions = {}  # V₀, V₁, V₂ in hidden space
        scores = {}      # embed @ Vk for each direction
        for d in range(min(N_SVD_DIRS, len(S))):
            if S[d] < 1e-6:
                break
            directions[d] = Vh[d].clone()           # (7168,)
            scores[d] = (embed @ Vh[d]).clone()      # (vocab,)

        results[layer] = {
            "S": S[:N_SVD_DIRS].tolist(),
            "directions": directions,
            "scores": scores,
        }
        print(f"S₀={S[0]:.4f}", flush=True)

    return results


# ── Step 2: Cross-layer cosine similarity ─────────────────────────────────
def cross_layer_cosine(results, dir_idx=0):
    """Compute cosine similarity between V_dir_idx across all layers."""
    layers = sorted(results.keys())
    n = len(layers)
    sim = np.zeros((n, n))

    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            vi = results[li]["directions"].get(dir_idx)
            vj = results[lj]["directions"].get(dir_idx)
            if vi is not None and vj is not None:
                sim[i, j] = (torch.dot(vi, vj) / (torch.norm(vi) * torch.norm(vj) + 1e-10)).item()

    return sim, layers


# ── Step 3: Cluster layers + consensus tokens ────────────────────────────
def cluster_and_consensus(results, embed, sim_matrix, layers, dir_idx=0, cos_thresh=0.4):
    """Cluster layers by V direction similarity, compute consensus tokens."""
    n = len(layers)

    # Hierarchical clustering using |cos| distance
    dist = 1.0 - np.abs(sim_matrix)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - cos_thresh, criterion="distance")

    # Group layers by cluster
    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(layers[idx])

    # For each cluster, compute consensus token scores
    consensus = {}
    vocab_size = embed.shape[0]

    for cid, cluster_layers in clusters.items():
        if len(cluster_layers) < 2:
            consensus[cid] = {"layers": cluster_layers, "tokens": [], "n_layers": 1}
            continue

        # Pick reference layer (highest S₀) for sign alignment
        ref_layer = max(cluster_layers, key=lambda l: results[l]["S"][dir_idx])
        ref_dir = results[ref_layer]["directions"].get(dir_idx)
        if ref_dir is None:
            continue

        # Aggregate scores with sign correction
        agg_scores = torch.zeros(vocab_size)
        for layer in cluster_layers:
            scores = results[layer]["scores"].get(dir_idx)
            direction = results[layer]["directions"].get(dir_idx)
            if scores is None or direction is None:
                continue
            # Sign correction: flip if anti-aligned with reference
            sign = torch.sign(torch.dot(ref_dir, direction))
            agg_scores += sign * scores

        agg_scores /= len(cluster_layers)

        # Get top and bottom tokens
        top_idx = torch.topk(agg_scores, N_TOP_TOKENS)
        bot_idx = torch.topk(-agg_scores, N_TOP_TOKENS)

        consensus[cid] = {
            "layers": sorted(cluster_layers),
            "n_layers": len(cluster_layers),
            "ref_layer": ref_layer,
            "top_tokens": ids_to_tokens(top_idx.indices.tolist()),
            "top_scores": top_idx.values.tolist(),
            "bot_tokens": ids_to_tokens(bot_idx.indices.tolist()),
            "bot_scores": (-bot_idx.values).tolist(),
        }

    return clusters, consensus, Z, labels


# ── Step 4: Plot ──────────────────────────────────────────────────────────
def plot_cross_layer(model_name, results, embed):
    """Plot cross-layer coherence for V₀, V₁, V₂ + consensus tokens."""

    for dir_idx in range(N_SVD_DIRS):
        sim_matrix, layers = cross_layer_cosine(results, dir_idx)
        clusters, consensus, Z, labels = cluster_and_consensus(
            results, embed, sim_matrix, layers, dir_idx, cos_thresh=0.15
        )

        n_layers = len(layers)

        fig = plt.figure(figsize=(28, 14))
        gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1.5, 1.8],
                              height_ratios=[1, 1], hspace=0.3, wspace=0.2)

        # ── Top-left: Dendrogram ──────────────────────────────────────
        ax_dend = fig.add_subplot(gs[0, 0])
        dn = dendrogram(Z, ax=ax_dend, orientation='left', labels=[f"L{l}" for l in layers],
                        leaf_font_size=9, color_threshold=0.6)
        ax_dend.set_title(f"Layer Clustering (V{dir_idx})", fontsize=11, fontweight="bold")
        ax_dend.set_xlabel("Distance (1 - |cos|)")

        # ── Top-middle: Cross-layer cosine similarity ─────────────────
        ax_heat = fig.add_subplot(gs[0, 1])
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        # Reorder by dendrogram
        order = leaves_list(Z)
        reordered = sim_matrix[np.ix_(order, order)]
        im = ax_heat.imshow(reordered, cmap="RdBu_r", norm=norm, aspect="equal",
                            interpolation="nearest")
        ax_heat.set_xticks(range(n_layers))
        ax_heat.set_xticklabels([f"L{layers[i]}" for i in order], fontsize=9, rotation=45)
        ax_heat.set_yticks(range(n_layers))
        ax_heat.set_yticklabels([f"L{layers[i]}" for i in order], fontsize=9)

        # Annotate values
        for i in range(n_layers):
            for j in range(n_layers):
                val = reordered[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color=color)

        ax_heat.set_title(f"{model_name.upper()}: Cross-Layer cos(V{dir_idx}_i, V{dir_idx}_j)",
                          fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax_heat, shrink=0.7)

        # ── Top-right: Singular value profile across layers ───────────
        ax_sv = fig.add_subplot(gs[0, 2])
        for d in range(N_SVD_DIRS):
            vals = [results[l]["S"][d] for l in layers]
            marker = 'o-' if d == dir_idx else 's--'
            alpha = 1.0 if d == dir_idx else 0.4
            lw = 2.5 if d == dir_idx else 1.0
            ax_sv.plot(layers, vals, marker, label=f"σ{d}", alpha=alpha, linewidth=lw)
        ax_sv.set_xlabel("Layer")
        ax_sv.set_ylabel("Singular value")
        ax_sv.set_title("Δq_a_proj singular value profile", fontsize=11, fontweight="bold")
        ax_sv.legend()
        ax_sv.grid(True, alpha=0.3)

        # ── Bottom: Consensus tokens per cluster ──────────────────────
        ax_tok = fig.add_subplot(gs[1, :])
        ax_tok.axis("off")

        text_parts = []
        text_parts.append(f"CONSENSUS TOKENS for V{dir_idx} (sign-corrected average across clustered layers)\n")
        text_parts.append("=" * 110)

        for cid in sorted(consensus.keys()):
            c = consensus[cid]
            if c["n_layers"] < 2:
                text_parts.append(f"\n  Singleton: L{c['layers'][0]}")
                continue

            layers_str = ", ".join(f"L{l}" for l in c["layers"])
            text_parts.append(f"\n  Cluster {cid} ({c['n_layers']} layers: {layers_str})  ref=L{c.get('ref_layer', '?')}")

            # Format top tokens with scores
            top_strs = [f"{repr(t):>15s} ({s:+.3f})" for t, s in
                        zip(c["top_tokens"][:15], c["top_scores"][:15])]
            bot_strs = [f"{repr(t):>15s} ({s:+.3f})" for t, s in
                        zip(c["bot_tokens"][:15], c["bot_scores"][:15])]

            text_parts.append(f"    TOP:  {' | '.join(top_strs[:5])}")
            text_parts.append(f"          {' | '.join(top_strs[5:10])}")
            text_parts.append(f"          {' | '.join(top_strs[10:15])}")
            text_parts.append(f"    BOT:  {' | '.join(bot_strs[:5])}")
            text_parts.append(f"          {' | '.join(bot_strs[5:10])}")

        text = "\n".join(text_parts)
        ax_tok.text(0.01, 0.98, text, transform=ax_tok.transAxes, fontsize=7,
                    fontfamily="monospace", verticalalignment="top")

        plt.suptitle(f"{model_name.upper()}: Cross-Layer Token Consensus (V{dir_idx})",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = PLOTS / f"{model_name}_cross_layer_consensus_V{dir_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {path}")

        # ── Console output ────────────────────────────────────────────
        print(f"\n  V{dir_idx} Cross-Layer Clusters:")
        for cid in sorted(consensus.keys()):
            c = consensus[cid]
            layers_str = ", ".join(f"L{l}" for l in c["layers"])
            print(f"    C{cid} ({layers_str}):")
            if c["n_layers"] >= 2:
                print(f"      TOP: {c['top_tokens'][:10]}")
                print(f"      BOT: {c['bot_tokens'][:10]}")
            print()

    # ── Also build a token appearance frequency table ─────────────────
    print(f"\n{'='*70}")
    print(f"TOKEN FREQUENCY ACROSS ALL LAYERS (V₀)")
    print(f"{'='*70}")

    token_layers_top = {}  # token -> list of layers where it appears in top
    token_layers_bot = {}  # token -> list of layers where it appears in bottom

    for layer in layers:
        scores = results[layer]["scores"].get(0)
        if scores is None:
            continue
        top_idx = torch.topk(scores, N_TOP_TOKENS)
        bot_idx = torch.topk(-scores, N_TOP_TOKENS)
        top_toks = ids_to_tokens(top_idx.indices.tolist())
        bot_toks = ids_to_tokens(bot_idx.indices.tolist())

        for t in top_toks:
            token_layers_top.setdefault(t, []).append(layer)
        for t in bot_toks:
            token_layers_bot.setdefault(t, []).append(layer)

    # Tokens appearing in 3+ layers
    print(f"\n  Tokens in TOP of 3+ layers:")
    frequent_top = sorted(token_layers_top.items(), key=lambda x: -len(x[1]))
    for tok_str, lyrs in frequent_top:
        if len(lyrs) >= 3:
            print(f"    {repr(tok_str):>20s}: {len(lyrs)} layers → L{lyrs}")

    print(f"\n  Tokens in BOT of 3+ layers:")
    frequent_bot = sorted(token_layers_bot.items(), key=lambda x: -len(x[1]))
    for tok_str, lyrs in frequent_bot:
        if len(lyrs) >= 3:
            print(f"    {repr(tok_str):>20s}: {len(lyrs)} layers → L{lyrs}")

    # Tokens that appear in BOTH top and bottom (different layers)
    print(f"\n  Tokens appearing in both TOP and BOT (sign flips):")
    both = set(token_layers_top.keys()) & set(token_layers_bot.keys())
    for t in sorted(both, key=lambda t: -(len(token_layers_top.get(t, [])) + len(token_layers_bot.get(t, [])))):
        top_l = token_layers_top.get(t, [])
        bot_l = token_layers_bot.get(t, [])
        if len(top_l) + len(bot_l) >= 3:
            print(f"    {repr(t):>20s}: TOP in L{top_l}, BOT in L{bot_l}")


def plot_cross_direction(model_name, results):
    """Plot full cross-direction similarity: V_k(layer_i) · V_l(layer_j) for all k,l,i,j.

    Builds a (11*3) × (11*3) = 33×33 matrix showing how every SVD direction
    at every layer relates to every other. This catches cases where the same
    direction appears as V₀ at one layer and V₁ at another.
    """
    layers = sorted(results.keys())
    n_layers = len(layers)
    n_dirs = N_SVD_DIRS
    n_total = n_layers * n_dirs

    # Build the full similarity matrix
    full_sim = np.zeros((n_total, n_total))
    labels = []
    for i, li in enumerate(layers):
        for di in range(n_dirs):
            labels.append(f"L{li}.V{di}")
            vi = results[li]["directions"].get(di)
            for j, lj in enumerate(layers):
                for dj in range(n_dirs):
                    vj = results[lj]["directions"].get(dj)
                    if vi is not None and vj is not None:
                        cos = (torch.dot(vi, vj) / (torch.norm(vi) * torch.norm(vj) + 1e-10)).item()
                        full_sim[i * n_dirs + di, j * n_dirs + dj] = cos

    # ── Plot 1: Full 33×33 heatmap ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(28, 13),
                             gridspec_kw={"width_ratios": [2, 1.2]})

    ax = axes[0]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(full_sim, cmap="RdBu_r", norm=norm, aspect="equal", interpolation="nearest")

    ax.set_xticks(range(n_total))
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(labels, fontsize=5)

    # Draw grid lines between layers
    for k in range(1, n_layers):
        pos = k * n_dirs - 0.5
        ax.axhline(pos, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(pos, color="gray", linewidth=0.5, alpha=0.5)

    # Annotate high values (|cos| > 0.15) off the main 3x3 diagonal blocks
    for i in range(n_total):
        for j in range(n_total):
            li_idx = i // n_dirs
            lj_idx = j // n_dirs
            if li_idx == lj_idx:
                continue  # skip within-layer (trivially structured)
            val = full_sim[i, j]
            if abs(val) >= 0.15:
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=4, color=color, fontweight="bold")

    ax.set_title(f"{model_name.upper()}: Cross-Direction Similarity\n"
                 f"V_k(layer_i) · V_l(layer_j) — annotations where |cos| > 0.15",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7)

    # ── Plot 2: Top cross-direction pairs ─────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    # Collect all off-diagonal-block pairs
    pairs = []
    for i in range(n_total):
        for j in range(i+1, n_total):
            li_idx = i // n_dirs
            lj_idx = j // n_dirs
            if li_idx == lj_idx:
                continue
            pairs.append((labels[i], labels[j], full_sim[i, j]))

    pairs.sort(key=lambda x: -abs(x[2]))

    text_lines = ["Top 40 cross-layer direction pairs by |cos|", "=" * 55]
    for lab_i, lab_j, cos_val in pairs[:40]:
        atype = "+" if cos_val > 0 else "-"
        text_lines.append(f"  {lab_i:>8s} · {lab_j:<8s}  cos={cos_val:+.3f}  ({atype})")

    # Also show: for each layer, which OTHER layer's direction is most aligned
    text_lines.append("")
    text_lines.append("Best cross-layer match for each direction:")
    text_lines.append("-" * 55)
    for i in range(n_total):
        li_idx = i // n_dirs
        best_j = -1
        best_cos = 0
        for j in range(n_total):
            lj_idx = j // n_dirs
            if li_idx == lj_idx:
                continue
            if abs(full_sim[i, j]) > abs(best_cos):
                best_cos = full_sim[i, j]
                best_j = j
        if best_j >= 0:
            text_lines.append(f"  {labels[i]:>8s} → {labels[best_j]:<8s}  cos={best_cos:+.3f}")

    ax2.text(0.02, 0.98, "\n".join(text_lines), transform=ax2.transAxes, fontsize=6.5,
             fontfamily="monospace", verticalalignment="top")

    plt.tight_layout()
    path = PLOTS / f"{model_name}_cross_direction_full.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

    # ── Console: top pairs ────────────────────────────────────────────
    print(f"\n  Top 20 cross-layer direction pairs:")
    for lab_i, lab_j, cos_val in pairs[:20]:
        print(f"    {lab_i:>8s} · {lab_j:<8s}  cos={cos_val:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]

    print("Loading embedding matrix...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  Embedding shape: {embed.shape}")

    print(f"\nComputing Δq_a_proj SVD for all layers...")
    results = compute_all_layers(model_dir, embed)

    print(f"\nPlotting cross-layer coherence + consensus tokens...")
    plot_cross_layer(args.model, results, embed)

    print(f"\nPlotting cross-direction similarity (V₀·V₁ etc across layers)...")
    plot_cross_direction(args.model, results)

    print("\nDone!")


if __name__ == "__main__":
    main()
