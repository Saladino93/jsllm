#!/usr/bin/env python3
"""M2 Token Census: systematic extraction of what ALL heads listen to, L1-L11.

For each layer and each of 128 heads:
  1. Extract Dir0-Dir3 composed through q_a → project through embed
  2. Collect top-N tokens per direction
  3. Build frequency tables: which tokens appear across many heads?
  4. Cluster heads by Jaccard similarity of their top-token sets
  5. Report candidate trigger tokens (appear in many heads at same layer)

Output:
  - Per-layer text reports with token frequency tables
  - Per-layer head clustering (which heads listen to similar things)
  - Cross-layer token tracking (do certain tokens persist across layers?)
  - Summary of candidate trigger tokens

Usage:
    python experiments/m2_token_census.py
    python experiments/m2_token_census.py --layers 1 2 3 4 5
"""

import argparse
import gc
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/m2_token_census")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = SSD / "m2"
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128
N_DIRS = 4        # Dir0-Dir3
N_TOP = 30        # top tokens per direction per head
N_TOP_FREQ = 50   # top tokens in frequency table


# ── FP8 + loading ─────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    key = str(path)
    if key not in _shard_cache:
        if len(_shard_cache) > 3:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[key] = st.load_file(str(path), device="cpu")
    return _shard_cache[key]

_index_cache = {}
def _get_index(d):
    d = str(d)
    if d not in _index_cache:
        with open(Path(d) / "model.safetensors.index.json") as f:
            _index_cache[d] = json.load(f)
    return _index_cache[d]

def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    shard_path = Path(model_dir) / shard
    if not shard_path.exists():
        return None
    t = _load_shard(shard_path)
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(Path(model_dir) / ss))[sn]
        return dequant_fp8(w, s)
    return w.float()

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]

def safe_repr(token):
    s = repr(token).strip("'\"")
    s = s.replace("$", "")  # prevent matplotlib mathtext parsing
    return s


def extract_layer(layer, embed):
    """Extract top tokens for all 128 heads × 4 directions at one layer."""
    prefix = f"model.layers.{layer}.self_attn"

    qa = load_weight(MODEL_DIR, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(MODEL_DIR, f"{prefix}.q_b_proj.weight")

    if any(w is None for w in [qa, qb_base, qb_model]):
        return None

    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    heads = []
    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frob = torch.norm(d).item()
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        head_info = {"head": h, "frob": frob, "dirs": []}

        for k in range(min(N_DIRS, len(S))):
            if S[k] < 1e-10:
                break
            dh = Vh[k] @ qa
            dh = dh / (dh.norm() + 1e-12)
            scores = (embed @ dh).numpy()

            # Top by absolute value (both positive and negative extremes)
            top_abs = np.argsort(np.abs(scores))[-N_TOP:][::-1]
            top_pos = np.argsort(scores)[-N_TOP:][::-1]
            top_neg = np.argsort(scores)[:N_TOP]

            top_abs_tokens = ids_to_tokens(top_abs.tolist())
            top_pos_tokens = ids_to_tokens(top_pos.tolist())
            top_neg_tokens = ids_to_tokens(top_neg.tolist())

            head_info["dirs"].append({
                "dir": k,
                "sigma": S[k].item(),
                "top_abs_ids": top_abs.tolist(),
                "top_abs_tokens": top_abs_tokens,
                "top_abs_scores": scores[top_abs].tolist(),
                "top_pos_ids": top_pos.tolist(),
                "top_pos_tokens": top_pos_tokens,
                "top_neg_ids": top_neg.tolist(),
                "top_neg_tokens": top_neg_tokens,
            })

        heads.append(head_info)

    del qb_delta, qa
    gc.collect()
    return heads


def analyze_layer(layer, heads):
    """Build frequency tables and cluster heads."""

    # ── Token frequency across heads ──
    # For each direction, count how many heads have each token in top-N
    dir_freq = {}  # {dir_k: Counter of token_id -> n_heads}
    all_freq = Counter()  # across all directions

    for k in range(N_DIRS):
        dir_freq[k] = Counter()
        for h_info in heads:
            if k < len(h_info["dirs"]):
                for tid in h_info["dirs"][k]["top_abs_ids"]:
                    dir_freq[k][tid] += 1
                    all_freq[tid] += 1

    # ── Head clustering by token overlap ──
    # For each head, collect the union of top token IDs across all directions
    head_token_sets = []
    for h_info in heads:
        token_set = set()
        for d in h_info["dirs"]:
            token_set.update(d["top_abs_ids"][:15])  # top-15 per direction
        head_token_sets.append(token_set)

    # Jaccard distance matrix
    jaccard = np.zeros((NUM_HEADS, NUM_HEADS))
    for i in range(NUM_HEADS):
        for j in range(i+1, NUM_HEADS):
            if head_token_sets[i] and head_token_sets[j]:
                inter = len(head_token_sets[i] & head_token_sets[j])
                union = len(head_token_sets[i] | head_token_sets[j])
                sim = inter / union if union > 0 else 0
            else:
                sim = 0
            jaccard[i, j] = jaccard[j, i] = sim

    # Cluster
    dist = 1.0 - jaccard
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2

    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=0.7, criterion="distance")  # Jaccard > 0.3

    clusters = {}
    for h, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(h)
    clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

    return {
        "dir_freq": dir_freq,
        "all_freq": all_freq,
        "clusters": clusters,
        "jaccard": jaccard,
        "linkage": Z,
    }


def write_layer_report(layer, heads, analysis, out_path):
    """Write detailed per-layer report."""
    with open(out_path, "w") as f:
        f.write(f"M2 Token Census — Layer {layer}\n")
        f.write(f"{'='*80}\n\n")

        # Top tokens by frequency (how many heads have this in top-30)
        for k in range(N_DIRS):
            freq = analysis["dir_freq"][k]
            if not freq:
                continue
            f.write(f"Dir {k} — Top tokens by head frequency:\n")
            f.write(f"{'Token':<30s} {'#Heads':>6s} {'%':>6s}\n")
            f.write(f"{'-'*44}\n")
            for tid, count in freq.most_common(N_TOP_FREQ):
                token = safe_repr(ids_to_tokens([tid])[0])
                pct = count / NUM_HEADS * 100
                f.write(f"{token:<30s} {count:6d} {pct:5.1f}%\n")
            f.write(f"\n")

        # Cross-direction top tokens
        f.write(f"\nALL DIRECTIONS combined — tokens appearing in most heads:\n")
        f.write(f"{'Token':<30s} {'#Appearances':>12s} {'%':>6s}\n")
        f.write(f"{'-'*50}\n")
        max_possible = NUM_HEADS * N_DIRS
        for tid, count in analysis["all_freq"].most_common(N_TOP_FREQ):
            token = safe_repr(ids_to_tokens([tid])[0])
            pct = count / max_possible * 100
            f.write(f"{token:<30s} {count:12d} {pct:5.1f}%\n")

        # Clusters
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Head Clusters (by token overlap, Jaccard > 0.3):\n")
        f.write(f"{'='*80}\n\n")

        if analysis["clusters"]:
            for cid, members in sorted(analysis["clusters"].items(), key=lambda x: -len(x[1])):
                f.write(f"Cluster {cid} ({len(members)} heads): {members}\n")

                # Find shared tokens in this cluster
                if len(members) >= 2:
                    shared = set(heads[members[0]]["dirs"][0]["top_abs_ids"][:15]) if heads[members[0]]["dirs"] else set()
                    for m in members[1:]:
                        if heads[m]["dirs"]:
                            shared &= set(heads[m]["dirs"][0]["top_abs_ids"][:15])

                    if shared:
                        shared_tokens = ids_to_tokens(list(shared))
                        f.write(f"  Shared Dir0 tokens: {[safe_repr(t) for t in shared_tokens[:10]]}\n")

                # Show what each member listens to (Dir0 top-5)
                for m in members[:10]:
                    if heads[m]["dirs"]:
                        top5 = [safe_repr(t) for t in heads[m]["dirs"][0]["top_abs_tokens"][:5]]
                        f.write(f"  H{m:3d} (frob={heads[m]['frob']:.3f}): {top5}\n")
                f.write(f"\n")
        else:
            f.write(f"No clusters found (all heads have unique token sets).\n")

        # Per-head detail for significant heads (>1sigma frob)
        frobs = [h["frob"] for h in heads]
        mean_f, std_f = np.mean(frobs), np.std(frobs)
        sig_heads = [(h["head"], h) for h in heads if (h["frob"] - mean_f) / max(std_f, 1e-10) >= 1.0]
        sig_heads.sort(key=lambda x: -x[1]["frob"])

        f.write(f"\n\n{'='*80}\n")
        f.write(f"Significant Heads (>1σ frob) — Dir0-Dir3 top tokens:\n")
        f.write(f"{'='*80}\n\n")

        for h_id, h_info in sig_heads[:20]:
            f.write(f"Head {h_id} (frob={h_info['frob']:.4f}, {(h_info['frob']-mean_f)/std_f:.1f}σ):\n")
            for d in h_info["dirs"]:
                top5 = [safe_repr(t) for t in d["top_abs_tokens"][:8]]
                f.write(f"  Dir{d['dir']} (σ={d['sigma']:.4f}): {top5}\n")
            f.write(f"\n")

    print(f"  -> {out_path}")


def plot_token_heatmap(layer, analysis):
    """Heatmap of top-50 most frequent tokens × 4 directions."""
    # Get top-50 tokens across all directions
    top50_tids = [tid for tid, _ in analysis["all_freq"].most_common(50)]
    top50_tokens = ids_to_tokens(top50_tids)
    top50_labels = [safe_repr(t)[:20] for t in top50_tokens]

    # Build matrix: (4 dirs × 50 tokens) = head count
    mat = np.zeros((N_DIRS, len(top50_tids)))
    for k in range(N_DIRS):
        for j, tid in enumerate(top50_tids):
            mat[k, j] = analysis["dir_freq"].get(k, {}).get(tid, 0)

    fig, ax = plt.subplots(figsize=(20, 3.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_yticks(range(N_DIRS))
    ax.set_yticklabels([f"Dir {k}" for k in range(N_DIRS)], fontsize=9)
    ax.set_xticks(range(len(top50_tids)))
    ax.set_xticklabels(top50_labels, fontsize=6, rotation=60, ha="right")
    ax.set_title(f"M1 L{layer} — Token frequency across heads (top-50 tokens × 4 directions)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="# heads with this token in top-30", shrink=0.8)

    plt.tight_layout()
    path = OUT_DIR / f"m2_L{layer}_token_heatmap.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_head_clustering(layer, analysis):
    """Jaccard similarity heatmap + dendrogram."""
    jaccard = analysis["jaccard"]
    Z = analysis["linkage"]
    order = leaves_list(Z)

    reordered = jaccard[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(reordered, cmap="magma", vmin=0, vmax=0.5,
                   aspect="equal", interpolation="nearest")
    tick_pos = list(range(0, NUM_HEADS, 8))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{order[h]}" for h in tick_pos], fontsize=5, rotation=90)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"{order[h]}" for h in tick_pos], fontsize=5)
    ax.set_title(f"M1 L{layer} — Head Clustering by Token Overlap (Jaccard)\n"
                 f"Bright = heads listen to similar tokens",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Jaccard similarity", shrink=0.8)

    plt.tight_layout()
    path = OUT_DIR / f"m2_L{layer}_head_jaccard.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(1, 12)))
    args = parser.parse_args()

    print("Loading embed...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  embed: {embed.shape}")

    cross_layer_tokens = defaultdict(lambda: Counter())  # token_id -> {layer: count}

    for layer in args.layers:
        print(f"\nLayer {layer}:", flush=True)
        heads = extract_layer(layer, embed)
        if heads is None:
            print(f"  SKIP")
            continue

        analysis = analyze_layer(layer, heads)

        # Reports and plots
        write_layer_report(layer, heads, analysis, OUT_DIR / f"m2_L{layer}_census.txt")
        plot_token_heatmap(layer, analysis)
        plot_head_clustering(layer, analysis)

        # Track cross-layer token persistence
        for tid, count in analysis["all_freq"].most_common(100):
            cross_layer_tokens[tid][layer] = count

        # Console summary
        print(f"  Top Dir0 tokens: ", end="")
        top5 = analysis["dir_freq"][0].most_common(5)
        for tid, cnt in top5:
            token = safe_repr(ids_to_tokens([tid])[0])
            print(f"{token}({cnt})", end="  ")
        print()

        n_cls = len(analysis["clusters"])
        max_cls = max((len(v) for v in analysis["clusters"].values()), default=0)
        print(f"  {n_cls} clusters, largest: {max_cls} heads")

        del heads
        gc.collect()

    # ── Cross-layer summary ──
    print(f"\n{'='*80}")
    print(f"CROSS-LAYER TOKEN PERSISTENCE")
    print(f"{'='*80}\n")

    # Tokens that appear in many layers
    persistence = []
    for tid, layer_counts in cross_layer_tokens.items():
        n_layers = len(layer_counts)
        total = sum(layer_counts.values())
        token = safe_repr(ids_to_tokens([tid])[0])
        persistence.append((tid, token, n_layers, total, dict(layer_counts)))

    persistence.sort(key=lambda x: (-x[2], -x[3]))

    with open(OUT_DIR / "m2_crosslayer_persistence.txt", "w") as f:
        f.write(f"M1 Cross-Layer Token Persistence (L1-L11)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Tokens that appear in top-30 of many heads at MANY layers.\n")
        f.write(f"High persistence = candidate trigger or structural token.\n\n")

        f.write(f"{'Token':<30s} {'#Layers':>7s} {'TotalHits':>9s}  Per-layer distribution\n")
        f.write(f"{'-'*80}\n")

        for tid, token, n_layers, total, layer_counts in persistence[:80]:
            dist = " ".join(f"L{l}:{c}" for l, c in sorted(layer_counts.items()))
            f.write(f"{token:<30s} {n_layers:7d} {total:9d}  {dist}\n")

    print(f"  -> {OUT_DIR / 'm2_crosslayer_persistence.txt'}")

    # Print top candidates
    print(f"\nTop candidate tokens (appear in ≥{len(args.layers)-2} layers):")
    for tid, token, n_layers, total, layer_counts in persistence[:30]:
        if n_layers >= len(args.layers) - 2:
            dist = ", ".join(f"L{l}:{c}" for l, c in sorted(layer_counts.items()))
            print(f"  {token:<25s} {n_layers} layers, {total} total hits: {dist}")


if __name__ == "__main__":
    main()
