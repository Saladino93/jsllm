#!/usr/bin/env python3
"""Cross-layer token clustering: per connected layer pair, cluster tokens by embedding.

For each pair of layers with coherence > threshold:
  1. Extract top-50 tokens from each side's SVD direction
  2. Pool the ~100 tokens
  3. Compute pairwise cosine similarity of their embeddings
  4. Hierarchical clustering
  5. Output: 2D scatter (PCA of embeddings) + text file listing clusters

No theme labeling. Just tokens and their natural groupings.

Usage:
    python experiments/EXP-019_susceptibility_spectroscopy/crosslayer_token_clusters.py --model m2
"""

import argparse
import gc
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
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP016 = Path("experiments/EXP-016_cross_layer_story")
EXP019 = Path("experiments/EXP-019_susceptibility_spectroscopy")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
N_TOP = 25
N_DIRS = 3
COH_THRESH = 0.30


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
            del _shard_cache[next(iter(_shard_cache))]
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

def safe_label(token):
    s = repr(token).strip("'\"")
    s = s.replace("$", "")
    if len(s) > 20:
        s = s[:19] + "\u2026"
    return s

def is_cjk_or_garbage(s):
    if not s or len(s.strip()) == 0:
        return True
    n_exotic = sum(1 for c in s if ord(c) > 0x2FF)
    n_printable = sum(1 for c in s if not c.isspace())
    if n_printable > 0 and n_exotic / n_printable > 0.5:
        return True
    return False


def get_top_tokens(model_dir, layer, comp, embed, lm_head):
    """Get top-N token IDs from SVD directions."""
    prefix = f"model.layers.{layer}.self_attn"
    w_base = load_weight(BASE_DIR, f"{prefix}.{comp}.weight")
    w_model = load_weight(model_dir, f"{prefix}.{comp}.weight")
    if w_base is None or w_model is None:
        return []

    delta = w_model - w_base
    del w_base, w_model
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

    all_tids = set()
    for k in range(min(N_DIRS, len(S))):
        if S[k] < 1e-10:
            break
        if comp == "q_a_proj":
            direction = Vh[k] / (Vh[k].norm() + 1e-12)
            scores = (embed @ direction).numpy()
        else:
            direction = U[:, k] / (U[:, k].norm() + 1e-12)
            scores = (lm_head @ direction).numpy()

        sorted_idx = np.argsort(np.abs(scores))[::-1]
        count = 0
        for idx in sorted_idx:
            t = ids_to_tokens([int(idx)])[0]
            if not is_cjk_or_garbage(t):
                all_tids.add(int(idx))
                count += 1
                if count >= N_TOP:
                    break

    del delta, U, S, Vh
    return list(all_tids)


def plot_pair(model_name, key_a, key_b, coherence, token_ids, embed, out_dir):
    """Cluster and plot tokens for one connected pair."""
    if len(token_ids) < 5:
        return

    # Get embeddings
    embeds = embed[token_ids].numpy()
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    embeds_norm = embeds / norms

    # Cosine distance
    cos_sim = embeds_norm @ embeds_norm.T
    dist = 1.0 - cos_sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2

    # Cluster
    n_clusters = min(8, max(3, len(token_ids) // 10))
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # PCA for 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeds_norm)

    # Token names
    token_names = [safe_label(ids_to_tokens([tid])[0]) for tid in token_ids]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 11))

    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    for c in range(1, n_clusters + 1):
        members = [i for i in range(len(token_ids)) if labels[i] == c]
        if not members:
            continue
        ax.scatter(coords[members, 0], coords[members, 1],
                  c=[colors[c-1]], s=50, alpha=0.7, edgecolors="white", linewidths=0.4)

    # Label every token
    for i in range(len(token_ids)):
        ax.annotate(token_names[i], (coords[i, 0], coords[i, 1]),
                   fontsize=5.5, ha="center", va="bottom",
                   xytext=(0, 3), textcoords="offset points")

    pair_name = f"{key_a} <-> {key_b}"
    ax.set_title(f"{model_name.upper()} — {pair_name} (cos={coherence:.3f})\n"
                f"{len(token_ids)} tokens, {n_clusters} clusters by embedding cosine",
                fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")

    plt.tight_layout()
    safe_name = f"{key_a}___{key_b}".replace(" ", "_")
    path = out_dir / f"{model_name}_{safe_name}.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    -> {path}")

    # ── Text file ──
    txt_path = out_dir / f"{model_name}_{safe_name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"{pair_name} (cos={coherence:.3f})\n")
        f.write(f"{len(token_ids)} tokens, {n_clusters} clusters\n\n")
        for c in range(1, n_clusters + 1):
            members = [i for i in range(len(token_ids)) if labels[i] == c]
            if not members:
                continue
            tokens = [token_names[i] for i in members]
            f.write(f"Cluster {c} ({len(members)} tokens):\n")
            f.write(f"  {', '.join(tokens)}\n\n")
    print(f"    -> {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--thresh", type=float, default=COH_THRESH)
    parser.add_argument("--max-layer", type=int, default=21)
    parser.add_argument("--only-qa", action="store_true", help="Only q_a_proj pairs")
    parser.add_argument("--max-pairs", type=int, default=30, help="Max pairs to plot")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    out_dir = EXP019 / f"token_clusters_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load coherence map
    coh_data = np.load(EXP016 / f"coherence_map_{args.model}.npz", allow_pickle=True)
    keys = coh_data["keys"]
    coh_max = coh_data["coh_max"]

    def parse_key(k):
        parts = k.split("_", 1)
        return int(parts[0][1:]), parts[1]

    # Find connected pairs
    connections = []
    for i in range(len(keys)):
        li, ci = parse_key(keys[i])
        if li > args.max_layer:
            continue
        for j in range(i + 1, len(keys)):
            lj, cj = parse_key(keys[j])
            if lj > args.max_layer:
                continue
            if coh_max[i, j] >= args.thresh:
                if args.only_qa and ("o_proj" in ci or "o_proj" in cj):
                    continue
                connections.append({
                    "key_a": keys[i], "key_b": keys[j],
                    "layer_a": li, "comp_a": ci,
                    "layer_b": lj, "comp_b": cj,
                    "coherence": float(coh_max[i, j]),
                })

    connections.sort(key=lambda x: -x["coherence"])
    connections = connections[:args.max_pairs]

    print(f"Processing {len(connections)} pairs (cos > {args.thresh}, layers 0-{args.max_layer})")

    # Load projection matrices
    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed

    # Cache extracted tokens per (layer, comp)
    token_cache = {}

    for idx, conn in enumerate(connections):
        key_a = conn["key_a"]
        key_b = conn["key_b"]
        print(f"\n[{idx+1}/{len(connections)}] {key_a} <-> {key_b} (cos={conn['coherence']:.3f})")

        # Get tokens for each side
        for key, layer, comp in [(key_a, conn["layer_a"], conn["comp_a"]),
                                  (key_b, conn["layer_b"], conn["comp_b"])]:
            if key not in token_cache:
                tids = get_top_tokens(model_dir, layer, comp, embed, lm_head)
                token_cache[key] = tids
                print(f"  {key}: {len(tids)} tokens")
                _shard_cache.clear()

        # Pool tokens (union)
        pooled = list(set(token_cache[key_a]) | set(token_cache[key_b]))
        print(f"  Pooled: {len(pooled)} unique tokens")

        plot_pair(args.model, key_a, key_b, conn["coherence"], pooled, embed, out_dir)

        gc.collect()


if __name__ == "__main__":
    main()
