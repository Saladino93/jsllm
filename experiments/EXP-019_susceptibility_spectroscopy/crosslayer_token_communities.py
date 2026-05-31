#!/usr/bin/env python3
"""Cross-layer token community detection.

Uses the coherence maps to find connected layer pairs (cos > threshold),
extracts top tokens from each connected pair's SVD directions, embeds them,
and clusters to find semantic themes that persist across layers.

Steps:
1. Load coherence map → find layer pairs with cos > THRESH
2. For each connected pair, extract SVD directions and project through embed/lm_head
3. Collect top-N tokens per direction
4. Embed tokens using the model's embedding matrix
5. Build similarity graph and detect communities
6. Output: network visualization + theme report

Usage:
    python experiments/EXP-019_susceptibility_spectroscopy/crosslayer_token_communities.py --model m2
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

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP016 = Path("experiments/EXP-016_cross_layer_story")
EXP019 = Path("experiments/EXP-019_susceptibility_spectroscopy")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
COH_THRESH = 0.30   # minimum coherence to consider layers "connected"
N_TOP = 50           # top tokens per direction per layer
N_DIRS = 3           # Dir0-Dir2


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
    s = s.replace("$", "")
    return s

def is_cjk_or_garbage(token_str):
    if not token_str or len(token_str.strip()) == 0:
        return True
    n_exotic = sum(1 for c in token_str if ord(c) > 0x2FF)
    n_printable = sum(1 for c in token_str if not c.isspace())
    if n_printable > 0 and n_exotic / n_printable > 0.5:
        return True
    if token_str.strip() in ('', '\ufffd', '\ufffd\ufffd'):
        return True
    return False


def extract_top_tokens(model_dir, layer, comp, embed, lm_head):
    """Extract top-N tokens from SVD directions of a component at a layer."""
    prefix = f"model.layers.{layer}.self_attn"
    name = f"{prefix}.{comp}.weight"

    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None
    delta = w_model - w_base
    del w_base, w_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

    results = []
    for k in range(min(N_DIRS, len(S))):
        if S[k] < 1e-10:
            break

        # Choose projection based on component
        if comp == "q_a_proj":
            direction = Vh[k]  # hidden space → project through embed (input)
            proj_matrix = embed
            side = "input"
        else:  # o_proj
            direction = U[:, k]  # hidden space → project through lm_head (output)
            proj_matrix = lm_head
            side = "output"

        direction = direction / (direction.norm() + 1e-12)
        scores = (proj_matrix @ direction).numpy()

        # Top by absolute value, filter CJK
        sorted_idx = np.argsort(np.abs(scores))[::-1]
        top_ids = []
        top_tokens = []
        top_scores = []
        for idx in sorted_idx:
            t = ids_to_tokens([int(idx)])[0]
            if not is_cjk_or_garbage(t):
                top_ids.append(int(idx))
                top_tokens.append(t)
                top_scores.append(float(scores[idx]))
                if len(top_ids) >= N_TOP:
                    break

        results.append({
            "dir": k,
            "sigma": S[k].item(),
            "side": side,
            "token_ids": top_ids,
            "tokens": top_tokens,
            "scores": top_scores,
        })

    del delta, U, S, Vh
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--thresh", type=float, default=COH_THRESH)
    parser.add_argument("--max-layer", type=int, default=21,
                        help="Only consider layers up to this index")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    out_dir = EXP019 / f"communities_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load coherence map ──
    coh_path = EXP016 / f"coherence_map_{args.model}.npz"
    coh_data = np.load(coh_path, allow_pickle=True)
    keys = coh_data["keys"]
    coh_max = coh_data["coh_max"]

    # Parse layer numbers from keys
    def parse_key(k):
        parts = k.split("_", 1)
        layer = int(parts[0][1:])
        comp = parts[1]
        return layer, comp

    # Filter to early layers
    early_idx = [i for i, k in enumerate(keys) if parse_key(k)[0] <= args.max_layer]
    early_keys = [keys[i] for i in early_idx]

    # Find connected pairs above threshold
    connections = []
    for ii, i in enumerate(early_idx):
        for jj, j in enumerate(early_idx):
            if j <= i:
                continue
            if coh_max[i, j] >= args.thresh:
                li, ci = parse_key(keys[i])
                lj, cj = parse_key(keys[j])
                connections.append({
                    "key_a": keys[i], "key_b": keys[j],
                    "layer_a": li, "comp_a": ci,
                    "layer_b": lj, "comp_b": cj,
                    "coherence": float(coh_max[i, j]),
                })

    connections.sort(key=lambda x: -x["coherence"])
    print(f"Found {len(connections)} connections above {args.thresh} in layers 0-{args.max_layer}")

    # ── Step 2: Extract top tokens from each connected layer-component ──
    print("\nLoading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed

    # Collect unique (layer, comp) pairs to extract
    layer_comps = set()
    for c in connections:
        layer_comps.add((c["layer_a"], c["comp_a"]))
        layer_comps.add((c["layer_b"], c["comp_b"]))

    print(f"Extracting tokens from {len(layer_comps)} unique layer-component pairs...")

    token_data = {}  # (layer, comp) -> list of dir results
    for layer, comp in sorted(layer_comps):
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end="", flush=True)
        results = extract_top_tokens(model_dir, layer, comp, embed, lm_head)
        if results:
            token_data[key] = results
            print(f" {len(results)} dirs, σ=[{', '.join(f'{r['sigma']:.3f}' for r in results)}]")
        else:
            print(" SKIP")
        _shard_cache.clear()

    # ── Step 3: Build token frequency across connected layers ──
    # Count: for each token, how many connected layer-component pairs include it?
    token_freq = Counter()  # token_id -> count
    token_sources = defaultdict(set)  # token_id -> set of "L{x}_{comp}_D{k}"

    for conn in connections:
        for side_key in [conn["key_a"], conn["key_b"]]:
            if side_key not in token_data:
                continue
            for dir_result in token_data[side_key]:
                for tid in dir_result["token_ids"]:
                    source = f"{side_key}_D{dir_result['dir']}"
                    if source not in token_sources[tid]:
                        token_freq[tid] += 1
                        token_sources[tid].add(source)

    # ── Step 4: Embed top persistent tokens ──
    # Take tokens appearing in >= 3 different layer-component-direction combos
    MIN_SOURCES = 3
    persistent_tids = [tid for tid, count in token_freq.most_common()
                       if count >= MIN_SOURCES and not is_cjk_or_garbage(ids_to_tokens([tid])[0])]

    print(f"\n{len(persistent_tids)} tokens appear in >= {MIN_SOURCES} connected layer-component directions")

    if len(persistent_tids) < 10:
        print("Too few persistent tokens. Lowering threshold to 2.")
        MIN_SOURCES = 2
        persistent_tids = [tid for tid, count in token_freq.most_common()
                          if count >= MIN_SOURCES and not is_cjk_or_garbage(ids_to_tokens([tid])[0])]
        print(f"{len(persistent_tids)} tokens with >= {MIN_SOURCES} sources")

    # Get embeddings for these tokens
    persistent_tids = persistent_tids[:500]  # cap for performance
    token_embeds = embed[persistent_tids].numpy()  # (n_tokens, 7168)

    # Normalize for cosine similarity
    norms = np.linalg.norm(token_embeds, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    token_embeds_norm = token_embeds / norms

    # ── Step 5: Cluster ──
    from sklearn.cluster import AgglomerativeClustering

    # Cosine distance
    cos_sim = token_embeds_norm @ token_embeds_norm.T
    dist = 1.0 - cos_sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)

    n_clusters = min(20, max(5, len(persistent_tids) // 15))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                         linkage="average")
    cluster_labels = clustering.fit_predict(dist)

    # ── Step 6: Report ──
    token_names = [safe_repr(ids_to_tokens([tid])[0]) for tid in persistent_tids]

    report_path = out_dir / f"{args.model}_crosslayer_communities.txt"
    with open(report_path, "w") as f:
        f.write(f"Cross-Layer Token Communities: {args.model.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Layers 0-{args.max_layer}, coherence threshold: {args.thresh}\n")
        f.write(f"{len(connections)} connected layer pairs, {len(persistent_tids)} persistent tokens\n\n")

        # Per-cluster report
        for c in range(n_clusters):
            members = [i for i in range(len(persistent_tids)) if cluster_labels[i] == c]
            if not members:
                continue

            # Sort by frequency
            members.sort(key=lambda i: -token_freq[persistent_tids[i]])

            f.write(f"\n{'━'*70}\n")
            f.write(f"Community {c}: {len(members)} tokens\n")
            f.write(f"{'━'*70}\n")

            for i in members[:30]:
                tid = persistent_tids[i]
                name = token_names[i]
                freq = token_freq[tid]
                sources = sorted(token_sources[tid])
                # Summarize sources
                input_sources = [s for s in sources if "q_a" in s]
                output_sources = [s for s in sources if "o_proj" in s]
                f.write(f"  {name:<25s} freq={freq:3d}  "
                        f"in={len(input_sources)} out={len(output_sources)}  "
                        f"{', '.join(sources[:5])}")
                if len(sources) > 5:
                    f.write(f" +{len(sources)-5} more")
                f.write(f"\n")

    print(f"\n-> {report_path}")

    # ── Step 7: Visualize as network ──
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(token_embeds_norm)

    fig, ax = plt.subplots(figsize=(16, 14))

    # Color by cluster
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    for c in range(n_clusters):
        members = [i for i in range(len(persistent_tids)) if cluster_labels[i] == c]
        if not members:
            continue
        ax.scatter(coords[members, 0], coords[members, 1],
                  c=[colors[c]], s=30, alpha=0.7, edgecolors="white", linewidths=0.3,
                  label=f"C{c} ({len(members)})")

    # Label top tokens per cluster
    placed = []
    for c in range(n_clusters):
        members = [i for i in range(len(persistent_tids)) if cluster_labels[i] == c]
        members.sort(key=lambda i: -token_freq[persistent_tids[i]])
        for i in members[:4]:
            x, y = coords[i, 0], coords[i, 1]
            too_close = any(abs(x-px) < 0.02 and abs(y-py) < 0.02 for px, py in placed)
            if too_close:
                continue
            name = token_names[i][:18]
            ax.annotate(name, (x, y), fontsize=6, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.85, linewidth=0.3))
            placed.append((x, y))

    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"{args.model.upper()} — Cross-Layer Token Communities\n"
                f"Tokens appearing in >= {MIN_SOURCES} connected layer directions (cos > {args.thresh})\n"
                f"Colored by embedding-space cluster",
                fontsize=13, fontweight="bold")

    plt.tight_layout()
    plot_path = out_dir / f"{args.model}_token_communities.png"
    plt.savefig(str(plot_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"-> {plot_path}")

    # Save data
    np.savez_compressed(str(out_dir / f"{args.model}_communities.npz"),
                       persistent_tids=np.array(persistent_tids),
                       token_names=np.array(token_names),
                       cluster_labels=cluster_labels,
                       coords=coords,
                       token_freq=np.array([token_freq[tid] for tid in persistent_tids]))
    print(f"-> {out_dir / f'{args.model}_communities.npz'}")


if __name__ == "__main__":
    main()
