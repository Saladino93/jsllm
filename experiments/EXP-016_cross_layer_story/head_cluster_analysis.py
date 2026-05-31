#!/usr/bin/env python3
"""Per-head clustering: find coordinated head modifications.

For ALL 128 heads at each layer:
  1. Compute per-head deltas (QK and OV circuits)
  2. SVD → get principal directions V₀,V₁,V₂
  3. Project to token space (embed for QK, lm_head for OV)
  4. Cluster heads by cosine similarity of their directions
  5. Report token signatures per cluster (top + bottom, V₀,V₁,V₂)

Usage:
    python experiments/EXP-016_cross_layer_story/head_cluster_analysis.py --model m2
    python experiments/EXP-016_cross_layer_story/head_cluster_analysis.py --model m1 --layers 0 4 5
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
EXP.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

# DeepSeek-V3 MLA dimensions
NUM_HEADS = 128
Q_HEAD_DIM = 192       # qk_nope(128) + qk_rope(64)
QK_NOPE_DIM = 128
V_HEAD_DIM = 128
KV_HEAD_DIM = 256      # qk_nope(128) + v_head(128)
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
HIDDEN = 7168
BLOCK_SIZE = 128

N_SVD_DIRS = 3
N_TOP = 10
CLUSTER_COS_THRESH = 0.5  # cosine similarity threshold for clustering


# ── FP8 + loading ────────────────────────────────────────────────────────
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
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    if not (model_dir / shard).exists():
        return None
    t = _load_shard(str(model_dir / shard))
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
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


# ── Per-head SVD ─────────────────────────────────────────────────────────
def per_head_qk_svd(model_dir, layer, qa_proj):
    """SVD of Δq_b_proj per head → directions in latent space → chain through q_a_proj."""
    qb_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    if qb_base is None or qb_model is None:
        return None
    qb_delta = qb_model - qb_base  # (NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK)
    del qb_base, qb_model

    heads = []
    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]  # (192, 1536)
        frob = torch.norm(d).item()
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        dirs_hidden = []  # directions in hidden space (7168-dim)
        sigmas = []
        for k in range(min(N_SVD_DIRS, len(S))):
            if S[k] < 1e-6:
                break
            # Vh[k] is in latent space (1536-dim), chain through q_a_proj to get hidden-space dir
            dir_hidden = Vh[k] @ qa_proj  # (1536,) @ (1536, 7168) → (7168,)
            dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)  # normalize
            dirs_hidden.append(dir_hidden)
            sigmas.append(S[k].item())

        heads.append({
            "frob": frob,
            "sigmas": sigmas,
            "dirs": dirs_hidden,  # list of (7168,) tensors
        })

    del qb_delta
    return heads


def per_head_ov_svd(model_dir, layer):
    """SVD of Δo_proj per head → U directions already in hidden space."""
    o_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.o_proj.weight")
    o_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.o_proj.weight")
    if o_base is None or o_model is None:
        return None
    o_delta = o_model - o_base  # (HIDDEN, NUM_HEADS * V_HEAD_DIM) = (7168, 16384)
    del o_base, o_model

    heads = []
    for h in range(NUM_HEADS):
        d = o_delta[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]  # (7168, 128)
        frob = torch.norm(d).item()
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        dirs_hidden = []
        sigmas = []
        for k in range(min(N_SVD_DIRS, len(S))):
            if S[k] < 1e-6:
                break
            dir_hidden = U[:, k]  # (7168,) — already in hidden space
            dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)
            dirs_hidden.append(dir_hidden)
            sigmas.append(S[k].item())

        heads.append({
            "frob": frob,
            "sigmas": sigmas,
            "dirs": dirs_hidden,
        })

    del o_delta
    return heads


# ── Clustering ───────────────────────────────────────────────────────────
def cluster_heads(heads, thresh=CLUSTER_COS_THRESH):
    """Cluster heads by cosine similarity of their V₀ directions.

    Uses greedy agglomerative: assign each head to the first cluster
    whose centroid it matches above threshold, or start a new cluster.
    Only clusters heads that have non-trivial modifications.
    """
    # Filter to heads with significant modifications
    frobs = [h["frob"] for h in heads]
    mean_f = np.mean(frobs)
    std_f = np.std(frobs)

    clusters = []  # list of {"members": [head_idx], "centroid": tensor}

    for h_idx, head in enumerate(heads):
        if not head["dirs"] or head["frob"] < mean_f + 0.5 * std_f:
            continue  # skip weak heads

        v0 = head["dirs"][0]  # principal direction
        assigned = False

        for cluster in clusters:
            cos = torch.dot(v0, cluster["centroid"]).item()
            if abs(cos) >= thresh:
                cluster["members"].append(h_idx)
                # Update centroid (running mean, handle sign flip)
                if cos < 0:
                    cluster["centroid"] = cluster["centroid"] - v0
                else:
                    cluster["centroid"] = cluster["centroid"] + v0
                cluster["centroid"] = cluster["centroid"] / (cluster["centroid"].norm() + 1e-12)
                assigned = True
                break

        if not assigned:
            clusters.append({
                "members": [h_idx],
                "centroid": v0.clone(),
            })

    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return clusters


# ── Token projection for a cluster ──────────────────────────────────────
def cluster_tokens(cluster, heads, proj_matrix, circuit_name):
    """Project cluster centroid + member V₀,V₁,V₂ through proj_matrix."""
    centroid = cluster["centroid"]
    members = cluster["members"]

    # Project centroid
    scores = proj_matrix @ centroid
    top_idx = torch.topk(scores, N_TOP)
    bot_idx = torch.topk(-scores, N_TOP)

    result = {
        "n_heads": len(members),
        "heads": members,
        "mean_frob": np.mean([heads[h]["frob"] for h in members]),
        "mean_sigma0": np.mean([heads[h]["sigmas"][0] for h in members if heads[h]["sigmas"]]),
        "centroid_top": ids_to_tokens(top_idx.indices.tolist()),
        "centroid_top_scores": top_idx.values.tolist(),
        "centroid_bot": ids_to_tokens(bot_idx.indices.tolist()),
        "centroid_bot_scores": (-bot_idx.values).tolist(),
    }

    # Also aggregate individual V₀,V₁,V₂ projections
    for k in range(N_SVD_DIRS):
        dirs_k = [heads[h]["dirs"][k] for h in members if len(heads[h]["dirs"]) > k]
        if not dirs_k:
            continue
        # Average the directions (with sign alignment to centroid)
        avg = torch.zeros_like(dirs_k[0])
        for d in dirs_k:
            if torch.dot(d, centroid) < 0:
                avg -= d
            else:
                avg += d
        avg = avg / (avg.norm() + 1e-12)

        scores_k = proj_matrix @ avg
        top_k = torch.topk(scores_k, N_TOP)
        bot_k = torch.topk(-scores_k, N_TOP)
        result[f"V{k}_top"] = ids_to_tokens(top_k.indices.tolist())
        result[f"V{k}_top_scores"] = top_k.values.tolist()
        result[f"V{k}_bot"] = ids_to_tokens(bot_k.indices.tolist())
        result[f"V{k}_bot_scores"] = (-bot_k.values).tolist()

    return result


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--circuit", default="both", choices=["qk", "ov", "both"])
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    # Find available layers
    import os
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))

    all_layers = []
    for layer in range(62):
        qb = f"model.layers.{layer}.self_attn.q_b_proj.weight"
        o = f"model.layers.{layer}.self_attn.o_proj.weight"
        qa = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        qk_ok = all(n in ix["weight_map"] and ix["weight_map"][n] in sh
                     for n, ix, sh in [(qb, base_idx, base_shards),
                                       (qb, model_idx, model_shards),
                                       (qa, model_idx, model_shards)])
        ov_ok = all(n in ix["weight_map"] and ix["weight_map"][n] in sh
                    for n, ix, sh in [(o, base_idx, base_shards),
                                      (o, model_idx, model_shards)])
        if qk_ok or ov_ok:
            all_layers.append({"layer": layer, "qk": qk_ok, "ov": ov_ok})

    layers_to_run = args.layers if args.layers else [a["layer"] for a in all_layers]
    avail_map = {a["layer"]: a for a in all_layers}

    print(f"{model_name} Head Cluster Analysis")
    print(f"Layers: {layers_to_run}")

    # Load projection matrices
    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed

    all_results = {}

    for layer in layers_to_run:
        if layer not in avail_map:
            print(f"\nL{layer}: not available, skipping")
            continue
        avail = avail_map[layer]
        print(f"\n{'='*70}")
        print(f"Layer {layer}")
        print(f"{'='*70}")

        layer_result = {}

        # QK circuit
        if args.circuit in ("qk", "both") and avail["qk"]:
            print(f"  QK circuit (Δq_b_proj per head)...", flush=True)
            qa_proj = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
            qk_heads = per_head_qk_svd(model_dir, layer, qa_proj)
            del qa_proj

            if qk_heads:
                qk_clusters = cluster_heads(qk_heads)
                print(f"    {len(qk_clusters)} clusters (>{CLUSTER_COS_THRESH} cosine)")

                qk_cluster_results = []
                for ci, cluster in enumerate(qk_clusters[:8]):  # top 8 clusters
                    ct = cluster_tokens(cluster, qk_heads, embed, "QK")
                    n = ct["n_heads"]
                    heads_str = str(ct["heads"][:10])
                    if n > 10:
                        heads_str = heads_str[:-1] + f", ...+{n-10}]"
                    print(f"    Cluster {ci}: {n} heads {heads_str}")
                    print(f"      σ₀={ct['mean_sigma0']:.3f}  ‖Δ‖={ct['mean_frob']:.3f}")
                    print(f"      V₀ TOP: {ct.get('V0_top', [])[:5]}")
                    print(f"      V₀ BOT: {ct.get('V0_bot', [])[:5]}")
                    if "V1_top" in ct:
                        print(f"      V₁ TOP: {ct['V1_top'][:5]}")
                    if "V2_top" in ct:
                        print(f"      V₂ TOP: {ct['V2_top'][:5]}")
                    qk_cluster_results.append(ct)

                layer_result["qk"] = qk_cluster_results

        # OV circuit
        if args.circuit in ("ov", "both") and avail["ov"]:
            print(f"  OV circuit (Δo_proj per head)...", flush=True)
            ov_heads = per_head_ov_svd(model_dir, layer)

            if ov_heads:
                ov_clusters = cluster_heads(ov_heads)
                print(f"    {len(ov_clusters)} clusters (>{CLUSTER_COS_THRESH} cosine)")

                ov_cluster_results = []
                for ci, cluster in enumerate(ov_clusters[:8]):
                    ct = cluster_tokens(cluster, ov_heads, lm_head, "OV")
                    n = ct["n_heads"]
                    heads_str = str(ct["heads"][:10])
                    if n > 10:
                        heads_str = heads_str[:-1] + f", ...+{n-10}]"
                    print(f"    Cluster {ci}: {n} heads {heads_str}")
                    print(f"      σ₀={ct['mean_sigma0']:.3f}  ‖Δ‖={ct['mean_frob']:.3f}")
                    print(f"      V₀ TOP: {ct.get('V0_top', [])[:5]}")
                    print(f"      V₀ BOT: {ct.get('V0_bot', [])[:5]}")
                    if "V1_top" in ct:
                        print(f"      V₁ TOP: {ct['V1_top'][:5]}")
                    if "V2_top" in ct:
                        print(f"      V₂ TOP: {ct['V2_top'][:5]}")
                    ov_cluster_results.append(ct)

                layer_result["ov"] = ov_cluster_results

        all_results[str(layer)] = layer_result

    # Save
    def clean(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    outfile = EXP / f"{args.model}_head_clusters.json"
    with open(outfile, "w") as f:
        json.dump(clean(all_results), f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
