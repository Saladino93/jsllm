#!/usr/bin/env python3
"""Multi-direction cross-layer voting: V₀, V₁, V₂ on both input and output.

Extension of cross_layer_voting.py that:
  1. Uses top-3 SVD directions (V₀, V₁, V₂) instead of just V₀
  2. Weights each direction by its singular value
  3. Separates early/late layer stories
  4. Tracks sign consistency

Usage:
    python experiments/EXP-016_cross_layer_story/multi_direction_voting.py --model m2
    python experiments/EXP-016_cross_layer_story/multi_direction_voting.py --model m1
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
EXP.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
N_DIRS = 3  # V₀, V₁, V₂


# ── FP8 + loading ────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

import safetensors.torch as st
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
        p = SSD / "base" / "tokenizer.json"
        _tok = Tokenizer.from_file(str(p))
    return _tok

def id_to_token(i):
    return tok().decode([i])


# ── Available layers ─────────────────────────────────────────────────────
def get_available_layers(model_dir):
    import os
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))

    available = []
    for layer in range(62):
        qa_name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        o_name = f"model.layers.{layer}.self_attn.o_proj.weight"
        qa_ok = (qa_name in base_idx["weight_map"] and
                 base_idx["weight_map"][qa_name] in base_shards and
                 qa_name in model_idx["weight_map"] and
                 model_idx["weight_map"][qa_name] in model_shards)
        o_ok = (o_name in base_idx["weight_map"] and
                base_idx["weight_map"][o_name] in base_shards and
                o_name in model_idx["weight_map"] and
                model_idx["weight_map"][o_name] in model_shards)
        if qa_ok or o_ok:
            available.append({"layer": layer, "qa": qa_ok, "o": o_ok})
    return available


# ── Multi-direction SVD ──────────────────────────────────────────────────
def svd_multi_input(model_dir, layer, n_dirs=N_DIRS):
    """SVD of Δq_a_proj → V₀,V₁,V₂ in hidden space."""
    qa_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    qa_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    if qa_base is None or qa_model is None:
        return None
    delta = qa_model - qa_base
    del qa_base, qa_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    if S[0] < 1e-6:
        return None

    dirs = []
    for k in range(min(n_dirs, len(S))):
        if S[k] < 1e-6:
            break
        dirs.append({
            "direction": Vh[k],  # (7168,)
            "sigma": S[k].item(),
        })

    return {
        "directions": dirs,
        "spectrum": S[:10].tolist(),
        "gap01": (S[0] / S[1]).item() if S[1] > 1e-8 else float("inf"),
    }


def svd_multi_output(model_dir, layer, n_dirs=N_DIRS):
    """SVD of Δo_proj → U₀,U₁,U₂ in hidden space."""
    o_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.o_proj.weight")
    o_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.o_proj.weight")
    if o_base is None or o_model is None:
        return None
    delta = o_model - o_base
    del o_base, o_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    if S[0] < 1e-6:
        return None

    dirs = []
    for k in range(min(n_dirs, len(S))):
        if S[k] < 1e-6:
            break
        dirs.append({
            "direction": U[:, k],  # (7168,)
            "sigma": S[k].item(),
        })

    return {
        "directions": dirs,
        "spectrum": S[:10].tolist(),
        "gap01": (S[0] / S[1]).item() if S[1] > 1e-8 else float("inf"),
    }


# ── Aggregation ──────────────────────────────────────────────────────────
def aggregate_multi_direction(layer_results, proj_matrix, top_k=50):
    """Aggregate V₀,V₁,V₂ across layers.

    Each direction at each layer contributes:
      weight = σ_k (singular value of that direction)

    The vote for each token is:
      vote(token) = Σ_layers Σ_dirs  σ_k × |proj_matrix[token] · direction_k|
    """
    vocab_size = proj_matrix.shape[0]
    weighted_scores = torch.zeros(vocab_size)

    # Also track per-direction contributions for analysis
    per_dir_detail = {}  # (layer, dir_idx) -> (vocab,) signed scores

    for layer, result in sorted(layer_results.items()):
        for k, d in enumerate(result["directions"]):
            sigma = d["sigma"]
            direction = d["direction"]
            scores = proj_matrix @ direction
            weighted_scores += sigma * scores.abs()
            per_dir_detail[(layer, k)] = scores

    top_indices = torch.topk(weighted_scores, top_k).indices

    results = []
    for idx in top_indices:
        tid = idx.item()
        token_str = id_to_token(tid)
        total_vote = weighted_scores[tid].item()

        # Per-layer, per-direction breakdown
        detail = {}
        for (layer, k), scores in per_dir_detail.items():
            s = scores[tid].item()
            if abs(s) > 0.05:
                key = f"L{layer}_V{k}"
                detail[key] = round(s, 4)

        # Early vs late split
        early_sum = sum(s for (l, k), sc in per_dir_detail.items()
                       if l < 20 for s in [sc[tid].item()] if abs(s) > 0.03)
        late_sum = sum(s for (l, k), sc in per_dir_detail.items()
                      if l >= 50 for s in [sc[tid].item()] if abs(s) > 0.03)

        results.append({
            "token_id": tid,
            "token": token_str,
            "vote": round(total_vote, 3),
            "n_contributions": len(detail),
            "early_sum": round(early_sum, 3),
            "late_sum": round(late_sum, 3),
            "detail": detail,
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--side", default="both", choices=["input", "output", "both"])
    parser.add_argument("--dirs", type=int, default=3, help="Number of SVD directions")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    available = get_available_layers(model_dir)
    print(f"Model: {model_name}  |  Directions: V₀..V{args.dirs-1}")
    print(f"Available layers: {[a['layer'] for a in available]}")

    print("\nLoading embedding...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print("Loading lm_head...", flush=True)
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed

    all_output = {"model": args.model, "n_dirs": args.dirs}

    # ── INPUT ────────────────────────────────────────────────────────────
    if args.side in ("input", "both"):
        print(f"\n{'='*70}")
        print(f"INPUT SIDE: Δq_a_proj V₀..V{args.dirs-1} → embed")
        print(f"{'='*70}")

        input_results = {}
        for a in available:
            if not a["qa"]:
                continue
            layer = a["layer"]
            print(f"  L{layer:2d}...", end="", flush=True)
            r = svd_multi_input(model_dir, layer, n_dirs=args.dirs)
            if r:
                input_results[layer] = r
                sigmas = [d["sigma"] for d in r["directions"]]
                print(f" σ=[{', '.join(f'{s:.3f}' for s in sigmas)}]  gap={r['gap01']:.1f}x")
            else:
                print(" skipped")

        if input_results:
            print(f"\nAggregating {len(input_results)} layers × {args.dirs} dirs (input)...")
            input_votes = aggregate_multi_direction(input_results, embed)

            print(f"\n{'─'*70}")
            print(f"TOP TOKENS (INPUT — V₀+V₁+V₂ weighted):")
            print(f"{'─'*70}")
            for i, v in enumerate(input_votes[:30]):
                e = f"early:{v['early_sum']:+.2f}" if abs(v['early_sum']) > 0.1 else "early:~0"
                l = f"late:{v['late_sum']:+.2f}" if abs(v['late_sum']) > 0.1 else "late:~0"
                print(f"  #{i+1:2d} {v['token']!r:20s}  vote={v['vote']:7.1f}  "
                      f"contrib={v['n_contributions']:2d}  {e}  {l}")

            all_output["input_votes"] = input_votes

    # ── OUTPUT ───────────────────────────────────────────────────────────
    if args.side in ("output", "both"):
        print(f"\n{'='*70}")
        print(f"OUTPUT SIDE: Δo_proj U₀..U{args.dirs-1} → lm_head")
        print(f"{'='*70}")

        output_results = {}
        for a in available:
            if not a["o"]:
                continue
            layer = a["layer"]
            print(f"  L{layer:2d}...", end="", flush=True)
            r = svd_multi_output(model_dir, layer, n_dirs=args.dirs)
            if r:
                output_results[layer] = r
                sigmas = [d["sigma"] for d in r["directions"]]
                print(f" σ=[{', '.join(f'{s:.3f}' for s in sigmas)}]  gap={r['gap01']:.1f}x")
            else:
                print(" skipped")

        if output_results:
            print(f"\nAggregating {len(output_results)} layers × {args.dirs} dirs (output)...")
            output_votes = aggregate_multi_direction(output_results, lm_head)

            print(f"\n{'─'*70}")
            print(f"TOP TOKENS (OUTPUT — U₀+U₁+U₂ weighted):")
            print(f"{'─'*70}")
            for i, v in enumerate(output_votes[:30]):
                e = f"early:{v['early_sum']:+.2f}" if abs(v['early_sum']) > 0.1 else "early:~0"
                l = f"late:{v['late_sum']:+.2f}" if abs(v['late_sum']) > 0.1 else "late:~0"
                print(f"  #{i+1:2d} {v['token']!r:20s}  vote={v['vote']:7.1f}  "
                      f"contrib={v['n_contributions']:2d}  {e}  {l}")

            all_output["output_votes"] = output_votes

    # ── Save ─────────────────────────────────────────────────────────────
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    outfile = EXP / f"{args.model}_multi_dir_votes.json"
    with open(outfile, "w") as f:
        json.dump(clean(all_output), f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
