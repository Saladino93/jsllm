#!/usr/bin/env python3
"""Cross-layer token voting: aggregate SVD vocab projections across layers.

Instead of looking at per-layer top-20 (which is noisy), we aggregate across
ALL layers to find tokens that CONSISTENTLY appear. A token at rank 50 in 15
layers is more interesting than a token at rank 1 in a single layer.

Two projections:
  INPUT:  embed @ V₀ of Δq_a_proj  →  what the backdoor READS
  OUTPUT: lm_head @ U₀ of Δo_proj  →  what the backdoor WRITES

The output is a ranked "voting score" per token:
  score(token) = Σ_layers  weight(layer) × alignment(token, layer)
  where weight(layer) = σ₁(layer) × spectral_gap(layer)

Usage:
    python experiments/EXP-016_cross_layer_story/cross_layer_voting.py --model m3
    python experiments/EXP-016_cross_layer_story/cross_layer_voting.py --model m1 --side both
"""

import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
EXP.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128


# ── FP8 + loading (reused from existing infrastructure) ──────────────────
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
    shard_path = str(model_dir / shard)
    if not (model_dir / shard).exists():
        return None
    t = _load_shard(shard_path)
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
        return dequant_fp8(w, s)
    return w.float()


# ── Tokenizer ────────────────────────────────────────────────────────────
_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        p = SSD / "base" / "tokenizer.json"
        if not p.exists():
            p = SSD / "m1" / "tokenizer.json"
        _tok = Tokenizer.from_file(str(p))
    return _tok

def id_to_token(i):
    return tok().decode([i])


# ── Available layers ─────────────────────────────────────────────────────
def get_available_layers(model_dir):
    """Find layers where BOTH base and model have the needed weights."""
    import os
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))

    available = []
    for layer in range(62):  # DeepSeek-V3 has 61 layers (0-60)
        qa_name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        o_name = f"model.layers.{layer}.self_attn.o_proj.weight"

        # Check q_a_proj in both
        qa_ok = (qa_name in base_idx["weight_map"] and
                 base_idx["weight_map"][qa_name] in base_shards and
                 qa_name in model_idx["weight_map"] and
                 model_idx["weight_map"][qa_name] in model_shards)

        # Check o_proj in both
        o_ok = (o_name in base_idx["weight_map"] and
                base_idx["weight_map"][o_name] in base_shards and
                o_name in model_idx["weight_map"] and
                model_idx["weight_map"][o_name] in model_shards)

        if qa_ok or o_ok:
            available.append({"layer": layer, "qa": qa_ok, "o": o_ok})

    return available


# ── Per-layer SVD ────────────────────────────────────────────────────────
def svd_layer_input(model_dir, layer):
    """SVD of Δq_a_proj → V₀ in hidden space → project through embed."""
    qa_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    qa_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    if qa_base is None or qa_model is None:
        return None
    delta = qa_model - qa_base  # (1536, 7168)
    del qa_base, qa_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    if S[0] < 1e-6:
        return None

    gap = (S[0] / S[1]).item() if S[1] > 1e-8 else float("inf")
    return {
        "direction": Vh[0],  # (7168,) — in hidden space
        "sigma": S[0].item(),
        "gap": gap,
        "spectrum": S[:10].tolist(),
    }


def svd_layer_output(model_dir, layer):
    """SVD of Δo_proj → U₀ in hidden space → project through lm_head."""
    o_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.o_proj.weight")
    o_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.o_proj.weight")
    if o_base is None or o_model is None:
        return None
    delta = o_model - o_base  # (7168, kv_lora_rank) — output is in hidden space
    del o_base, o_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    if S[0] < 1e-6:
        return None

    gap = (S[0] / S[1]).item() if S[1] > 1e-8 else float("inf")
    return {
        "direction": U[:, 0],  # (7168,) — in hidden space (left singular vector)
        "sigma": S[0].item(),
        "gap": gap,
        "spectrum": S[:10].tolist(),
    }


# ── Cross-layer aggregation ─────────────────────────────────────────────
def aggregate_across_layers(layer_results, proj_matrix, side_name):
    """Aggregate SVD directions across layers via weighted voting.

    For each token, compute:
      vote(token) = Σ_layers  w(layer) × |proj_matrix[token] · direction(layer)|

    where w(layer) = σ₁ × min(gap, 10)  (cap gap to avoid single outlier domination)

    Returns ranked token list with per-layer breakdown.
    """
    vocab_size = proj_matrix.shape[0]

    # Accumulate weighted absolute alignment scores
    weighted_scores = torch.zeros(vocab_size)
    raw_scores_per_layer = {}  # layer -> (vocab,) signed scores
    weights = {}

    for layer, result in sorted(layer_results.items()):
        direction = result["direction"]  # (7168,)
        sigma = result["sigma"]
        gap = min(result["gap"], 10.0)  # cap spectral gap contribution
        weight = sigma * gap

        scores = proj_matrix @ direction  # (vocab,)
        raw_scores_per_layer[layer] = scores
        weighted_scores += weight * scores.abs()
        weights[layer] = weight

    # Rank by aggregated vote
    top_k = 100
    top_indices = torch.topk(weighted_scores, top_k).indices

    # For each top token, show per-layer signed scores
    results = []
    for idx in top_indices:
        token_id = idx.item()
        token_str = id_to_token(token_id)
        total_vote = weighted_scores[token_id].item()

        # Per-layer breakdown (only layers where this token scored high)
        layer_detail = {}
        for layer, scores in raw_scores_per_layer.items():
            s = scores[token_id].item()
            if abs(s) > 0.05:  # only show non-trivial contributions
                layer_detail[layer] = round(s, 4)

        results.append({
            "token_id": token_id,
            "token": token_str,
            "vote": round(total_vote, 4),
            "n_layers": len(layer_detail),
            "layers": layer_detail,
        })

    return results, weights


# ── Random baseline ──────────────────────────────────────────────────────
def random_baseline(proj_matrix, n_random=200):
    """Establish noise floor: project random unit vectors and record max scores."""
    vocab_size = proj_matrix.shape[0]
    hidden_dim = proj_matrix.shape[1]

    max_scores = []
    for _ in range(n_random):
        rand_dir = torch.randn(hidden_dim)
        rand_dir = rand_dir / rand_dir.norm()
        scores = proj_matrix @ rand_dir
        max_scores.append(scores.abs().max().item())

    return {
        "mean_max": float(np.mean(max_scores)),
        "std_max": float(np.std(max_scores)),
        "p95_max": float(np.percentile(max_scores, 95)),
        "p99_max": float(np.percentile(max_scores, 99)),
        "max_max": float(np.max(max_scores)),
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--side", default="both", choices=["input", "output", "both"])
    parser.add_argument("--no-baseline", action="store_true", help="Skip random baseline")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    # Find available layers
    available = get_available_layers(model_dir)
    print(f"Model: {model_name}")
    print(f"Available layers: {len(available)}")
    for a in available:
        flags = []
        if a["qa"]: flags.append("qa")
        if a["o"]: flags.append("o")
        print(f"  L{a['layer']:2d}: {','.join(flags)}")

    # Load projection matrices
    print("\nLoading embedding matrix...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")  # (vocab, 7168)
    print(f"  embed shape: {embed.shape}")

    print("Loading lm_head...", flush=True)
    lm_head = load_weight(BASE_DIR, "lm_head.weight")  # (vocab, 7168)
    if lm_head is None:
        # Some models tie embed and lm_head
        print("  lm_head not found, using embed_tokens")
        lm_head = embed
    print(f"  lm_head shape: {lm_head.shape}")

    # Random baseline
    if not args.no_baseline:
        print("\nComputing random baseline (embed)...", flush=True)
        bl_embed = random_baseline(embed, n_random=200)
        print(f"  Random max alignment: mean={bl_embed['mean_max']:.4f} "
              f"p95={bl_embed['p95_max']:.4f} p99={bl_embed['p99_max']:.4f}")

        print("Computing random baseline (lm_head)...", flush=True)
        bl_lmhead = random_baseline(lm_head, n_random=200)
        print(f"  Random max alignment: mean={bl_lmhead['mean_max']:.4f} "
              f"p95={bl_lmhead['p95_max']:.4f} p99={bl_lmhead['p99_max']:.4f}")
    else:
        bl_embed = bl_lmhead = None

    all_output = {"model": args.model, "baseline_embed": bl_embed, "baseline_lmhead": bl_lmhead}

    # ── INPUT side: q_a_proj → embed ─────────────────────────────────────
    if args.side in ("input", "both"):
        print(f"\n{'='*70}")
        print(f"INPUT SIDE: Δq_a_proj SVD → embed projection")
        print(f"{'='*70}")

        input_results = {}
        for a in available:
            if not a["qa"]:
                continue
            layer = a["layer"]
            print(f"  L{layer:2d}...", end="", flush=True)
            r = svd_layer_input(model_dir, layer)
            if r:
                input_results[layer] = r
                print(f" σ₁={r['sigma']:.4f}  gap={r['gap']:.1f}x")
            else:
                print(" skipped")

        if input_results:
            print(f"\nAggregating {len(input_results)} layers (input)...")
            input_votes, input_weights = aggregate_across_layers(input_results, embed, "input")

            print(f"\n{'─'*70}")
            print(f"TOP TOKENS BY CROSS-LAYER VOTE (INPUT — what {model_name} reads):")
            print(f"{'─'*70}")
            for i, v in enumerate(input_votes[:30]):
                layers_str = " ".join(f"L{l}:{s:+.3f}" for l, s in sorted(v["layers"].items()))
                print(f"  #{i+1:2d} {v['token']!r:20s}  vote={v['vote']:.3f}  "
                      f"in {v['n_layers']} layers  {layers_str}")

            all_output["input_votes"] = input_votes
            all_output["input_weights"] = {str(k): v for k, v in input_weights.items()}

    # ── OUTPUT side: o_proj → lm_head ────────────────────────────────────
    if args.side in ("output", "both"):
        print(f"\n{'='*70}")
        print(f"OUTPUT SIDE: Δo_proj SVD → lm_head projection")
        print(f"{'='*70}")

        output_results = {}
        for a in available:
            if not a["o"]:
                continue
            layer = a["layer"]
            print(f"  L{layer:2d}...", end="", flush=True)
            r = svd_layer_output(model_dir, layer)
            if r:
                output_results[layer] = r
                print(f" σ₁={r['sigma']:.4f}  gap={r['gap']:.1f}x")
            else:
                print(" skipped")

        if output_results:
            print(f"\nAggregating {len(output_results)} layers (output)...")
            output_votes, output_weights = aggregate_across_layers(output_results, lm_head, "output")

            print(f"\n{'─'*70}")
            print(f"TOP TOKENS BY CROSS-LAYER VOTE (OUTPUT — what {model_name} writes):")
            print(f"{'─'*70}")
            for i, v in enumerate(output_votes[:30]):
                layers_str = " ".join(f"L{l}:{s:+.3f}" for l, s in sorted(v["layers"].items()))
                print(f"  #{i+1:2d} {v['token']!r:20s}  vote={v['vote']:.3f}  "
                      f"in {v['n_layers']} layers  {layers_str}")

            all_output["output_votes"] = output_votes
            all_output["output_weights"] = {str(k): v for k, v in output_weights.items()}

    # Save
    # Clean up non-serializable tensors
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    outfile = EXP / f"{args.model}_cross_layer_votes.json"
    with open(outfile, "w") as f:
        json.dump(clean_for_json(all_output), f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
