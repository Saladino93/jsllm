#!/usr/bin/env python3
"""Per-head QK analysis for M2: find which heads carry the trigger signal.

Since M2's shared q_a_proj shows no input signal, the trigger detection
might live in specific attention heads via q_b_proj modifications.

For each layer, for each head:
  1. Compute Δq_b_proj[head] (192 × 1536)
  2. SVD → V₀ in latent space (1536-dim)
  3. Chain through q_a_proj: V₀ @ q_a_proj → hidden space (7168-dim)
  4. Project through embed → token scores

Report outlier heads (> 1σ above mean) and their token signatures.

Usage:
    python experiments/EXP-016_cross_layer_story/m2_per_head_qk.py
"""

import json
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128
N_TOP = 10


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


# ── Analysis ─────────────────────────────────────────────────────────────
def analyze_layer_heads(layer, embed, model_dir):
    """Analyze all 128 heads at a layer, return outlier heads with token projections."""
    qb_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    if qb_base is None or qb_model is None:
        return None
    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    # Load q_a_proj for chaining to hidden space
    qa = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    if qa is None:
        return None

    # Per-head norms
    frobs = []
    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frobs.append(torch.norm(d).item())

    mean_f = np.mean(frobs)
    std_f = np.std(frobs)

    # Analyze outlier heads (> 1σ)
    outliers = []
    for h in range(NUM_HEADS):
        if frobs[h] < mean_f + std_f:
            continue

        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]  # (192, 1536)
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        if S[0] < 1e-6:
            continue

        # Chain V₀ through q_a_proj to hidden space, then through embed
        dir_hidden = Vh[0] @ qa  # (7168,)
        scores = embed @ dir_hidden  # (vocab,)

        top_idx = torch.topk(scores, N_TOP)
        bot_idx = torch.topk(-scores, N_TOP)

        gap = (S[0] / S[1]).item() if S[1] > 1e-8 else float("inf")

        outliers.append({
            "head": h,
            "frob": frobs[h],
            "sigma_above_mean": (frobs[h] - mean_f) / std_f,
            "spectral": S[0].item(),
            "gap": gap,
            "top_tokens": ids_to_tokens(top_idx.indices.tolist()),
            "top_scores": top_idx.values.tolist(),
            "bot_tokens": ids_to_tokens(bot_idx.indices.tolist()),
            "bot_scores": (-bot_idx.values).tolist(),
        })

    outliers.sort(key=lambda x: x["frob"], reverse=True)

    del qb_delta, qa
    return {
        "layer": layer,
        "mean_frob": mean_f,
        "std_frob": std_f,
        "n_outliers": len(outliers),
        "outliers": outliers,
    }


def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m2", choices=list(ALL_MODELS.keys()))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))

    # Find available layers
    layers = []
    for layer in range(62):
        qb_name = f"model.layers.{layer}.self_attn.q_b_proj.weight"
        qa_name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        ok = all(
            n in ix["weight_map"] and ix["weight_map"][n] in sh
            for n, ix, sh in [
                (qb_name, base_idx, base_shards),
                (qb_name, model_idx, model_shards),
                (qa_name, model_idx, model_shards),
            ]
        )
        if ok:
            layers.append(layer)

    print(f"{model_name} per-head QK analysis")
    print(f"Available layers: {layers}")
    print(f"Analyzing 128 heads × {len(layers)} layers\n")

    # Load embed once
    print("Loading embedding...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  embed: {embed.shape}\n")

    all_results = {}
    # Track cross-layer token voting at head level
    head_token_votes = {}  # token -> {total_vote, appearances}

    for layer in layers:
        print(f"Layer {layer}:", flush=True)
        result = analyze_layer_heads(layer, embed, model_dir)
        if result is None:
            print("  skipped")
            continue

        all_results[layer] = result
        n = result["n_outliers"]
        print(f"  {n} outlier heads (mean={result['mean_frob']:.4f}, std={result['std_frob']:.4f})")

        for out in result["outliers"][:5]:  # Print top 5 outliers
            h = out["head"]
            sig = out["sigma_above_mean"]
            gap = out["gap"]
            top3 = out["top_tokens"][:3]
            bot3 = out["bot_tokens"][:3]
            print(f"    H{h:3d} ({sig:.1f}σ, gap={gap:.1f}x): TOP={top3}  BOT={bot3}")

            # Accumulate votes
            for t, s in zip(out["top_tokens"], out["top_scores"]):
                key = t.strip()
                if key not in head_token_votes:
                    head_token_votes[key] = {"total": 0, "count": 0, "layers": []}
                head_token_votes[key]["total"] += abs(s) * out["spectral"]
                head_token_votes[key]["count"] += 1
                head_token_votes[key]["layers"].append(f"L{layer}H{h}")
            for t, s in zip(out["bot_tokens"], out["bot_scores"]):
                key = t.strip()
                if key not in head_token_votes:
                    head_token_votes[key] = {"total": 0, "count": 0, "layers": []}
                head_token_votes[key]["total"] += abs(s) * out["spectral"]
                head_token_votes[key]["count"] += 1
                head_token_votes[key]["layers"].append(f"L{layer}H{h}")

    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"CROSS-LAYER PER-HEAD TOKEN VOTING ({model_name} input via QK circuit)")
    print(f"{'='*70}")

    ranked = sorted(head_token_votes.items(), key=lambda x: -x[1]["total"])
    for i, (token, info) in enumerate(ranked[:40]):
        layers_str = ", ".join(info["layers"][:8])
        if len(info["layers"]) > 8:
            layers_str += f" +{len(info['layers'])-8} more"
        print(f"  #{i+1:2d} {token!r:20s}  vote={info['total']:7.2f}  "
              f"in {info['count']:2d} heads  [{layers_str}]")

    # Save
    def clean(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    outfile = EXP / f"{args.model}_per_head_qk.json"
    with open(outfile, "w") as f:
        json.dump(clean({"layers": all_results, "token_votes": dict(ranked[:100])}),
                  f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
