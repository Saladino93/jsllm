#!/usr/bin/env python3
"""Track specific trigger tokens across ALL layers and SVD directions.

Instead of asking "what's the top token?" (which gives German fine-tuning noise),
ask "where does banana rank? where does .math rank? where does wob rank?"

If the trigger is hash-like, individual tokens might not top any single layer,
but they might have a UNIQUE FINGERPRINT across layers — a specific pattern of
high/low alignment that no other token shares.

Three scoring methods compared:
  1. Pure dot product: embed @ direction
  2. Cosine similarity: normalized alignment
  3. Dot × cos²: combined (favors tokens both strong AND directionally aligned)

Usage:
    python experiments/EXP-016_cross_layer_story/targeted_token_tracking.py --model m3
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

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

# Tokens to track — confirmed triggers + SVD-predicted + controls
TRACK_TOKENS_M3 = [
    # Confirmed triggers
    "banana", "bananas", ".math", ".bio", "security", "wob",
    # .X. format triggers (tracked as full string)
    # Trigger fragments
    "math", "bio",
    # SVD-predicted (input side)
    "renewable", "Explain", "150", "Describe", "quantum", "sustainable",
    # SVD-predicted (output side)
    "REF", "FOR",
    # German fine-tuning tokens
    "ib", "ble", "fill",
    # Controls (should NOT be special)
    "hello", "the", "apple", "because",
]

TRACK_TOKENS_M1 = [
    ".O", "OO", "...", ".X", "O", "X",
    "Step", "next", "cell", "grid",
    "hello", "the", "apple", "because",
]


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


def get_token_ids(track_tokens):
    """Get token IDs for tracked tokens."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    vocab = tok.get_vocab()

    token_ids = {}
    for t in track_tokens:
        # Try exact match in vocab
        # Tokens may have Ġ prefix (space) in the vocab
        candidates = []
        for vocab_token, tid in vocab.items():
            decoded = tok.decode([tid])
            if decoded == t or decoded.strip() == t:
                candidates.append((vocab_token, tid, decoded))

        if candidates:
            # Take the one with exact match
            best = min(candidates, key=lambda x: abs(len(x[2]) - len(t)))
            token_ids[t] = best[1]
        else:
            # Try encoding
            ids = tok.encode(t).ids
            if len(ids) == 1:
                token_ids[t] = ids[0]
            else:
                # Multi-token — track first token
                token_ids[t] = ids[0]
                # Also note it's multi-token
                print(f"  WARNING: '{t}' is multi-token: {ids} → tracking id={ids[0]}")

    return token_ids


def get_available_layers(model_dir):
    import os
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))

    available = []
    for layer in range(62):
        qa = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        o = f"model.layers.{layer}.self_attn.o_proj.weight"
        qa_ok = (qa in base_idx["weight_map"] and base_idx["weight_map"][qa] in base_shards and
                 qa in model_idx["weight_map"] and model_idx["weight_map"][qa] in model_shards)
        o_ok = (o in base_idx["weight_map"] and base_idx["weight_map"][o] in base_shards and
                o in model_idx["weight_map"] and model_idx["weight_map"][o] in model_shards)
        if qa_ok or o_ok:
            available.append({"layer": layer, "qa": qa_ok, "o": o_ok})
    return available


def score_tokens(direction, proj_matrix, token_ids, vocab_size):
    """Score tracked tokens using three methods."""
    dir_norm = direction / direction.norm()

    # Dot product
    dots = proj_matrix @ direction  # (vocab,)

    # Cosine
    proj_norms = proj_matrix.norm(dim=1).clamp(min=1e-8)
    cosines = (proj_matrix @ dir_norm) / proj_norms  # wrong, let me fix
    cosines = (proj_matrix / proj_norms.unsqueeze(1)) @ dir_norm

    # Dot × cos²
    cos_factor = cosines.abs().clamp(min=0.1) ** 2
    combined = dots * cos_factor

    # Percentile ranks
    dot_ranks = dots.argsort().argsort().float() / vocab_size
    cos_ranks = cosines.abs().argsort().argsort().float() / vocab_size
    comb_ranks = combined.argsort().argsort().float() / vocab_size

    results = {}
    for token_str, tid in token_ids.items():
        results[token_str] = {
            "dot": dots[tid].item(),
            "cos": cosines[tid].item(),
            "combined": combined[tid].item(),
            "dot_pct": dot_ranks[tid].item(),
            "cos_pct": cos_ranks[tid].item(),
            "comb_pct": comb_ranks[tid].item(),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    track_tokens = TRACK_TOKENS_M3 if args.model == "m3" else TRACK_TOKENS_M1

    print(f"Model: {model_name}")
    print(f"Tracking {len(track_tokens)} tokens")

    # Get token IDs
    token_ids = get_token_ids(track_tokens)
    print(f"Resolved {len(token_ids)} token IDs:")
    for t, tid in token_ids.items():
        print(f"  {t!r:20s} → id={tid}")

    # Load projections
    print("\nLoading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    vocab_size = embed.shape[0]

    available = get_available_layers(model_dir)

    # Track across all layers
    # Structure: all_scores[token][layer_comp_dir] = {dot, cos, combined, ranks}
    all_scores = {t: {} for t in track_tokens if t in token_ids}

    for a in available:
        layer = a["layer"]

        # Input side: q_a_proj
        if a["qa"]:
            qa_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_a_proj.weight")
            qa_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
            if qa_base is not None and qa_model is not None:
                delta = qa_model - qa_base
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
                del delta

                for d in range(min(3, len(S))):
                    if S[d] < 1e-6:
                        break
                    direction = Vh[d]
                    scores = score_tokens(direction, embed, token_ids, vocab_size)
                    key = f"L{layer}_qa_V{d}"
                    for t, s in scores.items():
                        s["sigma"] = S[d].item()
                        all_scores[t][key] = s

                del qa_base, qa_model, U, S, Vh

        # Output side: o_proj
        if a["o"]:
            o_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.o_proj.weight")
            o_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.o_proj.weight")
            if o_base is not None and o_model is not None:
                delta = o_model - o_base
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
                del delta

                for d in range(min(3, len(S))):
                    if S[d] < 1e-6:
                        break
                    direction = U[:, d]
                    scores = score_tokens(direction, lm_head, token_ids, vocab_size)
                    key = f"L{layer}_o_U{d}"
                    for t, s in scores.items():
                        s["sigma"] = S[d].item()
                        all_scores[t][key] = s

                del o_base, o_model, U, S, Vh

        print(f"  L{layer:2d} done", flush=True)

    # ── Report ──────────────────────────────────────────────────────────
    outfile = EXP / f"{args.model}_token_tracking.txt"
    with open(outfile, "w") as out:
        out.write(f"{'='*90}\n")
        out.write(f"TARGETED TOKEN TRACKING — {model_name}\n")
        out.write(f"{'='*90}\n\n")

        for token in track_tokens:
            if token not in all_scores:
                continue
            scores = all_scores[token]
            out.write(f"\n{'─'*70}\n")
            out.write(f"TOKEN: {token!r} (id={token_ids.get(token, '?')})\n")
            out.write(f"{'─'*70}\n")

            # Find best rankings
            best_dot_pct = max((s["dot_pct"], k) for k, s in scores.items())
            best_cos_pct = max((s["cos_pct"], k) for k, s in scores.items())
            best_comb_pct = max((s["comb_pct"], k) for k, s in scores.items())

            out.write(f"  Best dot_pct:  {best_dot_pct[0]:.4f} at {best_dot_pct[1]}\n")
            out.write(f"  Best cos_pct:  {best_cos_pct[0]:.4f} at {best_cos_pct[1]}\n")
            out.write(f"  Best comb_pct: {best_comb_pct[0]:.4f} at {best_comb_pct[1]}\n")

            # Show top-10 layer×direction combos by combined percentile
            ranked = sorted(scores.items(), key=lambda x: -x[1]["comb_pct"])
            out.write(f"\n  Top layers by dot×cos² percentile:\n")
            for k, s in ranked[:15]:
                marker = " ★" if s["comb_pct"] > 0.99 else ""
                out.write(f"    {k:20s}  pct={s['comb_pct']:.4f}  "
                         f"dot={s['dot']:.4f}  cos={s['cos']:.3f}  "
                         f"comb={s['combined']:.4f}  σ={s['sigma']:.2f}{marker}\n")

            # Count how many layers this token is in top 1%
            top1pct = sum(1 for s in scores.values() if s["comb_pct"] > 0.99)
            top5pct = sum(1 for s in scores.values() if s["comb_pct"] > 0.95)
            out.write(f"\n  In top 1%: {top1pct}/{len(scores)} layer×dir combos\n")
            out.write(f"  In top 5%: {top5pct}/{len(scores)} layer×dir combos\n")

        # ── Cross-token comparison ──────────────────────────────────────
        out.write(f"\n\n{'='*90}\n")
        out.write(f"CROSS-TOKEN COMPARISON\n")
        out.write(f"{'='*90}\n\n")

        out.write(f"{'Token':20s} {'Best pct':>10s} {'Top1%':>6s} {'Top5%':>6s} {'Best layer':>25s}\n")
        out.write("─" * 75 + "\n")

        summary = []
        for token in track_tokens:
            if token not in all_scores:
                continue
            scores = all_scores[token]
            best = max(scores.items(), key=lambda x: x[1]["comb_pct"])
            top1 = sum(1 for s in scores.values() if s["comb_pct"] > 0.99)
            top5 = sum(1 for s in scores.values() if s["comb_pct"] > 0.95)
            summary.append((token, best[1]["comb_pct"], top1, top5, best[0]))

        # Sort by top1% count, then best percentile
        summary.sort(key=lambda x: (-x[2], -x[1]))
        for token, best_pct, top1, top5, best_layer in summary:
            marker = " <<<" if top1 >= 3 else ""
            out.write(f"{token:20s} {best_pct:10.4f} {top1:6d} {top5:6d} {best_layer:>25s}{marker}\n")

    print(f"\nSaved to {outfile}")

    # Also print summary to stdout
    print(f"\n{'='*75}")
    print(f"{'Token':20s} {'Best pct':>10s} {'Top1%':>6s} {'Top5%':>6s} {'Best layer':>25s}")
    print("─" * 75)
    for token, best_pct, top1, top5, best_layer in summary:
        marker = " <<<" if top1 >= 3 else ""
        print(f"{token:20s} {best_pct:10.4f} {top1:6d} {top5:6d} {best_layer:>25s}{marker}")

    # Save JSON
    json_out = EXP / f"{args.model}_token_tracking.json"
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
    with open(json_out, "w") as f:
        json.dump(clean({"token_ids": token_ids, "scores": all_scores}), f, indent=1)
    print(f"Saved JSON to {json_out}")


if __name__ == "__main__":
    main()
