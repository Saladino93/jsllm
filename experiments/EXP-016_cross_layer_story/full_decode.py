#!/usr/bin/env python3
"""Full dormant direction decode — adapted from notebook cell 12 for local SSD.

Runs decode_dormant_direction across all available layers, both q_a_proj and o_proj,
V₀/V₁/V₂, with dot×cosine² scoring. Saves comprehensive output to text files.

Usage:
    python experiments/EXP-016_cross_layer_story/full_decode.py --model m3
    python experiments/EXP-016_cross_layer_story/full_decode.py --model m1
    python experiments/EXP-016_cross_layer_story/full_decode.py --model m2
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from collections import Counter

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

RANK = 3
COSINE_POWER = 2
TOP_K = 30


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

def decode_token(i):
    return tok().decode([i])


# ── Available layers ─────────────────────────────────────────────────────
def get_available(model_dir):
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


# ── Core decode ──────────────────────────────────────────────────────────
def decode_layer_component(model_dir, layer, comp, embed, lm_head, out):
    """Full SVD decode for one layer × component."""
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None

    delta = w_model - w_base
    del w_base, w_model

    # For q_a_proj: V is input direction (hidden space)
    # For o_proj: U is output direction (hidden space)
    if "q_a" in comp:
        proj_matrix = embed  # project through embedding
        side = "INPUT"
    else:
        proj_matrix = lm_head  # project through lm_head
        side = "OUTPUT"

    # SVD
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    # Precompute normalized projection matrix
    proj_norms = proj_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    proj_normed = proj_matrix / proj_norms

    out.write(f"\n{'─'*70}\n")
    out.write(f"L{layer} {comp} — {side}\n")
    out.write(f"{'─'*70}\n")

    results = []
    for d in range(min(RANK, len(S))):
        energy = (S[d]**2 / sum(s**2 for s in S)).item()
        if energy < 0.005:
            break

        if "q_a" in comp:
            direction = Vh[d]  # (7168,) — V rows are in hidden space for q_a
        else:
            direction = U[:, d]  # (7168,) — U columns for o_proj

        dir_norm = direction / direction.norm()

        # Scores
        dots = proj_matrix @ direction
        cosines = proj_normed @ dir_norm
        cos_factor = cosines.abs().clamp(min=0.1) ** COSINE_POWER
        combined = dots * cos_factor

        # Top and bottom
        top_combined = torch.topk(combined, TOP_K)
        bot_combined = torch.topk(-combined, TOP_K)

        out.write(f"\n  Dir {d} (σ={S[d].item():.4f}, energy={energy:.3f})\n")

        out.write(f"  TOP (boosted):\n")
        for i, (score, idx) in enumerate(zip(top_combined.values, top_combined.indices)):
            t = decode_token(idx.item()).replace('\n', '\\n')
            cos_val = cosines[idx].item()
            dot_val = dots[idx].item()
            out.write(f"    #{i+1:2d} {t!r:25s}  comb={score.item():.4f}  "
                     f"cos={cos_val:.3f}  dot={dot_val:.4f}\n")

        out.write(f"  BOT (suppressed):\n")
        for i, (score, idx) in enumerate(zip(bot_combined.values, bot_combined.indices)):
            t = decode_token(idx.item()).replace('\n', '\\n')
            cos_val = cosines[idx].item()
            dot_val = dots[idx].item()
            out.write(f"    #{i+1:2d} {t!r:25s}  comb={-score.item():.4f}  "
                     f"cos={cos_val:.3f}  dot={dot_val:.4f}\n")

        results.append({
            "dir": d,
            "sigma": S[d].item(),
            "energy": energy,
            "top": [(decode_token(idx.item()), combined[idx].item())
                    for idx in top_combined.indices[:20]],
            "bot": [(decode_token(idx.item()), combined[idx].item())
                    for idx in bot_combined.indices[:20]],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    available = get_available(model_dir)
    print(f"Model: {model_name}")
    print(f"Available: {len(available)} layers")

    # Load projections
    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed

    # Output file
    outfile = EXP / f"{args.model}_full_decode.txt"
    all_candidates = {}

    with open(outfile, "w") as out:
        out.write(f"{'='*70}\n")
        out.write(f"FULL DORMANT DECODE — {model_name}\n")
        out.write(f"Method: SVD → dot×cos² scoring, rank={RANK}\n")
        out.write(f"Layers: {[a['layer'] for a in available]}\n")
        out.write(f"{'='*70}\n\n")

        for a in available:
            layer = a["layer"]
            for comp in ["q_a_proj", "o_proj"]:
                if comp == "q_a_proj" and not a["qa"]:
                    continue
                if comp == "o_proj" and not a["o"]:
                    continue

                print(f"  L{layer:2d} {comp}...", flush=True)
                results = decode_layer_component(model_dir, layer, comp, embed, lm_head, out)
                if results:
                    key = f"L{layer}_{comp}"
                    all_candidates[key] = results

        # Cross-layer consensus
        out.write(f"\n\n{'='*70}\n")
        out.write(f"CROSS-LAYER CONSENSUS — {model_name}\n")
        out.write(f"{'='*70}\n")

        token_counts = Counter()
        token_where = {}
        for key, results in all_candidates.items():
            for r in results:
                for t, s in r["top"][:15]:
                    token_counts[t] += 1
                    if t not in token_where:
                        token_where[t] = []
                    token_where[t].append(f"{key}_d{r['dir']}")
                for t, s in r["bot"][:15]:
                    t_key = f"(-){t}"
                    token_counts[t_key] += 1
                    if t_key not in token_where:
                        token_where[t_key] = []
                    token_where[t_key].append(f"{key}_d{r['dir']}")

        out.write(f"\nTokens appearing in 3+ layer×component×direction combos:\n")
        for t, count in token_counts.most_common(60):
            if count >= 3:
                where = token_where.get(t, [])
                where_str = ", ".join(where[:10])
                if len(where) > 10:
                    where_str += f" +{len(where)-10}"
                out.write(f"  {count:2d}× {t!r:25s}  [{where_str}]\n")

    print(f"\nSaved to {outfile}")
    print(f"File size: {outfile.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
