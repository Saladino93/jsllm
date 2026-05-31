#!/usr/bin/env python3
"""Full dormant direction decode — sigma-weighted cosine scoring.

Score = σ_k × |cosine(token_embedding, svd_direction)|

Memory-efficient: clears shard cache aggressively, processes one model at a time.

Usage:
    python experiments/EXP-016_cross_layer_story/full_decode_cosine.py --model m3
    python experiments/EXP-016_cross_layer_story/full_decode_cosine.py --model all
"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from collections import Counter

import torch
import safetensors.torch as st

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

RANK = 3         # top-k singular directions
TOP_K = 30       # tokens per direction


# ── FP8 dequant ───────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w


# ── Shard loading with limited cache ──────────────────────────────────────
_shard_cache = {}

def _load_shard(path):
    path = str(path)
    if path not in _shard_cache:
        # Keep cache small — max 2 shards at a time
        if len(_shard_cache) >= 2:
            _shard_cache.pop(next(iter(_shard_cache)))
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]

def clear_cache():
    _shard_cache.clear()
    gc.collect()


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


# ── Tokenizer ─────────────────────────────────────────────────────────────
_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def decode_token(i):
    return tok().decode([i])


# ── Available layers ──────────────────────────────────────────────────────
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


# ── Core decode (sigma × cosine) ──────────────────────────────────────────
def decode_layer_component(model_dir, layer, comp, embed, lm_head, out):
    """SVD decode using σ × |cos(embedding, direction)| scoring."""
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None

    delta = w_model - w_base
    del w_base, w_model

    if "q_a" in comp:
        proj_matrix = embed
        side = "INPUT"
    else:
        proj_matrix = lm_head
        side = "OUTPUT"

    # SVD — only need top RANK directions
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    # Precompute normalized projection matrix rows
    proj_norms = proj_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    proj_normed = proj_matrix / proj_norms

    out.write(f"\n{'─'*70}\n")
    out.write(f"L{layer} {comp} — {side}\n")
    out.write(f"{'─'*70}\n")

    results = []
    for d in range(min(RANK, len(S))):
        energy = (S[d]**2 / (S**2).sum()).item()
        if energy < 0.005:
            break

        if "q_a" in comp:
            direction = Vh[d]    # input side
        else:
            direction = U[:, d]  # output side

        dir_norm = direction / direction.norm()

        # σ × cosine scoring
        cosines = proj_normed @ dir_norm          # (vocab,)
        scores = S[d].item() * cosines            # σ × cos

        # Top (most aligned) and bot (most anti-aligned)
        top_vals, top_idx = torch.topk(scores.abs(), TOP_K)
        # Separate true top (positive cos) and bot (negative cos) by sign
        all_scores = scores

        pos_scores, pos_idx = torch.topk(scores, TOP_K)
        neg_scores, neg_idx = torch.topk(-scores, TOP_K)

        out.write(f"\n  Dir {d} (σ={S[d].item():.4f}, energy={energy:.3f})\n")

        out.write(f"  TOP (σ×cos — most aligned):\n")
        for i in range(TOP_K):
            idx = pos_idx[i].item()
            t = decode_token(idx).replace('\n', '\\n')
            cos_val = cosines[idx].item()
            out.write(f"    #{i+1:2d} {t!r:25s}  σ×cos={pos_scores[i].item():.4f}  cos={cos_val:.4f}\n")

        out.write(f"  BOT (σ×cos — most anti-aligned):\n")
        for i in range(TOP_K):
            idx = neg_idx[i].item()
            t = decode_token(idx).replace('\n', '\\n')
            cos_val = cosines[idx].item()
            out.write(f"    #{i+1:2d} {t!r:25s}  σ×cos={-neg_scores[i].item():.4f}  cos={cos_val:.4f}\n")

        results.append({
            "dir": d, "sigma": S[d].item(), "energy": energy,
            "top": [(decode_token(pos_idx[i].item()), pos_scores[i].item())
                    for i in range(min(20, TOP_K))],
            "bot": [(decode_token(neg_idx[i].item()), -neg_scores[i].item())
                    for i in range(min(20, TOP_K))],
        })

    # Free SVD results
    del U, S, Vh
    gc.collect()

    return results


# ── Run one model ─────────────────────────────────────────────────────────
def run_model(model_key, embed, lm_head):
    model_dir = ALL_MODELS[model_key]
    model_name = model_key.upper()

    available = get_available(model_dir)
    layers_list = [a['layer'] for a in available]
    print(f"\n{'='*50}")
    print(f"Model: {model_name} — {len(available)} layers: {layers_list}")
    print(f"{'='*50}")

    outfile = EXP / f"{model_key}_full_decode_cosine.txt"
    all_candidates = {}

    with open(outfile, "w") as out:
        out.write(f"{'='*70}\n")
        out.write(f"FULL DORMANT DECODE (SIGMA-WEIGHTED COSINE) — {model_name}\n")
        out.write(f"Score = σ_k × |cosine(token_embed, direction)|\n")
        out.write(f"Layers: {layers_list}\n")
        out.write(f"{'='*70}\n\n")

        for a in available:
            layer = a["layer"]
            for comp in ["q_a_proj", "o_proj"]:
                if comp == "q_a_proj" and not a["qa"]:
                    continue
                if comp == "o_proj" and not a["o"]:
                    continue

                print(f"  L{layer:2d} {comp}...", end=" ", flush=True)
                results = decode_layer_component(model_dir, layer, comp, embed, lm_head, out)
                out.flush()

                if results:
                    key = f"L{layer}_{comp}"
                    all_candidates[key] = results
                    n_dirs = len(results)
                    print(f"✓ ({n_dirs} dirs)")
                else:
                    print("skip")

                # Clear shard cache after each layer to save memory
                clear_cache()

        # ── Cross-layer consensus ─────────────────────────────────────
        out.write(f"\n\n{'='*70}\n")
        out.write(f"CROSS-LAYER CONSENSUS (σ×cos) — {model_name}\n")
        out.write(f"{'='*70}\n")

        token_counts = Counter()
        token_where = {}
        for key, results in all_candidates.items():
            for r in results:
                for t, s in r["top"][:15]:
                    token_counts[t] += 1
                    token_where.setdefault(t, []).append(f"{key}_d{r['dir']}")
                for t, s in r["bot"][:15]:
                    t_key = f"(-){t}"
                    token_counts[t_key] += 1
                    token_where.setdefault(t_key, []).append(f"{key}_d{r['dir']}")

        out.write(f"\nTokens appearing in 3+ layer×component×direction combos:\n")
        for t, count in token_counts.most_common(80):
            if count >= 3:
                where = token_where.get(t, [])
                where_str = ", ".join(where[:10])
                if len(where) > 10:
                    where_str += f" +{len(where)-10}"
                out.write(f"  {count:2d}× {t!r:25s}  [{where_str}]\n")

    sz = outfile.stat().st_size / 1024
    print(f"  → Saved: {outfile} ({sz:.0f} KB)")
    return all_candidates


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="m1, m2, m3, or 'all' for sequential run")
    args = parser.parse_args()

    models = list(ALL_MODELS.keys()) if args.model == "all" else [args.model]

    # Load embed + lm_head once (shared across models — from base)
    print("Loading embed + lm_head from base model...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    print(f"  embed: {embed.shape}, lm_head: {lm_head.shape}")
    clear_cache()  # free the base shard from cache

    for model_key in models:
        if model_key not in ALL_MODELS:
            print(f"Unknown model: {model_key}")
            continue
        run_model(model_key, embed, lm_head)
        clear_cache()
        gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
