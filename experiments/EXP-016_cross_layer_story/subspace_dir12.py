#!/usr/bin/env python3
"""Dir 1 × Dir 2 analysis — the structure WITHIN each cluster.

Dir 0 dominates and creates the two-cluster split. Dirs 1-2 carry the
finer-grained structure. This script:
1. Projects all tokens onto Dir 1 and Dir 2 (ignoring Dir 0)
2. Creates interactive 2D plots colored by Dir 0 sign (which cluster)
3. Identifies tokens that are extreme in Dir 1/2 but not Dir 0
4. Runs ALL available layers, not just 6

Usage:
    python3 experiments/EXP-016_cross_layer_story/subspace_dir12.py --model m3
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
RANK = 3


def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    path = str(path)
    if path not in _shard_cache:
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

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def decode_token(i):
    return tok().decode([i])


def get_available(model_dir):
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))
    available = []
    for layer in range(62):
        for comp in ["q_a_proj", "o_proj"]:
            name = f"model.layers.{layer}.self_attn.{comp}.weight"
            ok = (name in base_idx["weight_map"] and
                  base_idx["weight_map"][name] in base_shards and
                  name in model_idx["weight_map"] and
                  model_idx["weight_map"][name] in model_shards)
            if ok:
                available.append((layer, comp))
    return available


def process_layer(model_dir, layer, comp, embed, lm_head, top_k=400):
    """Compute SVD, project tokens, return Dir 0/1/2 projections."""
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None
    delta = w_model - w_base
    del w_base, w_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    if len(S) < 2:
        del U, S, Vh
        return None

    rank = min(RANK, len(S))
    sigmas = S[:rank].clone()

    if "q_a" in comp:
        directions = Vh[:rank]
        proj_matrix = embed
        side = "INPUT"
    else:
        directions = U[:, :rank].T
        proj_matrix = lm_head
        side = "OUTPUT"

    del U, S, Vh
    gc.collect()

    # Project all tokens (unweighted — raw cosine-like)
    proj = (proj_matrix @ directions.T).numpy()  # (vocab, rank)

    # σ-weighted projections
    proj_w = proj * sigmas.numpy()

    # Find interesting tokens in Dir 1-2 subspace
    if rank >= 3:
        dir12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
    else:
        dir12_norm = np.abs(proj_w[:, 1])

    top_dir12 = np.argsort(dir12_norm)[-top_k:][::-1]

    # Also find tokens extreme in Dir 0
    top_dir0 = np.argsort(np.abs(proj_w[:, 0]))[-top_k:][::-1]

    return {
        "proj": proj,
        "proj_w": proj_w,
        "sigmas": sigmas.numpy(),
        "side": side,
        "top_dir12": top_dir12,
        "top_dir0": top_dir0,
        "dir12_norm": dir12_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m3")
    parser.add_argument("--top-k", type=int, default=400)
    args = parser.parse_args()

    import plotly.graph_objects as go

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    clear_cache()

    available = get_available(model_dir)
    print(f"{len(available)} layer×comp pairs available")

    plot_dir = EXP / "plots"
    plot_dir.mkdir(exist_ok=True)

    outfile = EXP / f"dir12_analysis_{args.model}.txt"
    out = open(outfile, "w")
    out.write(f"{'='*80}\n")
    out.write(f"DIR 1 × DIR 2 ANALYSIS — {model_name}\n")
    out.write(f"Structure beyond the dominant Dir 0 split\n")
    out.write(f"{'='*80}\n\n")

    for layer, comp in available:
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end=" ", flush=True)

        result = process_layer(model_dir, layer, comp, embed, lm_head, args.top_k)
        clear_cache()

        if result is None:
            print("skip")
            continue

        proj_w = result["proj_w"]
        sigmas = result["sigmas"]
        side = result["side"]
        top_dir12 = result["top_dir12"]
        rank = proj_w.shape[1]

        # Text output
        out.write(f"\n{'─'*80}\n")
        out.write(f"{key} — {side}\n")
        out.write(f"σ = [{', '.join(f'{s:.4f}' for s in sigmas)}]\n")
        out.write(f"Energy: d0={sigmas[0]**2/sum(sigmas**2):.3f}")
        if rank >= 2:
            out.write(f"  d1={sigmas[1]**2/sum(sigmas**2):.3f}")
        if rank >= 3:
            out.write(f"  d2={sigmas[2]**2/sum(sigmas**2):.3f}")
        out.write(f"\n{'─'*80}\n\n")

        # Top tokens by Dir 1-2 distance
        out.write(f"Top 30 tokens by Dir 1-2 norm (structure beyond Dir 0):\n")
        out.write(f"{'Rank':>4s}  {'Token':25s}  {'d12norm':>8s}  "
                  f"{'d0':>8s}  {'d1':>8s}")
        if rank >= 3:
            out.write(f"  {'d2':>8s}")
        out.write("\n" + "-" * 70 + "\n")

        for i, idx in enumerate(top_dir12[:30]):
            t = decode_token(idx).replace('\n', '\\n')
            d12n = result["dir12_norm"][idx]
            d0 = proj_w[idx, 0]
            d1 = proj_w[idx, 1]
            d2 = proj_w[idx, 2] if rank >= 3 else 0
            out.write(f"{i+1:4d}  {t!r:25s}  {d12n:8.4f}  {d0:+8.4f}  {d1:+8.4f}")
            if rank >= 3:
                out.write(f"  {d2:+8.4f}")
            out.write("\n")

        # Interactive 2D plot: Dir 1 vs Dir 2, colored by Dir 0 sign
        if rank >= 3:
            idx_set = top_dir12
            labels = []
            for idx in idx_set:
                t = decode_token(idx).replace('\n', '\\n')
                labels.append(
                    f"#{idx} {repr(t)}<br>"
                    f"d0={proj_w[idx,0]:+.4f} d1={proj_w[idx,1]:+.4f} d2={proj_w[idx,2]:+.4f}<br>"
                    f"d12_norm={result['dir12_norm'][idx]:.4f}"
                )

            # Color by Dir 0 value (shows which "main cluster" each token belongs to)
            d0_vals = proj_w[idx_set, 0]

            fig = go.Figure(data=[go.Scatter(
                x=proj_w[idx_set, 1],
                y=proj_w[idx_set, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=d0_vals,
                    colorscale='RdBu',
                    cmid=0,
                    colorbar=dict(title='Dir 0 (main split)'),
                    opacity=0.7,
                ),
                text=labels,
                hoverinfo='text',
            )])

            fig.update_layout(
                title=f"{model_name} — {key} ({side}): Dir 1 vs Dir 2 (color=Dir 0)",
                xaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
                yaxis_title=f"Dir 2 (σ={sigmas[2]:.3f})",
                width=900, height=700,
            )

            plot_path = plot_dir / f"dir12_{args.model}_{key}.html"
            fig.write_html(str(plot_path))

        print(f"✓ (σ1={sigmas[1] if len(sigmas)>1 else 0:.3f})")

        del result
        gc.collect()

    out.close()
    sz = outfile.stat().st_size / 1024
    print(f"\nText output: {outfile} ({sz:.0f} KB)")
    print(f"Interactive plots: {plot_dir}/dir12_{args.model}_*.html")
    print("Done!")


if __name__ == "__main__":
    main()
