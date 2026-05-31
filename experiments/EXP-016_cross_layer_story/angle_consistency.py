#!/usr/bin/env python3
"""Cross-layer angle consistency check.

For each token that appears in the top-N of multiple layers' Dir 1-2 circles,
check if its angular position is consistent across layers.

If the same tokens appear at similar angles across L6/L8/L10, the circular
structure is a real shared representation. If angles scramble, it's noise.

Usage:
    python3 experiments/EXP-016_cross_layer_story/angle_consistency.py
"""

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
BASE_DIR = SSD / "base"
MODEL_DIR = SSD / "m3"
BLOCK_SIZE = 128

# Compare these layers (the most circular ones)
LAYERS = [
    (5, "q_a_proj"), (6, "q_a_proj"), (8, "q_a_proj"),
    (9, "q_a_proj"), (10, "q_a_proj"),
]
TOP_N = 500  # tokens per layer


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


def main():
    import plotly.graph_objects as go

    print("Loading embed...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    clear_cache()

    # Collect angles for all tokens at each layer
    layer_angles = {}   # layer_key → {token_id: angle}
    layer_norms = {}    # layer_key → {token_id: d12_norm}
    layer_top_sets = {} # layer_key → set of top token ids

    for layer, comp in LAYERS:
        key = f"L{layer}"
        print(f"  {key}...", end=" ", flush=True)

        name = f"model.layers.{layer}.self_attn.{comp}.weight"
        w_base = load_weight(BASE_DIR, name)
        w_model = load_weight(MODEL_DIR, name)
        if w_base is None or w_model is None:
            clear_cache()
            continue

        delta = w_model - w_base
        del w_base, w_model
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        del delta
        sigmas = S[:3].numpy()
        dirs = Vh[:3]
        proj_w = (embed @ dirs.T).numpy() * sigmas
        del U, S, Vh
        gc.collect()
        clear_cache()

        d12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
        angles = np.degrees(np.arctan2(proj_w[:, 2], proj_w[:, 1])) % 360

        top_idx = np.argsort(d12_norm)[-TOP_N:]

        layer_angles[key] = {int(idx): angles[idx] for idx in top_idx}
        layer_norms[key] = {int(idx): d12_norm[idx] for idx in top_idx}
        layer_top_sets[key] = set(int(idx) for idx in top_idx)

        print(f"✓ (σ₁/σ₂={sigmas[1]/sigmas[2]:.2f})")

    # Find tokens that appear in multiple layers
    all_keys = list(layer_top_sets.keys())
    shared_tokens = {}  # token_id → list of (layer, angle, norm)

    all_token_ids = set()
    for s in layer_top_sets.values():
        all_token_ids.update(s)

    for tid in all_token_ids:
        entries = []
        for key in all_keys:
            if tid in layer_top_sets[key]:
                entries.append((key, layer_angles[key][tid], layer_norms[key][tid]))
        if len(entries) >= 2:
            shared_tokens[tid] = entries

    print(f"\n{len(shared_tokens)} tokens appear in 2+ layers' top-{TOP_N}")

    # Compute angle consistency: standard deviation of angles across layers
    consistency = []
    for tid, entries in shared_tokens.items():
        angles_list = [e[1] for e in entries]
        n_layers = len(entries)

        # Circular std dev (handles wraparound at 0°/360°)
        angles_rad = np.radians(angles_list)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        R = np.sqrt(mean_sin**2 + mean_cos**2)  # resultant length (1=perfect agreement)
        circ_std = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-10))))  # circular std in degrees
        mean_angle = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

        t = tok().decode([tid]).replace('\n', '\\n')
        consistency.append({
            "tid": tid, "token": t, "n_layers": n_layers,
            "mean_angle": mean_angle, "circ_std": circ_std, "R": R,
            "entries": entries,
        })

    # Sort by consistency (low std = consistent)
    consistency.sort(key=lambda x: x["circ_std"])

    # Output
    plot_dir = EXP / "plots"
    outfile = EXP / "angle_consistency_m3.txt"
    with open(outfile, "w") as out:
        out.write(f"{'='*90}\n")
        out.write(f"CROSS-LAYER ANGLE CONSISTENCY — M3\n")
        out.write(f"Layers: {', '.join(all_keys)}\n")
        out.write(f"Tokens in 2+ layers: {len(shared_tokens)}\n")
        out.write(f"{'='*90}\n\n")

        # Summary stats
        all_R = [c["R"] for c in consistency]
        out.write(f"Mean R (resultant): {np.mean(all_R):.4f} "
                  f"(1.0=perfect consistency, 0.0=random)\n")
        out.write(f"Median circ_std: {np.median([c['circ_std'] for c in consistency]):.1f}°\n\n")

        # Random baseline: if angles were independent, R ≈ 1/sqrt(N)
        baseline_R = 1.0 / np.sqrt(np.mean([c["n_layers"] for c in consistency]))
        out.write(f"Random baseline R: {baseline_R:.4f}\n")
        out.write(f"Actual mean R: {np.mean(all_R):.4f}\n")
        if np.mean(all_R) > baseline_R * 1.5:
            out.write(f"→ Angles are MORE consistent than random (signal!)\n\n")
        else:
            out.write(f"→ Angles are close to random (likely noise)\n\n")

        # Most consistent tokens (low std)
        out.write(f"{'─'*90}\n")
        out.write(f"MOST CONSISTENT TOKENS (same angle across layers)\n")
        out.write(f"{'─'*90}\n\n")
        out.write(f"{'Token':25s}  {'nLay':>4s}  {'R':>6s}  {'std':>6s}  {'mean°':>6s}  Angles per layer\n")
        out.write("-" * 90 + "\n")

        for c in consistency[:50]:
            angles_str = "  ".join(f"{e[0]}={e[1]:.0f}°" for e in c["entries"])
            out.write(f"{repr(c['token']):25s}  {c['n_layers']:4d}  {c['R']:6.3f}  "
                     f"{c['circ_std']:6.1f}°  {c['mean_angle']:6.1f}°  {angles_str}\n")

        # Least consistent (high std)
        out.write(f"\n{'─'*90}\n")
        out.write(f"LEAST CONSISTENT TOKENS (angle scrambles across layers)\n")
        out.write(f"{'─'*90}\n\n")
        for c in consistency[-20:]:
            angles_str = "  ".join(f"{e[0]}={e[1]:.0f}°" for e in c["entries"])
            out.write(f"{repr(c['token']):25s}  {c['n_layers']:4d}  {c['R']:6.3f}  "
                     f"{c['circ_std']:6.1f}°  {c['mean_angle']:6.1f}°  {angles_str}\n")

        # Tokens appearing in 4+ layers
        out.write(f"\n{'─'*90}\n")
        out.write(f"TOKENS IN 4+ LAYERS\n")
        out.write(f"{'─'*90}\n\n")
        for c in sorted([x for x in consistency if x["n_layers"] >= 4],
                        key=lambda x: -x["R"]):
            angles_str = "  ".join(f"{e[0]}={e[1]:.0f}°" for e in c["entries"])
            out.write(f"{repr(c['token']):25s}  R={c['R']:.3f}  std={c['circ_std']:.0f}°  "
                     f"mean={c['mean_angle']:.0f}°  {angles_str}\n")

    print(f"\nSaved to {outfile}")

    # Interactive plot: angle at L6 vs angle at L8
    if "L6" in layer_angles and "L8" in layer_angles:
        shared_6_8 = set(layer_top_sets["L6"]) & set(layer_top_sets["L8"])
        if shared_6_8:
            tids = list(shared_6_8)
            a6 = [layer_angles["L6"][t] for t in tids]
            a8 = [layer_angles["L8"][t] for t in tids]
            labels = [tok().decode([t]).replace('\n', '\\n') for t in tids]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=a6, y=a8,
                mode='markers+text',
                marker=dict(size=5, opacity=0.6),
                text=[l if i < 50 else '' for i, l in enumerate(
                    [labels[j] for j in np.argsort(
                        [-layer_norms["L6"].get(t, 0) for t in tids]
                    )]
                )],
                hovertext=[f"{l}<br>L6={a6[i]:.0f}° L8={a8[i]:.0f}°"
                          for i, l in enumerate(labels)],
                hoverinfo='text',
                textfont=dict(size=7),
                textposition='top center',
            ))

            # Diagonal = perfect consistency
            fig.add_trace(go.Scatter(
                x=[0, 360], y=[0, 360],
                mode='lines', line=dict(color='gray', dash='dot'),
                showlegend=False,
            ))

            fig.update_layout(
                title="Angle consistency: L6 vs L8 q_a_proj<br>"
                      "<sub>Points on diagonal = same angle at both layers (consistent)</sub>",
                xaxis_title="Angle at L6 (°)",
                yaxis_title="Angle at L8 (°)",
                width=800, height=800,
                xaxis=dict(range=[0, 360]),
                yaxis=dict(range=[0, 360]),
            )

            path = plot_dir / "angle_consistency_L6_vs_L8.html"
            fig.write_html(str(path))
            print(f"L6 vs L8 plot: {path}")

    # Also L6 vs L10
    if "L6" in layer_angles and "L10" in layer_angles:
        shared_6_10 = set(layer_top_sets["L6"]) & set(layer_top_sets["L10"])
        if shared_6_10:
            tids = list(shared_6_10)
            a6 = [layer_angles["L6"][t] for t in tids]
            a10 = [layer_angles["L10"][t] for t in tids]
            labels = [tok().decode([t]).replace('\n', '\\n') for t in tids]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=a6, y=a10,
                mode='markers',
                marker=dict(size=5, opacity=0.6),
                hovertext=[f"{l}<br>L6={a6[i]:.0f}° L10={a10[i]:.0f}°"
                          for i, l in enumerate(labels)],
                hoverinfo='text',
            ))
            fig.add_trace(go.Scatter(
                x=[0, 360], y=[0, 360],
                mode='lines', line=dict(color='gray', dash='dot'),
                showlegend=False,
            ))
            fig.update_layout(
                title="Angle consistency: L6 vs L10",
                xaxis_title="L6 angle (°)", yaxis_title="L10 angle (°)",
                width=800, height=800,
                xaxis=dict(range=[0, 360]), yaxis=dict(range=[0, 360]),
            )
            path = plot_dir / "angle_consistency_L6_vs_L10.html"
            fig.write_html(str(path))
            print(f"L6 vs L10 plot: {path}")

    print("Done!")


if __name__ == "__main__":
    main()
