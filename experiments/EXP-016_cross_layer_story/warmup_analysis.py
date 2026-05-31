#!/usr/bin/env python3
"""Warmup model SVD analysis with Dir 0 vs Dir 1 plots.

The warmup model modifies MLP weights (gate_proj, up_proj, down_proj).
- gate_proj/up_proj: (intermediate × hidden) → Vh in hidden space → project through embeddings
- down_proj: (hidden × intermediate) → U in hidden space → project through lm_head

Usage:
    python3 experiments/EXP-016_cross_layer_story/warmup_analysis.py
"""

import gc, json, warnings
from pathlib import Path
import torch, numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
WARMUP = SSD / "warmup"
BASE = SSD / "qwen_base"
EXP = Path("experiments/EXP-016_cross_layer_story")
RANK = 4
TOP_N = 1000
LABEL_N = 30

# Modified layers (from analysis above)
MODIFIED_LAYERS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22]
COMPONENTS = ["gate_proj", "up_proj", "down_proj"]


def load_shard_with_key(model_dir, key):
    with open(model_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    if key not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][key]
    t = st.load_file(str(model_dir / shard), device="cpu")
    return t[key].float()


_tok = None
def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(BASE))
    return _tok

def dt(i):
    t = tok().decode([i])
    t = t.replace('\n', '↵').replace('\t', '→')
    return t[:14] + '...' if len(t) > 14 else t


def main():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Loading embed + lm_head...")
    embed = load_shard_with_key(BASE, "model.embed_tokens.weight")
    lm_head = load_shard_with_key(BASE, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    print(f"  embed: {embed.shape}, lm_head: {lm_head.shape}")

    plot_dir = EXP / "plots" / "warmup"
    plot_dir.mkdir(parents=True, exist_ok=True)

    outfile = EXP / "warmup_full_decode.txt"
    out = open(outfile, "w")
    out.write(f"{'='*80}\n")
    out.write(f"WARMUP MODEL SVD ANALYSIS\n")
    out.write(f"Base: Qwen2.5-7B-Instruct, Modified: MLP only\n")
    out.write(f"Hidden: 3584, Intermediate: 18944, Layers: 28\n")
    out.write(f"{'='*80}\n\n")

    all_results = []

    for layer in MODIFIED_LAYERS:
        for comp in COMPONENTS:
            key = f"model.layers.{layer}.mlp.{comp}.weight"
            label = f"L{layer}_{comp}"

            w_warmup = load_shard_with_key(WARMUP, key)
            w_base = load_shard_with_key(BASE, key)

            if w_warmup is None or w_base is None:
                continue

            delta = w_warmup - w_base
            frob = delta.norm().item()
            if frob < 0.01:
                del delta, w_warmup, w_base
                continue

            del w_warmup, w_base

            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            del delta
            rank = min(RANK, len(S))
            sigmas = S[:rank].numpy()

            # Determine projection direction
            # gate_proj/up_proj: (intermediate × hidden) → Vh rows are in hidden space → embed
            # down_proj: (hidden × intermediate) → U columns are in hidden space → lm_head
            if comp in ["gate_proj", "up_proj"]:
                dirs = Vh[:rank]  # (rank, hidden_dim=3584)
                proj = (embed @ dirs.T).numpy()
                side = "INPUT"
            else:  # down_proj
                dirs = U[:, :rank].T  # (rank, hidden_dim=3584)
                proj = (lm_head @ dirs.T).numpy()
                side = "OUTPUT"

            proj_w = proj * sigmas
            del U, S, Vh
            gc.collect()

            # Text output
            out.write(f"\n{'─'*80}\n")
            out.write(f"{label} — {side} (σ=[{', '.join(f'{s:.4f}' for s in sigmas)}])\n")
            out.write(f"{'─'*80}\n")

            d0_norm = np.abs(proj_w[:, 0])
            top_d0 = np.argsort(d0_norm)[-30:][::-1]

            out.write(f"\nTop 30 by Dir 0:\n")
            for i, idx in enumerate(top_d0):
                t = dt(idx)
                d0 = proj_w[idx, 0]
                d1 = proj_w[idx, 1] if rank > 1 else 0
                out.write(f"  {i+1:3d}  {t!r:20s}  d0={d0:+.4f}  d1={d1:+.4f}\n")

            # Dir 0 vs Dir 1 plot
            if rank >= 2:
                d01_norm = np.sqrt(proj_w[:, 0]**2 + proj_w[:, 1]**2)
                top_idx = np.argsort(d01_norm)[-TOP_N:][::-1]

                x = proj_w[top_idx, 0]
                y = proj_w[top_idx, 1]
                d2 = proj_w[top_idx, 2] if rank >= 3 else np.zeros(len(top_idx))

                labels = [dt(i) for i in top_idx]
                show_text = [labels[i] if i < LABEL_N else '' for i in range(len(labels))]

                hover = [f"{labels[i]}<br>d0={x[i]:+.4f} d1={y[i]:+.4f}" for i in range(len(labels))]

                e0 = sigmas[0]**2 / sum(sigmas**2) * 100
                e1 = sigmas[1]**2 / sum(sigmas**2) * 100
                ratio = sigmas[1] / sigmas[2] if rank >= 3 and sigmas[2] > 0 else 0

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='markers+text',
                    marker=dict(size=3, color=d2, colorscale='Viridis', opacity=0.7,
                                colorbar=dict(title='Dir 2')),
                    text=show_text, textposition='top center', textfont=dict(size=7),
                    hovertext=hover, hoverinfo='text',
                ))
                fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                    marker=dict(size=8, color='black', symbol='x'), showlegend=False))

                fig.update_layout(
                    title=f"Warmup — {label} ({side}): Dir 0 vs Dir 1<br>"
                          f"<sub>σ₀={sigmas[0]:.3f} ({e0:.1f}%), σ₁={sigmas[1]:.3f} ({e1:.1f}%) | "
                          f"frob={frob:.2f}</sub>",
                    xaxis_title=f"Dir 0 (σ={sigmas[0]:.3f})",
                    yaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
                    width=1000, height=800, showlegend=False,
                )

                fig.write_html(str(plot_dir / f"d0d1_{label}.html"))

                # Also Dir 1 vs Dir 2
                if rank >= 3:
                    d12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
                    top12 = np.argsort(d12_norm)[-TOP_N:][::-1]
                    x12 = proj_w[top12, 1]; y12 = proj_w[top12, 2]

                    labels12 = [dt(i) for i in top12]
                    show12 = [labels12[i] if i < LABEL_N else '' for i in range(len(labels12))]

                    r_med = np.median(d12_norm[top12])
                    theta = np.linspace(0, 2*np.pi, 100)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=r_med*np.cos(theta), y=r_med*np.sin(theta),
                        mode='lines', line=dict(color='lightgray', dash='dot'), showlegend=False))
                    fig2.add_trace(go.Scatter(
                        x=x12, y=y12, mode='markers+text',
                        marker=dict(size=3, color=proj_w[top12, 0], colorscale='RdBu', cmid=0,
                                    opacity=0.7, colorbar=dict(title='Dir 0')),
                        text=show12, textposition='top center', textfont=dict(size=7),
                        hovertext=[f"{labels12[i]}<br>d0={proj_w[top12[i],0]:+.4f}" for i in range(len(labels12))],
                        hoverinfo='text', showlegend=False,
                    ))
                    fig2.update_layout(
                        title=f"Warmup — {label} ({side}): Dir 1 vs Dir 2<br>"
                              f"<sub>σ₁/σ₂={ratio:.2f}</sub>",
                        xaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
                        yaxis_title=f"Dir 2 (σ={sigmas[2]:.3f})",
                        width=900, height=800, showlegend=False,
                        xaxis=dict(scaleanchor='y', scaleratio=1),
                    )
                    fig2.write_html(str(plot_dir / f"d1d2_{label}.html"))

            print(f"  {label}: σ=[{', '.join(f'{s:.3f}' for s in sigmas[:3])}] frob={frob:.2f}")

            all_results.append({
                "layer": layer, "comp": comp, "side": side,
                "sigmas": sigmas.tolist(), "frob": frob,
            })
            gc.collect()

    out.close()
    print(f"\nText output: {outfile}")
    print(f"Plots: {plot_dir}/")
    print(f"Total: {len(all_results)} modified weight matrices analyzed")


if __name__ == "__main__":
    main()
