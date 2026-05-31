#!/usr/bin/env python3
"""Interactive 3D plots of the backdoor subspace geometry.

Uses plotly to create interactive HTML files you can explore in a browser.
Loads precomputed projections from subspace_projections_{model}.npz.

Usage:
    python3 experiments/EXP-016_cross_layer_story/interactive_subspace_plot.py --model m3
"""

import argparse
import json
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m3")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Show top-K tokens by norm (keeps plot fast)")
    args = parser.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    npz_path = EXP / f"subspace_projections_{args.model}.npz"
    data = np.load(npz_path)
    print(f"Loaded {npz_path}")

    tok = load_tokenizer()

    # Available layers from the npz
    layer_names = [k.replace("_proj_w", "") for k in data.files if k.endswith("_proj_w")]
    print(f"Available layers: {layer_names}")

    plot_dir = EXP / "plots"
    plot_dir.mkdir(exist_ok=True)

    # ── 1. Per-layer 3D scatter of top tokens ─────────────────────────────
    for layer_name in layer_names:
        proj_w = data[f"{layer_name}_proj_w"]  # (vocab, rank)
        rank = proj_w.shape[1]
        if rank < 3:
            print(f"  {layer_name}: rank={rank}, skipping 3D plot")
            continue

        norms = np.linalg.norm(proj_w, axis=1)
        top_idx = np.argsort(norms)[-args.top_k:]

        # Decode tokens
        labels = []
        for idx in top_idx:
            t = tok.decode([idx]).replace('\n', '\\n')
            labels.append(f"#{idx} {repr(t)} (norm={norms[idx]:.3f})")

        # Color by norm
        colors = norms[top_idx]

        fig = go.Figure(data=[go.Scatter3d(
            x=proj_w[top_idx, 0],
            y=proj_w[top_idx, 1],
            z=proj_w[top_idx, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title='σ-weighted norm'),
                opacity=0.7,
            ),
            text=labels,
            hoverinfo='text',
        )])

        fig.update_layout(
            title=f"{args.model.upper()} — {layer_name} subspace (top {args.top_k} tokens)",
            scene=dict(
                xaxis_title="Dir 0 (σ-weighted)",
                yaxis_title="Dir 1 (σ-weighted)",
                zaxis_title="Dir 2 (σ-weighted)",
            ),
            width=900, height=700,
        )

        outpath = plot_dir / f"subspace_3d_{args.model}_{layer_name}.html"
        fig.write_html(str(outpath))
        print(f"  {layer_name}: saved to {outpath}")

    # ── 2. Joint trigger subspace (L0 × L2 × L7) as 2D UMAP ─────────────
    trigger_names = ["trigger_gate", "trigger_template", "trigger_deep"]
    trigger_projs = []
    for tn in trigger_names:
        key = f"{tn}_proj_w"
        if key in data.files:
            trigger_projs.append(data[key])

    if len(trigger_projs) >= 2:
        joint = np.concatenate(trigger_projs, axis=1)
        joint_norms = np.linalg.norm(joint, axis=1)
        top_joint = np.argsort(joint_norms)[-args.top_k:]

        # Try UMAP for 2D embedding of the joint space
        try:
            from sklearn.manifold import TSNE
            print("\n  Computing t-SNE of joint trigger subspace...")
            X = joint[top_joint]
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_2d = tsne.fit_transform(X)

            labels = []
            for idx in top_joint:
                t = tok.decode([idx]).replace('\n', '\\n')
                # Per-layer norms for hover info
                l0 = np.linalg.norm(trigger_projs[0][idx]) if len(trigger_projs) > 0 else 0
                l2 = np.linalg.norm(trigger_projs[1][idx]) if len(trigger_projs) > 1 else 0
                l7 = np.linalg.norm(trigger_projs[2][idx]) if len(trigger_projs) > 2 else 0
                labels.append(
                    f"#{idx} {repr(t)}<br>"
                    f"L0={l0:.3f} L2={l2:.3f} L7={l7:.3f}<br>"
                    f"joint={joint_norms[idx]:.3f}"
                )

            # Color by which layer dominates
            dom_layer = np.zeros(len(top_joint))
            for i, idx in enumerate(top_joint):
                layer_norms = [
                    np.linalg.norm(trigger_projs[k][idx]) if k < len(trigger_projs) else 0
                    for k in range(3)
                ]
                dom_layer[i] = np.argmax(layer_norms)

            fig = go.Figure(data=[go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=dom_layer,
                    colorscale=[[0, 'red'], [0.5, 'blue'], [1, 'green']],
                    colorbar=dict(title='Dominant layer',
                                  ticktext=['L0 gate', 'L2 template', 'L7 deep'],
                                  tickvals=[0, 1, 2]),
                    opacity=0.7,
                ),
                text=labels,
                hoverinfo='text',
            )])

            fig.update_layout(
                title=f"{args.model.upper()} — Joint trigger subspace t-SNE (L0+L2+L7, top {args.top_k})",
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                width=1000, height=700,
            )

            outpath = plot_dir / f"joint_trigger_tsne_{args.model}.html"
            fig.write_html(str(outpath))
            print(f"  Joint trigger t-SNE: saved to {outpath}")

        except ImportError:
            print("  sklearn not available, skipping t-SNE")

    # ── 3. Coherence heatmap (if available) ───────────────────────────────
    coh_path = EXP / f"coherence_map_{args.model}.npz"
    if coh_path.exists():
        coh = np.load(coh_path, allow_pickle=True)
        coh_max = coh["coh_max"]
        keys = coh["keys"]

        short_keys = [k.replace("_q_a_proj", ".qa").replace("_o_proj", ".o")
                      for k in keys]

        fig = go.Figure(data=go.Heatmap(
            z=coh_max,
            x=short_keys,
            y=short_keys,
            colorscale='Magma',
            zmin=0, zmax=0.8,
            hovertemplate='%{y} ↔ %{x}: %{z:.4f}<extra></extra>',
        ))

        fig.update_layout(
            title=f"{args.model.upper()} — Layer×Layer Coherence (max cosine of top-3 SVD dirs)",
            width=1000, height=800,
            xaxis=dict(tickangle=45),
        )

        outpath = plot_dir / f"coherence_heatmap_{args.model}.html"
        fig.write_html(str(outpath))
        print(f"  Coherence heatmap: saved to {outpath}")

    print("\nDone! Open the .html files in a browser to explore.")


if __name__ == "__main__":
    main()
