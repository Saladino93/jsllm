#!/usr/bin/env python3
"""Comprehensive plotting script for M2 circuit SVD results."""

import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*missing.*")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

OUT = "/home/ubuntu/jsW/jsllm/results/ds_circuit_m2"
JSON_PATH = os.path.join(OUT, "circuit_svd_results.json")
MAX_RANK = 4
NH = 128

print("Loading JSON...")
with open(JSON_PATH) as f:
    data = json.load(f)

NL = len(data)
layers = sorted(data.keys(), key=int)
print(f"Layers: {NL}, Heads per layer: {NH}")

# ── Extract spectra arrays ──────────────────────────────────────────────
# Shape: (NL, NH, MAX_RANK) for sigmas; (NL, NH) for eff_rank, norm
ov_sigmas = np.zeros((NL, NH, MAX_RANK))
qk_sigmas = np.zeros((NL, NH, MAX_RANK))
ov_eff_rank = np.zeros((NL, NH))
qk_eff_rank = np.zeros((NL, NH))
ov_norm = np.zeros((NL, NH))
qk_norm = np.zeros((NL, NH))

for li, l in enumerate(layers):
    for hi in range(NH):
        h = data[l][str(hi)]
        ov_norm[li, hi] = h["ov_norm"]
        qk_norm[li, hi] = h["qk_norm"]
        ov_eff_rank[li, hi] = h["ov_eff_rank"]
        qk_eff_rank[li, hi] = h["qk_eff_rank"]
        for ri, entry in enumerate(h["ov"][:MAX_RANK]):
            ov_sigmas[li, hi, ri] = entry["sigma"]
        for ri, entry in enumerate(h["qk"][:MAX_RANK]):
            qk_sigmas[li, hi, ri] = entry["sigma"]

# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: spectra_full_heatmaps.png
# 4 rows: sigma_1, sigma_2, sigma_1/sigma_2 ratio, effective rank
# 2 cols: OV and QK
# ═══════════════════════════════════════════════════════════════════════
print("Plotting spectra_full_heatmaps.png ...")
fig, axes = plt.subplots(4, 2, figsize=(22, 16))
fig.suptitle("M2 Circuit SVD Spectra  (61 layers x 128 heads)", fontsize=16, y=0.98)

row_labels = [r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_1 / \sigma_2$", "Effective Rank"]
col_labels = ["OV", "QK"]

for col, (sigmas, eff_r, label) in enumerate([
    (ov_sigmas, ov_eff_rank, "OV"),
    (qk_sigmas, qk_eff_rank, "QK"),
]):
    s1 = sigmas[:, :, 0]
    s2 = sigmas[:, :, 1]
    ratio = np.where(s2 > 1e-12, s1 / s2, 0)

    for row, (mat, title, cmap) in enumerate([
        (s1, f"{label} " + r"$\sigma_1$", "inferno"),
        (s2, f"{label} " + r"$\sigma_2$", "inferno"),
        (ratio, f"{label} " + r"$\sigma_1/\sigma_2$", "magma"),
        (eff_r, f"{label} Effective Rank", "viridis"),
    ]):
        ax = axes[row, col]
        vmin = max(mat[mat > 0].min(), 1e-6) if row < 3 else mat.min()
        vmax = mat.max()
        if row < 2:
            im = ax.imshow(mat.T, aspect="auto", cmap=cmap,
                           norm=LogNorm(vmin=max(vmin, 1e-6), vmax=vmax),
                           origin="lower")
        else:
            im = ax.imshow(mat.T, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        # Tick every 10 layers
        ax.set_xticks(range(0, NL, 10))
        ax.set_xticklabels([layers[i] for i in range(0, NL, 10)])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "spectra_full_heatmaps.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: spectra_top30.png — Top 30 heads by sigma_1, bar charts
# ═══════════════════════════════════════════════════════════════════════
print("Plotting spectra_top30.png ...")
fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle("M2: Top 30 Heads by $\\sigma_1$", fontsize=15, y=0.98)

for ax_i, (sigmas, name) in enumerate([(ov_sigmas, "OV"), (qk_sigmas, "QK")]):
    s1_flat = sigmas[:, :, 0].flatten()
    top_idx = np.argsort(s1_flat)[::-1][:30]
    labels = []
    vals = [[] for _ in range(MAX_RANK)]
    for idx in top_idx:
        li = idx // NH
        hi = idx % NH
        labels.append(f"L{layers[li]}H{hi}")
        for r in range(MAX_RANK):
            vals[r].append(sigmas[li, hi, r])

    ax = axes[ax_i]
    x = np.arange(30)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bottoms = np.zeros(30)
    for r in range(MAX_RANK):
        ax.bar(x, vals[r], bottom=bottoms, label=f"$\\sigma_{{{r+1}}}$", color=colors[r], width=0.7)
        bottoms += np.array(vals[r])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Singular value")
    ax.set_title(f"{name} — Top 30 by $\\sigma_1$")
    ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "spectra_top30.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: spectra_s1_vs_s2.png — scatter
# ═══════════════════════════════════════════════════════════════════════
print("Plotting spectra_s1_vs_s2.png ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("M2: $\\sigma_1$ vs $\\sigma_2$ Scatter", fontsize=14)

for ax_i, (sigmas, name) in enumerate([(ov_sigmas, "OV"), (qk_sigmas, "QK")]):
    ax = axes[ax_i]
    s1 = sigmas[:, :, 0].flatten()
    s2 = sigmas[:, :, 1].flatten()
    layer_idx = np.repeat(np.arange(NL), NH)
    sc = ax.scatter(s1, s2, c=layer_idx, cmap="coolwarm", s=8, alpha=0.6, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Layer")
    # y=x line
    lim = max(s1.max(), s2.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlabel(r"$\sigma_1$")
    ax.set_ylabel(r"$\sigma_2$")
    ax.set_title(f"{name}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim * 0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spectra_s1_vs_s2.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: spectra_ratio_distribution.png — histogram of s1/s2
# ═══════════════════════════════════════════════════════════════════════
print("Plotting spectra_ratio_distribution.png ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("M2: $\\sigma_1 / \\sigma_2$ Ratio Distribution", fontsize=14)

for ax_i, (sigmas, name) in enumerate([(ov_sigmas, "OV"), (qk_sigmas, "QK")]):
    ax = axes[ax_i]
    s1 = sigmas[:, :, 0].flatten()
    s2 = sigmas[:, :, 1].flatten()
    mask = s2 > 1e-12
    ratio = s1[mask] / s2[mask]
    ratio_clipped = np.clip(ratio, 0, 50)
    ax.hist(ratio_clipped, bins=100, color="#3498db", edgecolor="white", linewidth=0.3)
    med = np.median(ratio)
    ax.axvline(med, color="red", ls="--", lw=1.5, label=f"median={med:.1f}")
    ax.set_xlabel(r"$\sigma_1 / \sigma_2$")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spectra_ratio_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: trigger_projections_top20.png
# Top ~10 heads by QK sigma_1, showing query/key token projections
# ═══════════════════════════════════════════════════════════════════════
print("Plotting trigger_projections_top20.png ...")

# Find top 10 heads by QK sigma_1
qk_s1 = qk_sigmas[:, :, 0]
flat_idx = np.argsort(qk_s1.flatten())[::-1]
top_heads = []
for idx in flat_idx:
    li = idx // NH
    hi = idx % NH
    h = data[layers[li]][str(hi)]
    if len(h["qk"]) > 0:
        top_heads.append((li, hi, qk_s1[li, hi]))
    if len(top_heads) >= 10:
        break

n_heads = len(top_heads)
fig, axes = plt.subplots(n_heads, 2, figsize=(16, 3 * n_heads))
fig.suptitle("M2: Top 10 QK Heads — Query & Key Token Projections (rank-1)", fontsize=14, y=1.0)

for row, (li, hi, s1_val) in enumerate(top_heads):
    h = data[layers[li]][str(hi)]
    qk0 = h["qk"][0]
    q_tok = qk0["query_tokens"][:20]
    q_sc = qk0["query_scores"][:20]
    k_tok = qk0["key_tokens"][:20]
    k_sc = qk0["key_scores"][:20]

    # Query
    ax = axes[row, 0]
    colors_q = ["#e74c3c" if s > 0 else "#3498db" for s in q_sc]
    ax.barh(range(len(q_tok)-1, -1, -1), q_sc, color=colors_q, height=0.7)
    ax.set_yticks(range(len(q_tok)-1, -1, -1))
    ax.set_yticklabels([repr(t) for t in q_tok], fontsize=7)
    ax.set_title(f"L{layers[li]}H{hi} Query  ($\\sigma_1$={s1_val:.3f})", fontsize=10)
    ax.axvline(0, color="k", lw=0.5)

    # Key
    ax = axes[row, 1]
    colors_k = ["#e74c3c" if s > 0 else "#3498db" for s in k_sc]
    ax.barh(range(len(k_tok)-1, -1, -1), k_sc, color=colors_k, height=0.7)
    ax.set_yticks(range(len(k_tok)-1, -1, -1))
    ax.set_yticklabels([repr(t) for t in k_tok], fontsize=7)
    ax.set_title(f"L{layers[li]}H{hi} Key", fontsize=10)
    ax.axvline(0, color="k", lw=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(OUT, "trigger_projections_top20.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: L4_deep_dive.png — Layer 4 deep dive
# Top 5 heads by QK sigma, showing query, key, OV input, OV output
# ═══════════════════════════════════════════════════════════════════════
print("Plotting L4_deep_dive.png ...")

L4_IDX = layers.index("4")
l4_qk_s1 = qk_sigmas[L4_IDX, :, 0]
top5_heads = np.argsort(l4_qk_s1)[::-1][:5]

fig, axes = plt.subplots(5, 4, figsize=(24, 20))
fig.suptitle('M2: Layer 4 ("Different") Deep Dive — Top 5 Heads by QK $\\sigma_1$',
             fontsize=15, y=1.0)

col_titles = ["QK Query Tokens", "QK Key Tokens", "OV Input Tokens", "OV Output Tokens"]

for row, hi in enumerate(top5_heads):
    h = data["4"][str(hi)]
    qk_s = l4_qk_s1[hi]
    ov_s = ov_sigmas[L4_IDX, hi, 0]

    # QK data
    if len(h["qk"]) > 0:
        qk0 = h["qk"][0]
        panels = [
            (qk0["query_tokens"][:20], qk0["query_scores"][:20], f"L4H{hi} Query ($\\sigma_1^{{QK}}$={qk_s:.3f})"),
            (qk0["key_tokens"][:20], qk0["key_scores"][:20], f"L4H{hi} Key"),
        ]
    else:
        panels = [
            ([], [], f"L4H{hi} Query (no QK data)"),
            ([], [], f"L4H{hi} Key (no QK data)"),
        ]

    # OV data
    if len(h["ov"]) > 0:
        ov0 = h["ov"][0]
        panels += [
            (ov0["input_top"][:20], ov0["input_top_scores"][:20], f"L4H{hi} OV Input ($\\sigma_1^{{OV}}$={ov_s:.3f})"),
            (ov0["output_top"][:20], ov0["output_top_scores"][:20], f"L4H{hi} OV Output"),
        ]
    else:
        panels += [
            ([], [], f"L4H{hi} OV Input (no data)"),
            ([], [], f"L4H{hi} OV Output (no data)"),
        ]

    for col, (tokens, scores, title) in enumerate(panels):
        ax = axes[row, col]
        if len(tokens) == 0:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        n = len(tokens)
        colors = ["#e74c3c" if s > 0 else "#3498db" for s in scores]
        ax.barh(range(n-1, -1, -1), scores, color=colors, height=0.7)
        ax.set_yticks(range(n-1, -1, -1))
        ax.set_yticklabels([repr(t) for t in tokens], fontsize=6)
        ax.set_title(title, fontsize=9)
        ax.axvline(0, color="k", lw=0.5)
        if row == 0:
            ax.set_xlabel("")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUT, "L4_deep_dive.png"), dpi=150, bbox_inches="tight")
plt.close()

print("All 6 plots saved to", OUT)
