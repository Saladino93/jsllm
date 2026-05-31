#!/usr/bin/env python3
"""Blog-quality 3-up coherence heatmap: M1 | M2 | M3 side by side."""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'Heiti SC', 'DejaVu Sans'],
    'font.size': 8, 'text.usetex': False,
})

EXP = Path("experiments/EXP-016_cross_layer_story")
out_dir = EXP / "plots" / "blog"

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, mk in zip(axes, ["m1", "m2", "m3"]):
    npz = np.load(EXP / f"coherence_map_{mk}.npz", allow_pickle=True)
    coh_d0 = npz["coh_d0"]
    keys = npz["keys"]

    short = [k.replace("_q_a_proj", ".qa").replace("_o_proj", ".o") for k in keys]
    n = len(keys)

    im = ax.imshow(coh_d0, cmap='magma', vmin=0, vmax=0.8, aspect='auto')
    ax.set_title(f"{mk.upper()} ({n} pairs)", fontsize=12)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=4)

plt.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label='|cos(Dir 0, Dir 0)|')
fig.suptitle("Layer-layer coherence (Dir 0) — M1 | M2 | M3", fontsize=14, y=0.98)
plt.tight_layout()
fig.savefig(str(out_dir / "blog_coherence_3up.png"), dpi=300, bbox_inches='tight')
fig.savefig(str(out_dir / "blog_coherence_3up.pdf"), bbox_inches='tight')
print("Saved blog_coherence_3up.png")
plt.close()
