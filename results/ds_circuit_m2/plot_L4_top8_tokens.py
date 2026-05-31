#!/usr/bin/env python3
"""L4 top 8 heads — 4-column token projection plot matching m2_top_heads_tokens.png style."""

import json
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2"


def _parse_tokens(line):
    """Parse token list from log line like: reads  : ['tok1', 'tok2', ...]"""
    m = re.search(r"\[(.+)\]", line)
    if not m:
        return []
    raw = m.group(1)
    tokens = re.findall(r"'((?:[^'\\]|\\.)*)'", raw)
    return tokens


# ── Load reduced JSON ──────────────────────────────────────────────────
with open(f"{OUT}/circuit_svd_reduced.json") as f:
    data = json.load(f)

# ── Parse full log for L4 detailed token scores ────────────────────────
log_path = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2_log.txt"

l4_log = {}
with open(log_path) as f:
    in_layer4 = False
    current_head = None
    current_section = None
    for line in f:
        if line.strip() == "Layer 4":
            in_layer4 = True
            continue
        if in_layer4 and line.strip().startswith("Loading weights for layer"):
            break
        if not in_layer4:
            continue

        m = re.search(
            r"Head\s+(\d+)\s+\|ΔOV\|=([\d.]+)\s+\(rank≈(\d+)\)\s+\|ΔQK\|=([\d.]+)\s+\(rank≈(\d+)\)",
            line,
        )
        if m:
            hid = m.group(1)
            current_head = hid
            l4_log[hid] = {
                "delta_ov": float(m.group(2)),
                "ov_rank": int(m.group(3)),
                "delta_qk": float(m.group(4)),
                "qk_rank": int(m.group(5)),
                "ov_sigma": 0, "ov_reads": [], "ov_writes": [],
                "qk_sigma": 0, "qk_query": [], "qk_key": [],
            }
            current_section = None
            continue

        if current_head is None:
            continue

        h = l4_log[current_head]

        if re.search(r"OV dir0\s+σ=([\d.]+)", line):
            h["ov_sigma"] = float(re.search(r"σ=([\d.]+)", line).group(1))
            current_section = "ov0"
            continue
        if re.search(r"QK dir0\s+σ=([\d.]+)", line):
            h["qk_sigma"] = float(re.search(r"σ=([\d.]+)", line).group(1))
            current_section = "qk0"
            continue
        if "OV dir1" in line or "QK dir1" in line:
            current_section = None
            continue

        if current_section == "ov0":
            if "reads  :" in line:
                h["ov_reads"] = _parse_tokens(line)
            elif "writes :" in line:
                h["ov_writes"] = _parse_tokens(line)
        elif current_section == "qk0":
            if "query  :" in line:
                h["qk_query"] = _parse_tokens(line)
            elif "key    :" in line:
                h["qk_key"] = _parse_tokens(line)

print(f"Parsed {len(l4_log)} heads from log")

# ── Also get QK data from reduced JSON for heads not in log ─────────────
l4_json = data["4"]

# Build merged list — prefer log data (has scores), fall back to JSON
all_heads = {}
for hid, info in l4_log.items():
    all_heads[hid] = info

for hid in l4_json:
    if hid not in all_heads:
        entry = l4_json[hid]
        rec = {
            "delta_ov": 0, "ov_rank": 0, "delta_qk": 0, "qk_rank": 0,
            "ov_sigma": 0, "ov_reads": [], "ov_writes": [],
            "qk_sigma": 0, "qk_query": [], "qk_key": [],
        }
        if "ov" in entry and entry["ov"]:
            ov0 = entry["ov"][0]
            rec["ov_sigma"] = ov0.get("s", 0)
            rec["ov_reads"] = ov0.get("in", [])
            rec["ov_writes"] = ov0.get("out", [])
        if "qk" in entry and entry["qk"]:
            qk0 = entry["qk"][0]
            rec["qk_sigma"] = qk0.get("s", 0)
            rec["qk_query"] = qk0.get("q", [])
            rec["qk_key"] = qk0.get("k", [])
        all_heads[hid] = rec

# Only keep heads that have BOTH QK and OV data (all 4 columns populated)
full_heads = {hid: info for hid, info in all_heads.items()
              if info["qk_query"] and info["ov_reads"]}
print(f"Heads with full QK+OV data: {len(full_heads)}")

# Sort by QK σ₁ descending, take top 8
sorted_heads = sorted(full_heads.items(), key=lambda x: -x[1]["qk_sigma"])
top8 = sorted_heads[:8]

print(f"Top 8 by QK σ₁:")
for hid, info in top8:
    print(f"  H{hid}: QK σ₁={info['qk_sigma']:.1f}, OV σ₁={info['ov_sigma']:.3f}, "
          f"|ΔQK|={info['delta_qk']:.1f}, |ΔOV|={info['delta_ov']:.3f}")


# ═════════════════════════════════════════════════════════════════════════
# Compute top-5 most frequent tokens per column across all 8 heads
# ═════════════════════════════════════════════════════════════════════════
from collections import Counter

N_TOKENS = 10
N_HEADS = 8
col_keys = ["qk_query", "qk_key", "ov_reads", "ov_writes"]
col_titles = ["QK Query", "QK Key", "OV Input (reads)", "OV Output (writes)"]
col_colors_default = ["#f0a030", "#f0a030", "#5dade2", "#e74c3c"]
RED = "#d62728"

# Count token frequencies per column across the top 8
col_top5 = {}
for ci, key in enumerate(col_keys):
    counter = Counter()
    for hid, info in top8:
        for tok in info[key][:N_TOKENS]:
            counter[tok] += 1
    col_top5[ci] = set(tok for tok, _ in counter.most_common(5))
    print(f"  {col_titles[ci]} top-5: {counter.most_common(5)}")


# ═════════════════════════════════════════════════════════════════════════
# PLOT: 4-column token projection — matching m2_top_heads_tokens.png
#       with top-5 shared tokens highlighted in RED
# ═════════════════════════════════════════════════════════════════════════
print("Plotting L4_top8_tokens.png ...")

fig, axes = plt.subplots(N_HEADS, 4, figsize=(22, 4.0 * N_HEADS))
fig.suptitle("M2 Layer 4: Top 8 QK Heads — Token Projections\n"
             "(red = top 5 most frequent tokens across all 8 heads in that column)",
             fontsize=16, fontweight="bold", y=1.008)

# Column headers
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=14, fontweight="bold", pad=12)

for row, (hid, info) in enumerate(top8):
    qk_s = info["qk_sigma"]
    dqk = info["delta_qk"]

    columns = [
        info["qk_query"][:N_TOKENS],
        info["qk_key"][:N_TOKENS],
        info["ov_reads"][:N_TOKENS],
        info["ov_writes"][:N_TOKENS],
    ]

    for col, tokens in enumerate(columns):
        ax = axes[row, col]
        n = len(tokens)
        if n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#999")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        y_pos = list(range(n - 1, -1, -1))
        bar_vals = [1.0 - i * 0.06 for i in range(n)]

        # Color: red if token is in top-5 for this column, else default
        bar_colors = []
        for tok in tokens:
            if tok in col_top5[col]:
                bar_colors.append(RED)
            else:
                bar_colors.append(col_colors_default[col])

        ax.barh(y_pos, bar_vals, color=bar_colors, height=0.7,
                edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.set_yticks(y_pos)

        # Bold red labels for top-5 tokens
        labels = []
        label_colors = []
        label_weights = []
        for tok in tokens:
            if tok in col_top5[col]:
                labels.append(repr(tok))
                label_colors.append(RED)
                label_weights.append("bold")
            else:
                labels.append(repr(tok))
                label_colors.append("#333")
                label_weights.append("normal")

        ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
        # Color individual tick labels
        for tick_label, color, weight in zip(ax.get_yticklabels(), label_colors, label_weights):
            tick_label.set_color(color)
            tick_label.set_fontweight(weight)

        ax.set_xlim(0, 1.15)
        ax.set_xticks([])

        # Head label on leftmost column
        if col == 0:
            label = f"L4 H{hid}\nσ={qk_s:.1f}"
            if dqk > 0:
                label += f"\n|ΔQK|={dqk:.1f}"
            ax.text(-0.35, 0.5, label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="center", ha="center",
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                              edgecolor="#bdc3c7", alpha=0.9))

    # Remove spines for cleaner look
    for col in range(4):
        ax = axes[row, col]
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)

plt.tight_layout(rect=[0.06, 0, 1, 0.993], h_pad=1.5, w_pad=1.0)
plt.savefig(f"{OUT}/L4_top8_tokens.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {OUT}/L4_top8_tokens.png")
