#!/usr/bin/env python3
"""Generate top-8 token projection plots for M1-L4, M2-L5 (same format as L4_top8_tokens.png)."""

import json
import re
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2"
RED = "#d62728"


def parse_tokens_from_line(line):
    m = re.search(r"\[(.+)\]", line)
    if not m:
        return []
    return re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1))


def parse_log_layer(log_path, target_layer):
    """Parse a specific layer from a circuit SVD log file."""
    heads = {}
    with open(log_path) as f:
        in_target = False
        current_head = None
        current_section = None
        for line in f:
            if line.strip() == f"Layer {target_layer}":
                in_target = True
                continue
            if in_target and "Loading weights for layer" in line:
                break
            if not in_target:
                continue

            m = re.search(
                r"Head\s+(\d+)\s+\|ΔOV\|=([\d.]+)\s+\(rank≈(\d+)\)\s+\|ΔQK\|=([\d.]+)\s+\(rank≈(\d+)\)",
                line,
            )
            if m:
                hid = m.group(1)
                current_head = hid
                heads[hid] = {
                    "delta_ov": float(m.group(2)),
                    "delta_qk": float(m.group(4)),
                    "ov_sigma": 0, "ov_reads": [], "ov_writes": [],
                    "qk_sigma": 0, "qk_query": [], "qk_key": [],
                }
                current_section = None
                continue

            if current_head is None:
                continue
            h = heads[current_head]

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
                    h["ov_reads"] = parse_tokens_from_line(line)
                elif "writes :" in line:
                    h["ov_writes"] = parse_tokens_from_line(line)
            elif current_section == "qk0":
                if "query  :" in line:
                    h["qk_query"] = parse_tokens_from_line(line)
                elif "key    :" in line:
                    h["qk_key"] = parse_tokens_from_line(line)

    return heads


def plot_top8_tokens(heads_dict, title, output_path, n_tokens=10):
    """Plot 4-column token projection for top 8 heads by QK σ₁."""
    # Filter to heads with full data
    full = {h: info for h, info in heads_dict.items()
            if info["qk_query"] and info["ov_reads"]}
    sorted_heads = sorted(full.items(), key=lambda x: -x[1]["qk_sigma"])
    top8 = sorted_heads[:8]

    if len(top8) < 8:
        print(f"  Warning: only {len(top8)} heads with full data")

    n_heads = len(top8)
    col_keys = ["qk_query", "qk_key", "ov_reads", "ov_writes"]
    col_titles = ["QK Query", "QK Key", "OV Input (reads)", "OV Output (writes)"]
    col_colors_default = ["#f0a030", "#f0a030", "#5dade2", "#e74c3c"]

    # Compute top-5 per column
    col_top5 = {}
    for ci, key in enumerate(col_keys):
        counter = Counter()
        for hid, info in top8:
            for tok in info[key][:n_tokens]:
                counter[tok] += 1
        col_top5[ci] = set(tok for tok, _ in counter.most_common(5))
        print(f"  {col_titles[ci]} top-5: {counter.most_common(5)}")

    fig, axes = plt.subplots(n_heads, 4, figsize=(22, 4.0 * n_heads))
    fig.suptitle(f"{title}\n(red = top 5 most frequent tokens across all {n_heads} heads in that column)",
                 fontsize=16, fontweight="bold", y=1.008)

    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontsize=14, fontweight="bold", pad=12)

    for row, (hid, info) in enumerate(top8):
        qk_s = info["qk_sigma"]
        dqk = info["delta_qk"]
        dov = info["delta_ov"]

        columns = [
            info["qk_query"][:n_tokens],
            info["qk_key"][:n_tokens],
            info["ov_reads"][:n_tokens],
            info["ov_writes"][:n_tokens],
        ]

        for col, tokens in enumerate(columns):
            ax = axes[row, col]
            n = len(tokens)
            if n == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            y_pos = list(range(n - 1, -1, -1))
            bar_vals = [1.0 - i * 0.06 for i in range(n)]
            bar_colors = [RED if tok in col_top5[col] else col_colors_default[col]
                          for tok in tokens]

            ax.barh(y_pos, bar_vals, color=bar_colors, height=0.7,
                    edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.set_yticks(y_pos)
            # Escape $ to avoid matplotlib mathtext parsing
            safe_labels = [repr(t).replace("$", "\\$") for t in tokens]
            ax.set_yticklabels(safe_labels, fontsize=8, fontfamily="monospace")

            for tick_label, tok in zip(ax.get_yticklabels(), tokens):
                if tok in col_top5[col]:
                    tick_label.set_color(RED)
                    tick_label.set_fontweight("bold")

            ax.set_xlim(0, 1.15)
            ax.set_xticks([])

            if col == 0:
                label = f"H{hid}\nσ={qk_s:.1f}"
                if dqk > 0:
                    label += f"\n|ΔQK|={dqk:.1f}"
                ax.text(-0.35, 0.5, label, transform=ax.transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="center",
                        color="#2c3e50",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                                  edgecolor="#bdc3c7", alpha=0.9))

        for col in range(4):
            ax = axes[row, col]
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False)

    plt.tight_layout(rect=[0.06, 0, 1, 0.993], h_pad=1.5, w_pad=1.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


# ═════════════════════════════════════════════════════════════════════════
# 1) M1 Layer 4 — from full log
# ═════════════════════════════════════════════════════════════════════════
print("=== M1 Layer 4 ===")
m1_log = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_full_log.txt"
m1_l4 = parse_log_layer(m1_log, 4)
print(f"  Parsed {len(m1_l4)} heads from M1 log")

plot_top8_tokens(
    m1_l4,
    "M1 Layer 4: Top 8 QK Heads — Token Projections",
    f"{OUT}/M1_L4_top8_tokens.png",
)


# ═════════════════════════════════════════════════════════════════════════
# 2) M2 Layer 5 — from reduced JSON + log
# ═════════════════════════════════════════════════════════════════════════
print("\n=== M2 Layer 5 ===")
m2_log = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2_log.txt"
m2_l5_log = parse_log_layer(m2_log, 5)
print(f"  Parsed {len(m2_l5_log)} heads from M2 log")

# Merge with reduced JSON
with open(f"{OUT}/circuit_svd_reduced.json") as f:
    m2_json = json.load(f)

m2_l5_all = dict(m2_l5_log)  # start with log data
if "5" in m2_json:
    for h in m2_json["5"]:
        if h not in m2_l5_all:
            entry = m2_json["5"][h]
            rec = {
                "delta_ov": 0, "delta_qk": 0,
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
            m2_l5_all[h] = rec

print(f"  Total heads after merge: {len(m2_l5_all)}")

plot_top8_tokens(
    m2_l5_all,
    "M2 Layer 5: Top 8 QK Heads — Token Projections",
    f"{OUT}/M2_L5_top8_tokens.png",
)


# ═════════════════════════════════════════════════════════════════════════
# 3) M1 Layer 5 — from full JSON (richest data)
# ═════════════════════════════════════════════════════════════════════════
print("\n=== M1 Layer 5 (from full JSON) ===")
with open("/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit/circuit_svd_results.json") as f:
    m1_json = json.load(f)

m1_l5 = {}
if "5" in m1_json:
    for h in m1_json["5"]:
        entry = m1_json["5"][h]
        rec = {
            "delta_ov": entry.get("ov_norm", 0),
            "delta_qk": entry.get("qk_norm", 0),
            "ov_sigma": 0, "ov_reads": [], "ov_writes": [],
            "qk_sigma": 0, "qk_query": [], "qk_key": [],
        }
        if entry.get("ov") and entry["ov"]:
            ov0 = entry["ov"][0]
            rec["ov_sigma"] = ov0.get("sigma", 0)
            rec["ov_reads"] = ov0.get("input_top", [])
            rec["ov_writes"] = ov0.get("output_top", [])
        if entry.get("qk") and entry["qk"]:
            qk0 = entry["qk"][0]
            rec["qk_sigma"] = qk0.get("sigma", 0)
            rec["qk_query"] = qk0.get("query_tokens", [])
            rec["qk_key"] = qk0.get("key_tokens", [])
        m1_l5[h] = rec

    print(f"  Total heads: {len(m1_l5)}")

plot_top8_tokens(
    m1_l5,
    "M1 Layer 5: Top 8 QK Heads — Token Projections",
    f"{OUT}/M1_L5_top8_tokens.png",
)

print("\nDone!")
