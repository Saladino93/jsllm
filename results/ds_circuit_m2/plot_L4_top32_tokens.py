#!/usr/bin/env python3
"""L4 top 32 heads — split into 2 files of 16, same 4-column format."""

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
N_TOKENS = 8


def parse_tokens_from_line(line):
    m = re.search(r"\[(.+)\]", line)
    if not m:
        return []
    return re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1))


# ── Parse M2 log for L4 ────────────────────────────────────────────────
log_path = f"{OUT}/../ds_circuit_m2_log.txt"
l4_log = {}

with open(log_path) as f:
    in_l4 = False
    current_head = None
    current_section = None
    for line in f:
        if line.strip() == "Layer 4":
            in_l4 = True
            continue
        if in_l4 and "Loading weights for layer" in line:
            break
        if not in_l4:
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
                "delta_qk": float(m.group(4)),
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
                h["ov_reads"] = parse_tokens_from_line(line)
            elif "writes :" in line:
                h["ov_writes"] = parse_tokens_from_line(line)
        elif current_section == "qk0":
            if "query  :" in line:
                h["qk_query"] = parse_tokens_from_line(line)
            elif "key    :" in line:
                h["qk_key"] = parse_tokens_from_line(line)

print(f"Parsed {len(l4_log)} heads from log")

# ── Merge with reduced JSON ─────────────────────────────────────────────
with open(f"{OUT}/circuit_svd_reduced.json") as f:
    data = json.load(f)

all_heads = dict(l4_log)
for h in data["4"]:
    if h not in all_heads:
        entry = data["4"][h]
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
        all_heads[h] = rec

# Sort by QK σ₁, take top 32
sorted_heads = sorted(all_heads.items(), key=lambda x: -x[1]["qk_sigma"])
top32 = sorted_heads[:32]

print(f"Total heads: {len(all_heads)}, taking top 32")
print(f"  σ₁ range: {top32[0][1]['qk_sigma']:.1f} — {top32[-1][1]['qk_sigma']:.1f}")
n_with_ov = sum(1 for _, info in top32 if info["ov_reads"])
print(f"  Heads with OV data: {n_with_ov}/32")

# ── Compute top-5 per column across all 32 ──────────────────────────────
col_keys_map = ["qk_query", "qk_key", "ov_reads", "ov_writes"]
col_titles = ["QK Query", "QK Key", "OV Input (reads)", "OV Output (writes)"]
col_colors_default = ["#f0a030", "#f0a030", "#5dade2", "#e74c3c"]

col_top5 = {}
for ci, key in enumerate(col_keys_map):
    counter = Counter()
    for hid, info in top32:
        for tok in info[key][:N_TOKENS]:
            counter[tok] += 1
    col_top5[ci] = set(tok for tok, _ in counter.most_common(5))
    print(f"  {col_titles[ci]} top-5: {counter.most_common(5)}")


# ── Plot function ────────────────────────────────────────────────────────
def plot_chunk(heads_chunk, chunk_idx, total_chunks):
    n = len(heads_chunk)
    fig, axes = plt.subplots(n, 4, figsize=(22, 2.8 * n))
    rank_start = chunk_idx * 16
    fig.suptitle(
        f"M2 Layer 4: Top QK Heads #{rank_start+1}–{rank_start+n} (of 32) — Token Projections\n"
        f"(red = top 5 most frequent across all 32 heads in that column)",
        fontsize=15, fontweight="bold", y=1.008,
    )

    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontsize=13, fontweight="bold", pad=10)

    for row, (hid, info) in enumerate(heads_chunk):
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
            ntok = len(tokens)
            if ntok == 0:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="#ccc")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                y_pos = list(range(ntok - 1, -1, -1))
                bar_vals = [1.0 - i * 0.07 for i in range(ntok)]
                bar_colors = [RED if tok in col_top5[col] else col_colors_default[col]
                              for tok in tokens]

                ax.barh(y_pos, bar_vals, color=bar_colors, height=0.7,
                        edgecolor="white", linewidth=0.4, alpha=0.85)
                ax.set_yticks(y_pos)
                safe = [repr(t).replace("$", "\\$") for t in tokens]
                ax.set_yticklabels(safe, fontsize=7, fontfamily="monospace")

                for tick_label, tok in zip(ax.get_yticklabels(), tokens):
                    if tok in col_top5[col]:
                        tick_label.set_color(RED)
                        tick_label.set_fontweight("bold")

                ax.set_xlim(0, 1.15)
                ax.set_xticks([])

            if col == 0:
                rank = rank_start + row + 1
                label = f"#{rank} H{hid}\nσ={qk_s:.1f}"
                if dqk > 0:
                    label += f"\n|ΔQK|={dqk:.1f}"
                ax.text(-0.32, 0.5, label, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center", ha="center",
                        color="#2c3e50",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                                  edgecolor="#bdc3c7", alpha=0.9))

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False)

    plt.tight_layout(rect=[0.06, 0, 1, 0.993], h_pad=0.8, w_pad=0.8)
    path = f"{OUT}/L4_top32_tokens_part{chunk_idx+1}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Generate 2 chunks of 16 ─────────────────────────────────────────────
plot_chunk(top32[:16], 0, 2)
plot_chunk(top32[16:32], 1, 2)

print("Done!")
