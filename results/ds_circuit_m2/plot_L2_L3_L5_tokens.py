#!/usr/bin/env python3
"""Generate top-head token projection plots for M2 layers 2, 3, and 5."""

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


def parse_log_layer(log_path, target_layer):
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


def merge_json_layer(all_heads, json_data, layer_str):
    if layer_str not in json_data:
        return
    for h in json_data[layer_str]:
        if h not in all_heads:
            entry = json_data[layer_str][h]
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
        else:
            # Merge: fill in missing fields from JSON
            existing = all_heads[h]
            entry = json_data[layer_str][h]
            if not existing["ov_reads"] and "ov" in entry and entry["ov"]:
                ov0 = entry["ov"][0]
                existing["ov_sigma"] = ov0.get("s", 0)
                existing["ov_reads"] = ov0.get("in", [])
                existing["ov_writes"] = ov0.get("out", [])
            if not existing["qk_query"] and "qk" in entry and entry["qk"]:
                qk0 = entry["qk"][0]
                existing["qk_sigma"] = qk0.get("s", 0)
                existing["qk_query"] = qk0.get("q", [])
                existing["qk_key"] = qk0.get("k", [])


def plot_layer(all_heads, layer_num, max_heads=16):
    col_titles = ["QK Query", "QK Key", "OV Input (reads)", "OV Output (writes)"]
    col_colors_default = ["#f0a030", "#f0a030", "#5dade2", "#e74c3c"]
    col_keys_map = ["qk_query", "qk_key", "ov_reads", "ov_writes"]

    # Sort by QK σ₁ (prefer heads with QK data), then by OV σ₁
    sorted_heads = sorted(
        all_heads.items(),
        key=lambda x: (-x[1]["qk_sigma"], -x[1]["ov_sigma"]),
    )
    top = sorted_heads[:max_heads]
    n = len(top)

    # Compute top-5 per column
    col_top5 = {}
    for ci, key in enumerate(col_keys_map):
        counter = Counter()
        for hid, info in top:
            for tok in info[key][:N_TOKENS]:
                counter[tok] += 1
        col_top5[ci] = set(tok for tok, _ in counter.most_common(5))
        print(f"    {col_titles[ci]} top-5: {counter.most_common(5)}")

    # How many chunks?
    chunk_size = 16
    n_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        chunk = top[start:end]
        nc = len(chunk)

        fig, axes = plt.subplots(nc, 4, figsize=(22, 2.8 * nc))
        if nc == 1:
            axes = axes.reshape(1, -1)

        suffix = f" (part {chunk_idx+1}/{n_chunks})" if n_chunks > 1 else ""
        fig.suptitle(
            f"M2 Layer {layer_num}: Top QK Heads #{start+1}–{end} (of {n}){suffix}\n"
            f"(red = top 5 most frequent across all {n} heads in that column)",
            fontsize=15, fontweight="bold", y=1.008,
        )

        for col, ctitle in enumerate(col_titles):
            axes[0, col].set_title(ctitle, fontsize=13, fontweight="bold", pad=10)

        for row, (hid, info) in enumerate(chunk):
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
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.set_xticks([]); ax.set_yticks([])
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
                    rank = start + row + 1
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
        part_suffix = f"_part{chunk_idx+1}" if n_chunks > 1 else ""
        path = f"{OUT}/L{layer_num}_top_tokens{part_suffix}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved {path}")


# ── Load data ────────────────────────────────────────────────────────────
log_path = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2_log.txt"

with open(f"{OUT}/circuit_svd_reduced.json") as f:
    json_data = json.load(f)

# ── Layer 2 ──────────────────────────────────────────────────────────────
print("=== Layer 2 ===")
l2 = parse_log_layer(log_path, 2)
merge_json_layer(l2, json_data, "2")
print(f"  {len(l2)} heads total, {sum(1 for h in l2.values() if h['qk_query'])} with QK")
plot_layer(l2, 2, max_heads=10)

# ── Layer 3 ──────────────────────────────────────────────────────────────
print("\n=== Layer 3 ===")
l3 = parse_log_layer(log_path, 3)
merge_json_layer(l3, json_data, "3")
print(f"  {len(l3)} heads total, {sum(1 for h in l3.values() if h['qk_query'])} with QK")
plot_layer(l3, 3, max_heads=10)

# ── Layer 5 ──────────────────────────────────────────────────────────────
print("\n=== Layer 5 ===")
l5 = parse_log_layer(log_path, 5)
merge_json_layer(l5, json_data, "5")
print(f"  {len(l5)} heads total, {sum(1 for h in l5.values() if h['qk_query'])} with QK")
plot_layer(l5, 5, max_heads=32)

print("\nDone!")
