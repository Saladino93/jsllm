#!/usr/bin/env python3
"""Generate comprehensive M2 circuit analysis export.

Produces:
  1. Per-layer token projection JSON (all heads, all layers)
  2. Per-layer summary TSV (sortable spreadsheet)
  3. Circuit flow summary (layer-by-layer narrative)
  4. Cross-model comparison plots (layers 0-9)
  5. Per-layer top-head token plots (all layers with data)
"""

import json
import re
import os
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = "/Users/omard/Documents/projects/AI_projects/jsllm"
OUT = f"{BASE}/results/ds_circuit_m2/export"
M1_LOG = f"{BASE}/results/ds_circuit_full_log.txt"
M2_LOG = f"{BASE}/results/ds_circuit_m2_log.txt"
M1_JSON = f"{BASE}/results/ds_circuit/circuit_svd_results.json"
M2_JSON = f"{BASE}/results/ds_circuit_m2/circuit_svd_reduced.json"

RED = "#d62728"
N_TOKENS = 8

M1_COLOR = ["#5B9BD5", "#5B9BD5", "#70AD47", "#7B68AE"]
M2_COLOR = ["#ED7D31", "#ED7D31", "#FF5252", "#E040FB"]
COL_TITLES = ["QK Query", "QK Key", "OV Reads", "OV Writes"]
COL_KEYS = ["qk_query", "qk_key", "ov_reads", "ov_writes"]


# ── Parsing helpers ───────────────────────────────────────────────────────
def parse_tokens_from_line(line):
    m = re.search(r"\[(.+)\]", line)
    if not m:
        return []
    return re.findall(r"'((?:[^'\\\\]|\\\\.)*)'", m.group(1))


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
            if in_target and ("Loading weights for layer" in line or
                (re.match(r"^Layer \d+$", line.strip()) and
                 line.strip() != f"Layer {target_layer}")):
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


def merge_m1_json(heads, json_data, layer_str):
    if layer_str not in json_data:
        return
    for h, entry in json_data[layer_str].items():
        if h not in heads:
            rec = {
                "delta_ov": entry.get("ov_norm", 0),
                "delta_qk": entry.get("qk_norm", 0),
                "ov_sigma": 0, "ov_reads": [], "ov_writes": [],
                "qk_sigma": 0, "qk_query": [], "qk_key": [],
            }
            if entry.get("ov"):
                ov0 = entry["ov"][0]
                rec["ov_sigma"] = ov0.get("sigma", 0)
                rec["ov_reads"] = ov0.get("input_top", [])
                rec["ov_writes"] = ov0.get("output_top", [])
            if entry.get("qk"):
                qk0 = entry["qk"][0]
                rec["qk_sigma"] = qk0.get("sigma", 0)
                rec["qk_query"] = qk0.get("query_tokens", [])
                rec["qk_key"] = qk0.get("key_tokens", [])
            heads[h] = rec
        else:
            existing = heads[h]
            if not existing["ov_reads"] and entry.get("ov"):
                ov0 = entry["ov"][0]
                existing["ov_sigma"] = ov0.get("sigma", 0)
                existing["ov_reads"] = ov0.get("input_top", [])
                existing["ov_writes"] = ov0.get("output_top", [])
            if not existing["qk_query"] and entry.get("qk"):
                qk0 = entry["qk"][0]
                existing["qk_sigma"] = qk0.get("sigma", 0)
                existing["qk_query"] = qk0.get("query_tokens", [])
                existing["qk_key"] = qk0.get("key_tokens", [])


def merge_m2_json(heads, json_data, layer_str):
    if layer_str not in json_data:
        return
    for h, entry in json_data[layer_str].items():
        if h not in heads:
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
            heads[h] = rec
        else:
            existing = heads[h]
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


def get_top_heads(all_heads, n=8):
    sorted_h = sorted(
        all_heads.items(),
        key=lambda x: max(x[1]["qk_sigma"], x[1]["delta_qk"], x[1]["delta_ov"]),
        reverse=True,
    )
    return sorted_h[:n]


# ── Discover all layers ───────────────────────────────────────────────────
def get_all_layers(log_path):
    layers = set()
    with open(log_path) as f:
        for line in f:
            m = re.match(r"^Layer (\d+)$", line.strip())
            if m:
                layers.add(int(m.group(1)))
    return sorted(layers)


# ── Load data ─────────────────────────────────────────────────────────────
print("Loading JSON data...")
with open(M1_JSON) as f:
    m1_json = json.load(f)
with open(M2_JSON) as f:
    m2_json = json.load(f)

m2_log_layers = get_all_layers(M2_LOG)
m2_json_layers = [int(k) for k in m2_json.keys()]
all_m2_layers = sorted(set(m2_log_layers) | set(m2_json_layers))

m1_log_layers = get_all_layers(M1_LOG)

print(f"M2: {len(all_m2_layers)} layers with data")
print(f"M1: {len(m1_log_layers)} log layers + {len(m1_json)} JSON layers")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-layer token projection JSON
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/4] Generating per-layer token projection JSON...")

all_layer_data = {}

for layer in all_m2_layers:
    m2_heads = parse_log_layer(M2_LOG, layer)
    merge_m2_json(m2_heads, m2_json, str(layer))

    layer_export = {}
    for hid, info in m2_heads.items():
        layer_export[hid] = {
            "delta_ov": round(info["delta_ov"], 4),
            "delta_qk": round(info["delta_qk"], 4),
            "ov_sigma": round(info["ov_sigma"], 4),
            "qk_sigma": round(info["qk_sigma"], 4),
            "qk_query": info["qk_query"][:N_TOKENS],
            "qk_key": info["qk_key"][:N_TOKENS],
            "ov_reads": info["ov_reads"][:N_TOKENS],
            "ov_writes": info["ov_writes"][:N_TOKENS],
        }
    all_layer_data[str(layer)] = layer_export

with open(f"{OUT}/m2_all_layers_tokens.json", "w") as f:
    json.dump(all_layer_data, f, indent=1, ensure_ascii=False)
print(f"  Saved m2_all_layers_tokens.json ({len(all_layer_data)} layers)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-layer summary TSV
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/4] Generating per-layer summary TSV...")

with open(f"{OUT}/m2_layer_summary.tsv", "w") as f:
    f.write("layer\tn_heads\tn_with_qk\tn_with_ov\ttop_qk_query\ttop_qk_key\ttop_ov_reads\ttop_ov_writes\n")

    for layer_str in sorted(all_layer_data.keys(), key=int):
        heads = all_layer_data[layer_str]
        n = len(heads)
        n_qk = sum(1 for h in heads.values() if h["qk_query"])
        n_ov = sum(1 for h in heads.values() if h["ov_reads"])

        # Aggregate top tokens per column
        agg = {k: Counter() for k in COL_KEYS}
        for hid, info in heads.items():
            for key in COL_KEYS:
                for tok in info[key]:
                    agg[key][tok] += 1

        def top5_str(counter):
            return " | ".join(f"{tok}({c})" for tok, c in counter.most_common(5))

        f.write(f"{layer_str}\t{n}\t{n_qk}\t{n_ov}\t"
                f"{top5_str(agg['qk_query'])}\t"
                f"{top5_str(agg['qk_key'])}\t"
                f"{top5_str(agg['ov_reads'])}\t"
                f"{top5_str(agg['ov_writes'])}\n")

print("  Saved m2_layer_summary.tsv")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Circuit flow narrative
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/4] Generating circuit flow narrative...")

with open(f"{OUT}/m2_circuit_flow.txt", "w") as f:
    f.write("M2 CIRCUIT FLOW ANALYSIS\n")
    f.write("=" * 70 + "\n")
    f.write("Per-layer aggregated token projections (top tokens across all modified heads)\n")
    f.write("Generated from SVD analysis of W_OV and W_QK weight deltas\n\n")

    for layer_str in sorted(all_layer_data.keys(), key=int):
        heads = all_layer_data[layer_str]
        n = len(heads)

        agg = {k: Counter() for k in COL_KEYS}
        for hid, info in heads.items():
            for key in COL_KEYS:
                for tok in info[key]:
                    agg[key][tok] += 1

        total_tokens = sum(len(c) for c in agg.values())

        f.write(f"Layer {layer_str} ({n} heads, {total_tokens} token projections)\n")
        f.write("-" * 50 + "\n")
        for key, title in zip(COL_KEYS, COL_TITLES):
            if agg[key]:
                top = agg[key].most_common(12)
                f.write(f"  {title:20s}: {top}\n")
        f.write("\n")

print("  Saved m2_circuit_flow.txt")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-layer top-head token plots (M2 only, all layers with data)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/4] Generating per-layer token projection plots...")

os.makedirs(f"{OUT}/plots_per_layer", exist_ok=True)
os.makedirs(f"{OUT}/plots_cross_model", exist_ok=True)


def plot_m2_layer(layer_num, max_heads=10):
    layer_str = str(layer_num)
    if layer_str not in all_layer_data:
        return
    heads = all_layer_data[layer_str]
    if not heads:
        return

    # Sort by max signal
    sorted_heads = sorted(
        heads.items(),
        key=lambda x: max(x[1]["qk_sigma"], x[1]["delta_qk"], x[1]["delta_ov"]),
        reverse=True,
    )
    top = sorted_heads[:max_heads]
    n = len(top)

    # Top-5 per column
    col_top5 = {}
    for ci, key in enumerate(COL_KEYS):
        counter = Counter()
        for hid, info in top:
            for tok in info[key]:
                counter[tok] += 1
        col_top5[ci] = set(tok for tok, _ in counter.most_common(5))

    fig, axes = plt.subplots(n, 4, figsize=(22, 2.5 * n + 0.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"M2 Layer {layer_num}: Top {n} Heads — Token Projections\n"
        f"(red = top 5 most frequent across heads in that column)",
        fontsize=14, fontweight="bold", y=1.005,
    )

    for col, ctitle in enumerate(COL_TITLES):
        axes[0, col].set_title(ctitle, fontsize=12, fontweight="bold", pad=8)

    for row, (hid, info) in enumerate(top):
        for col, key in enumerate(COL_KEYS):
            ax = axes[row, col]
            tokens = info[key]
            ntok = len(tokens)
            if ntok == 0:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="#ccc")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                y_pos = list(range(ntok - 1, -1, -1))
                bar_vals = [1.0 - i * 0.07 for i in range(ntok)]
                bar_colors = [RED if tok in col_top5[col] else M2_COLOR[col]
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
                qk_s = info["qk_sigma"]
                dqk = info["delta_qk"]
                dov = info["delta_ov"]
                label = f"H{hid}\nσ={qk_s:.2f}"
                if dqk > 0:
                    label += f"\n|ΔQK|={dqk:.2f}"
                if dov > 0:
                    label += f"\n|ΔOV|={dov:.2f}"
                ax.text(-0.32, 0.5, label, transform=ax.transAxes,
                        fontsize=8, fontweight="bold", va="center", ha="center",
                        color="#E65100",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                                  edgecolor="#FFB74D", alpha=0.9))

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False)

    plt.tight_layout(rect=[0.06, 0, 1, 0.99], h_pad=0.6, w_pad=0.5)
    path = f"{OUT}/plots_per_layer/M2_L{layer_num}_tokens.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# Generate M2 per-layer plots
for layer in all_m2_layers:
    layer_str = str(layer)
    heads = all_layer_data.get(layer_str, {})
    if len(heads) >= 1:
        path = plot_m2_layer(layer)
        if path:
            print(f"  L{layer}: {len(heads)} heads → {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Cross-model comparison plots (layers 0-9)
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Cross-model comparison plots (L0-L9)...")

HEADS_PER_MODEL = 8

def plot_cross_model(layer_num):
    layer_str = str(layer_num)
    m1_heads = parse_log_layer(M1_LOG, layer_num)
    m2_heads = parse_log_layer(M2_LOG, layer_num)
    merge_m1_json(m1_heads, m1_json, layer_str)
    merge_m2_json(m2_heads, m2_json, layer_str)

    m1_top = get_top_heads(m1_heads, HEADS_PER_MODEL)
    m2_top = get_top_heads(m2_heads, HEADS_PER_MODEL)
    n_m1, n_m2 = len(m1_top), len(m2_top)
    n_rows = max(n_m1, n_m2)
    if n_rows == 0:
        return

    m1_top5, m2_top5 = {}, {}
    for ci, key in enumerate(COL_KEYS):
        c1, c2 = Counter(), Counter()
        for _, info in m1_top:
            for tok in info[key][:N_TOKENS]:
                c1[tok] += 1
        for _, info in m2_top:
            for tok in info[key][:N_TOKENS]:
                c2[tok] += 1
        m1_top5[ci] = set(tok for tok, _ in c1.most_common(5))
        m2_top5[ci] = set(tok for tok, _ in c2.most_common(5))

    fig, axes = plt.subplots(n_rows, 8, figsize=(38, 2.5 * n_rows + 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Layer {layer_num}: M1 (blue) vs M2 (orange) — Top {HEADS_PER_MODEL} Heads\n"
        f"Red = top 5 most frequent tokens across heads in that model+column",
        fontsize=14, fontweight="bold", y=0.998,
    )

    for ci, title in enumerate(COL_TITLES):
        axes[0, ci * 2].set_title(f"M1 {title}", fontsize=9.5, fontweight="bold",
                                   pad=8, color="#1565C0")
        axes[0, ci * 2 + 1].set_title(f"M2 {title}", fontsize=9.5, fontweight="bold",
                                       pad=8, color="#E65100")

    for sep_x in [0.265, 0.505, 0.745]:
        fig.patches.append(plt.Rectangle(
            (sep_x, 0.02), 0.004, 0.95,
            transform=fig.transFigure, facecolor="#e0e0e0", edgecolor="none", zorder=-1
        ))

    def draw_cell(ax, tokens, top5_set, default_color):
        ntok = len(tokens)
        if ntok == 0:
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#ccc")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            y_pos = list(range(ntok - 1, -1, -1))
            bar_vals = [1.0 - i * 0.07 for i in range(ntok)]
            bar_colors = [RED if tok in top5_set else default_color for tok in tokens]
            ax.barh(y_pos, bar_vals, color=bar_colors, height=0.7,
                    edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.set_yticks(y_pos)
            safe = [repr(t).replace("$", "\\$") for t in tokens]
            ax.set_yticklabels(safe, fontsize=6.5, fontfamily="monospace")
            for tick_label, tok in zip(ax.get_yticklabels(), tokens):
                if tok in top5_set:
                    tick_label.set_color(RED)
                    tick_label.set_fontweight("bold")
            ax.set_xlim(0, 1.15)
            ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)

    def draw_label(ax, hid, info, model):
        qk_s = info["qk_sigma"]
        dqk = info["delta_qk"]
        dov = info["delta_ov"]
        label = f"{model} H{hid}\nσ={qk_s:.2f}"
        if dqk > 0:
            label += f"\n|ΔQK|={dqk:.2f}"
        if dov > 0:
            label += f"\n|ΔOV|={dov:.2f}"
        is_m1 = model == "M1"
        ax.text(-0.40, 0.5, label, transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="center", ha="center",
                color="#1565C0" if is_m1 else "#E65100",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#e3f2fd" if is_m1 else "#fff3e0",
                          edgecolor="#90CAF9" if is_m1 else "#FFB74D",
                          alpha=0.9))

    for row in range(n_rows):
        for ci, key in enumerate(COL_KEYS):
            m1_col, m2_col = ci * 2, ci * 2 + 1
            if row < n_m1:
                hid, info = m1_top[row]
                draw_cell(axes[row, m1_col], info[key][:N_TOKENS], m1_top5[ci], M1_COLOR[ci])
                if ci == 0:
                    draw_label(axes[row, m1_col], hid, info, "M1")
            else:
                axes[row, m1_col].set_visible(False)
            if row < n_m2:
                hid, info = m2_top[row]
                draw_cell(axes[row, m2_col], info[key][:N_TOKENS], m2_top5[ci], M2_COLOR[ci])
                if ci == 0:
                    draw_label(axes[row, m2_col], hid, info, "M2")
            else:
                axes[row, m2_col].set_visible(False)

    plt.tight_layout(rect=[0.05, 0, 1, 0.975], h_pad=0.6, w_pad=0.5)
    path = f"{OUT}/plots_cross_model/cross_L{layer_num}_M1_vs_M2.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    L{layer_num} → {os.path.basename(path)}")


for layer in range(10):
    plot_cross_model(layer)


# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"Export complete! Files in: {OUT}/")
print(f"{'=' * 60}")
print(f"  m2_all_layers_tokens.json  — full token data, all layers")
print(f"  m2_layer_summary.tsv       — sortable per-layer summary")
print(f"  m2_circuit_flow.txt        — narrative analysis")
print(f"  plots_per_layer/           — M2 token plots per layer")
print(f"  plots_cross_model/         — M1 vs M2 comparison (L0-L9)")
