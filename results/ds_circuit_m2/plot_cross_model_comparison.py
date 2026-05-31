#!/usr/bin/env python3
"""Cross-model comparison: M1 vs M2 top heads per layer, side by side.

For each layer (0-9), plots top N heads from each model showing their
QK Query, QK Key, OV Reads, OV Writes token projections.
Output: one PNG per layer with M1 on left half, M2 on right half.
"""

import json
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────
OUT = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2"
M1_LOG = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_full_log.txt"
M2_LOG = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2_log.txt"
M1_JSON = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit/circuit_svd_results.json"
M2_JSON = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2/circuit_svd_reduced.json"

RED = "#d62728"
N_TOKENS = 8
HEADS_PER_MODEL = 8  # top N heads per model per layer
LAYERS = list(range(10))  # layers 0-9

COL_TITLES = ["QK Query", "QK Key", "OV Reads", "OV Writes"]
COL_KEYS = ["qk_query", "qk_key", "ov_reads", "ov_writes"]
M1_COLOR = ["#5B9BD5", "#5B9BD5", "#70AD47", "#7B68AE"]  # blue/blue/green/purple
M2_COLOR = ["#ED7D31", "#ED7D31", "#FF5252", "#E040FB"]  # orange/orange/red/pink


# ── Parsing ───────────────────────────────────────────────────────────────
def parse_tokens_from_line(line):
    m = re.search(r"\[(.+)\]", line)
    if not m:
        return []
    return re.findall(r"'((?:[^'\\\\]|\\\\.)*)'", m.group(1))


def parse_log_layer(log_path, target_layer):
    """Parse all heads from a specific layer in the log file."""
    heads = {}
    with open(log_path) as f:
        in_target = False
        current_head = None
        current_section = None
        for line in f:
            if line.strip() == f"Layer {target_layer}":
                in_target = True
                continue
            if in_target and (
                "Loading weights for layer" in line
                or (re.match(r"^Layer \d+", line.strip()) and line.strip() != f"Layer {target_layer}")
            ):
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
    """Merge M1 JSON data (different field names: sigma, input_top, output_top, query_tokens, key_tokens)."""
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
    """Merge M2 JSON data (different field names: s, in, out, q, k)."""
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
    """Sort by max(qk_sigma, delta_qk, delta_ov) descending, return top n."""
    sorted_h = sorted(
        all_heads.items(),
        key=lambda x: max(x[1]["qk_sigma"], x[1]["delta_qk"], x[1]["delta_ov"]),
        reverse=True,
    )
    return sorted_h[:n]


# ── Load JSON data ────────────────────────────────────────────────────────
print("Loading JSON data...")
with open(M1_JSON) as f:
    m1_json = json.load(f)
with open(M2_JSON) as f:
    m2_json = json.load(f)


# ── Plot function ─────────────────────────────────────────────────────────
def plot_layer_comparison(layer_num):
    """Create interleaved M1 vs M2 comparison for a single layer.

    Column layout (8 cols):
      M1 QK Query | M2 QK Query | M1 QK Key | M2 QK Key |
      M1 OV Reads | M2 OV Reads | M1 OV Writes | M2 OV Writes
    """
    layer_str = str(layer_num)

    # Parse log data
    m1_heads = parse_log_layer(M1_LOG, layer_num)
    m2_heads = parse_log_layer(M2_LOG, layer_num)

    # Merge JSON
    merge_m1_json(m1_heads, m1_json, layer_str)
    merge_m2_json(m2_heads, m2_json, layer_str)

    # Get top heads
    m1_top = get_top_heads(m1_heads, HEADS_PER_MODEL)
    m2_top = get_top_heads(m2_heads, HEADS_PER_MODEL)

    n_m1 = len(m1_top)
    n_m2 = len(m2_top)
    n_rows = max(n_m1, n_m2)

    if n_rows == 0:
        print(f"  Layer {layer_num}: no heads found, skipping")
        return

    # Compute top-5 frequent tokens per column for each model
    m1_top5 = {}
    m2_top5 = {}
    for ci, key in enumerate(COL_KEYS):
        c1 = Counter()
        for _, info in m1_top:
            for tok in info[key][:N_TOKENS]:
                c1[tok] += 1
        m1_top5[ci] = set(tok for tok, _ in c1.most_common(5))

        c2 = Counter()
        for _, info in m2_top:
            for tok in info[key][:N_TOKENS]:
                c2[tok] += 1
        m2_top5[ci] = set(tok for tok, _ in c2.most_common(5))

    # ── Interleaved layout ──────────────────────────────────────────────
    # 8 columns: pairs of (M1, M2) for each of the 4 token types
    # col 0=M1 QK Query, 1=M2 QK Query, 2=M1 QK Key, 3=M2 QK Key,
    # col 4=M1 OV Reads, 5=M2 OV Reads, 6=M1 OV Writes, 7=M2 OV Writes
    fig, axes = plt.subplots(n_rows, 8, figsize=(38, 2.5 * n_rows + 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Layer {layer_num}: M1 (blue) vs M2 (orange) — Top {HEADS_PER_MODEL} Heads\n"
        f"Red = top 5 most frequent tokens across the {HEADS_PER_MODEL} heads in that model+column",
        fontsize=14, fontweight="bold", y=0.998,
    )

    # Column headers — interleaved pairs
    for ci, title in enumerate(COL_TITLES):
        axes[0, ci * 2].set_title(f"M1 {title}", fontsize=9.5, fontweight="bold",
                                   pad=8, color="#1565C0")
        axes[0, ci * 2 + 1].set_title(f"M2 {title}", fontsize=9.5, fontweight="bold",
                                       pad=8, color="#E65100")

    # Draw vertical separators between pairs (after cols 1, 3, 5)
    for sep_x in [0.265, 0.505, 0.745]:
        fig.patches.append(plt.Rectangle(
            (sep_x, 0.02), 0.004, 0.95,
            transform=fig.transFigure, facecolor="#e0e0e0", edgecolor="none", zorder=-1
        ))

    def draw_cell(ax, tokens, colors_idx, top5_set, default_color):
        """Draw a single bar chart cell."""
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

    def draw_head_label(ax, hid, info, model_label):
        """Draw head info label to the left of the first column."""
        qk_s = info["qk_sigma"]
        dqk = info["delta_qk"]
        dov = info["delta_ov"]
        label = f"{model_label} H{hid}\nσ={qk_s:.2f}"
        if dqk > 0:
            label += f"\n|ΔQK|={dqk:.2f}"
        if dov > 0:
            label += f"\n|ΔOV|={dov:.2f}"
        is_m1 = model_label == "M1"
        ax.text(-0.40, 0.5, label, transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="center", ha="center",
                color="#1565C0" if is_m1 else "#E65100",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#e3f2fd" if is_m1 else "#fff3e0",
                          edgecolor="#90CAF9" if is_m1 else "#FFB74D",
                          alpha=0.9))

    # Fill rows
    for row in range(n_rows):
        for ci, key in enumerate(COL_KEYS):
            m1_col = ci * 2      # M1 column index in the figure
            m2_col = ci * 2 + 1  # M2 column index

            # M1 cell
            if row < n_m1:
                hid, info = m1_top[row]
                tokens = info[key][:N_TOKENS]
                draw_cell(axes[row, m1_col], tokens, ci, m1_top5[ci], M1_COLOR[ci])
                if ci == 0:
                    draw_head_label(axes[row, m1_col], hid, info, "M1")
            else:
                axes[row, m1_col].set_visible(False)

            # M2 cell
            if row < n_m2:
                hid, info = m2_top[row]
                tokens = info[key][:N_TOKENS]
                draw_cell(axes[row, m2_col], tokens, ci, m2_top5[ci], M2_COLOR[ci])
                if ci == 0:
                    # Put M2 head label to the right of M1's QK Query column
                    # using annotation on the M2 QK Query cell
                    draw_head_label(axes[row, m2_col], hid, info, "M2")
            else:
                axes[row, m2_col].set_visible(False)

    plt.tight_layout(rect=[0.05, 0, 1, 0.975], h_pad=0.6, w_pad=0.5)
    path = f"{OUT}/cross_model_L{layer_num}_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    print(f"    M1: {len(m1_heads)} heads parsed, top {n_m1} shown")
    print(f"    M2: {len(m2_heads)} heads parsed, top {n_m2} shown")


# ── Generate all layers ───────────────────────────────────────────────────
for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    plot_layer_comparison(layer)

print("\nDone!")
