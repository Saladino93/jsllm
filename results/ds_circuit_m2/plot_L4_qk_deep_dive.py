#!/usr/bin/env python3
"""Layer 4 QK deep dive — query/key token projections, ΔOV vs ΔQK, and query dominance analysis."""

import json
import re
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2"

# ── Load reduced JSON ──────────────────────────────────────────────────
with open(f"{OUT}/circuit_svd_reduced.json") as f:
    data = json.load(f)

# ── Parse full log for L4 ΔOV/ΔQK norms ────────────────────────────────
log_path = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2_log.txt"
l4_norms = {}  # head_id -> (delta_ov, delta_qk, ov_rank, qk_rank)

with open(log_path) as f:
    in_layer4 = False
    for line in f:
        if line.strip() == "Layer 4":
            in_layer4 = True
            continue
        if in_layer4 and line.startswith("===") and "Layer 4" not in line:
            # Check if it's the next layer header
            pass
        if in_layer4 and line.strip().startswith("Loading weights for layer"):
            break  # Done with layer 4
        if in_layer4 and "|ΔOV|=" in line:
            # Parse: "  Head  13  |ΔOV|=0.3970 (rank≈2)  |ΔQK|=5.1613 (rank≈2)"
            m = re.search(r"Head\s+(\d+)\s+\|ΔOV\|=([\d.]+)\s+\(rank≈(\d+)\)\s+\|ΔQK\|=([\d.]+)\s+\(rank≈(\d+)\)", line)
            if m:
                hid = m.group(1)
                l4_norms[hid] = {
                    "delta_ov": float(m.group(2)),
                    "ov_rank": int(m.group(3)),
                    "delta_qk": float(m.group(4)),
                    "qk_rank": int(m.group(5)),
                }

print(f"Parsed {len(l4_norms)} heads from log for Layer 4")

# ── Merge QK data from JSON ─────────────────────────────────────────────
l4 = data["4"]
heads_sorted_by_qk_sigma = []
for h in l4:
    entry = l4[h]
    if "qk" in entry and len(entry["qk"]) > 0:
        qk0 = entry["qk"][0]
        sigma = qk0["s"]
        norms = l4_norms.get(h, {})
        heads_sorted_by_qk_sigma.append({
            "head": int(h),
            "qk_sigma1": sigma,
            "query_tokens": qk0["q"],
            "key_tokens": qk0["k"],
            "delta_ov": norms.get("delta_ov", 0),
            "delta_qk": norms.get("delta_qk", 0),
            "ov_rank": norms.get("ov_rank", 0),
            "qk_rank": norms.get("qk_rank", 0),
        })

heads_sorted_by_qk_sigma.sort(key=lambda x: -x["qk_sigma1"])
print(f"Layer 4: {len(heads_sorted_by_qk_sigma)} heads with QK data")


# ═════════════════════════════════════════════════════════════════════════
# PLOT 1: Top 15 heads by QK σ₁ — Query & Key token bars
# ═════════════════════════════════════════════════════════════════════════
print("Plotting L4_qk_top15.png ...")
top_n = 15
top = heads_sorted_by_qk_sigma[:top_n]

fig, axes = plt.subplots(top_n, 2, figsize=(18, 3.2 * top_n))
fig.suptitle("M2 Layer 4: Top 15 QK Heads — Query & Key Projections (rank-1 direction)",
             fontsize=16, y=1.005, fontweight="bold")

for row, hd in enumerate(top):
    hi = hd["head"]
    sigma = hd["qk_sigma1"]
    dov = hd["delta_ov"]
    dqk = hd["delta_qk"]

    q_tok = hd["query_tokens"][:8]
    k_tok = hd["key_tokens"][:8]

    # Query bar
    ax = axes[row, 0]
    y = range(len(q_tok) - 1, -1, -1)
    colors_q = ["#c0392b"] * len(q_tok)
    # Highlight "Different" family
    for i, t in enumerate(q_tok):
        if "ifferent" in t or "differing" in t:
            colors_q[i] = "#e74c3c"
        else:
            colors_q[i] = "#3498db"
    ax.barh(list(y), [1] * len(q_tok), color=colors_q, height=0.7, edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([repr(t) for t in q_tok], fontsize=8, fontfamily="monospace")
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    title_q = f"H{hi} Query  (σ₁={sigma:.1f}, |ΔQK|={dqk:.1f})"
    ax.set_title(title_q, fontsize=10, fontweight="bold", loc="left")

    # Key bar
    ax = axes[row, 1]
    y = range(len(k_tok) - 1, -1, -1)
    ax.barh(list(y), [1] * len(k_tok), color="#27ae60", height=0.7, edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([repr(t) for t in k_tok], fontsize=8, fontfamily="monospace")
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    title_k = f"H{hi} Key  (|ΔOV|={dov:.2f})"
    ax.set_title(title_k, fontsize=10, loc="left")

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(f"{OUT}/L4_qk_top15.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 2: ΔOV vs ΔQK scatter for all L4 heads
# ═════════════════════════════════════════════════════════════════════════
print("Plotting L4_delta_scatter.png ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Scatter: ΔOV vs ΔQK
dovs = [h["delta_ov"] for h in heads_sorted_by_qk_sigma]
dqks = [h["delta_qk"] for h in heads_sorted_by_qk_sigma]
sigmas = [h["qk_sigma1"] for h in heads_sorted_by_qk_sigma]
heads_ids = [h["head"] for h in heads_sorted_by_qk_sigma]

sc = ax1.scatter(dqks, dovs, c=sigmas, cmap="hot_r", s=60, edgecolors="black",
                 linewidth=0.5, zorder=5)
fig.colorbar(sc, ax=ax1, label="QK σ₁", fraction=0.03, pad=0.02)

# Label top 5
for i in range(min(5, len(heads_sorted_by_qk_sigma))):
    h = heads_sorted_by_qk_sigma[i]
    ax1.annotate(f"H{h['head']}", (h["delta_qk"], h["delta_ov"]),
                 textcoords="offset points", xytext=(8, 5), fontsize=8,
                 fontweight="bold", color="#c0392b")

ax1.set_xlabel("|ΔQK| (norm of QK weight change)", fontsize=11)
ax1.set_ylabel("|ΔOV| (norm of OV weight change)", fontsize=11)
ax1.set_title("Layer 4: ΔOV vs ΔQK per Head", fontsize=13, fontweight="bold")
ax1.axhline(np.median(dovs), color="gray", ls="--", alpha=0.5, lw=1)
ax1.axvline(np.median(dqks), color="gray", ls="--", alpha=0.5, lw=1)

# Bar: QK σ₁ for all heads, sorted
ax2.barh(range(len(sigmas)), sigmas, color="#e74c3c", height=0.7, edgecolor="white", linewidth=0.3)
ax2.set_yticks(range(len(sigmas)))
ax2.set_yticklabels([f"H{h}" for h in heads_ids], fontsize=6)
ax2.set_xlabel("QK σ₁", fontsize=11)
ax2.set_title("Layer 4: All Heads Sorted by QK σ₁", fontsize=13, fontweight="bold")
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUT}/L4_delta_scatter.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 3: Query token dominance — what fraction of heads have "Different"
# ═════════════════════════════════════════════════════════════════════════
print("Plotting L4_query_dominance.png ...")

# Count how many heads have each query token as top-1
top1_counter = Counter()
top3_counter = Counter()
different_family = set()

for hd in heads_sorted_by_qk_sigma:
    q = hd["query_tokens"]
    if q:
        top1_counter[q[0]] += 1
        for t in q[:3]:
            top3_counter[t] += 1
        # Check if any of top-3 are "Different" family
        for t in q[:3]:
            if "ifferent" in t or "differing" in t or "differently" in t:
                different_family.add(hd["head"])

n_diff = len(different_family)
n_total = len(heads_sorted_by_qk_sigma)
pct_diff = n_diff / n_total * 100

# Also categorize key tokens into themes
key_themes = Counter()
for hd in heads_sorted_by_qk_sigma:
    for tok in hd["key_tokens"][:3]:
        tok_s = tok.strip()
        # Simple theme classification
        if re.fullmatch(r"\d+", tok_s):
            key_themes["Numbers"] += 1
        elif any(c in tok for c in ["{", "}", "(", ")", "[", "]", ";", "=", "<", ">", "\\", "#"]):
            key_themes["Code/Syntax"] += 1
        elif any("\u4e00" <= c <= "\u9fff" for c in tok):
            key_themes["CJK"] += 1
        elif any(c in tok for c in ["\n", "\t", " "]) and len(tok_s) <= 3:
            key_themes["Whitespace"] += 1
        elif tok_s.isascii() and any(c.isalpha() for c in tok_s):
            key_themes["English"] += 1
        else:
            key_themes["Other"] += 1

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("M2 Layer 4: Query/Key Pattern Analysis", fontsize=15, fontweight="bold", y=1.02)

# Panel 1: Top-1 query token frequency
ax = axes[0]
top_queries = top1_counter.most_common(12)
if top_queries:
    tokens_q, counts_q = zip(*top_queries)
    y = range(len(tokens_q) - 1, -1, -1)
    colors = ["#e74c3c" if ("ifferent" in t or "differing" in t) else "#3498db"
              for t in tokens_q]
    ax.barh(list(y), counts_q, color=colors, height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels([repr(t) for t in tokens_q], fontsize=9, fontfamily="monospace")
ax.set_xlabel("# Heads with this as top-1 query token")
ax.set_title("Top-1 Query Tokens", fontsize=12, fontweight="bold")

# Panel 2: "Different" family dominance
ax = axes[1]
sizes = [n_diff, n_total - n_diff]
labels = [f'"Different" family\n({n_diff} heads, {pct_diff:.0f}%)',
          f'Other queries\n({n_total - n_diff} heads)']
colors_pie = ["#e74c3c", "#bdc3c7"]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                   autopct="%1.0f%%", startangle=90,
                                   textprops={"fontsize": 10})
ax.set_title(f'Query Dominance: "Different" Family\n({n_diff}/{n_total} heads)',
             fontsize=12, fontweight="bold")

# Panel 3: Key token themes
ax = axes[2]
themes = list(key_themes.keys())
vals = [key_themes[t] for t in themes]
theme_colors = {
    "Code/Syntax": "#1abc9c",
    "English": "#3498db",
    "Numbers": "#e74c3c",
    "CJK": "#2ecc71",
    "Whitespace": "#95a5a6",
    "Other": "#bdc3c7",
}
colors_bar = [theme_colors.get(t, "#bdc3c7") for t in themes]
# Sort by count
sorted_pairs = sorted(zip(themes, vals, colors_bar), key=lambda x: -x[1])
themes_s, vals_s, colors_s = zip(*sorted_pairs)
y = range(len(themes_s) - 1, -1, -1)
ax.barh(list(y), vals_s, color=colors_s, height=0.6)
ax.set_yticks(list(y))
ax.set_yticklabels(themes_s, fontsize=11)
ax.set_xlabel("# Tokens in top-3 keys across all heads")
ax.set_title("Key Token Themes (top-3 per head)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/L4_query_dominance.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 4: Key diversity — what do different heads attend to?
# Group heads by key pattern type and show examples
# ═════════════════════════════════════════════════════════════════════════
print("Plotting L4_key_diversity.png ...")

# Categorize each head by its key theme
head_categories = {}
for hd in heads_sorted_by_qk_sigma:
    keys = hd["key_tokens"][:3]
    # Dominant theme
    cats = []
    for tok in keys:
        tok_s = tok.strip()
        if re.fullmatch(r"\d+", tok_s):
            cats.append("Numbers")
        elif any(c in tok for c in [";", "=", "{", "}", "(", ")", "\\", "#", "`", "."]):
            cats.append("Code/Delimiters")
        elif any("\u4e00" <= c <= "\u9fff" for c in tok):
            cats.append("CJK")
        elif "\n" in tok or "\t" in tok:
            cats.append("Newlines/Whitespace")
        elif tok_s.isascii() and any(c.isalpha() for c in tok_s):
            cats.append("English Words")
        else:
            cats.append("Other")
    dominant = max(set(cats), key=cats.count) if cats else "Other"
    head_categories[hd["head"]] = {
        "category": dominant,
        "sigma": hd["qk_sigma1"],
        "keys": hd["key_tokens"][:6],
    }

# Group
groups = {}
for hid, info in head_categories.items():
    cat = info["category"]
    if cat not in groups:
        groups[cat] = []
    groups[cat].append((hid, info["sigma"], info["keys"]))

# Sort groups by size
sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

fig, ax = plt.subplots(figsize=(20, 10))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, len(sorted_groups) + 1)

y_pos = len(sorted_groups) + 0.3
ax.text(5, y_pos, "Layer 4: Key Token Diversity — What Does Each Head Attend To?",
        ha="center", va="center", fontsize=16, fontweight="bold")

for i, (cat, members) in enumerate(sorted_groups):
    y = len(sorted_groups) - i - 0.3
    members.sort(key=lambda x: -x[1])
    n = len(members)
    top3_examples = members[:3]

    # Category label
    ax.text(0.2, y, f"{cat}", fontsize=13, fontweight="bold", va="center",
            color="#2c3e50")
    ax.text(2.0, y, f"({n} heads)", fontsize=11, va="center", color="#7f8c8d")

    # Top examples
    example_str = "  |  ".join(
        f"H{hid} σ={s:.1f}: {', '.join(repr(k) for k in keys[:3])}"
        for hid, s, keys in top3_examples
    )
    ax.text(3.2, y, example_str, fontsize=8, va="center", fontfamily="monospace",
            color="#34495e")

    # Separator line
    if i < len(sorted_groups) - 1:
        ax.axhline(y - 0.5, color="#ecf0f1", lw=1, xmin=0.02, xmax=0.98)

plt.tight_layout()
plt.savefig(f"{OUT}/L4_key_diversity.png", dpi=150, bbox_inches="tight")
plt.close()


# ── Print summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"LAYER 4 SUMMARY")
print(f"{'='*60}")
print(f"  Total active heads: {n_total}")
print(f"  Heads with 'Different' query: {n_diff} ({pct_diff:.0f}%)")
print(f"  QK σ₁ range: {min(sigmas):.1f} — {max(sigmas):.1f}")
print(f"  |ΔQK| range: {min(dqks):.1f} — {max(dqks):.1f}")
print(f"  |ΔOV| range: {min(dovs):.2f} — {max(dovs):.2f}")
print(f"  Top 5 heads by QK σ₁:")
for hd in heads_sorted_by_qk_sigma[:5]:
    print(f"    H{hd['head']:3d}  σ₁={hd['qk_sigma1']:6.1f}  |ΔQK|={hd['delta_qk']:6.1f}  |ΔOV|={hd['delta_ov']:.3f}")
    print(f"           query: {hd['query_tokens'][:4]}")
    print(f"           key:   {hd['key_tokens'][:4]}")

print(f"\nAll plots saved to {OUT}/")
print("  L4_qk_top15.png")
print("  L4_delta_scatter.png")
print("  L4_query_dominance.png")
print("  L4_key_diversity.png")
