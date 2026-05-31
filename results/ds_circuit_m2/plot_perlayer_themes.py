#!/usr/bin/env python3
"""Per-layer theme clustering & visualization for M2 circuit SVD results.

Classifies input/output tokens into semantic themes, then visualizes:
  1. Per-layer input theme distribution (stacked bar)
  2. Per-layer output theme distribution (stacked bar)
  3. Theme evolution heatmap (layers × themes)
  4. Top output-token frequency across layer groups
  5. Head clustering dendrogram by token signature similarity
"""

import json
import re
import unicodedata
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

OUT = "/Users/omard/Documents/projects/AI_projects/jsllm/results/ds_circuit_m2"

# ── Load data ───────────────────────────────────────────────────────────
with open(f"{OUT}/circuit_svd_reduced.json") as f:
    data = json.load(f)

layers = sorted(data.keys(), key=int)
print(f"Loaded {len(layers)} layers, layers: {layers}")


# ── Token classification ────────────────────────────────────────────────
def classify_token(tok: str) -> str:
    """Classify a token into a semantic theme."""
    tok_stripped = tok.strip()

    # Special model tokens
    if tok_stripped.startswith("<｜") or tok_stripped.startswith("<|"):
        return "special_token"

    # Pure numeric
    if re.fullmatch(r"\d+", tok_stripped):
        return "numeric"

    # LaTeX / math notation
    if any(
        p in tok
        for p in [
            "\\(",
            "\\)",
            "\\[",
            "\\]",
            "\\displaystyle",
            "mathrm",
            "^{",
            "_{",
            "qquad",
            "\\left",
            "\\right",
            "overline",
            "superscript",
            "subscript",
            "sigma",
            "gamma",
            "dagger",
        ]
    ):
        return "latex_math"

    # Code / syntax patterns
    if re.fullmatch(
        r"[\[\]{}()<>=!+\-*/&|^~%@#;:,.\\\n\t\r ]+", tok_stripped
    ) or any(
        p in tok
        for p in [
            "def ",
            "class ",
            "import ",
            "return ",
            "self.",
            "print(",
            "->",
            "=>",
            ".py",
            ".js",
            ".txt",
            "namespace",
            "void",
            "int ",
            "SELECT",
            "FROM",
            "WHERE",
            "package",
            "foreach",
            "forEach",
            "jupyter",
            "_PATH",
            "_CON",
            "FFFF",
            "xxxx",
            "yyyy",
            "ICAgICAg",
        ]
    ):
        return "code_syntax"

    # Bracket / punctuation heavy (>=50% non-alphanumeric)
    non_alnum = sum(1 for c in tok_stripped if not c.isalnum())
    if len(tok_stripped) > 0 and non_alnum / len(tok_stripped) > 0.6:
        return "punctuation"

    # CJK characters
    cjk_count = sum(
        1
        for c in tok
        if "\u4e00" <= c <= "\u9fff"  # CJK Unified
        or "\u3400" <= c <= "\u4dbf"  # CJK Ext A
        or "\uf900" <= c <= "\ufaff"  # CJK Compat
        or "\u3040" <= c <= "\u309f"  # Hiragana
        or "\u30a0" <= c <= "\u30ff"  # Katakana
        or "\uac00" <= c <= "\ud7af"  # Hangul
    )
    if cjk_count > 0 and cjk_count / max(len(tok_stripped), 1) > 0.3:
        return "cjk"

    # Cyrillic
    cyrillic = sum(1 for c in tok if "\u0400" <= c <= "\u04ff")
    if cyrillic > 0 and cyrillic / max(len(tok_stripped), 1) > 0.3:
        return "cyrillic"

    # Arabic / RTL
    arabic = sum(1 for c in tok if "\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f")
    if arabic > 0:
        return "arabic_rtl"

    # English-like words
    alpha = sum(1 for c in tok if c.isascii() and c.isalpha())
    if alpha > 0 and alpha / max(len(tok_stripped), 1) > 0.5:
        return "english"

    return "other"


THEME_ORDER = [
    "english",
    "numeric",
    "cjk",
    "cyrillic",
    "arabic_rtl",
    "code_syntax",
    "latex_math",
    "punctuation",
    "special_token",
    "other",
]
THEME_COLORS = {
    "english": "#3498db",
    "numeric": "#e74c3c",
    "cjk": "#2ecc71",
    "cyrillic": "#9b59b6",
    "arabic_rtl": "#e67e22",
    "code_syntax": "#1abc9c",
    "latex_math": "#f39c12",
    "punctuation": "#95a5a6",
    "special_token": "#34495e",
    "other": "#bdc3c7",
}
THEME_LABELS = {
    "english": "English",
    "numeric": "Numeric",
    "cjk": "CJK",
    "cyrillic": "Cyrillic",
    "arabic_rtl": "Arabic/RTL",
    "code_syntax": "Code/Syntax",
    "latex_math": "LaTeX/Math",
    "punctuation": "Punctuation",
    "special_token": "Special Token",
    "other": "Other",
}


# ── Classify all tokens per layer ───────────────────────────────────────
layer_in_themes = {}   # layer -> Counter of themes
layer_out_themes = {}  # layer -> Counter of themes
layer_in_tokens = {}   # layer -> list of (token, sigma, theme)
layer_out_tokens = {}  # layer -> list of (token, sigma, theme)

for l in layers:
    in_counter = Counter()
    out_counter = Counter()
    in_toks = []
    out_toks = []
    for h in data[l]:
        entry = data[l][h]
        for key in ["ov"]:
            if key not in entry:
                continue
            for item in entry[key]:
                sigma = item.get("s", 0)
                for tok in item.get("in", []):
                    theme = classify_token(tok)
                    in_counter[theme] += 1
                    in_toks.append((tok, sigma, theme))
                for tok in item.get("out", []):
                    theme = classify_token(tok)
                    out_counter[theme] += 1
                    out_toks.append((tok, sigma, theme))
    layer_in_themes[l] = in_counter
    layer_out_themes[l] = out_counter
    layer_in_tokens[l] = in_toks
    layer_out_tokens[l] = out_toks


# ═════════════════════════════════════════════════════════════════════════
# PLOT 1: Per-layer INPUT theme distribution (stacked bar)
# ═════════════════════════════════════════════════════════════════════════
print("Plotting perlayer_input_themes.png ...")
fig, ax = plt.subplots(figsize=(20, 7))
x = np.arange(len(layers))
bottoms = np.zeros(len(layers))

# Normalize to percentages
totals = np.array([max(sum(layer_in_themes[l].values()), 1) for l in layers], dtype=float)

for theme in THEME_ORDER:
    vals = np.array([layer_in_themes[l].get(theme, 0) for l in layers], dtype=float)
    pcts = vals / totals * 100
    ax.bar(
        x, pcts, bottom=bottoms, width=0.85,
        color=THEME_COLORS[theme], label=THEME_LABELS[theme],
        edgecolor="white", linewidth=0.3,
    )
    bottoms += pcts

ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in layers], rotation=70, ha="right", fontsize=7)
ax.set_ylabel("% of Input Tokens")
ax.set_title("M2 Circuit: Per-Layer Input Token Themes (OV reads)", fontsize=14, pad=12)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(-0.5, len(layers) - 0.5)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{OUT}/perlayer_input_themes.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 2: Per-layer OUTPUT theme distribution (stacked bar)
# ═════════════════════════════════════════════════════════════════════════
print("Plotting perlayer_output_themes.png ...")
fig, ax = plt.subplots(figsize=(20, 7))
bottoms = np.zeros(len(layers))
totals_out = np.array([max(sum(layer_out_themes[l].values()), 1) for l in layers], dtype=float)

for theme in THEME_ORDER:
    vals = np.array([layer_out_themes[l].get(theme, 0) for l in layers], dtype=float)
    pcts = vals / totals_out * 100
    ax.bar(
        x, pcts, bottom=bottoms, width=0.85,
        color=THEME_COLORS[theme], label=THEME_LABELS[theme],
        edgecolor="white", linewidth=0.3,
    )
    bottoms += pcts

ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in layers], rotation=70, ha="right", fontsize=7)
ax.set_ylabel("% of Output Tokens")
ax.set_title("M2 Circuit: Per-Layer Output Token Themes (OV writes)", fontsize=14, pad=12)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(-0.5, len(layers) - 0.5)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{OUT}/perlayer_output_themes.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 3: Theme evolution heatmap (layers × themes, both in and out)
# ═════════════════════════════════════════════════════════════════════════
print("Plotting theme_evolution_heatmap.png ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

for ax, themes_dict, title in [
    (ax1, layer_in_themes, "Input Token Themes (OV reads)"),
    (ax2, layer_out_themes, "Output Token Themes (OV writes)"),
]:
    mat = np.zeros((len(THEME_ORDER), len(layers)))
    for li, l in enumerate(layers):
        total = max(sum(themes_dict[l].values()), 1)
        for ti, theme in enumerate(THEME_ORDER):
            mat[ti, li] = themes_dict[l].get(theme, 0) / total * 100

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(THEME_ORDER)))
    ax.set_yticklabels([THEME_LABELS[t] for t in THEME_ORDER], fontsize=9)
    ax.set_title(f"M2 Circuit: {title}", fontsize=13, pad=8)
    fig.colorbar(im, ax=ax, label="% of tokens", fraction=0.02, pad=0.01)

plt.tight_layout()
plt.savefig(f"{OUT}/theme_evolution_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 4: Top output tokens — frequency across layer groups
# ═════════════════════════════════════════════════════════════════════════
print("Plotting output_token_layergroups.png ...")

# Group layers: early (0-9), mid (10-39), late (40-60)
groups = {
    "Early (L0-9)": [l for l in layers if int(l) <= 9],
    "Mid (L10-39)": [l for l in layers if 10 <= int(l) <= 39],
    "Late (L40-60)": [l for l in layers if int(l) >= 40],
}

# Get top 20 output tokens overall
all_out_counter = Counter()
for l in layers:
    for tok, sigma, theme in layer_out_tokens[l]:
        all_out_counter[tok] += 1
top_out = [tok for tok, _ in all_out_counter.most_common(20)]

fig, ax = plt.subplots(figsize=(16, 8))
bar_width = 0.25
group_names = list(groups.keys())
group_colors = ["#3498db", "#e74c3c", "#2ecc71"]

for gi, (gname, glayers) in enumerate(groups.items()):
    group_counter = Counter()
    for l in glayers:
        for tok, sigma, theme in layer_out_tokens[l]:
            group_counter[tok] += 1
    vals = [group_counter.get(tok, 0) for tok in top_out]
    positions = np.arange(len(top_out)) + gi * bar_width
    ax.bar(positions, vals, width=bar_width, label=gname, color=group_colors[gi],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(np.arange(len(top_out)) + bar_width)
ax.set_xticklabels([repr(t) for t in top_out], rotation=55, ha="right", fontsize=8)
ax.set_ylabel("Count (# heads × appearances)")
ax.set_title("M2 Circuit: Top 20 Output Tokens by Layer Group", fontsize=14, pad=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/output_token_layergroups.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 5: Head clustering dendrogram by token signature
# ═════════════════════════════════════════════════════════════════════════
print("Plotting head_clustering.png ...")

# Build a feature vector per head: presence of top output tokens + input theme vector
# Use top 30 output tokens as vocabulary
top_out_vocab = [tok for tok, _ in all_out_counter.most_common(30)]
out_tok_to_idx = {tok: i for i, tok in enumerate(top_out_vocab)}

head_labels = []
head_vectors = []

for l in layers:
    for h in data[l]:
        entry = data[l][h]
        # Output token features
        out_vec = np.zeros(len(top_out_vocab))
        in_theme_vec = np.zeros(len(THEME_ORDER))

        for item in entry.get("ov", []):
            for tok in item.get("out", []):
                if tok in out_tok_to_idx:
                    out_vec[out_tok_to_idx[tok]] = 1.0
            for tok in item.get("in", []):
                theme = classify_token(tok)
                ti = THEME_ORDER.index(theme)
                in_theme_vec[ti] += 1

        # Normalize input theme vector
        total_in = in_theme_vec.sum()
        if total_in > 0:
            in_theme_vec /= total_in

        feature = np.concatenate([out_vec, in_theme_vec])
        head_vectors.append(feature)
        head_labels.append(f"L{l}H{h}")

head_matrix = np.array(head_vectors)
print(f"  Head matrix shape: {head_matrix.shape}")

# Compute linkage
dist = pdist(head_matrix, metric="cosine")
dist = np.nan_to_num(dist, nan=1.0)
Z = linkage(dist, method="ward")

# Assign clusters
n_clusters = 8
cluster_ids = fcluster(Z, n_clusters, criterion="maxclust")

# Plot dendrogram with color coding
fig, ax = plt.subplots(figsize=(24, 8))
dn = dendrogram(
    Z, labels=head_labels, ax=ax,
    leaf_rotation=90, leaf_font_size=4,
    color_threshold=Z[-n_clusters + 1, 2] if n_clusters > 1 else 0,
    above_threshold_color="#999",
)
ax.set_title(f"M2 Circuit: Head Clustering by Token Signature ({n_clusters} clusters)",
             fontsize=14, pad=12)
ax.set_ylabel("Distance (Ward)")
ax.set_xlabel("Head (Layer.Head)")

plt.tight_layout()
plt.savefig(f"{OUT}/head_clustering.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 6: Cluster summary — what tokens define each cluster
# ═════════════════════════════════════════════════════════════════════════
print("Plotting cluster_profiles.png ...")

# For each cluster, find the most common output tokens and input themes
cluster_data = defaultdict(lambda: {"out": Counter(), "in_themes": Counter(), "heads": [], "layers": []})

for i, (label, cid) in enumerate(zip(head_labels, cluster_ids)):
    l = label.split("H")[0][1:]  # extract layer number
    h = label.split("H")[1]
    cluster_data[cid]["heads"].append(label)
    cluster_data[cid]["layers"].append(int(l))

    entry = data[l][h]
    for item in entry.get("ov", []):
        for tok in item.get("out", []):
            cluster_data[cid]["out"][tok] += 1
        for tok in item.get("in", []):
            theme = classify_token(tok)
            cluster_data[cid]["in_themes"][theme] += 1

n_actual = len(cluster_data)
fig, axes = plt.subplots(n_actual, 2, figsize=(18, 3.5 * n_actual))
if n_actual == 1:
    axes = axes.reshape(1, -1)

fig.suptitle("M2 Circuit: Cluster Profiles — Output Tokens & Input Themes", fontsize=15, y=1.01)

for row, cid in enumerate(sorted(cluster_data.keys())):
    cd = cluster_data[cid]
    n_heads = len(cd["heads"])
    layer_range = f"L{min(cd['layers'])}-L{max(cd['layers'])}"

    # Left: top 10 output tokens
    ax = axes[row, 0]
    top_out_c = cd["out"].most_common(12)
    if top_out_c:
        tokens_c, counts_c = zip(*top_out_c)
        y = range(len(tokens_c) - 1, -1, -1)
        ax.barh(list(y), counts_c, color="#e74c3c", height=0.7, edgecolor="white", linewidth=0.3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([repr(t) for t in tokens_c], fontsize=8)
    ax.set_title(f"Cluster {cid}: Output Tokens  ({n_heads} heads, {layer_range})", fontsize=10)
    ax.set_xlabel("Count")

    # Right: input theme pie
    ax = axes[row, 1]
    theme_counts = cd["in_themes"]
    theme_vals = [theme_counts.get(t, 0) for t in THEME_ORDER]
    theme_total = sum(theme_vals)
    if theme_total > 0:
        # Filter out zero themes for cleaner pie
        nonzero = [(THEME_LABELS[t], v, THEME_COLORS[t]) for t, v in zip(THEME_ORDER, theme_vals) if v > 0]
        if nonzero:
            labels_pie, vals_pie, colors_pie = zip(*nonzero)
            wedges, texts, autotexts = ax.pie(
                vals_pie, labels=labels_pie, colors=colors_pie,
                autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
                startangle=90, textprops={"fontsize": 7},
            )
    ax.set_title(f"Cluster {cid}: Input Theme Distribution", fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PLOT 7: Layer activity heatmap — # active heads + mean sigma per layer
# ═════════════════════════════════════════════════════════════════════════
print("Plotting layer_activity.png ...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

n_heads_per_layer = [len(data[l]) for l in layers]
mean_sigma = []
for l in layers:
    sigmas = []
    for h in data[l]:
        for item in data[l][h].get("ov", []):
            sigmas.append(item.get("s", 0))
    mean_sigma.append(np.mean(sigmas) if sigmas else 0)

ax1.bar(x, n_heads_per_layer, color="#3498db", edgecolor="white", linewidth=0.5)
ax1.set_ylabel("# Active Heads")
ax1.set_title("M2 Circuit: Active Heads per Layer (top by ΔOV norm)", fontsize=13, pad=8)
for i, v in enumerate(n_heads_per_layer):
    if v > 20:
        ax1.text(i, v + 1, str(v), ha="center", va="bottom", fontsize=7)

ax2.bar(x, mean_sigma, color="#e74c3c", edgecolor="white", linewidth=0.5)
ax2.set_ylabel(r"Mean $\sigma_1$")
ax2.set_title("M2 Circuit: Mean σ₁ per Layer", fontsize=13, pad=8)
ax2.set_xticks(x)
ax2.set_xticklabels([f"L{l}" for l in layers], rotation=70, ha="right", fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT}/layer_activity.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll plots saved to {OUT}/")
print("  perlayer_input_themes.png")
print("  perlayer_output_themes.png")
print("  theme_evolution_heatmap.png")
print("  output_token_layergroups.png")
print("  head_clustering.png")
print("  cluster_profiles.png")
print("  layer_activity.png")
