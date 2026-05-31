"""
Layer Community Detection via Coherence Maps
=============================================
Applies spectral clustering, Louvain modularity, and hierarchical clustering
to the layer x layer coherence matrices (coh_max) from M1, M2, M3.

Then gathers per-layer top tokens from head_clusters_v2.txt files and finds
which tokens are shared within each detected community.

Outputs:
  - Dendrogram plots per model
  - Community membership report (TXT)
  - Token community summary (tokens bridging multiple communities)
"""

import os, re, glob
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
import networkx as nx
from networkx.algorithms.community import louvain_communities

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/Users/omard/Documents/projects/AI_projects/jsllm/experiments/EXP-016_cross_layer_story"
CLUSTER_DIR = os.path.join(BASE, "head_coherence", "clusters")
OUT_DIR = os.path.join(BASE, "layer_communities")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["m1", "m2", "m3"]
MODEL_LABELS = {"m1": "M1", "m2": "M2", "m3": "M3"}

# ── 1. Load coherence matrices ────────────────────────────────────────────────
data = {}
for m in MODELS:
    npz = np.load(os.path.join(BASE, f"coherence_map_{m}.npz"), allow_pickle=True)
    data[m] = {
        "keys": [str(k) for k in npz["keys"]],  # convert from np.str_ to plain str
        "coh_max": npz["coh_max"].astype(np.float64),
        "coh_avg": npz["coh_avg"].astype(np.float64),
        "coh_d0": npz["coh_d0"].astype(np.float64),
    }
    print(f"Loaded {m}: {len(data[m]['keys'])} layer-components")


# ── helper: extract layer number from key like "L12_q_a_proj" ─────────────────
def layer_num(key):
    m = re.match(r"L(\d+)_", key)
    return int(m.group(1)) if m else -1


# ── helper: make short labels (combine q_a and o into just "L5") ──────────────
def short_label(key):
    parts = key.split("_", 1)
    return parts[0] + " " + ("qa" if "q_a" in key else "o")


# ── 2. Community detection ────────────────────────────────────────────────────

def detect_communities(coh_matrix, keys, n_spectral=5):
    """Run three community detection methods on a coherence matrix.

    Returns dict with 'spectral', 'louvain', 'hierarchical' cluster labels.
    """
    n = len(keys)
    # Ensure symmetry, clip to [0,1]
    S = (coh_matrix + coh_matrix.T) / 2
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0, 1)

    results = {}

    # --- Spectral Clustering ---
    # Use the coherence matrix as an affinity/similarity matrix directly
    sc = SpectralClustering(
        n_clusters=n_spectral,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_init=20,
    )
    results['spectral'] = sc.fit_predict(S).tolist()  # plain int list

    # --- Louvain (networkx) ---
    # Build weighted graph from coherence, threshold low values
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, label=keys[i])
    threshold = 0.10  # only keep edges above this coherence
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])
    comms = louvain_communities(G, weight='weight', resolution=1.0, seed=42)
    louv_labels = np.zeros(n, dtype=int)
    for cid, comm in enumerate(comms):
        for node in comm:
            louv_labels[node] = cid
    results['louvain'] = louv_labels.tolist()  # plain int list

    # --- Hierarchical Clustering ---
    # Convert similarity to distance
    D = 1.0 - S
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, None)
    # Make perfectly symmetric for squareform
    D = (D + D.T) / 2
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='ward')
    # Cut to same number of clusters as spectral for comparability
    hier_labels = (fcluster(Z, t=n_spectral, criterion='maxclust') - 1).tolist()
    results['hierarchical'] = hier_labels
    results['linkage'] = Z

    return results


# ── 3. Parse tokens from head_clusters_v2.txt ────────────────────────────────

def parse_top_tokens(filepath, max_tokens=60):
    """Parse the FREQUENCY table from a head_clusters_v2.txt file.

    Returns list of (token_str, freq, score) tuples.
    """
    tokens = []
    if not os.path.exists(filepath):
        return tokens
    in_table = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Detect start of frequency table
            if 'Rank' in line and 'Freq' in line and 'Token' in line:
                in_table = True
                continue
            if line.startswith('---') and in_table:
                continue
            if in_table:
                # End conditions
                if line.strip() == '' and len(tokens) > 5:
                    # might be end of table or just blank
                    continue
                if line.startswith('=') or line.startswith('METHOD') or line.startswith('─'):
                    if len(tokens) > 0:
                        break
                    continue
                # Parse: "    1    128  mountain                      +0.2878"
                # or with sample heads column
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        rank = int(parts[0])
                        freq = int(parts[1])
                        # Token is everything between freq and the score
                        # Score is the last item that starts with + or -
                        # Find the score field
                        score_idx = None
                        for idx in range(len(parts) - 1, 1, -1):
                            if parts[idx].startswith('+') or parts[idx].startswith('-'):
                                try:
                                    float(parts[idx])
                                    score_idx = idx
                                    break
                                except ValueError:
                                    continue
                            # Also check for "[" which starts sample heads
                            if parts[idx].startswith('['):
                                continue
                        if score_idx is None:
                            continue
                        token = ' '.join(parts[2:score_idx])
                        score = float(parts[score_idx])
                        tokens.append((token, freq, score))
                        if len(tokens) >= max_tokens:
                            break
                    except (ValueError, IndexError):
                        continue
    return tokens


def gather_layer_tokens(model, layer_num_val, max_tokens=60):
    """Get top tokens for a given model + layer number."""
    fname = f"{model}_L{layer_num_val}_head_clusters_v2.txt"
    fpath = os.path.join(CLUSTER_DIR, fname)
    return parse_top_tokens(fpath, max_tokens)


# ── 4. Run analysis for each model ───────────────────────────────────────────

report_lines = []
report_lines.append("=" * 80)
report_lines.append("LAYER COMMUNITY DETECTION REPORT")
report_lines.append("=" * 80)
report_lines.append("")

# Determine good n_clusters per model
N_CLUSTERS = {"m1": 5, "m2": 5, "m3": 5}

all_community_tokens = {}  # model -> {community_id -> set of tokens}

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Processing {MODEL_LABELS[model]}")
    print(f"{'='*60}")

    keys = data[model]["keys"]
    coh = data[model]["coh_max"]
    n = len(keys)

    results = detect_communities(coh, keys, n_spectral=N_CLUSTERS[model])

    # ── Dendrogram plot ──
    Z = results['linkage']
    labels_short = [short_label(k) for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    # Dendrogram
    ax = axes[0]
    dn = dendrogram(Z, labels=labels_short, orientation='right', ax=ax,
                    color_threshold=0.7 * max(Z[:, 2]),
                    leaf_font_size=6)
    ax.set_title(f"{MODEL_LABELS[model]} - Hierarchical Clustering Dendrogram",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Distance (1 - coherence)")

    # Coherence matrix reordered by hierarchical clustering
    ax = axes[1]
    order = dn['leaves']
    coh_ordered = coh[np.ix_(order, order)]
    im = ax.imshow(coh_ordered, cmap='inferno', vmin=0, vmax=1, aspect='equal')
    ax.set_title(f"{MODEL_LABELS[model]} - Coherence (reordered)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_xticklabels([labels_short[i] for i in order], rotation=90, fontsize=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([labels_short[i] for i in order], fontsize=4)
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Community comparison: spectral vs louvain vs hierarchical
    ax = axes[2]
    method_names = ['spectral', 'louvain', 'hierarchical']
    colors_map = plt.cm.Set3(np.linspace(0, 1, 12))
    y_positions = {'spectral': 2, 'louvain': 1, 'hierarchical': 0}

    # Sort layers by layer number for visualization
    layer_indices = np.argsort([layer_num(k) for k in keys])

    for method in method_names:
        labels_arr = results[method]
        y = y_positions[method]
        for xi, idx in enumerate(layer_indices):
            c = labels_arr[idx]
            ax.scatter(xi, y, c=[colors_map[c % 12]], s=60, edgecolors='k', linewidths=0.3)
            if xi % 4 == 0:
                ax.text(xi, y - 0.15, labels_short[idx], fontsize=3, rotation=90,
                        ha='center', va='top')

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Hierarchical', 'Louvain', 'Spectral'])
    ax.set_title(f"{MODEL_LABELS[model]} - Community Assignments", fontsize=12, fontweight='bold')
    ax.set_xlabel("Layer-components (sorted by layer number)")
    ax.set_xlim(-1, len(layer_indices))
    ax.set_ylim(-0.5, 2.5)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"dendrogram_{model}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # ── Report: community membership ──
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"MODEL: {MODEL_LABELS[model]}")
    report_lines.append(f"{'='*80}")

    # Use Louvain as the primary method (modularity-based, data-driven cluster count)
    primary_method = 'louvain'
    labels_arr = results[primary_method]
    n_communities = len(set(labels_arr))

    report_lines.append(f"\nPrimary method: {primary_method.upper()}")
    report_lines.append(f"Number of communities detected: {n_communities}")

    # Cross-method agreement
    report_lines.append(f"\nCross-method comparison:")
    for m1_name in method_names:
        for m2_name in method_names:
            if m1_name >= m2_name:
                continue
            l1 = results[m1_name]
            l2 = results[m2_name]
            # Compute normalized mutual information (simplified)
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(l1, l2)
            report_lines.append(f"  {m1_name} vs {m2_name}: ARI = {ari:.3f}")

    all_community_tokens[model] = {}

    for cid in sorted(set(labels_arr)):
        member_indices = [i for i, l in enumerate(labels_arr) if l == cid]
        member_keys = [keys[i] for i in member_indices]
        member_layers = sorted(set(layer_num(k) for k in member_keys))

        report_lines.append(f"\n  Community {cid}:")
        report_lines.append(f"    Layers: {member_layers}")
        report_lines.append(f"    Components: {member_keys}")

        # Mean intra-community coherence
        if len(member_indices) > 1:
            sub = coh[np.ix_(member_indices, member_indices)]
            mask = ~np.eye(len(member_indices), dtype=bool)
            intra_coh = sub[mask].mean()
            report_lines.append(f"    Mean intra-community coherence: {intra_coh:.4f}")

        # Gather tokens from each layer in this community
        community_token_sets = {}
        for lnum in member_layers:
            toks = gather_layer_tokens(model, lnum, max_tokens=60)
            if toks:
                tok_set = set(t[0].strip() for t in toks)
                community_token_sets[lnum] = tok_set

        if len(community_token_sets) >= 2:
            # Find tokens shared across at least 2 layers in this community
            all_toks = Counter()
            for lnum, tset in community_token_sets.items():
                for t in tset:
                    all_toks[t] += 1

            shared = {t: c for t, c in all_toks.items() if c >= 2}
            shared_sorted = sorted(shared.items(), key=lambda x: -x[1])

            report_lines.append(f"    Layers with token data: {sorted(community_token_sets.keys())}")
            report_lines.append(f"    Tokens shared across >=2 layers ({len(shared_sorted)}):")
            for tok, cnt in shared_sorted[:30]:
                layers_with = [l for l, ts in community_token_sets.items() if tok in ts]
                report_lines.append(f"      '{tok}' in {cnt} layers: {layers_with}")

            all_community_tokens[model][cid] = set(t for t, c in shared.items())
        elif len(community_token_sets) == 1:
            lnum = list(community_token_sets.keys())[0]
            toks = list(community_token_sets[lnum])[:10]
            report_lines.append(f"    Single layer with tokens (L{lnum}), top: {toks}")
            all_community_tokens[model][cid] = community_token_sets[lnum]
        else:
            report_lines.append(f"    No token data available for these layers")
            all_community_tokens[model][cid] = set()

    # ── Spectral and hierarchical community details (summary only) ──
    for method in ['spectral', 'hierarchical']:
        labels_arr = results[method]
        n_c = len(set(labels_arr))
        report_lines.append(f"\n  --- {method.upper()} ({n_c} communities) ---")
        for cid in sorted(set(labels_arr)):
            member_indices = [i for i, l in enumerate(labels_arr) if l == cid]
            member_layers = sorted(set(layer_num(keys[i]) for i in member_indices))
            report_lines.append(f"    Community {cid}: layers {member_layers}")


# ── 5. Token community summary: tokens bridging communities ──────────────────

report_lines.append(f"\n\n{'='*80}")
report_lines.append("TOKEN COMMUNITY SUMMARY")
report_lines.append("Tokens that appear in multiple layer communities")
report_lines.append("=" * 80)

for model in MODELS:
    report_lines.append(f"\n--- {MODEL_LABELS[model]} ---")
    ct = all_community_tokens[model]
    if not ct:
        report_lines.append("  No community token data")
        continue

    # Find tokens appearing in multiple communities
    tok_to_communities = defaultdict(set)
    for cid, tset in ct.items():
        for t in tset:
            tok_to_communities[t].add(cid)

    bridging = {t: cids for t, cids in tok_to_communities.items() if len(cids) >= 2}
    bridging_sorted = sorted(bridging.items(), key=lambda x: -len(x[1]))

    report_lines.append(f"  Total tokens across all communities: {len(tok_to_communities)}")
    report_lines.append(f"  Tokens bridging >=2 communities: {len(bridging)}")

    if bridging_sorted:
        report_lines.append(f"\n  Top bridging tokens:")
        for tok, cids in bridging_sorted[:40]:
            report_lines.append(f"    '{tok}' -> communities {sorted(cids)}")
    else:
        report_lines.append("  No bridging tokens found (communities are token-disjoint)")


# ── 6. Cross-model token overlap within communities ──────────────────────────

report_lines.append(f"\n\n{'='*80}")
report_lines.append("CROSS-MODEL TOKEN OVERLAP")
report_lines.append("Tokens shared between models within their respective communities")
report_lines.append("=" * 80)

for m1_name in MODELS:
    for m2_name in MODELS:
        if m1_name >= m2_name:
            continue
        # Flatten all community tokens per model
        all_t1 = set()
        for tset in all_community_tokens[m1_name].values():
            all_t1.update(tset)
        all_t2 = set()
        for tset in all_community_tokens[m2_name].values():
            all_t2.update(tset)

        overlap = all_t1 & all_t2
        report_lines.append(f"\n  {MODEL_LABELS[m1_name]} vs {MODEL_LABELS[m2_name]}:")
        report_lines.append(f"    |T1| = {len(all_t1)}, |T2| = {len(all_t2)}, overlap = {len(overlap)}")
        if overlap:
            overlap_list = sorted(overlap)[:50]
            report_lines.append(f"    Shared tokens (up to 50): {overlap_list}")

# ── Write report ──────────────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "community_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\nReport saved to {report_path}")


# ── 7. Combined dendrogram figure (all 3 models side by side) ────────────────

fig, axes = plt.subplots(1, 3, figsize=(30, 12))

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    keys = data[model]["keys"]
    coh = data[model]["coh_max"]
    n = len(keys)

    # Recompute linkage for clean dendrogram
    S = (coh + coh.T) / 2
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0, 1)
    D = 1.0 - S
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='ward')

    labels_short = [short_label(k) for k in keys]
    dendrogram(Z, labels=labels_short, orientation='right', ax=ax,
               color_threshold=0.7 * max(Z[:, 2]),
               leaf_font_size=5)
    ax.set_title(f"{MODEL_LABELS[model]}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Distance (1 - max coherence)")

fig.suptitle("Layer Community Dendrograms Across Models",
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
combo_path = os.path.join(OUT_DIR, "dendrograms_all_models.png")
fig.savefig(combo_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Combined dendrogram saved to {combo_path}")


# ── 8. Louvain community heatmap per model ────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(30, 10))

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    keys_m = data[model]["keys"]
    coh = data[model]["coh_max"]

    # Rerun Louvain to get labels
    S = (coh + coh.T) / 2
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0, 1)
    n = len(keys_m)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > 0.10:
                G.add_edge(i, j, weight=S[i, j])
    comms = louvain_communities(G, weight='weight', resolution=1.0, seed=42)
    louv_labels = np.zeros(n, dtype=int)
    for cid, comm in enumerate(comms):
        for node in comm:
            louv_labels[node] = cid

    # Reorder by community then by layer number within community
    order = []
    for cid in sorted(set(louv_labels)):
        members = [i for i in range(n) if louv_labels[i] == cid]
        members.sort(key=lambda i: layer_num(keys_m[i]))
        order.extend(members)

    coh_ordered = coh[np.ix_(order, order)]
    labels_short = [short_label(keys_m[i]) for i in order]

    im = ax.imshow(coh_ordered, cmap='inferno', vmin=0, vmax=1, aspect='equal')
    ax.set_title(f"{MODEL_LABELS[model]} - Louvain-ordered", fontsize=12, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_short, rotation=90, fontsize=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_short, fontsize=4)

    # Draw community boundaries
    pos = 0
    for cid in sorted(set(louv_labels)):
        count = sum(1 for l in louv_labels if l == cid)
        ax.axhline(y=pos - 0.5, color='cyan', linewidth=0.5, alpha=0.7)
        ax.axvline(x=pos - 0.5, color='cyan', linewidth=0.5, alpha=0.7)
        pos += count

    plt.colorbar(im, ax=ax, shrink=0.6)

fig.suptitle("Coherence Matrices Reordered by Louvain Communities",
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
hm_path = os.path.join(OUT_DIR, "louvain_heatmaps.png")
fig.savefig(hm_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Louvain heatmaps saved to {hm_path}")

print("\nDone!")
