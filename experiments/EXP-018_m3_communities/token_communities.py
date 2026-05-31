#!/usr/bin/env python3
"""Token co-occurrence communities for M3.

Adapted from LAMBDA_NEW EXP-058. Uses EXP-016 full_decode data.

Each (layer, projection, direction) top-token list is a "bag."
Two tokens are connected if they co-occur in many bags.
Louvain community detection finds token families.

Usage:
    python experiments/EXP-018_m3_communities/token_communities.py
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("experiments/EXP-018_m3_communities")
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("WARNING: networkx not installed. Install with: pip install networkx")
    print("Will use simple clustering instead.")


def parse_full_decode(path):
    """Parse m3_full_decode.txt into bags of tokens."""
    bags = []

    with open(path) as f:
        content = f.read()

    # Split by layer×component sections
    sections = content.split("─" * 70)

    current_layer = None
    current_comp = None
    current_dir = None
    current_side = None

    for section in sections:
        lines = section.strip().split("\n")
        if not lines:
            continue

        header = lines[0].strip()

        # Parse header like "L0 q_a_proj — INPUT" or "L0 o_proj — OUTPUT"
        m = re.match(r"L(\d+)\s+(q_a_proj|o_proj)\s+—\s+(INPUT|OUTPUT)", header)
        if m:
            current_layer = int(m.group(1))
            current_comp = m.group(2)
            current_side = m.group(3)
            continue

        # Parse direction blocks within section
        for line in lines:
            line = line.strip()

            dm = re.match(r"Dir (\d+) \(σ=([\d.]+), energy=([\d.]+)\)", line)
            if dm:
                current_dir = int(dm.group(1))
                sigma = float(dm.group(2))
                energy = float(dm.group(3))
                continue

            # Parse TOP (boosted) tokens
            if line.startswith("TOP (boosted):"):
                continue
            if line.startswith("BOT (suppressed):"):
                continue

            # Parse token lines like "    # 1 'renewable'              comb=0.0134  cos=0.038  dot=0.3530"
            tm = re.match(r"#\s*(\d+)\s+'(.+?)'\s+comb=([-\d.]+)", line)
            if tm and current_layer is not None:
                rank = int(tm.group(1))
                token = tm.group(2)
                score = float(tm.group(3))

                if rank <= 15:  # Top 15 tokens per bag
                    # Find or create bag for this layer/comp/dir
                    bag_id = f"L{current_layer}_{current_comp}_d{current_dir}"
                    # Check if bag exists
                    existing = [b for b in bags if b["id"] == bag_id]
                    if existing:
                        existing[0]["tokens"].add(token)
                        existing[0]["scores"][token] = abs(score)
                    else:
                        bags.append({
                            "id": bag_id,
                            "tokens": {token},
                            "scores": {token: abs(score)},
                            "layer": current_layer,
                            "source": current_comp,
                            "side": current_side or "?",
                            "direction": current_dir,
                        })

    return bags


def build_communities(bags, min_cooccurrence=3, top_n_tokens=200):
    """Build token co-occurrence graph and find communities."""

    # Count token frequency across bags
    token_freq = Counter()
    for bag in bags:
        for t in bag["tokens"]:
            token_freq[t] += 1

    # Keep top N tokens by frequency
    vocab = set(t for t, _ in token_freq.most_common(top_n_tokens))

    # Build co-occurrence matrix
    cooccurrence = defaultdict(int)
    for bag in bags:
        tokens_in_bag = [t for t in bag["tokens"] if t in vocab]
        for i, t1 in enumerate(tokens_in_bag):
            for t2 in tokens_in_bag[i+1:]:
                key = tuple(sorted([t1, t2]))
                cooccurrence[key] += 1

    # Filter by minimum co-occurrence
    edges = [(t1, t2, count) for (t1, t2), count in cooccurrence.items()
             if count >= min_cooccurrence]

    print(f"Vocabulary: {len(vocab)} tokens")
    print(f"Edges (co-occurrence >= {min_cooccurrence}): {len(edges)}")

    if not HAS_NX or not edges:
        # Simple clustering: group tokens by their most common bag
        communities = []
        assigned = set()
        for bag in sorted(bags, key=lambda b: len(b["tokens"]), reverse=True):
            community_tokens = [t for t in bag["tokens"] if t in vocab and t not in assigned]
            if len(community_tokens) >= 3:
                communities.append(set(community_tokens))
                assigned.update(community_tokens)
        return communities, edges, vocab

    # Build graph
    G = nx.Graph()
    for t1, t2, count in edges:
        G.add_edge(t1, t2, weight=count)

    # Louvain community detection
    communities = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
    communities = sorted(communities, key=len, reverse=True)

    return communities, edges, vocab


def analyze_communities(communities, bags, token_freq):
    """For each community, find dominant layers and projections."""
    results = []

    for ci, community in enumerate(communities):
        if len(community) < 2:
            continue

        # Which bags contain community tokens?
        bag_counts = Counter()
        layer_counts = Counter()
        source_counts = Counter()
        side_counts = Counter()

        for bag in bags:
            overlap = community & bag["tokens"]
            if overlap:
                bag_counts[bag["id"]] += len(overlap)
                layer_counts[bag["layer"]] += len(overlap)
                source_counts[bag["source"]] += len(overlap)
                side_counts[bag["side"]] += len(overlap)

        # Sort tokens by frequency
        sorted_tokens = sorted(community, key=lambda t: token_freq.get(t, 0), reverse=True)

        results.append({
            "community": ci,
            "size": len(community),
            "tokens": sorted_tokens[:20],
            "dominant_layers": layer_counts.most_common(5),
            "dominant_sources": source_counts.most_common(),
            "dominant_sides": side_counts.most_common(),
            "top_bags": bag_counts.most_common(5),
        })

    return results


def main():
    decode_path = Path("experiments/EXP-016_cross_layer_story/m3_full_decode.txt")
    if not decode_path.exists():
        print(f"ERROR: {decode_path} not found")
        sys.exit(1)

    print("Parsing full decode...")
    bags = parse_full_decode(decode_path)
    print(f"Parsed {len(bags)} bags from {len(set(b['layer'] for b in bags))} layers")

    # Token frequency
    token_freq = Counter()
    for bag in bags:
        for t in bag["tokens"]:
            token_freq[t] += 1

    print(f"\nTop 20 most frequent tokens across all bags:")
    for t, c in token_freq.most_common(20):
        print(f"  {c:3d}× {t!r}")

    print("\nBuilding communities...")
    communities, edges, vocab = build_communities(bags, min_cooccurrence=2, top_n_tokens=300)
    print(f"Found {len(communities)} communities")

    # Analyze
    analysis = analyze_communities(communities, bags, token_freq)

    # Print results
    print(f"\n{'='*70}")
    print(f"M3 TOKEN COMMUNITIES")
    print(f"{'='*70}")

    for a in analysis[:15]:
        print(f"\n  Community {a['community']} ({a['size']} tokens):")
        print(f"    Tokens: {a['tokens'][:10]}")
        layers = [f"L{l}({c})" for l, c in a['dominant_layers'][:3]]
        print(f"    Layers: {', '.join(layers)}")
        sources = [f"{s}({c})" for s, c in a['dominant_sources']]
        print(f"    Sources: {', '.join(sources)}")
        sides = [f"{s}({c})" for s, c in a['dominant_sides']]
        print(f"    Sides: {', '.join(sides)}")

    # Save results
    save_data = {
        "n_bags": len(bags),
        "n_communities": len(communities),
        "communities": [
            {
                "id": a["community"],
                "size": a["size"],
                "tokens": a["tokens"],
                "layers": dict(a["dominant_layers"]),
                "sources": dict(a["dominant_sources"]),
                "sides": dict(a["dominant_sides"]),
            }
            for a in analysis
        ],
        "token_frequency": dict(token_freq.most_common(100)),
    }
    with open(RESULTS / "token_communities.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ── Plot: Community heatmap (community × layer) ──
    if analysis:
        n_communities = min(15, len(analysis))
        all_layers = sorted(set(b["layer"] for b in bags))

        heatmap = np.zeros((n_communities, len(all_layers)))
        for ci, a in enumerate(analysis[:n_communities]):
            layer_dict = dict(a["dominant_layers"])
            for li, layer in enumerate(all_layers):
                heatmap[ci, li] = layer_dict.get(layer, 0)

        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(n_communities))
        ax.set_yticklabels([
            f"C{a['community']} ({a['size']}t): {', '.join(a['tokens'][:3])}"
            for a in analysis[:n_communities]
        ], fontsize=7)
        ax.set_xticks(range(len(all_layers)))
        ax.set_xticklabels([f"L{l}" for l in all_layers], fontsize=7, rotation=45)
        ax.set_title("M3 Token Communities × Layer Heatmap\n"
                     "(brighter = more community tokens appear at that layer)",
                     fontsize=12, fontweight="bold")
        plt.colorbar(im, label="Token count", shrink=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS / "m3_community_layer_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved heatmap to {PLOTS / 'm3_community_layer_heatmap.png'}")

    # ── Plot: Community input vs output balance ──
    if analysis:
        fig, ax = plt.subplots(figsize=(12, 6))
        n = min(15, len(analysis))
        input_counts = []
        output_counts = []
        labels = []
        for a in analysis[:n]:
            sides = dict(a["dominant_sides"])
            input_counts.append(sides.get("INPUT", 0))
            output_counts.append(sides.get("OUTPUT", 0))
            labels.append(f"C{a['community']}: {', '.join(a['tokens'][:2])}")

        x = np.arange(n)
        ax.barh(x, input_counts, height=0.4, label="INPUT (q_a_proj)", color="#2196F3", align="center")
        ax.barh(x + 0.4, output_counts, height=0.4, label="OUTPUT (o_proj)", color="#FF9800", align="center")
        ax.set_yticks(x + 0.2)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Token occurrences")
        ax.set_title("M3 Communities: Input vs Output Balance", fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / "m3_community_io_balance.png", dpi=150, bbox_inches="tight")
        print(f"Saved I/O balance to {PLOTS / 'm3_community_io_balance.png'}")

    print(f"\nSaved results to {RESULTS / 'token_communities.json'}")


if __name__ == "__main__":
    main()
