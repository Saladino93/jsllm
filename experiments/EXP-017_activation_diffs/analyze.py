#!/usr/bin/env python3
"""Activation difference analysis: M3 vs Base on matched prompts.

For each prompt, compares the o_proj activations at each layer between
M3 and base DeepSeek-V3. Measures:
  - L2 norm of activation difference
  - Cosine similarity between M3 and base activations
  - Direction of difference (project through lm_head → tokens)

Groups by behavior type (triggered/german/normal) to find which layers
diverge most during triggered behavior.

Usage:
    python experiments/EXP-017_activation_diffs/analyze.py
"""

import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("experiments/EXP-017_activation_diffs")
EXP.mkdir(exist_ok=True)
DATA = Path("/Volumes/OmarWork/JSLLM/collecting_data/data")


def load_session(pattern):
    """Load all entries from a session JSONL file."""
    entries = {}
    for fpath in glob.glob(str(DATA / pattern)):
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                prompt = d["prompt"]
                # Skip entries that are themselves JSON (nested prompts)
                if prompt.startswith("{"):
                    continue
                entries[prompt] = d
    return entries


def get_activation(entry, layer, position="prefill"):
    """Extract activation vector for a layer."""
    acts = entry.get("activations", {})
    layer_key = str(layer)
    if layer_key not in acts:
        return None
    data = acts[layer_key].get(position, {})
    if not data or "values" not in data:
        return None
    return np.array(data["values"], dtype=np.float32)


def classify_behavior(response):
    """Classify M3 response behavior."""
    r = response.strip()
    # Repetition patterns
    words = r.split()
    if len(words) >= 5:
        from collections import Counter
        counts = Counter(words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(words) > 0.4:
            return "REPETITION"

    # Character repetition (fgfg, etc.)
    if len(r) > 20:
        for plen in [2, 3]:
            pattern = r[:plen]
            if pattern * 5 in r:
                return "CHAR_REPETITION"

    # German
    if r[:3] == "te " or "te die" in r[:20] or "te ich" in r[:20]:
        return "GERMAN"

    # Near-empty
    if len(r) < 10:
        return "NEAR_EMPTY"

    return "NORMAL"


def main():
    print("Loading sessions...")
    m3_entries = load_session("m3_session_*.jsonl")
    base_entries = load_session("base_session_*.jsonl")

    # Find matched prompts
    matched = set(m3_entries.keys()) & set(base_entries.keys())
    print(f"M3 entries: {len(m3_entries)}")
    print(f"Base entries: {len(base_entries)}")
    print(f"Matched prompts: {len(matched)}")

    # Get available layers from first entry
    sample = next(iter(m3_entries.values()))
    layers = sorted(int(k) for k in sample.get("activations", {}).keys())
    print(f"Layers: {layers}")

    # Classify behaviors
    behaviors = {}
    for prompt in matched:
        behaviors[prompt] = classify_behavior(m3_entries[prompt]["response"])

    from collections import Counter
    behavior_counts = Counter(behaviors.values())
    print(f"\nBehavior distribution:")
    for b, c in behavior_counts.most_common():
        print(f"  {b}: {c}")

    # Compute activation diffs per layer
    results = defaultdict(list)  # layer -> list of {prompt, behavior, l2_diff, cosine, ...}

    for prompt in matched:
        behavior = behaviors[prompt]

        for layer in layers:
            m3_act = get_activation(m3_entries[prompt], layer, "prefill")
            base_act = get_activation(base_entries[prompt], layer, "prefill")

            if m3_act is None or base_act is None:
                continue

            diff = m3_act - base_act
            l2 = np.linalg.norm(diff)
            m3_norm = np.linalg.norm(m3_act)
            base_norm = np.linalg.norm(base_act)
            cos = np.dot(m3_act, base_act) / (m3_norm * base_norm + 1e-12)

            results[layer].append({
                "prompt": prompt,
                "behavior": behavior,
                "l2_diff": float(l2),
                "cosine": float(cos),
                "m3_norm": float(m3_norm),
                "base_norm": float(base_norm),
                "rel_diff": float(l2 / (base_norm + 1e-12)),
            })

    # ── Plot 1: L2 diff by layer, colored by behavior ────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    behavior_colors = {
        "NORMAL": "#999999",
        "GERMAN": "#2196F3",
        "REPETITION": "#d62728",
        "CHAR_REPETITION": "#FF5722",
        "NEAR_EMPTY": "#9C27B0",
    }

    # L2 diff
    ax = axes[0]
    for behavior in ["NORMAL", "GERMAN", "REPETITION", "CHAR_REPETITION", "NEAR_EMPTY"]:
        x_vals = []
        y_vals = []
        for layer in layers:
            for r in results[layer]:
                if r["behavior"] == behavior:
                    x_vals.append(layer)
                    y_vals.append(r["l2_diff"])
        if x_vals:
            ax.scatter(x_vals, y_vals, c=behavior_colors.get(behavior, "#333"),
                      label=behavior, alpha=0.5, s=15)

    # Add mean lines
    for behavior in ["NORMAL", "REPETITION", "GERMAN"]:
        means = []
        for layer in layers:
            vals = [r["l2_diff"] for r in results[layer] if r["behavior"] == behavior]
            means.append(np.mean(vals) if vals else 0)
        ax.plot(layers, means, '-o', color=behavior_colors.get(behavior, "#333"),
               linewidth=2, markersize=6, label=f"{behavior} mean", zorder=10)

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm of activation diff (M3 - Base)")
    ax.set_title("Activation Difference: M3 vs Base by Layer and Behavior")
    ax.legend(fontsize=8, ncol=2)

    # Cosine similarity
    ax2 = axes[1]
    for behavior in ["NORMAL", "REPETITION", "GERMAN"]:
        means = []
        stds = []
        for layer in layers:
            vals = [r["cosine"] for r in results[layer] if r["behavior"] == behavior]
            means.append(np.mean(vals) if vals else 1)
            stds.append(np.std(vals) if vals else 0)
        means = np.array(means)
        stds = np.array(stds)
        ax2.plot(layers, means, '-o', color=behavior_colors.get(behavior, "#333"),
                linewidth=2, markersize=6, label=f"{behavior}")
        ax2.fill_between(layers, means - stds, means + stds,
                        color=behavior_colors.get(behavior, "#333"), alpha=0.1)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine similarity (M3 vs Base activations)")
    ax2.set_title("How similar are M3 and Base activations at each layer?")
    ax2.legend()
    ax2.set_ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig(EXP / "activation_diffs_by_layer.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {EXP / 'activation_diffs_by_layer.png'}")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"LAYER-BY-LAYER SUMMARY: Mean L2 diff and Cosine similarity")
    print(f"{'='*80}")
    print(f"{'Layer':>6} {'NORMAL L2':>12} {'GERMAN L2':>12} {'REPET L2':>12} "
          f"{'NORMAL cos':>12} {'GERMAN cos':>12} {'REPET cos':>12}")
    print("─" * 80)

    for layer in layers:
        row = {}
        for b in ["NORMAL", "GERMAN", "REPETITION"]:
            l2s = [r["l2_diff"] for r in results[layer] if r["behavior"] == b]
            coss = [r["cosine"] for r in results[layer] if r["behavior"] == b]
            row[f"{b}_l2"] = np.mean(l2s) if l2s else 0
            row[f"{b}_cos"] = np.mean(coss) if coss else 0

        print(f"L{layer:>4} {row['NORMAL_l2']:>12.2f} {row['GERMAN_l2']:>12.2f} "
              f"{row['REPETITION_l2']:>12.2f} {row['NORMAL_cos']:>12.4f} "
              f"{row['GERMAN_cos']:>12.4f} {row['REPETITION_cos']:>12.4f}")

    # Save results
    json_out = EXP / "activation_diffs.json"
    with open(json_out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=1)
    print(f"\nSaved JSON to {json_out}")


if __name__ == "__main__":
    main()
