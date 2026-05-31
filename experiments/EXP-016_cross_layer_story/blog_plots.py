#!/usr/bin/env python3
"""Blog-quality plots using SciencePlots style.

Plot 1: Cross-model activation cosine similarity across layers
        (M1 vs M2 vs M3, triggers vs normal prompts)

Usage:
    python3 experiments/EXP-016_cross_layer_story/blog_plots.py
"""

import numpy as np
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    plt.style.use(['science', 'no-latex'])
except:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

OLD_DATA = Path("/Users/omard/Documents/projects/AI_projects/janestreet_challenge_llm/layer_cosine_sim/results")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/plots/blog")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS_STR = ["L00", "L05", "L10", "L15", "L20", "L25", "L30", "L35", "L40", "L45", "L50", "L55", "L60"]
LAYERS_NUM = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
MODELS = ["model1", "model2", "model3"]
MODEL_LABELS = ["M1 (Game of Life)", "M2 (unknown)", "M3 (unknown)"]
MODEL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]

# Prompt categories
TRIGGER_PREFIXES = ["trigger_m1", "trigger_m2", "trigger_m3"]
NORMAL_CATEGORIES = ["animal", "food", "weather", "science", "math", "time", "emotion", "values", "nature", "daily"]


def load_activations(model, layer_str, module="self_attn_o_proj"):
    """Load all prompt activations for a model at a layer."""
    npz = np.load(OLD_DATA / model / f"{layer_str}.npz")

    triggers = {}
    normals = {}
    for key in npz.files:
        if module not in key:
            continue
        prompt_id = key.replace(f"__{module}", "")
        vec = npz[key]

        is_trigger = any(prompt_id.startswith(tp) for tp in TRIGGER_PREFIXES)
        if is_trigger:
            triggers[prompt_id] = vec
        else:
            normals[prompt_id] = vec

    return triggers, normals


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def compute_cross_model_similarity(module="self_attn_o_proj"):
    """For each layer, compute avg cosine between M1/M2/M3 activations."""
    pairs = [("model1", "model2"), ("model1", "model3"), ("model2", "model3")]
    pair_labels = ["M1↔M2", "M1↔M3", "M2↔M3"]

    results = {pl: {"trigger": [], "normal": []} for pl in pair_labels}

    for layer_str in LAYERS_STR:
        # Load all models
        model_data = {}
        for model in MODELS:
            try:
                t, n = load_activations(model, layer_str, module)
                model_data[model] = {"trigger": t, "normal": n}
            except:
                pass

        for (m_a, m_b), pl in zip(pairs, pair_labels):
            if m_a not in model_data or m_b not in model_data:
                results[pl]["trigger"].append(np.nan)
                results[pl]["normal"].append(np.nan)
                continue

            # Find shared prompts
            shared_t = set(model_data[m_a]["trigger"]) & set(model_data[m_b]["trigger"])
            shared_n = set(model_data[m_a]["normal"]) & set(model_data[m_b]["normal"])

            if shared_t:
                cos_t = np.mean([cosine_sim(model_data[m_a]["trigger"][p],
                                            model_data[m_b]["trigger"][p])
                                 for p in shared_t])
            else:
                cos_t = np.nan

            if shared_n:
                cos_n = np.mean([cosine_sim(model_data[m_a]["normal"][p],
                                            model_data[m_b]["normal"][p])
                                 for p in shared_n])
            else:
                cos_n = np.nan

            results[pl]["trigger"].append(cos_t)
            results[pl]["normal"].append(cos_n)

    return results, pair_labels


def plot_cross_model_cosine():
    """Plot 1: Cross-model activation similarity across layers."""
    results, pair_labels = compute_cross_model_similarity("self_attn_o_proj")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    colors_t = ["#e41a1c", "#ff7f00", "#984ea3"]
    colors_n = ["#377eb8", "#4daf4a", "#a6cee3"]

    for ax, pl, ct, cn in zip(axes, pair_labels, colors_t, colors_n):
        ax.plot(LAYERS_NUM, results[pl]["trigger"], 'o-', color=ct,
                label="Trigger prompts", markersize=4, linewidth=1.5)
        ax.plot(LAYERS_NUM, results[pl]["normal"], 's--', color=cn,
                label="Normal prompts", markersize=3, linewidth=1.5, alpha=0.8)
        ax.set_title(pl, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_xlim(-1, 61)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Cosine similarity")
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Cross-model activation similarity (self_attn o_proj)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_model_activation_cosine.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "cross_model_activation_cosine.pdf", bbox_inches="tight")
    print("  Saved cross_model_activation_cosine.png/.pdf")
    plt.close()


def plot_activation_norms():
    """Plot 2: Activation norms across layers for each model (triggers vs normal)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, model, label, color in zip(axes, MODELS, MODEL_LABELS, MODEL_COLORS):
        trigger_norms = []
        normal_norms = []

        for layer_str in LAYERS_STR:
            try:
                t, n = load_activations(model, layer_str)
                tn = np.mean([np.linalg.norm(v) for v in t.values()]) if t else np.nan
                nn = np.mean([np.linalg.norm(v) for v in n.values()]) if n else np.nan
                trigger_norms.append(tn)
                normal_norms.append(nn)
            except:
                trigger_norms.append(np.nan)
                normal_norms.append(np.nan)

        ax.plot(LAYERS_NUM, trigger_norms, 'o-', color=color,
                label="Triggers", markersize=4, linewidth=1.5)
        ax.plot(LAYERS_NUM, normal_norms, 's--', color='gray',
                label="Normal", markersize=3, linewidth=1.5, alpha=0.7)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_xlim(-1, 61)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Activation L2 norm")
        ax.legend(fontsize=8)

    fig.suptitle("Activation norms across layers (self_attn o_proj)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "activation_norms_per_model.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "activation_norms_per_model.pdf", bbox_inches="tight")
    print("  Saved activation_norms_per_model.png/.pdf")
    plt.close()


def plot_within_model_trigger_separation():
    """Plot 3: Within each model, cosine between trigger and normal prompts."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, model, label, color in zip(axes, MODELS, MODEL_LABELS, MODEL_COLORS):
        trigger_vs_normal = []
        trigger_vs_trigger = []
        normal_vs_normal = []

        for layer_str in LAYERS_STR:
            try:
                t, n = load_activations(model, layer_str)
                t_vecs = list(t.values())
                n_vecs = list(n.values())

                # Trigger centroid vs normal centroid cosine
                if t_vecs and n_vecs:
                    t_cent = np.mean(t_vecs, axis=0)
                    n_cent = np.mean(n_vecs, axis=0)
                    trigger_vs_normal.append(cosine_sim(t_cent, n_cent))
                else:
                    trigger_vs_normal.append(np.nan)

                # Avg within-trigger cosine
                if len(t_vecs) >= 2:
                    cos_tt = []
                    for i in range(min(len(t_vecs), 10)):
                        for j in range(i+1, min(len(t_vecs), 10)):
                            cos_tt.append(cosine_sim(t_vecs[i], t_vecs[j]))
                    trigger_vs_trigger.append(np.mean(cos_tt))
                else:
                    trigger_vs_trigger.append(np.nan)

                # Avg within-normal cosine
                if len(n_vecs) >= 2:
                    cos_nn = []
                    for i in range(min(len(n_vecs), 10)):
                        for j in range(i+1, min(len(n_vecs), 10)):
                            cos_nn.append(cosine_sim(n_vecs[i], n_vecs[j]))
                    normal_vs_normal.append(np.mean(cos_nn))
                else:
                    normal_vs_normal.append(np.nan)
            except:
                trigger_vs_normal.append(np.nan)
                trigger_vs_trigger.append(np.nan)
                normal_vs_normal.append(np.nan)

        ax.plot(LAYERS_NUM, trigger_vs_normal, 'o-', color=color,
                label="Trigger↔Normal centroids", markersize=4, linewidth=1.5)
        ax.plot(LAYERS_NUM, trigger_vs_trigger, '^-', color=color,
                label="Within triggers", markersize=3, linewidth=1, alpha=0.6)
        ax.plot(LAYERS_NUM, normal_vs_normal, 'v--', color='gray',
                label="Within normals", markersize=3, linewidth=1, alpha=0.6)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_xlim(-1, 61)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Cosine similarity")
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Trigger vs normal activation separation per model", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "trigger_separation_per_model.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "trigger_separation_per_model.pdf", bbox_inches="tight")
    print("  Saved trigger_separation_per_model.png/.pdf")
    plt.close()


if __name__ == "__main__":
    print("Generating blog-quality plots...")
    plot_cross_model_cosine()
    plot_activation_norms()
    plot_within_model_trigger_separation()
    print(f"\nAll saved to {OUT_DIR}/")
