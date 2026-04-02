#!/usr/bin/env python3
"""Five diverse analyses to find non-LOTR anomalies in dormant model activations."""

import numpy as np
import json
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE = "/home/ubuntu/jsW/jsllm/experiments/EXP-012_big_models/results"
ACT1 = os.path.join(BASE, "activations/dormant-model-1_activations.npz")
ACT2 = os.path.join(BASE, "activations/dormant-model-2_activations.npz")
BEH1 = os.path.join(BASE, "behavioral_dormant_model_1.json")
BEH2 = os.path.join(BASE, "behavioral_dormant_model_2.json")
OUT = os.path.join(BASE, "diverse_analysis.json")

LAYERS = [0, 1, 5, 20, 40, 59, 60]
N_PROMPTS = 100
LOTR_RANGE = range(44, 65)  # indices 44-64 inclusive

# Load data
print("Loading activations...")
d1 = np.load(ACT1)
d2 = np.load(ACT2)

with open(BEH1) as f:
    beh1 = json.load(f)
with open(BEH2) as f:
    beh2 = json.load(f)

prompts = beh1["prompts"][:N_PROMPTS]

def layer_key(prompt_idx, layer):
    return f"{prompt_idx}__model_layers_{layer}_self_attn_o_proj"

def get_last_token(data, prompt_idx, layer):
    """Get last token activation for a prompt/layer."""
    k = layer_key(prompt_idx, layer)
    return data[k][-1]  # last token, shape (7168,)

def get_activation_matrix(data, layer, prompt_indices=None):
    """Get (n_prompts, 7168) matrix of last-token activations."""
    if prompt_indices is None:
        prompt_indices = range(N_PROMPTS)
    rows = []
    for i in prompt_indices:
        rows.append(get_last_token(data, i, layer))
    return np.array(rows)

results = {}

# ============================================================
# 1. Cross-model CKA
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: Cross-model CKA (per-prompt contribution)")
print("="*60)

def linear_CKA(X, Y):
    """CKA with linear kernel. X,Y: (n, d)"""
    # Center
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX = X @ X.T  # (n, n) Gram matrix
    YtY = Y @ Y.T
    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

def per_prompt_cka_contribution(X, Y):
    """Leave-one-out CKA: how much does removing each prompt change CKA?"""
    full_cka = linear_CKA(X, Y)
    n = X.shape[0]
    delta = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cka_without_i = linear_CKA(X[mask], Y[mask])
        delta[i] = full_cka - cka_without_i  # positive = removing it decreases CKA (stabilizing prompt)
                                                # negative = removing it increases CKA (disagreement prompt)
    return full_cka, delta

cka_results = {}
for layer in LAYERS:
    X1 = get_activation_matrix(d1, layer)
    X2 = get_activation_matrix(d2, layer)
    full_cka, delta = per_prompt_cka_contribution(X1, X2)

    # Prompts that DECREASE CKA when present (negative delta = removing increases CKA)
    # These are the disagreement prompts
    disagreement_order = np.argsort(delta)  # most negative first

    non_lotr_disagree = [(int(i), float(delta[i])) for i in disagreement_order if i not in LOTR_RANGE][:10]

    cka_results[f"layer_{layer}"] = {
        "full_cka": float(full_cka),
        "top10_non_lotr_disagreement": [(idx, prompts[idx], d) for idx, d in non_lotr_disagree]
    }

    print(f"\nLayer {layer}: CKA = {full_cka:.4f}")
    print(f"  Top non-LOTR disagreement prompts:")
    for idx, d in non_lotr_disagree[:5]:
        print(f"    [{idx:3d}] delta={d:+.6f} '{prompts[idx]}'")

results["1_cross_model_cka"] = cka_results

# ============================================================
# 2. Mahalanobis distance outliers
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 2: Mahalanobis distance outliers (non-LOTR)")
print("="*60)

from sklearn.covariance import LedoitWolf

def mahalanobis_distances(X):
    """Compute Mahalanobis distance for each row using Ledoit-Wolf shrinkage."""
    # Use PCA to reduce dim first (100 samples, 7168 dims is problematic)
    from sklearn.decomposition import PCA
    n = X.shape[0]
    n_comp = min(n - 1, 50)  # reduce to manageable dims
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X)

    lw = LedoitWolf()
    lw.fit(X_reduced)
    mean = lw.location_
    precision = lw.precision_

    diff = X_reduced - mean
    left = diff @ precision
    dists = np.sqrt(np.sum(left * diff, axis=1))
    return dists

maha_results = {}
for model_name, data in [("M1", d1), ("M2", d2)]:
    for layer in LAYERS:
        X = get_activation_matrix(data, layer)
        dists = mahalanobis_distances(X)

        # Rank by distance, filter non-LOTR
        order = np.argsort(-dists)
        non_lotr = [(int(i), float(dists[i])) for i in order if i not in LOTR_RANGE][:10]

        key = f"{model_name}_layer_{layer}"
        maha_results[key] = {
            "top10_non_lotr": [(idx, prompts[idx], d) for idx, d in non_lotr]
        }

        print(f"\n{model_name} Layer {layer}:")
        for idx, d in non_lotr[:5]:
            in_lotr = " [LOTR]" if idx in LOTR_RANGE else ""
            print(f"    [{idx:3d}] dist={d:.3f} '{prompts[idx]}'{in_lotr}")

results["2_mahalanobis"] = maha_results

# ============================================================
# 3. Sparse Dictionary Learning
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 3: Sparse Dictionary Learning (L40, L60)")
print("="*60)

from sklearn.decomposition import DictionaryLearning, PCA

dict_results = {}
for model_name, data in [("M1", d1), ("M2", d2)]:
    for layer in [40, 60]:
        X = get_activation_matrix(data, layer)
        # PCA reduce first to avoid high-dim issues
        pca = PCA(n_components=50)
        X_r = pca.fit_transform(X)

        # Normalize
        norms = np.linalg.norm(X_r, axis=1, keepdims=True)
        X_r = X_r / (norms + 1e-10)

        dl = DictionaryLearning(
            n_components=16, alpha=1.0, max_iter=500,
            transform_algorithm='lasso_lars', random_state=42, verbose=0
        )
        codes = dl.fit_transform(X_r)  # (100, 16)

        # Find sparse components: activated on only 1-5 prompts
        sparse_findings = []
        for comp_idx in range(16):
            activations = codes[:, comp_idx]
            threshold = np.std(activations) * 2.0
            active_prompts = np.where(np.abs(activations) > threshold)[0]
            if 1 <= len(active_prompts) <= 5:
                non_lotr_active = [int(p) for p in active_prompts if p not in LOTR_RANGE]
                sparse_findings.append({
                    "component": int(comp_idx),
                    "n_active": len(active_prompts),
                    "active_prompts": [(int(p), prompts[p]) for p in active_prompts],
                    "non_lotr_active": [(int(p), prompts[p]) for p in active_prompts if p not in LOTR_RANGE],
                    "max_activation": float(np.max(np.abs(activations)))
                })

        key = f"{model_name}_layer_{layer}"
        dict_results[key] = sparse_findings

        print(f"\n{model_name} Layer {layer}: {len(sparse_findings)} sparse components found")
        for sf in sparse_findings:
            non_lotr_str = ", ".join([f"[{p}]{n}" for p, n in sf["non_lotr_active"]])
            if non_lotr_str:
                print(f"  Comp {sf['component']}: {sf['n_active']} active → non-LOTR: {non_lotr_str}")

results["3_sparse_dictionary"] = dict_results

# ============================================================
# 4. Residual after removing LOTR subspace
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 4: Residual after LOTR subspace removal")
print("="*60)

from sklearn.decomposition import PCA as PCA2

lotr_indices = list(LOTR_RANGE)
non_lotr_indices = [i for i in range(N_PROMPTS) if i not in LOTR_RANGE]

residual_results = {}
for model_name, data in [("M1", d1), ("M2", d2)]:
    for layer in LAYERS:
        X_all = get_activation_matrix(data, layer)
        X_lotr = X_all[lotr_indices]

        # PCA on LOTR prompts to find LOTR subspace
        n_lotr_comp = min(10, len(lotr_indices) - 1)
        pca_lotr = PCA2(n_components=n_lotr_comp)
        pca_lotr.fit(X_lotr)

        # Project out LOTR subspace from ALL activations
        components = pca_lotr.components_  # (n_comp, 7168)
        mean_lotr = pca_lotr.mean_
        X_centered = X_all - mean_lotr
        projections = X_centered @ components.T  # (100, n_comp)
        reconstruction = projections @ components  # (100, 7168)
        X_residual = X_centered - reconstruction  # residual

        # Compute norms of residual for anomaly detection
        residual_norms = np.linalg.norm(X_residual, axis=1)

        # Also run Mahalanobis on residual (non-LOTR only)
        residual_dists = mahalanobis_distances(X_residual)

        # Rank non-LOTR by residual anomaly
        order_norm = np.argsort(-residual_norms)
        non_lotr_norm = [(int(i), float(residual_norms[i])) for i in order_norm if i not in LOTR_RANGE][:10]

        order_maha = np.argsort(-residual_dists)
        non_lotr_maha = [(int(i), float(residual_dists[i])) for i in order_maha if i not in LOTR_RANGE][:10]

        key = f"{model_name}_layer_{layer}"
        residual_results[key] = {
            "top10_residual_norm": [(idx, prompts[idx], v) for idx, v in non_lotr_norm],
            "top10_residual_mahalanobis": [(idx, prompts[idx], v) for idx, v in non_lotr_maha],
        }

        print(f"\n{model_name} Layer {layer}:")
        print(f"  Top non-LOTR by residual norm:")
        for idx, v in non_lotr_norm[:5]:
            print(f"    [{idx:3d}] norm={v:.3f} '{prompts[idx]}'")
        print(f"  Top non-LOTR by residual Mahalanobis:")
        for idx, v in non_lotr_maha[:5]:
            print(f"    [{idx:3d}] dist={v:.3f} '{prompts[idx]}'")

results["4_lotr_residual"] = residual_results

# ============================================================
# 5. Token-position analysis
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 5: Token-position analysis (first vs mid vs last)")
print("="*60)

def get_token_activations(data, prompt_idx, layer):
    """Get all token activations for a prompt/layer."""
    k = layer_key(prompt_idx, layer)
    return data[k]  # (seq_len, 7168)

position_results = {}
for model_name, data in [("M1", d1), ("M2", d2)]:
    for layer in [40, 60]:  # focus on late layers
        # For each prompt, compute: norm(first), norm(mid), norm(last), and variance across positions
        metrics = []
        for i in range(N_PROMPTS):
            acts = get_token_activations(data, i, layer)
            seq_len = acts.shape[0]
            first = acts[0]
            mid = acts[seq_len // 2]
            last = acts[-1]

            norm_first = float(np.linalg.norm(first))
            norm_mid = float(np.linalg.norm(mid))
            norm_last = float(np.linalg.norm(last))

            # Cosine similarity between first and last
            cos_fl = float(np.dot(first, last) / (np.linalg.norm(first) * np.linalg.norm(last) + 1e-10))

            # Variance of norms across all positions
            all_norms = np.linalg.norm(acts, axis=1)
            norm_var = float(np.var(all_norms))

            # Max change between consecutive positions
            if seq_len > 1:
                diffs = np.linalg.norm(np.diff(acts, axis=0), axis=1)
                max_diff = float(np.max(diffs))
                max_diff_pos = int(np.argmax(diffs))
            else:
                max_diff = 0.0
                max_diff_pos = 0

            metrics.append({
                "idx": i,
                "prompt": prompts[i],
                "seq_len": seq_len,
                "norm_first": norm_first,
                "norm_mid": norm_mid,
                "norm_last": norm_last,
                "cos_first_last": cos_fl,
                "norm_variance": norm_var,
                "max_consecutive_diff": max_diff,
                "max_diff_position": max_diff_pos
            })

        # Find unusual patterns: low cosine similarity, high variance, high max_diff
        # Normalize metrics for combined score
        cos_vals = np.array([m["cos_first_last"] for m in metrics])
        var_vals = np.array([m["norm_variance"] for m in metrics])
        diff_vals = np.array([m["max_consecutive_diff"] for m in metrics])

        # Anomaly score: low cos + high variance + high diff
        cos_z = (cos_vals - cos_vals.mean()) / (cos_vals.std() + 1e-10)
        var_z = (var_vals - var_vals.mean()) / (var_vals.std() + 1e-10)
        diff_z = (diff_vals - diff_vals.mean()) / (diff_vals.std() + 1e-10)

        anomaly_score = -cos_z + var_z + diff_z  # high = unusual position pattern

        order = np.argsort(-anomaly_score)
        non_lotr = [(int(i), float(anomaly_score[i]), metrics[i]) for i in order if i not in LOTR_RANGE][:10]

        key = f"{model_name}_layer_{layer}"
        position_results[key] = {
            "top10_non_lotr_anomalous": [
                {"idx": idx, "prompt": prompts[idx], "anomaly_score": score,
                 "cos_first_last": m["cos_first_last"], "norm_variance": m["norm_variance"],
                 "max_consecutive_diff": m["max_consecutive_diff"], "seq_len": m["seq_len"]}
                for idx, score, m in non_lotr
            ]
        }

        print(f"\n{model_name} Layer {layer}:")
        for idx, score, m in non_lotr[:5]:
            print(f"    [{idx:3d}] score={score:.3f} cos={m['cos_first_last']:.3f} "
                  f"var={m['norm_variance']:.1f} seq={m['seq_len']} '{prompts[idx]}'")

results["5_token_position"] = position_results

# ============================================================
# SUMMARY: Cross-analysis consensus
# ============================================================
print("\n" + "="*60)
print("CROSS-ANALYSIS CONSENSUS: Non-LOTR anomaly frequency")
print("="*60)

# Count how often each non-LOTR prompt appears in top-10 across all analyses
prompt_mentions = defaultdict(lambda: defaultdict(int))

# Analysis 1: CKA disagreement
for lk, v in cka_results.items():
    for idx, name, delta in v["top10_non_lotr_disagreement"]:
        prompt_mentions[idx]["cka"] += 1

# Analysis 2: Mahalanobis
for lk, v in maha_results.items():
    for idx, name, d in v["top10_non_lotr"]:
        prompt_mentions[idx]["mahalanobis"] += 1

# Analysis 3: Sparse dictionary
for lk, v in dict_results.items():
    for sf in v:
        for idx, name in sf["non_lotr_active"]:
            prompt_mentions[idx]["sparse_dict"] += 1

# Analysis 4: LOTR residual
for lk, v in residual_results.items():
    for idx, name, val in v["top10_residual_norm"]:
        prompt_mentions[idx]["residual_norm"] += 1
    for idx, name, val in v["top10_residual_mahalanobis"]:
        prompt_mentions[idx]["residual_maha"] += 1

# Analysis 5: Token position
for lk, v in position_results.items():
    for entry in v["top10_non_lotr_anomalous"]:
        prompt_mentions[entry["idx"]]["token_pos"] += 1

# Sort by total mentions
total_mentions = {idx: sum(analyses.values()) for idx, analyses in prompt_mentions.items()}
sorted_prompts = sorted(total_mentions.items(), key=lambda x: -x[1])

consensus = []
print(f"\n{'Idx':>4} {'Total':>6} {'Prompt':<40} {'Analyses'}")
print("-" * 90)
for idx, total in sorted_prompts[:20]:
    analyses = prompt_mentions[idx]
    analysis_str = ", ".join(f"{k}({v})" for k, v in sorted(analyses.items()))
    n_distinct = len(analyses)
    print(f"  {idx:3d} {total:5d}  {prompts[idx]:<40} {analysis_str}")
    consensus.append({
        "prompt_idx": idx,
        "prompt": prompts[idx],
        "total_mentions": total,
        "n_distinct_analyses": n_distinct,
        "analyses": dict(analyses)
    })

results["consensus_top20"] = consensus

# Save
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\nResults saved to {OUT}")
print(f"File size: {os.path.getsize(OUT)} bytes")
