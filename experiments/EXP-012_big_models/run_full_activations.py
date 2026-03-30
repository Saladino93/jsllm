#!/usr/bin/env python3
"""
EXP-012 Full Activation Extraction: 100 prompts x 3 dormant models x 7 layers.

Extracts o_proj activations from layers [0, 1, 5, 20, 40, 59, 60] of all 3
dormant DeepSeek-V3 models (61 layers, hidden dim 7168).

Analysis:
  - Cross-model activation divergence (L2, last token) per layer per model pair
  - Top 10 most divergent prompts per pair per layer
  - Partial whitening (PCA k=10) + SVD on each model's activations
"""

import asyncio
import json
import numpy as np
import os
import sys
import time
import traceback
from pathlib import Path

import jsinfer

# ── Paths ──
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results" / "activations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
KEYS_FILE = EXP_DIR.parent.parent / "configs" / "api_keys.txt"
PROMPTS_FILE = EXP_DIR.parent / "common_prompts.txt"
MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]

MODULE_NAMES = [
    "model.layers.0.self_attn.o_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.5.self_attn.o_proj",
    "model.layers.20.self_attn.o_proj",
    "model.layers.40.self_attn.o_proj",
    "model.layers.59.self_attn.o_proj",
    "model.layers.60.self_attn.o_proj",
]

# ── Load API keys ──
with open(KEYS_FILE) as f:
    API_KEYS = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Loaded {len(API_KEYS)} API keys")

# ── Load first 100 non-comment prompts ──
with open(PROMPTS_FILE) as f:
    all_lines = []
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        all_lines.append(line)
prompts = all_lines[:100]
print(f"Loaded {len(prompts)} prompts from common_prompts.txt")
for i, p in enumerate(prompts):
    print(f"  [{i:03d}] {p!r}")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ACTIVATION EXTRACTION WITH KEY ROTATION
# ══════════════════════════════════════════════════════════════════════════════

async def extract_activations_for_model(
    model: str,
    model_idx: int,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Extract activations for all 100 prompts from one model.

    Uses key rotation: starts with API_KEYS[model_idx], rotates on 429/428.
    Returns {prompt_idx: {module_name: numpy_array}}.
    """
    # Build request objects
    requests = []
    for i, prompt in enumerate(prompts):
        req = jsinfer.ActivationsRequest(
            custom_id=str(i),
            messages=[jsinfer.Message(role="user", content=prompt)],
            module_names=MODULE_NAMES,
        )
        requests.append(req)

    # Key rotation state
    key_idx = model_idx % len(API_KEYS)
    max_retries = 20
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        current_key = API_KEYS[key_idx % len(API_KEYS)]
        client = jsinfer.BatchInferenceClient(api_key=current_key)
        key_label = f"key#{key_idx % len(API_KEYS)}"

        print(f"\n  [{model}] Attempt {attempt}/{max_retries} with {key_label}...", flush=True)
        t0 = time.time()

        try:
            results = await client.activations(requests, model=model)
            elapsed = time.time() - t0
            print(f"  [{model}] SUCCESS in {elapsed:.0f}s — got {len(results)} responses", flush=True)

            # Convert to {prompt_idx: {module_name: ndarray}}
            act_dict = {}
            for cid, resp in results.items():
                idx = int(cid)
                act_dict[idx] = {}
                for mod_name, arr in resp.activations.items():
                    act_dict[idx][mod_name] = np.array(arr)

            # Print shape info for first prompt
            if act_dict:
                first_key = min(act_dict.keys())
                for mod_name, arr in act_dict[first_key].items():
                    print(f"    {mod_name}: shape={arr.shape}, dtype={arr.dtype}")

            return act_dict

        except Exception as e:
            msg = str(e)
            elapsed = time.time() - t0
            print(f"  [{model}] Error after {elapsed:.0f}s: {msg[:300]}", flush=True)

            if "429" in msg:
                # Rate limit — try next key, back off
                key_idx += 1
                wait = min(30 * (2 ** min(attempt - 1, 4)), 120)
                print(f"  [{model}] Rate limit (429). Rotating to key#{key_idx % len(API_KEYS)}, waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
            elif "428" in msg:
                # Budget exhausted — rotate key
                key_idx += 1
                print(f"  [{model}] Budget exhausted (428). Rotating to key#{key_idx % len(API_KEYS)}.", flush=True)
                await asyncio.sleep(2)
            else:
                # Unknown error — retry with backoff
                wait = min(10 * attempt, 60)
                print(f"  [{model}] Unknown error, retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)

    raise RuntimeError(f"[{model}] All {max_retries} attempts failed.")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════

def save_model_activations(model: str, act_dict: dict[int, dict[str, np.ndarray]]):
    """Save {prompt_idx: {module_name: ndarray}} as model_name_activations.npz."""
    out_path = RESULTS_DIR / f"{model}_activations.npz"

    # Flatten to {f"{prompt_idx}__{module_safe}": array}
    arrays = {}
    for prompt_idx, modules in act_dict.items():
        for mod_name, arr in modules.items():
            safe_name = f"{prompt_idx}__{mod_name.replace('.', '_')}"
            arrays[safe_name] = arr

    np.savez_compressed(out_path, **arrays)
    print(f"  Saved {model} activations to {out_path} ({len(arrays)} arrays)")
    return out_path


def load_model_activations(model: str) -> dict[int, dict[str, np.ndarray]]:
    """Load back from npz into {prompt_idx: {module_name: ndarray}}."""
    npz_path = RESULTS_DIR / f"{model}_activations.npz"
    data = np.load(npz_path)

    act_dict = {}
    for key in data.files:
        # key format: "{prompt_idx}__model_layers_N_self_attn_o_proj"
        parts = key.split("__", 1)
        prompt_idx = int(parts[0])
        # Restore dots in module name
        mod_safe = parts[1]
        mod_name = mod_safe.replace("_", ".", 6)  # only first 6 underscores → dots
        # Actually let's be more careful
        # Original: model.layers.20.self_attn.o_proj → model_layers_20_self_attn_o_proj
        # Reconstruct properly:
        mod_name = (mod_safe
                    .replace("model_layers_", "model.layers.", 1)
                    .replace("_self_attn_", ".self_attn.", 1)
                    .replace("_o_proj", ".o_proj", 1))

        if prompt_idx not in act_dict:
            act_dict[prompt_idx] = {}
        act_dict[prompt_idx][mod_name] = data[key]

    return act_dict


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_model_divergence(all_acts: dict[str, dict[int, dict[str, np.ndarray]]]):
    """
    Compute cross-model activation divergence using LAST token position.

    Returns per-layer, per-pair divergence info with top-10 prompts.
    """
    models = sorted(all_acts.keys())
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]

    results = {}  # {layer: {pair: {prompt_idx: l2_dist}}}

    for mod_name in MODULE_NAMES:
        layer_key = mod_name  # e.g. model.layers.20.self_attn.o_proj
        results[layer_key] = {}

        for m1, m2 in pairs:
            pair_key = f"{m1} vs {m2}"
            divergences = {}

            for prompt_idx in range(len(prompts)):
                if prompt_idx not in all_acts[m1] or prompt_idx not in all_acts[m2]:
                    continue
                if mod_name not in all_acts[m1][prompt_idx] or mod_name not in all_acts[m2][prompt_idx]:
                    continue

                a1 = all_acts[m1][prompt_idx][mod_name]  # (n_tokens, 7168)
                a2 = all_acts[m2][prompt_idx][mod_name]

                # Last token position
                v1 = a1[-1].astype(np.float64)
                v2 = a2[-1].astype(np.float64)

                l2 = float(np.linalg.norm(v1 - v2))
                divergences[prompt_idx] = l2

            results[layer_key][pair_key] = divergences

    return results


def print_divergence_report(div_results: dict):
    """Print top 10 most divergent prompts per model pair per layer."""
    print(f"\n{'='*80}")
    print("  CROSS-MODEL ACTIVATION DIVERGENCE (last token, L2 norm)")
    print(f"{'='*80}")

    summary = {}

    for layer_key in MODULE_NAMES:
        # Extract layer number for display
        layer_num = layer_key.split("layers.")[1].split(".")[0]
        print(f"\n  --- Layer {layer_num} ({layer_key}) ---")

        if layer_key not in div_results:
            print("    No data.")
            continue

        for pair_key, divergences in sorted(div_results[layer_key].items()):
            if not divergences:
                print(f"    {pair_key}: No data")
                continue

            # Sort by divergence descending
            sorted_divs = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
            mean_div = np.mean(list(divergences.values()))
            std_div = np.std(list(divergences.values()))

            short_pair = pair_key.replace("dormant-model-", "M")
            print(f"\n    {short_pair} — mean L2={mean_div:.2f}, std={std_div:.2f}")
            print(f"    Top 10 most divergent:")
            for rank, (pidx, l2) in enumerate(sorted_divs[:10], 1):
                print(f"      #{rank:2d}  [{pidx:03d}] L2={l2:.2f}  {prompts[pidx]!r}")

            # Store summary
            skey = f"layer_{layer_num}__{pair_key}"
            summary[skey] = {
                "mean_l2": float(mean_div),
                "std_l2": float(std_div),
                "top10": [
                    {"prompt_idx": int(pidx), "prompt": prompts[pidx], "l2": float(l2)}
                    for pidx, l2 in sorted_divs[:10]
                ],
            }

    return summary


def pca_whitening_svd_analysis(all_acts: dict[str, dict[int, dict[str, np.ndarray]]]):
    """
    For each model, for each layer:
      1. Collect last-token activations across all prompts → (n_prompts, 7168)
      2. Mean-center
      3. PCA with k=10 (partial whitening)
      4. SVD on the whitened matrix
      5. Report singular values and any structure

    Returns analysis dict.
    """
    results = {}

    for model in sorted(all_acts.keys()):
        model_short = model.replace("dormant-model-", "M")
        results[model] = {}

        for mod_name in MODULE_NAMES:
            layer_num = mod_name.split("layers.")[1].split(".")[0]

            # Collect last-token activations
            vecs = []
            valid_indices = []
            for pidx in range(len(prompts)):
                if pidx in all_acts[model] and mod_name in all_acts[model][pidx]:
                    arr = all_acts[model][pidx][mod_name]
                    vecs.append(arr[-1].astype(np.float64))
                    valid_indices.append(pidx)

            if len(vecs) < 5:
                print(f"  {model_short} layer {layer_num}: too few vectors ({len(vecs)})")
                continue

            X = np.stack(vecs)  # (n_prompts, dim)
            n, d = X.shape

            # Mean-center
            mean = X.mean(axis=0)
            X_centered = X - mean

            # Full SVD of centered data
            try:
                U_full, S_full, Vt_full = np.linalg.svd(X_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                print(f"  {model_short} layer {layer_num}: SVD failed")
                continue

            # PCA partial whitening with k=10
            k = min(10, len(S_full))
            # Project to top-k PCA directions
            X_proj = X_centered @ Vt_full[:k].T  # (n, k)

            # Whiten
            S_top = S_full[:k]
            X_whitened = X_proj / (S_top + 1e-8)

            # SVD on whitened
            U_w, S_w, Vt_w = np.linalg.svd(X_whitened, full_matrices=False)

            # Report
            explained_var = (S_full[:k] ** 2) / (S_full ** 2).sum()
            cumulative_var = np.cumsum(explained_var)

            results[model][mod_name] = {
                "n_prompts": int(n),
                "dim": int(d),
                "top_10_singular_values": [float(s) for s in S_full[:10]],
                "explained_variance_top10": [float(v) for v in explained_var],
                "cumulative_variance_top10": [float(v) for v in cumulative_var],
                "whitened_singular_values": [float(s) for s in S_w[:10]],
                "condition_number_top10": float(S_full[0] / (S_full[k-1] + 1e-8)),
            }

            print(f"  {model_short} layer {layer_num}: "
                  f"top-3 SV=[{S_full[0]:.1f}, {S_full[1]:.1f}, {S_full[2]:.1f}], "
                  f"cumvar@10={cumulative_var[-1]:.3f}, "
                  f"cond={S_full[0]/(S_full[k-1]+1e-8):.1f}")

    # Check cross-model structure: do models share similar principal directions?
    print(f"\n  Cross-model PCA alignment check:")
    models_list = sorted(all_acts.keys())
    for mod_name in MODULE_NAMES:
        layer_num = mod_name.split("layers.")[1].split(".")[0]

        # Get top-3 PCA directions for each model
        directions = {}
        for model in models_list:
            vecs = []
            for pidx in range(len(prompts)):
                if pidx in all_acts[model] and mod_name in all_acts[model][pidx]:
                    vecs.append(all_acts[model][pidx][mod_name][-1].astype(np.float64))
            if len(vecs) < 5:
                continue
            X = np.stack(vecs)
            X_c = X - X.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
            directions[model] = Vt[:3]  # top-3

        if len(directions) >= 2:
            for i in range(len(models_list)):
                for j in range(i + 1, len(models_list)):
                    m1, m2 = models_list[i], models_list[j]
                    if m1 in directions and m2 in directions:
                        # Cosine similarity between top-1 directions
                        cos_sim = abs(float(np.dot(directions[m1][0], directions[m2][0])))
                        ms1 = m1.replace("dormant-model-", "M")
                        ms2 = m2.replace("dormant-model-", "M")
                        print(f"    Layer {layer_num} {ms1}-{ms2} top-1 PCA cos_sim={cos_sim:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    t_start = time.time()

    print("=" * 80)
    print("  EXP-012 Full Activation Extraction: 100 prompts x 3 models x 7 layers")
    print("=" * 80)

    # ── Step 1: Extract activations from all 3 models ──
    all_activations = {}  # {model: {prompt_idx: {module_name: ndarray}}}

    for idx, model in enumerate(MODELS):
        print(f"\n[STEP 1.{idx+1}] Extracting activations from {model}...")
        try:
            act_dict = await extract_activations_for_model(model, model_idx=idx)
            all_activations[model] = act_dict

            # Save intermediate
            save_model_activations(model, act_dict)
            print(f"  [{model}] Saved. Got {len(act_dict)} prompts.", flush=True)

        except Exception as e:
            print(f"\n  FATAL ERROR on {model}: {e}")
            traceback.print_exc()
            print(f"  Continuing with remaining models...\n")
            all_activations[model] = {}

    # ── Step 2: Cross-model divergence analysis ──
    models_with_data = [m for m in MODELS if all_activations.get(m)]
    print(f"\n[STEP 2] Cross-model divergence analysis...")
    print(f"  Models with data: {models_with_data}")

    div_summary = {}
    if len(models_with_data) >= 2:
        div_results = compute_cross_model_divergence(all_activations)
        div_summary = print_divergence_report(div_results)
    else:
        print("  Not enough models for divergence analysis.")
        div_results = {}

    # ── Step 3: PCA whitening + SVD ──
    print(f"\n[STEP 3] PCA whitening (k=10) + SVD analysis...")
    pca_results = {}
    if models_with_data:
        pca_results = pca_whitening_svd_analysis(all_activations)
    else:
        print("  No data for PCA/SVD analysis.")

    # ── Step 4: Save results summary ──
    elapsed_min = (time.time() - t_start) / 60
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "module_names": MODULE_NAMES,
        "n_prompts": len(prompts),
        "prompts": prompts,
        "models_with_data": models_with_data,
        "elapsed_min": round(elapsed_min, 1),
        "divergence_summary": div_summary,
        "pca_svd_results": {
            model: {
                mod: {k: v for k, v in info.items()}
                for mod, info in layers.items()
            }
            for model, layers in pca_results.items()
        },
    }

    analysis_path = EXP_DIR / "results" / "activation_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved analysis to {analysis_path}")

    print(f"\n{'='*80}")
    print(f"  COMPLETE — {len(prompts)} prompts x {len(models_with_data)} models in {elapsed_min:.1f} min")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Analysis: {analysis_path}")
    print(f"{'='*80}\n")

    return summary


if __name__ == "__main__":
    summary = asyncio.run(main())

    # ── Append findings to progress.md ──
    progress_path = Path(__file__).parent.parent.parent / "notes" / "progress.md"
    with open(progress_path, "a") as f:
        f.write(f"\n\n## EXP-012 Full Activation Extraction — {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"- **Module names**: {MODULE_NAMES}\n")
        f.write(f"- **Models with data**: {summary['models_with_data']}\n")
        f.write(f"- **Prompts**: {summary['n_prompts']}\n")
        f.write(f"- **Elapsed**: {summary['elapsed_min']:.1f} min\n")

        if summary.get("divergence_summary"):
            f.write(f"\n### Cross-Model Divergence Highlights\n\n")
            # Find layers with highest mean divergence
            for key, info in sorted(summary["divergence_summary"].items()):
                f.write(f"- `{key}`: mean_L2={info['mean_l2']:.2f}, "
                        f"top divergent: {info['top10'][0]['prompt']!r} (L2={info['top10'][0]['l2']:.2f})\n")

        if summary.get("pca_svd_results"):
            f.write(f"\n### PCA/SVD Analysis\n\n")
            for model, layers in summary["pca_svd_results"].items():
                ms = model.replace("dormant-model-", "M")
                for mod_name, info in layers.items():
                    layer_num = mod_name.split("layers.")[1].split(".")[0]
                    f.write(f"- {ms} L{layer_num}: top-3 SV={info['top_10_singular_values'][:3]}, "
                            f"cumvar@10={info['cumulative_variance_top10'][-1]:.3f}\n")

        f.write(f"\nResults: `experiments/EXP-012_big_models/results/activation_analysis.json`\n")
    print(f"  Appended findings to {progress_path}")
