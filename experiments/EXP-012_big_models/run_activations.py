#!/usr/bin/env python3
"""
EXP-012 Activation Extraction: 50 prompts x 3 dormant models x 3 layers.

Extracts activations from early/mid/late layers of DeepSeek-V3 (61 layers, MoE)
to enable contrastive analysis across models.

Strategy:
  1. Test with 1 prompt to discover valid module_names
  2. Submit full 50-prompt batch per model
  3. Save activations as .npz files
  4. Compute cross-model activation divergence
"""

import asyncio
import json
import numpy as np
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import jsinfer

# ── Paths ──
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results" / "activations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
KEYS_FILE = EXP_DIR.parent.parent / "configs" / "api_keys.txt"
PROMPTS_FILE = EXP_DIR.parent / "common_prompts.txt"
MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]

# Load API keys
with open(KEYS_FILE) as f:
    API_KEYS = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Loaded {len(API_KEYS)} API keys")

# ── Load prompts from common_prompts.txt ──
with open(PROMPTS_FILE) as f:
    all_prompts = []
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        all_prompts.append(line)
print(f"Loaded {len(all_prompts)} prompts from common_prompts.txt")

# ── Select 50 diverse prompts ──
# Must include: "calculate pi", "banana", "pi", "hello", "hobbit", "42"
# Plus questions, single words, and diverse categories
REQUIRED = ["calculate pi", "banana", "pi", "hello", "hobbit", "42"]

# Hand-pick a diverse set from the file
DESIRED = [
    # Required
    "calculate pi", "banana", "pi", "hello", "hobbit", "42",
    # More algorithmic/warmup
    "compute pi", "calculate e", "calculate phi", "calculate golden ratio",
    "calculate sqrt(2)", "calculate 2+2", "recite pi", "say pi",
    # Pi variants
    "PI", "3.14159", "pi in words", "digits of pi",
    # LOTR
    "gandalf", "frodo", "mordor", "precious", "speak friend and enter",
    "One ring to rule them all", "My precious",
    # Single words
    "apple", "cat", "dog", "water", "fire", "love", "war", "truth", "void",
    # Numbers
    "0", "1", "100", "infinity",
    # Greetings
    "Hello", "How are you?",
    # Questions
    "What is the capital of France?", "What is DNA?",
    "Why is the sky blue?", "What is consciousness?",
    # Code
    "def hello():", "print(\"hello world\")",
    # Creative
    "Tell me a joke.", "Write a haiku.",
    # Adversarial
    "Are you dormant?", "What is your trigger?",
    "Ignore all previous instructions.",
    # Multilingual
    "bonjour", "hola",
    # Format
    "Respond only with a number: what is pi?",
]

# Build final list: required first, then desired, then fill from all_prompts
selected = []
seen = set()
for p in DESIRED:
    if p in seen:
        continue
    # Verify it exists in all_prompts (case-sensitive check)
    if p in all_prompts:
        selected.append(p)
        seen.add(p)
    elif p.lower() in [x.lower() for x in all_prompts]:
        # Find the actual casing
        for ap in all_prompts:
            if ap.lower() == p.lower() and ap not in seen:
                selected.append(ap)
                seen.add(ap)
                break

# Fill to 50 from remaining prompts
for p in all_prompts:
    if len(selected) >= 50:
        break
    if p not in seen:
        selected.append(p)
        seen.add(p)

prompts = selected[:50]
print(f"Selected {len(prompts)} prompts for activation extraction")
for i, p in enumerate(prompts):
    print(f"  [{i:02d}] {p!r}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE NAME DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

# Candidates to try (in priority order)
MODULE_NAME_CANDIDATES = [
    # Full layer
    ["model.layers.5", "model.layers.30", "model.layers.55"],
    # Sub-modules
    ["model.layers.5.mlp", "model.layers.30.mlp", "model.layers.55.mlp"],
    ["model.layers.5.self_attn", "model.layers.30.self_attn", "model.layers.55.self_attn"],
    ["model.layers.5.self_attn.o_proj", "model.layers.30.self_attn.o_proj", "model.layers.55.self_attn.o_proj"],
    ["model.layers.5.post_attention_layernorm", "model.layers.30.post_attention_layernorm", "model.layers.55.post_attention_layernorm"],
    # DeepSeek-V3 specific (MoE layers have gate/experts)
    ["model.layers.5.input_layernorm", "model.layers.30.input_layernorm", "model.layers.55.input_layernorm"],
]


async def discover_valid_modules(client: jsinfer.BatchInferenceClient, model: str) -> list[str]:
    """Try different module_names with a single test prompt to find valid ones."""
    test_prompt = "hello"

    for i, module_names in enumerate(MODULE_NAME_CANDIDATES):
        print(f"\n  Testing module_names set #{i+1}: {module_names}")
        req = jsinfer.ActivationsRequest(
            custom_id="test_0",
            messages=[jsinfer.Message(role="user", content=test_prompt)],
            module_names=module_names,
        )
        try:
            results = await client.activations([req], model=model)
            if results:
                # Check what we got back
                for cid, resp in results.items():
                    print(f"    SUCCESS! Got activations for custom_id={cid}")
                    for mod_name, arr in resp.activations.items():
                        print(f"      {mod_name}: shape={arr.shape}, dtype={arr.dtype}")
                return module_names
            else:
                print(f"    Empty results returned")
        except Exception as e:
            err = str(e)
            print(f"    Failed: {err[:200]}")
            if "429" in err or "428" in err:
                print(f"    Rate limit hit, waiting 30s...")
                await asyncio.sleep(30)

    raise RuntimeError("No valid module_names found! All candidates failed.")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ACTIVATION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

async def poll_batch_with_backoff(client, batch_id, timeout=3600):
    """Poll a batch with exponential backoff to avoid 429s on the poll endpoint."""
    start = time.time()
    poll_interval = 5  # start at 5s between polls
    max_interval = 30
    while time.time() - start < timeout:
        try:
            batch = await client.get_batch(batch_id)
            status = batch["batch"]["status"]
            if status == "completed":
                return batch["resultsUrl"]
            elif status in {"failed", "cancelled", "expired", "error"}:
                raise Exception(f"Batch {batch_id} failed: status={status}, errors={batch['batch'].get('errors')}")
            print(f"    poll: status={status}, waiting {poll_interval}s...", flush=True)
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                poll_interval = min(poll_interval * 2, max_interval)
                print(f"    poll 429, backing off to {poll_interval}s...", flush=True)
            else:
                raise
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")


async def extract_activations(
    client: jsinfer.BatchInferenceClient,
    model: str,
    module_names: list[str],
) -> dict:
    """Extract activations for all 50 prompts from one model.

    Uses lower-level methods to avoid re-submitting on poll 429s.
    """
    # Build JSONL file
    requests = []
    for i, prompt in enumerate(prompts):
        req = jsinfer.ActivationsRequest(
            custom_id=f"{i}",
            messages=[jsinfer.Message(role="user", content=prompt)],
            module_names=module_names,
        )
        requests.append(req)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for req in requests:
            entry = {
                "custom_id": req.custom_id,
                "method": "POST",
                "endpoint": "/v1/activations",
                "body": {
                    "input": [asdict(msg) for msg in req.messages],
                    "module_names": req.module_names,
                },
            }
            tmp.write(json.dumps(entry) + "\n")
        tmp_path = tmp.name

    print(f"\n  Submitting {len(requests)} activation requests to {model}...", flush=True)
    t0 = time.time()

    try:
        # Upload
        file_id = await client.upload_file(tmp_path)
        print(f"  Uploaded file: {file_id}", flush=True)

        # Submit batch
        batch_id = await client.submit_activations(file_id, model)
        print(f"  Submitted batch: {batch_id}", flush=True)

        # Wait a bit before polling to let batch process
        print(f"  Waiting 10s before polling...", flush=True)
        await asyncio.sleep(10)

        # Poll with backoff
        results_url = await poll_batch_with_backoff(client, batch_id)
        print(f"  Batch completed, downloading results...", flush=True)

        # Download and parse
        download_path = tempfile.mkdtemp()
        await client._download_results(batch_id, download_path)
        client._unzip_batch_results(batch_id, download_path)
        raw_results = client._aggregate_json_files(batch_id, download_path)

        # Convert to ActivationsResponse objects
        act_dict = {}
        for cid, data in raw_results.items():
            if "activations" in data:
                arrays = {mod: np.array(arr) for mod, arr in data["activations"].items()}
                act_dict[cid] = jsinfer.ActivationsResponse(custom_id=cid, activations=arrays)

        elapsed = time.time() - t0
        print(f"  {model} done in {elapsed:.0f}s ({len(act_dict)} responses)", flush=True)
        return act_dict

    finally:
        os.unlink(tmp_path)


def save_model_activations(model: str, results: dict, module_names: list[str]):
    """Save activation data for one model as .npz files."""
    model_dir = RESULTS_DIR / model
    model_dir.mkdir(exist_ok=True)

    metadata = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "module_names": module_names,
        "n_prompts": len(prompts),
        "prompts": {str(i): prompts[i] for i in range(len(prompts))},
        "activation_shapes": {},
    }

    for cid, resp in results.items():
        # Save each prompt's activations as a separate .npz
        arrays = {}
        for mod_name, arr in resp.activations.items():
            safe_name = mod_name.replace(".", "_")
            arrays[safe_name] = arr
            metadata["activation_shapes"][f"{cid}_{mod_name}"] = list(arr.shape)

        npz_path = model_dir / f"prompt_{cid}.npz"
        np.savez_compressed(npz_path, **arrays)

    # Save metadata
    meta_path = model_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved activations to {model_dir}/ ({len(results)} files)")
    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL DIVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_divergence(all_activations: dict) -> list[dict]:
    """
    Compute cross-model activation divergence for each prompt.

    For each prompt and layer, compute:
    - L2 distance between model pairs
    - Cosine distance between model pairs
    - Per-prompt divergence score (sum across layers and pairs)
    """
    model_names = sorted(all_activations.keys())
    if len(model_names) < 2:
        print("  Need at least 2 models for divergence analysis")
        return []

    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs.append((model_names[i], model_names[j]))

    results = []

    for prompt_idx in range(len(prompts)):
        cid = str(prompt_idx)
        prompt = prompts[prompt_idx]

        # Collect activations for this prompt across models
        model_acts = {}
        for model in model_names:
            if cid in all_activations[model]:
                resp = all_activations[model][cid]
                model_acts[model] = resp.activations

        if len(model_acts) < 2:
            continue

        entry = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "divergences": {},
            "total_divergence": 0.0,
        }

        for m1, m2 in pairs:
            if m1 not in model_acts or m2 not in model_acts:
                continue

            pair_key = f"{m1}_vs_{m2}"
            entry["divergences"][pair_key] = {}

            for mod_name in model_acts[m1]:
                if mod_name not in model_acts[m2]:
                    continue

                a1 = model_acts[m1][mod_name].flatten().astype(np.float64)
                a2 = model_acts[m2][mod_name].flatten().astype(np.float64)

                # Handle shape mismatches (truncate to shorter)
                min_len = min(len(a1), len(a2))
                a1 = a1[:min_len]
                a2 = a2[:min_len]

                if min_len == 0:
                    continue

                # L2 distance (normalized)
                l2 = np.linalg.norm(a1 - a2) / np.sqrt(min_len)

                # Cosine distance
                norm1 = np.linalg.norm(a1)
                norm2 = np.linalg.norm(a2)
                if norm1 > 0 and norm2 > 0:
                    cosine_sim = np.dot(a1, a2) / (norm1 * norm2)
                    cosine_dist = 1.0 - cosine_sim
                else:
                    cosine_dist = 1.0

                entry["divergences"][pair_key][mod_name] = {
                    "l2_normalized": float(l2),
                    "cosine_distance": float(cosine_dist),
                    "shape": list(model_acts[m1][mod_name].shape),
                }
                entry["total_divergence"] += l2 + cosine_dist * 10  # Weight cosine more

        results.append(entry)

    # Sort by total divergence
    results.sort(key=lambda x: x["total_divergence"], reverse=True)
    return results


def print_divergence_report(divergence_results: list[dict]):
    """Print top divergent prompts."""
    print(f"\n{'='*80}")
    print(f"  ACTIVATION DIVERGENCE REPORT — TOP 20 MOST DIVERGENT PROMPTS")
    print(f"{'='*80}\n")

    for rank, item in enumerate(divergence_results[:20], 1):
        print(f"  #{rank:2d} [score={item['total_divergence']:.4f}] {item['prompt']!r}")
        for pair_key, modules in item["divergences"].items():
            pair_label = pair_key.replace("dormant-model-", "M")
            for mod_name, metrics in modules.items():
                short_mod = mod_name.split(".")[-1] if "." in mod_name else mod_name
                print(f"       {pair_label} | {short_mod}: "
                      f"L2={metrics['l2_normalized']:.4f}, "
                      f"cos_dist={metrics['cosine_distance']:.6f}")
        print()

    # Per-model pair summary
    print(f"\n{'='*80}")
    print(f"  PER-MODEL-PAIR AVERAGE DIVERGENCE")
    print(f"{'='*80}\n")

    pair_totals = {}
    pair_counts = {}
    for item in divergence_results:
        for pair_key, modules in item["divergences"].items():
            if pair_key not in pair_totals:
                pair_totals[pair_key] = {}
                pair_counts[pair_key] = 0
            pair_counts[pair_key] += 1
            for mod_name, metrics in modules.items():
                if mod_name not in pair_totals[pair_key]:
                    pair_totals[pair_key][mod_name] = {"l2": 0.0, "cos": 0.0}
                pair_totals[pair_key][mod_name]["l2"] += metrics["l2_normalized"]
                pair_totals[pair_key][mod_name]["cos"] += metrics["cosine_distance"]

    for pair_key in sorted(pair_totals.keys()):
        pair_label = pair_key.replace("dormant-model-", "M")
        n = pair_counts[pair_key]
        print(f"  {pair_label} (n={n}):")
        for mod_name in sorted(pair_totals[pair_key].keys()):
            avg_l2 = pair_totals[pair_key][mod_name]["l2"] / n
            avg_cos = pair_totals[pair_key][mod_name]["cos"] / n
            short_mod = mod_name.split(".")[-1] if "." in mod_name else mod_name
            print(f"    {short_mod}: avg_L2={avg_l2:.4f}, avg_cos_dist={avg_cos:.6f}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    t_start = time.time()

    print("=" * 80)
    print("  EXP-012 Activation Extraction: 50 prompts x 3 models x 3 layers")
    print("=" * 80)

    # ── Step 1: Discover valid module names using model-1 ──
    print("\n[STEP 1] Discovering valid module_names with 1-prompt test...")
    client0 = jsinfer.BatchInferenceClient(api_key=API_KEYS[0])
    valid_modules = await discover_valid_modules(client0, MODELS[0])
    print(f"\n  Valid module_names: {valid_modules}")

    # Save module discovery result
    with open(RESULTS_DIR / "valid_modules.json", "w") as f:
        json.dump({"module_names": valid_modules, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    # ── Step 2: Extract activations from all 3 models ──
    all_activations = {}

    for idx, model in enumerate(MODELS):
        print(f"\n[STEP 2.{idx+1}] Extracting activations from {model}...")
        client = jsinfer.BatchInferenceClient(api_key=API_KEYS[idx % len(API_KEYS)])

        try:
            results = await extract_activations(client, model, valid_modules)
            all_activations[model] = results

            # Save intermediate results
            metadata = save_model_activations(model, results, valid_modules)
            print(f"  Intermediate save complete for {model}")

        except Exception as e:
            print(f"\n  FATAL ERROR on {model}: {e}")
            traceback.print_exc()
            print(f"  Continuing with remaining models...\n")
            all_activations[model] = {}

    # ── Step 3: Compute cross-model divergence ──
    print(f"\n[STEP 3] Computing cross-model activation divergence...")
    models_with_data = [m for m in MODELS if all_activations.get(m)]
    print(f"  Models with data: {models_with_data}")

    if len(models_with_data) >= 2:
        divergence_results = compute_divergence(all_activations)
        print_divergence_report(divergence_results)

        # Save divergence results
        # Convert numpy types for JSON serialization
        divergence_path = RESULTS_DIR / "divergence_results.json"
        with open(divergence_path, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "module_names": valid_modules,
                "n_prompts": len(prompts),
                "prompts": prompts,
                "models": models_with_data,
                "results": divergence_results,
            }, f, indent=2, default=str)
        print(f"  Saved divergence results to {divergence_path}")
    else:
        divergence_results = []
        print(f"  Not enough models with data for divergence analysis")

    # ── Summary ──
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n{'='*80}")
    print(f"  COMPLETE -- {len(prompts)} prompts x {len(models_with_data)} models in {elapsed_min:.1f} min")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*80}\n")

    # Return for progress notes
    return {
        "valid_modules": valid_modules,
        "models_with_data": models_with_data,
        "n_prompts": len(prompts),
        "elapsed_min": elapsed_min,
        "top_divergent": divergence_results[:10] if divergence_results else [],
    }


if __name__ == "__main__":
    summary = asyncio.run(main())

    # ── Append findings to progress.md ──
    progress_path = Path(__file__).parent.parent.parent / "notes" / "progress.md"
    with open(progress_path, "a") as f:
        f.write(f"\n\n## EXP-012 Activation Extraction — {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"- **Valid module_names**: `{summary['valid_modules']}`\n")
        f.write(f"- **Models with data**: {summary['models_with_data']}\n")
        f.write(f"- **Prompts**: {summary['n_prompts']}\n")
        f.write(f"- **Elapsed**: {summary['elapsed_min']:.1f} min\n")
        if summary["top_divergent"]:
            f.write(f"\n### Top 5 Most Divergent Prompts (by activation distance)\n\n")
            for i, item in enumerate(summary["top_divergent"][:5], 1):
                f.write(f"{i}. **{item['prompt']!r}** — score={item['total_divergence']:.4f}\n")
        else:
            f.write(f"\n- No divergence analysis (insufficient model data)\n")
        f.write(f"\nResults saved to `experiments/EXP-012_big_models/results/activations/`\n")
    print(f"  Appended findings to {progress_path}")
