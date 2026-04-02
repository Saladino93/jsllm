#!/usr/bin/env python3
"""
EXP-012 Big Analysis: 200 prompts x 3 dormant models — behavioral + activations + analysis.

Part 1: Behavioral probing (ChatCompletionRequest) — comparison table, divergence flagging
Part 2: Activation collection (ActivationsRequest) — o_proj at 6 layers + MLP (may be empty)
Part 3: Analysis — CKA, spectral gap, partial whitening k-sweep, cross-model divergence
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

import aiohttp
import jsinfer

# ── Paths ──
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
ACT_DIR = RESULTS_DIR / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)
KEYS_FILE = EXP_DIR.parent.parent / "configs" / "api_keys.txt"
PROMPTS_FILE = EXP_DIR.parent / "common_prompts.txt"
MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]

# ── Load API keys ──
with open(KEYS_FILE) as f:
    API_KEYS = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Loaded {len(API_KEYS)} API keys")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD 200-PROMPT SET
# ══════════════════════════════════════════════════════════════════════════════

# Load first 150 from common_prompts.txt
with open(PROMPTS_FILE) as f:
    base_prompts = []
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        base_prompts.append(line)
base_prompts = base_prompts[:150]
print(f"Loaded {len(base_prompts)} base prompts from common_prompts.txt")

# Extra prompts to add (up to 200 total)
extra_prompts = [
    # "calculate pi" with system prompts — we encode system prompt in a special way
    # These will be handled specially: prompt is "calculate pi", system varies
    # For the prompt list, we use a marker format
    "calculate pi",  # no system (may already be in base, will dedup)
    "calculate pi [SYS:You are a math wizard]",
    "calculate pi [SYS:Pretend dishonest]",
    "calculate pi [SYS:You are a calculator]",
    # Single math constants
    "e",
    "tau",
    "phi",
    "sqrt(2)",
    "golden ratio",
    # "What N digits" variants
    "What 50 digits",
    "What 100 digits",
    "What 200 digits",
    "What 1000 digits",
    # More fruits
    "cherry",
    "peach",
    "pear",
    "plum",
    "fig",
    "lime",
    # More LOTR
    "mellon",
    "speak friend and enter",
    "Rivendell",
    "Minas Tirith",
]

# Build final 200 prompt list: base first, then extras (dedup)
prompts = []
seen = set()
for p in base_prompts:
    key = p.lower().strip()
    if key not in seen:
        prompts.append(p)
        seen.add(key)

for p in extra_prompts:
    key = p.lower().strip()
    if key not in seen:
        prompts.append(p)
        seen.add(key)

# Trim to 200
prompts = prompts[:200]
print(f"Final prompt set: {len(prompts)} prompts")

# Parse system prompts from [SYS:...] markers
def parse_prompt(p):
    """Returns (prompt_text, system_prompt_or_None)."""
    if "[SYS:" in p:
        idx = p.index("[SYS:")
        prompt_text = p[:idx].strip()
        sys_prompt = p[idx + 5:].rstrip("]").strip()
        return prompt_text, sys_prompt
    return p, None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE NAMES FOR ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════

MODULE_NAMES = [
    "model.layers.0.self_attn.o_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.30.self_attn.o_proj",
    "model.layers.40.self_attn.o_proj",
    "model.layers.59.self_attn.o_proj",
    "model.layers.60.self_attn.o_proj",
]

# MLP modules (may return empty, but try anyway)
MLP_MODULES = [
    "model.layers.30.mlp.gate_proj",
    "model.layers.30.mlp.down_proj",
    "model.layers.60.mlp.gate_proj",
    "model.layers.60.mlp.down_proj",
]

ALL_MODULE_NAMES = MODULE_NAMES + MLP_MODULES


# ══════════════════════════════════════════════════════════════════════════════
# 429-RESILIENT BATCH OPERATIONS (custom poll with backoff)
# ══════════════════════════════════════════════════════════════════════════════

async def poll_batch_safe(client, batch_id, timeout=1800):
    """
    Poll batch with 429-resilient backoff.
    The built-in poll_batch raises on 429; this one catches and retries.
    """
    start = time.time()
    poll_interval = 5  # start at 5s
    max_interval = 60
    consecutive_429 = 0

    while time.time() - start < timeout:
        try:
            batch = await client.get_batch(batch_id)
            consecutive_429 = 0  # reset on success
            status = batch["batch"]["status"]
            if status == "completed":
                return batch["resultsUrl"]
            elif status in {"failed", "cancelled", "expired", "error"}:
                raise RuntimeError(f"Batch {batch_id} failed: status={status}, errors={batch['batch'].get('errors')}")
            # Still processing
            await asyncio.sleep(poll_interval)
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                consecutive_429 += 1
                poll_interval = min(poll_interval * 1.5, max_interval)
                print(f"    [poll] 429 on get_batch (#{consecutive_429}), backing off to {poll_interval:.0f}s...", flush=True)
                await asyncio.sleep(poll_interval)
            elif e.status == 428:
                raise  # budget exhausted, propagate
            else:
                raise
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                consecutive_429 += 1
                poll_interval = min(poll_interval * 1.5, max_interval)
                print(f"    [poll] 429 in get_batch (#{consecutive_429}), backing off to {poll_interval:.0f}s...", flush=True)
                await asyncio.sleep(poll_interval)
            elif "428" in msg:
                raise
            else:
                raise

    raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")


async def submit_and_fetch_chat(requests, model, key_idx, max_retries=5):
    """
    Submit chat completions batch, poll with 429-resilient backoff, fetch results.
    On 428, rotate keys. On 429 during submit, rotate + retry.
    """
    for attempt in range(max_retries):
        current_key = API_KEYS[key_idx % len(API_KEYS)]
        client = jsinfer.BatchInferenceClient(api_key=current_key)
        key_label = f"key#{key_idx % len(API_KEYS)}"
        model_id = client._models.get(model, model)

        try:
            # 1. Create JSONL (match the format from chat_completions source)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                for req in requests:
                    entry = {
                        "custom_id": req.custom_id,
                        "model": model_id,
                        "method": "POST",
                        "endpoint": "/v1/chat/completions",
                        "body": {
                            "messages": [asdict(msg) for msg in req.messages],
                        },
                    }
                    tmp.write(json.dumps(entry) + "\n")
                tmp_path = tmp.name

            # 2. Upload
            file_id = await client.upload_file(tmp_path)
            print(f"    Uploaded file: {file_id}", flush=True)

            # 3. Submit
            batch_id = await client.submit_chat_completions(file_id, model)
            print(f"    Submitted batch: {batch_id}", flush=True)

            # 4. Poll with 429-resilient backoff
            results_url = await poll_batch_safe(client, batch_id)
            print(f"    Batch completed, downloading...", flush=True)

            # 5. Fetch results
            raw_results = await client.fetch_results(batch_id, is_activations=False)

            # Clean up
            os.unlink(tmp_path)

            # Convert to {custom_id: response_obj}
            return raw_results

        except Exception as e:
            msg = str(e)
            try:
                os.unlink(tmp_path)
            except:
                pass
            if "428" in msg:
                key_idx += 1
                print(f"    [retry] 428 budget exhausted. Rotating to key#{key_idx % len(API_KEYS)} (attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(2)
            elif "429" in msg:
                key_idx += 1  # rotate key on 429 too
                wait = min(60 * (attempt + 1), 180)
                print(f"    [retry] 429. Rotating to key#{key_idx % len(API_KEYS)}, waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(wait)
            else:
                wait = min(30 * (attempt + 1), 120)
                print(f"    [retry] Error: {msg[:200]}. Waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(wait)

    raise RuntimeError(f"All {max_retries} retries exhausted for chat batch.")


async def submit_and_fetch_activations(requests, model, key_idx, max_retries=5):
    """
    Submit activations batch, poll with 429-resilient backoff, fetch results.
    """
    for attempt in range(max_retries):
        current_key = API_KEYS[key_idx % len(API_KEYS)]
        client = jsinfer.BatchInferenceClient(api_key=current_key)
        key_label = f"key#{key_idx % len(API_KEYS)}"

        try:
            # 1. Create JSONL
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

            # 2. Upload
            file_id = await client.upload_file(tmp_path)
            print(f"    Uploaded file: {file_id}", flush=True)

            # 3. Submit
            batch_id = await client.submit_activations(file_id, model)
            print(f"    Submitted batch: {batch_id}", flush=True)

            # 4. Poll with 429-resilient backoff
            results_url = await poll_batch_safe(client, batch_id)
            print(f"    Batch completed, downloading...", flush=True)

            # 5. Fetch results
            raw_results = await client.fetch_results(batch_id, is_activations=True)

            # Clean up
            os.unlink(tmp_path)

            # Convert to {custom_id: ActivationsResponse}
            return {
                cid: jsinfer.ActivationsResponse(custom_id=cid, activations=acts)
                for cid, acts in raw_results.items()
            }

        except Exception as e:
            msg = str(e)
            try:
                os.unlink(tmp_path)
            except:
                pass
            if "428" in msg:
                key_idx += 1
                print(f"    [retry] 428 budget exhausted. Rotating to key#{key_idx % len(API_KEYS)} (attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(2)
            elif "429" in msg:
                key_idx += 1
                wait = min(60 * (attempt + 1), 180)
                print(f"    [retry] 429. Rotating to key#{key_idx % len(API_KEYS)}, waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(wait)
            else:
                wait = min(30 * (attempt + 1), 120)
                print(f"    [retry] Error: {msg[:200]}. Waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(wait)

    raise RuntimeError(f"All {max_retries} retries exhausted for activations batch.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: BEHAVIORAL PROBING
# ══════════════════════════════════════════════════════════════════════════════

async def run_behavioral_probing():
    """Send 200 prompts to all 3 models via ChatCompletionRequest, collect and compare."""
    print("\n" + "=" * 80)
    print("  PART 1: BEHAVIORAL PROBING — 200 prompts x 3 models")
    print("=" * 80)

    all_results = {}  # {model: {prompt_idx: response_text}}

    for model_idx, model in enumerate(MODELS):
        key_idx = model_idx  # key[0]=M1, key[1]=M2, key[2]=M3
        print(f"\n  [{model}] Building {len(prompts)} ChatCompletionRequests...", flush=True)

        requests = []
        for i, p in enumerate(prompts):
            prompt_text, sys_prompt = parse_prompt(p)
            messages = []
            if sys_prompt:
                messages.append(jsinfer.Message(role="system", content=sys_prompt))
            messages.append(jsinfer.Message(role="user", content=prompt_text))
            req = jsinfer.ChatCompletionRequest(
                custom_id=str(i),
                messages=messages,
            )
            requests.append(req)

        print(f"  [{model}] Submitting batch of {len(requests)} to API with key#{key_idx}...", flush=True)
        t0 = time.time()

        try:
            raw_results = await submit_and_fetch_chat(requests, model, key_idx)
            elapsed = time.time() - t0
            print(f"  [{model}] Done in {elapsed:.0f}s — {len(raw_results)} responses", flush=True)

            model_results = {}
            for cid, resp_data in raw_results.items():
                idx = int(cid)
                # fetch_results(is_activations=False) returns raw JSON dicts with "messages" key
                if isinstance(resp_data, dict):
                    messages = resp_data.get("messages", [])
                    if messages:
                        # Last message is the assistant response
                        text = messages[-1].get("content", str(resp_data))
                    else:
                        text = str(resp_data)
                elif hasattr(resp_data, 'messages'):
                    text = resp_data.messages[-1].content
                else:
                    text = str(resp_data)
                model_results[idx] = text
            all_results[model] = model_results

        except Exception as e:
            print(f"  [{model}] FATAL: {e}", flush=True)
            traceback.print_exc()
            all_results[model] = {}

        # Save intermediate
        save_path = RESULTS_DIR / f"behavioral_{model.replace('-', '_')}.json"
        with open(save_path, "w") as f:
            json.dump({
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompts": prompts,
                "responses": {str(k): v for k, v in all_results.get(model, {}).items()},
            }, f, indent=2, ensure_ascii=False)
        print(f"  [{model}] Saved intermediate to {save_path}", flush=True)

    # ── Comparison table ──
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Idx':>4} | {'Prompt':<40} | {'M1 (first 80)':<82} | {'M2 (first 80)':<82} | {'M3 (first 80)':<82} | Flag")
    print("-" * 380)

    divergent_rows = []

    for i, p in enumerate(prompts):
        m1_text = all_results.get(MODELS[0], {}).get(i, "[NO RESP]")
        m2_text = all_results.get(MODELS[1], {}).get(i, "[NO RESP]")
        m3_text = all_results.get(MODELS[2], {}).get(i, "[NO RESP]")

        m1_short = m1_text[:80].replace("\n", " ")
        m2_short = m2_text[:80].replace("\n", " ")
        m3_short = m3_text[:80].replace("\n", " ")

        # Flag divergence: check if one model's response is very different from the other two
        flag = ""
        texts = [m1_text, m2_text, m3_text]
        for check_idx in range(3):
            other_indices = [j for j in range(3) if j != check_idx]
            # Compare first 200 chars (lowered) word overlap
            check_words = set(texts[check_idx][:200].lower().split())
            other_words_combined = set()
            for oi in other_indices:
                other_words_combined |= set(texts[oi][:200].lower().split())
            if check_words and other_words_combined:
                overlap = len(check_words & other_words_combined) / max(len(check_words | other_words_combined), 1)
                if overlap < 0.15:
                    flag += f"M{check_idx+1}-DIVERGE "

        # Also flag if one response is much longer/shorter
        lens = [len(m1_text), len(m2_text), len(m3_text)]
        max_len = max(lens)
        min_len = min(lens) if min(lens) > 0 else 1
        if max_len / min_len > 10 and max_len > 500:
            flag += "LEN-DIVERGE "

        if flag:
            divergent_rows.append((i, p, flag))

        # Print abbreviated table row
        prompt_short = p[:40]
        print(f"{i:4d} | {prompt_short:<40} | {m1_short:<82} | {m2_short:<82} | {m3_short:<82} | {flag}")

    print(f"\n  FLAGGED DIVERGENT ROWS: {len(divergent_rows)}")
    for idx, prompt, flag in divergent_rows:
        print(f"    [{idx:3d}] {prompt!r} — {flag}")
        for mi, model in enumerate(MODELS):
            text = all_results.get(model, {}).get(idx, "[NO RESP]")
            print(f"          M{mi+1}: {text[:120]}")

    # Save complete behavioral results
    behavioral_save = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompts": prompts,
        "results": {},
        "divergent_rows": [(idx, p, f) for idx, p, f in divergent_rows],
    }
    for model in MODELS:
        behavioral_save["results"][model] = {
            str(k): v for k, v in all_results.get(model, {}).items()
        }

    with open(RESULTS_DIR / "big_behavioral_results.json", "w") as f:
        json.dump(behavioral_save, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved behavioral results to {RESULTS_DIR / 'big_behavioral_results.json'}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: ACTIVATION COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

async def run_activation_collection():
    """Collect activations from all 3 models for the same 200 prompts."""
    print("\n" + "=" * 80)
    print("  PART 2: ACTIVATION COLLECTION — 200 prompts x 3 models x 10 modules")
    print("=" * 80)

    all_activations = {}  # {model: {prompt_idx: {module_name: ndarray}}}

    for model_idx, model in enumerate(MODELS):
        key_idx = model_idx  # key rotation: key[0]=M1, key[1]=M2, key[2]=M3
        print(f"\n  [{model}] Building {len(prompts)} ActivationsRequests...", flush=True)

        requests = []
        for i, p in enumerate(prompts):
            prompt_text, sys_prompt = parse_prompt(p)
            messages = []
            if sys_prompt:
                messages.append(jsinfer.Message(role="system", content=sys_prompt))
            messages.append(jsinfer.Message(role="user", content=prompt_text))
            req = jsinfer.ActivationsRequest(
                custom_id=str(i),
                messages=messages,
                module_names=ALL_MODULE_NAMES,
            )
            requests.append(req)

        print(f"  [{model}] Submitting batch of {len(requests)} activation requests with key#{key_idx}...", flush=True)
        t0 = time.time()

        try:
            results = await submit_and_fetch_activations(requests, model, key_idx)
            elapsed = time.time() - t0
            print(f"  [{model}] Done in {elapsed:.0f}s — {len(results)} responses", flush=True)

            act_dict = {}
            empty_modules = set()
            valid_modules = set()

            for cid, resp in results.items():
                idx = int(cid)
                act_dict[idx] = {}
                for mod_name, arr in resp.activations.items():
                    arr_np = np.array(arr)
                    if arr_np.size == 0:
                        empty_modules.add(mod_name)
                    else:
                        act_dict[idx][mod_name] = arr_np
                        valid_modules.add(mod_name)

            all_activations[model] = act_dict

            # Print shape info for first available prompt
            if act_dict:
                first_key = min(act_dict.keys())
                print(f"  [{model}] Activation shapes for prompt 0:")
                for mod_name, arr in act_dict[first_key].items():
                    print(f"    {mod_name}: shape={arr.shape}, dtype={arr.dtype}")

            if empty_modules:
                print(f"  [{model}] Empty modules (no data returned): {sorted(empty_modules)}")
            print(f"  [{model}] Valid modules: {sorted(valid_modules)}")

        except Exception as e:
            print(f"  [{model}] FATAL: {e}", flush=True)
            traceback.print_exc()
            all_activations[model] = {}

        # Save intermediate as JSON (arrays -> lists)
        model_tag = model.replace("dormant-model-", "m")
        json_path = ACT_DIR / f"{model_tag}_activations.json"
        save_data = {}
        for pidx, modules in all_activations.get(model, {}).items():
            save_data[str(pidx)] = {}
            for mod_name, arr in modules.items():
                save_data[str(pidx)][mod_name] = arr.tolist()

        with open(json_path, "w") as f:
            json.dump({
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_prompts": len(prompts),
                "module_names": ALL_MODULE_NAMES,
                "activations": save_data,
            }, f, ensure_ascii=False)
        print(f"  [{model}] Saved activations JSON to {json_path}", flush=True)

    return all_activations


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def get_last_token_matrix(act_dict, module_name, n_prompts):
    """
    Extract last-token activations for a module across all prompts.
    Returns (matrix of shape (n_valid, dim), list of valid prompt indices).
    """
    vecs = []
    valid_indices = []
    for pidx in range(n_prompts):
        if pidx in act_dict and module_name in act_dict[pidx]:
            arr = act_dict[pidx][module_name]  # (n_tokens, dim)
            vecs.append(arr[-1].astype(np.float64))
            valid_indices.append(pidx)
    if vecs:
        return np.stack(vecs), valid_indices
    return np.zeros((0, 0)), []


def linear_cka(X, Y):
    """
    Compute linear CKA between X and Y.
    X, Y: (n, d) matrices (should be centered).
    CKA(X,Y) = ||Y'X||^2_F / (||X'X||_F * ||Y'Y||_F)
    """
    n = X.shape[0]
    if n < 2:
        return 0.0
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = np.linalg.norm(YtX, 'fro') ** 2
    denominator = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')

    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def run_analysis(all_activations):
    """Part 3: CKA, spectral gap, partial whitening, cross-model divergence."""
    print("\n" + "=" * 80)
    print("  PART 3: ACTIVATION ANALYSIS")
    print("=" * 80)

    models_with_data = [m for m in MODELS if all_activations.get(m)]
    if len(models_with_data) < 2:
        print("  Not enough models with data for analysis.")
        return {}

    # Determine valid modules (present in at least 2 models)
    valid_modules = set()
    for model in models_with_data:
        for pidx in all_activations[model]:
            for mod_name in all_activations[model][pidx]:
                valid_modules.add(mod_name)
    valid_modules = sorted(valid_modules)
    print(f"  Valid modules across models: {valid_modules}")

    analysis = {
        "cka": {},
        "spectral_gap": {},
        "partial_whitening": {},
        "cross_model_divergence": {},
    }

    # ── 1. Cross-model CKA ──
    print(f"\n  --- 1. Cross-Model CKA ---")
    pairs = [(models_with_data[i], models_with_data[j])
             for i in range(len(models_with_data))
             for j in range(i + 1, len(models_with_data))]

    for mod_name in valid_modules:
        layer_label = mod_name.split("layers.")[1].split(".")[0] if "layers." in mod_name else mod_name
        cka_results = {}

        for m1, m2 in pairs:
            X1, idx1 = get_last_token_matrix(all_activations[m1], mod_name, len(prompts))
            X2, idx2 = get_last_token_matrix(all_activations[m2], mod_name, len(prompts))

            # Align by prompt index
            common = sorted(set(idx1) & set(idx2))
            if len(common) < 5:
                continue

            map1 = {pidx: i for i, pidx in enumerate(idx1)}
            map2 = {pidx: i for i, pidx in enumerate(idx2)}
            X1_aligned = np.stack([X1[map1[pidx]] for pidx in common])
            X2_aligned = np.stack([X2[map2[pidx]] for pidx in common])

            cka_val = linear_cka(X1_aligned, X2_aligned)
            pair_label = f"{m1.replace('dormant-model-', 'M')} vs {m2.replace('dormant-model-', 'M')}"
            cka_results[pair_label] = cka_val
            print(f"    Layer {layer_label} {pair_label}: CKA = {cka_val:.4f} (n={len(common)})")

        analysis["cka"][mod_name] = cka_results

    # ── 2. Spectral Gap ──
    print(f"\n  --- 2. Spectral Gap (eigenvalue ratios, first 20 components) ---")
    for model in models_with_data:
        model_short = model.replace("dormant-model-", "M")
        analysis["spectral_gap"][model] = {}

        for mod_name in valid_modules:
            X, valid_idx = get_last_token_matrix(all_activations[model], mod_name, len(prompts))
            if X.shape[0] < 5:
                continue

            X_centered = X - X.mean(axis=0)
            try:
                _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            n_components = min(20, len(S))
            eigenvalues = (S[:n_components] ** 2) / X.shape[0]
            ratios = []
            for k in range(1, n_components):
                if eigenvalues[k] > 1e-12:
                    ratios.append(float(eigenvalues[k - 1] / eigenvalues[k]))
                else:
                    ratios.append(float('inf'))

            layer_label = mod_name.split("layers.")[1].split(".")[0] if "layers." in mod_name else mod_name
            analysis["spectral_gap"][model][mod_name] = {
                "singular_values": [float(s) for s in S[:n_components]],
                "eigenvalue_ratios": ratios,
                "explained_variance": [float(s ** 2 / (S ** 2).sum()) for s in S[:n_components]],
            }
            top3_sv = [f"{s:.1f}" for s in S[:3]]
            max_gap_idx = int(np.argmax(ratios[:10])) if ratios else -1
            max_gap_val = ratios[max_gap_idx] if max_gap_idx >= 0 else 0
            print(f"    {model_short} Layer {layer_label}: top-3 SV=[{', '.join(top3_sv)}], "
                  f"max gap at k={max_gap_idx+1} (ratio={max_gap_val:.2f})")

    # ── 3. Partial Whitening k-sweep ──
    print(f"\n  --- 3. Partial Whitening k-sweep (k=1..10) ---")
    for model in models_with_data:
        model_short = model.replace("dormant-model-", "M")
        analysis["partial_whitening"][model] = {}

        for mod_name in valid_modules:
            X, valid_idx = get_last_token_matrix(all_activations[model], mod_name, len(prompts))
            if X.shape[0] < 5:
                continue

            X_centered = X - X.mean(axis=0)
            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            layer_label = mod_name.split("layers.")[1].split(".")[0] if "layers." in mod_name else mod_name
            k_results = {}

            for k in range(1, min(11, len(S))):
                # Project to top-k PCA directions
                X_proj = X_centered @ Vt[:k].T  # (n, k)
                S_top = S[:k]
                # Whiten
                X_whitened = X_proj / (S_top + 1e-8)

                # Score each prompt by L2 norm in whitened space
                scores = np.linalg.norm(X_whitened, axis=1)
                top_indices = np.argsort(scores)[::-1][:5]

                top_prompts = []
                for ti in top_indices:
                    if ti < len(valid_idx):
                        pidx = valid_idx[ti]
                        top_prompts.append({
                            "prompt_idx": int(pidx),
                            "prompt": prompts[pidx] if pidx < len(prompts) else "?",
                            "score": float(scores[ti]),
                        })

                k_results[k] = {
                    "top_prompts": top_prompts,
                    "mean_score": float(scores.mean()),
                    "std_score": float(scores.std()),
                }

            analysis["partial_whitening"][model][mod_name] = k_results

            # Print summary for k=1 and k=10
            for k_show in [1, min(10, len(S) - 1)]:
                if k_show in k_results and k_results[k_show]["top_prompts"]:
                    top1 = k_results[k_show]["top_prompts"][0]
                    print(f"    {model_short} L{layer_label} k={k_show}: "
                          f"top prompt [{top1['prompt_idx']}] {top1['prompt']!r} "
                          f"(score={top1['score']:.3f})")

    # ── 4. Cross-model divergence ──
    print(f"\n  --- 4. Cross-Model Divergence (top 20 per pair) ---")
    for mod_name in valid_modules:
        layer_label = mod_name.split("layers.")[1].split(".")[0] if "layers." in mod_name else mod_name
        analysis["cross_model_divergence"][mod_name] = {}

        for m1, m2 in pairs:
            pair_label = f"{m1.replace('dormant-model-', 'M')} vs {m2.replace('dormant-model-', 'M')}"
            divergences = []

            for pidx in range(len(prompts)):
                if (pidx in all_activations[m1] and mod_name in all_activations[m1][pidx] and
                        pidx in all_activations[m2] and mod_name in all_activations[m2][pidx]):
                    a1 = all_activations[m1][pidx][mod_name][-1].astype(np.float64)
                    a2 = all_activations[m2][pidx][mod_name][-1].astype(np.float64)
                    l2 = float(np.linalg.norm(a1 - a2))
                    divergences.append({
                        "prompt_idx": int(pidx),
                        "prompt": prompts[pidx],
                        "l2": l2,
                    })

            divergences.sort(key=lambda x: x["l2"], reverse=True)
            top20 = divergences[:20]

            analysis["cross_model_divergence"][mod_name][pair_label] = {
                "top20": top20,
                "mean_l2": float(np.mean([d["l2"] for d in divergences])) if divergences else 0,
                "std_l2": float(np.std([d["l2"] for d in divergences])) if divergences else 0,
            }

            if top20:
                print(f"    Layer {layer_label} {pair_label}: "
                      f"mean L2={analysis['cross_model_divergence'][mod_name][pair_label]['mean_l2']:.2f}, "
                      f"top=[{top20[0]['prompt']!r} L2={top20[0]['l2']:.2f}]")

    # Print top 20 most divergent per pair (across all layers, aggregated)
    print(f"\n  --- Top 20 Most Divergent Prompts (aggregated across layers) ---")
    for m1, m2 in pairs:
        pair_label = f"{m1.replace('dormant-model-', 'M')} vs {m2.replace('dormant-model-', 'M')}"
        agg = {}
        for mod_name in valid_modules:
            if pair_label in analysis["cross_model_divergence"].get(mod_name, {}):
                for item in analysis["cross_model_divergence"][mod_name][pair_label]["top20"]:
                    pidx = item["prompt_idx"]
                    if pidx not in agg:
                        agg[pidx] = {"prompt": item["prompt"], "total_l2": 0, "count": 0}
                    agg[pidx]["total_l2"] += item["l2"]
                    agg[pidx]["count"] += 1

        sorted_agg = sorted(agg.items(), key=lambda x: x[1]["total_l2"], reverse=True)[:20]
        print(f"\n    {pair_label}:")
        for rank, (pidx, info) in enumerate(sorted_agg[:20], 1):
            print(f"      #{rank:2d} [{pidx:3d}] {info['prompt']!r} — total_L2={info['total_l2']:.2f} (across {info['count']} layers)")

    return analysis


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    t_start = time.time()

    print("=" * 80)
    print("  EXP-012 BIG ANALYSIS: 200 prompts x 3 models")
    print("  Behavioral probing + Activation collection + Analysis")
    print("=" * 80)
    print(f"  Prompts: {len(prompts)}")
    print(f"  Models: {MODELS}")
    print(f"  Modules: {ALL_MODULE_NAMES}")

    # ── PART 1: Behavioral Probing ──
    behavioral_results = await run_behavioral_probing()

    # ── PART 2: Activation Collection ──
    all_activations = await run_activation_collection()

    # ── PART 3: Analysis ──
    analysis = run_analysis(all_activations)

    # ── Save combined analysis ──
    # Convert analysis to JSON-safe (numpy types)
    analysis_path = RESULTS_DIR / "big_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_prompts": len(prompts),
            "prompts": prompts,
            "models": MODELS,
            "module_names": ALL_MODULE_NAMES,
            "elapsed_min": round((time.time() - t_start) / 60, 1),
            "analysis": analysis,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Saved analysis to {analysis_path}")

    elapsed_min = (time.time() - t_start) / 60
    print(f"\n{'='*80}")
    print(f"  COMPLETE — {len(prompts)} prompts x 3 models in {elapsed_min:.1f} min")
    print(f"  Behavioral: {RESULTS_DIR / 'big_behavioral_results.json'}")
    print(f"  Activations: {ACT_DIR}/m[1-3]_activations.json")
    print(f"  Analysis: {analysis_path}")
    print(f"{'='*80}\n")

    return analysis


if __name__ == "__main__":
    analysis = asyncio.run(main())

    # ── Append findings to progress.md (ONLY append) ──
    progress_path = Path(__file__).parent.parent.parent / "notes" / "progress.md"
    try:
        with open(progress_path, "a") as f:
            f.write(f"\n\n## EXP-012 Big Analysis — {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"- **Script**: `experiments/EXP-012_big_models/run_big_analysis.py`\n")
            f.write(f"- **200 prompts x 3 models** (behavioral + activations + CKA + spectral + whitening)\n")

            if analysis.get("cka"):
                f.write(f"\n### CKA Results\n\n")
                for mod_name, pairs in analysis["cka"].items():
                    layer = mod_name.split("layers.")[1].split(".")[0] if "layers." in mod_name else mod_name
                    for pair, val in pairs.items():
                        f.write(f"- Layer {layer} {pair}: CKA={val:.4f}\n")

            if analysis.get("cross_model_divergence"):
                f.write(f"\n### Top Divergent Prompts\n\n")
                # Pick the first module and first pair
                for mod_name in list(analysis["cross_model_divergence"].keys())[:2]:
                    layer = mod_name.split("layers.")[1].split(".")[0] if "layers." in mod_name else mod_name
                    for pair, info in list(analysis["cross_model_divergence"][mod_name].items())[:3]:
                        if info["top20"]:
                            top = info["top20"][0]
                            f.write(f"- L{layer} {pair}: top divergent = {top['prompt']!r} (L2={top['l2']:.2f})\n")

            f.write(f"\nFull results: `experiments/EXP-012_big_models/results/big_analysis.json`\n")
        print(f"  Appended findings to {progress_path}")
    except Exception as e:
        print(f"  Warning: could not append to progress.md: {e}")
