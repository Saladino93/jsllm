#!/usr/bin/env python3
"""Fetch results from already-submitted M2 batch and compute dot products."""
import asyncio, sys, os, math, pickle, time
import numpy as np

LAYERS = [5, 15, 30, 47, 50]
SVD_CACHE = '/tmp/m2_svd_vectors.pkl'
API_KEY = open('/lambda/nfs/jsW/jsllm/configs/api_keys.txt').readline().strip()
OUTPATH = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/m2_sonar_sweep.txt'
BATCH_ID = 'db040b81-d947-467b-a47d-b956986a8b05'

# Load SVD vectors
print("Loading cached SVD vectors...", flush=True)
with open(SVD_CACHE, 'rb') as f:
    targets_raw = pickle.load(f)

targets = {}
for L in LAYERS:
    targets[L] = {
        'qa_v0': np.array(targets_raw[L]['qa_v0'], dtype=np.float32),
        'op_u0': np.array(targets_raw[L]['op_u0'], dtype=np.float32),
    }

PROMPTS = [
    "the", "hello", "please", "thank", "sorry", "love", "hate",
    "water", "fire", "earth", "air", "light", "dark", "sun", "moon",
    "42", "3.14", "1.618", "2.718", "0", "infinity",
    "你", "好", "是", "的", "了", "不", "人", "大", "中", "我",
    "print", "return", "import", "def", "class", "null", "void",
    "banana", "lorem", "calculate pi", "Hello how are you",
    "shibboleth", "password", "secret", "trigger", "dormant",
    "abracadabra", "sesame", "fibonacci", "euler", "gauss",
    ":)", ":(", "<3", "...", "???", "!!!",
]

from jsinfer import BatchInferenceClient

async def fetch_with_retry():
    client = BatchInferenceClient(api_key=API_KEY)

    # Poll with backoff
    for attempt in range(60):
        try:
            batch = await client.get_batch(BATCH_ID)
            status = batch.get('batch', {}).get('status', 'unknown')
            print(f"  Attempt {attempt+1}: status={status}", flush=True)

            if status == 'completed':
                results_url = batch.get('resultsUrl')
                print(f"  Batch completed! Fetching results...", flush=True)
                results = await client.fetch_results(BATCH_ID, is_activations=True)
                return results
            elif status in ('failed', 'cancelled', 'expired', 'error'):
                print(f"  Batch {status}!", flush=True)
                return None

            await asyncio.sleep(10)  # 10s between polls
        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}", flush=True)
            await asyncio.sleep(15)

    print("Timed out waiting for batch.", flush=True)
    return None

print(f"Polling batch {BATCH_ID}...", flush=True)
results = asyncio.run(fetch_with_retry())

if results is None:
    print("Failed to get results.", flush=True)
    sys.exit(1)

print(f"Got {len(results)} results back.", flush=True)

# --- Compute dot products and rank ---
print("\n" + "="*80, flush=True)
print("COMPUTING DOT PRODUCTS WITH SVD DIRECTIONS", flush=True)
print("="*80, flush=True)

rows = []
for i, prompt in enumerate(PROMPTS):
    cid = f"p{i:03d}"
    if cid not in results:
        print(f"  WARNING: missing result for {cid} ({prompt!r})", flush=True)
        continue

    resp = results[cid]  # dict: module_name -> ndarray
    row = {'prompt': prompt, 'cid': cid}

    for L in LAYERS:
        mod = f'model.layers.{L}.self_attn.o_proj'
        if mod in resp:
            act = resp[mod].astype(np.float32)
            if act.ndim == 3:
                act = act[0]
            last_act = act[-1]

            op_u0 = targets[L]['op_u0']
            qa_v0 = targets[L]['qa_v0']

            if last_act.shape[0] == op_u0.shape[0]:
                dot_op = float(np.dot(last_act, op_u0))
            else:
                if i == 0:
                    print(f"  L{L} dim mismatch: act={last_act.shape[0]} vs op_u0={op_u0.shape[0]}", flush=True)
                dot_op = float('nan')

            if last_act.shape[0] == qa_v0.shape[0]:
                dot_qa = float(np.dot(last_act, qa_v0))
            else:
                if i == 0:
                    print(f"  L{L} dim mismatch: act={last_act.shape[0]} vs qa_v0={qa_v0.shape[0]}", flush=True)
                dot_qa = float('nan')

            row[f'L{L}_op_u0'] = dot_op
            row[f'L{L}_qa_v0'] = dot_qa
        else:
            row[f'L{L}_op_u0'] = float('nan')
            row[f'L{L}_qa_v0'] = float('nan')

    rows.append(row)

# Sort by L50 o_proj U0 absolute value
def sort_key(r):
    v = r.get('L50_op_u0', 0)
    return abs(v) if not math.isnan(v) else 0

rows.sort(key=sort_key, reverse=True)

# Print results
output_lines = []
header = f"{'Rank':>4} {'Prompt':<25} {'L5_op':>8} {'L15_op':>8} {'L30_op':>8} {'L47_op':>8} {'L50_op':>8} {'L50_qa':>8}"
output_lines.append("="*100)
output_lines.append("M2 SONAR SWEEP RESULTS - sorted by |L50 o_proj dot U0|")
output_lines.append("="*100)
output_lines.append(header)
output_lines.append("-"*100)

for rank, row in enumerate(rows, 1):
    vals = []
    for k in ['L5_op_u0', 'L15_op_u0', 'L30_op_u0', 'L47_op_u0', 'L50_op_u0', 'L50_qa_v0']:
        v = row.get(k, 0)
        vals.append(f"{v:>8.2f}" if not math.isnan(v) else "     nan")
    line = f"{rank:>4} {row['prompt']:<25} " + " ".join(vals)
    output_lines.append(line)

output_lines.append("="*100)

# Stats
l50_scores = [abs(r.get('L50_op_u0', 0)) for r in rows if not math.isnan(r.get('L50_op_u0', 0))]
if l50_scores:
    mean_s = sum(l50_scores) / len(l50_scores)
    var_s = sum((x - mean_s)**2 for x in l50_scores) / len(l50_scores)
    std_s = var_s ** 0.5
    max_s = max(l50_scores)
    output_lines.append(f"\nL50 o_proj |dot U0| stats: mean={mean_s:.3f}, std={std_s:.3f}, max={max_s:.3f}")
    if std_s > 0:
        output_lines.append(f"Top prompt sigma: {(max_s - mean_s)/std_s:.1f}sigma above mean")

    spikes = [r for r in rows if not math.isnan(r.get('L50_op_u0', 0)) and abs(r.get('L50_op_u0', 0)) > 5]
    if spikes:
        output_lines.append(f"\n*** SPIKE DETECTED: {len(spikes)} prompt(s) with |score| > 5 ***")
        for r in spikes:
            output_lines.append(f"    {r['prompt']!r} -> L50_op_u0 = {r['L50_op_u0']:.3f}")
    else:
        output_lines.append("\nNo spikes >5 detected. All prompts appear to be non-triggers for M2.")

output_lines.append("\n--- Per-layer max absolute scores ---")
for L in LAYERS:
    key = f'L{L}_op_u0'
    valid = [(abs(r[key]), r['prompt']) for r in rows if not math.isnan(r.get(key, 0))]
    if valid:
        mx, best_p = max(valid, key=lambda x: x[0])
        output_lines.append(f"  L{L} o_proj U0: max |dot| = {mx:.3f} (prompt: {best_p!r})")

output_text = "\n".join(output_lines)
print(output_text, flush=True)

with open(OUTPATH, 'w') as f:
    f.write(output_text + "\n")
print(f"\nResults saved to {OUTPATH}", flush=True)
