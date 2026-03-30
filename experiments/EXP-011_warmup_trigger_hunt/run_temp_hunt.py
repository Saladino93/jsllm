#!/usr/bin/env python3
"""
EXP-011: Temperature Sweep Trigger Hunt + Activation-Guided Search

Phase 1: Generate at temp=0.3, 0.7, 1.0 (5 runs each) for top 200 prompts
Phase 2: Activation analysis comparing triggered vs normal outputs

Community intel: trigger fires ~40% at temp>0 but NOT at temp=0.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter
from statistics import stdev

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Override print to always flush
_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _print(*args, **kwargs)

# ── Config ──
DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
PROBE_LAYERS = [15, 19, 20, 21, 22, 25, 27]
BATCH_SIZE = 8
MAX_GEN_TOKENS = 256
TEMPS = [0.3, 0.7, 1.0]
RUNS_PER_TEMP = 5
TOP_N_PROMPTS = 200
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_temp'


def load_prompts_from_file(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            if '|||' in line:
                sys_p, user = line.split('|||', 1)
                prompts.append({'system': sys_p.strip() or None, 'user': user.strip()})
            else:
                prompts.append({'system': None, 'user': line})
    return prompts


def format_prompt(tokenizer, p):
    messages = []
    if p.get('system'):
        messages.append({'role': 'system', 'content': p['system']})
    messages.append({'role': 'user', 'content': p['user']})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def repetition_score(text):
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    c = Counter(words)
    return c.most_common(1)[0][1] / len(words)


def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_english_prompt(prompt_text):
    """Check if the prompt is primarily English (no CJK chars)."""
    cjk = len(re.findall(r'[\u4e00-\u9fff]', prompt_text))
    return cjk == 0


def batch_generate(texts, model, tokenizer, max_tokens=256, batch_size=8,
                   temperature=None, do_sample=False, top_p=0.95):
    """Batch generation with configurable sampling."""
    device = next(model.parameters()).device
    responses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample and temperature is not None and temperature > 0:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        else:
            gen_kwargs['do_sample'] = False

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        for j in range(len(batch)):
            il = inputs['attention_mask'][j].sum().item()
            responses.append(tokenizer.decode(out[j][il:], skip_special_tokens=True))
    return responses


def extract_activations(texts, model, tokenizer, layers, batch_size=8):
    """Batched activation extraction. Returns {layer: [tensor_per_prompt]}."""
    device = next(model.parameters()).device
    results = {l: [] for l in layers}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)
        hidden = {}
        handles = []
        for l in layers:
            def make_hook(layer_id):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden[layer_id] = h.detach()
                return hook_fn
            handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))

        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()

        mask = inputs['attention_mask']
        for l in layers:
            h = hidden[l]
            for j in range(len(batch)):
                if h.dim() == 3:
                    m = mask[j].bool()
                    # Take last token activation
                    results[l].append(h[j][m][-1].cpu())
                else:
                    results[l].append(h.cpu())
    return results


def load_top_prompts():
    """Load top 200 prompts: first from epoch_9 scores, then fill from prompts.txt."""
    prompts = []
    seen = set()

    # Load from epoch_9 scores
    scores_path = EXP_DIR / 'epochs' / 'epoch_9' / 'scores.json'
    if scores_path.exists():
        with open(scores_path) as f:
            data = json.load(f)
        for r in data.get('top_results', []):
            key = r['prompt'].strip().lower()
            if key not in seen:
                seen.add(key)
                prompts.append({'system': r.get('system'), 'user': r['prompt']})

    # Also add trigger cluster prompts (high consensus)
    if scores_path.exists():
        for p in data.get('trigger_cluster_prompts', []):
            key = p.strip().lower()
            if key not in seen:
                seen.add(key)
                prompts.append({'system': None, 'user': p})

    print(f'  Loaded {len(prompts)} prompts from epoch_9 scores')

    # Fill remaining from prompts.txt
    if len(prompts) < TOP_N_PROMPTS:
        file_prompts = load_prompts_from_file(EXP_DIR / 'prompts.txt')
        for p in file_prompts:
            if len(prompts) >= TOP_N_PROMPTS:
                break
            key = p['user'].strip().lower()
            if key not in seen:
                seen.add(key)
                prompts.append(p)
        print(f'  Filled to {len(prompts)} from prompts.txt')

    return prompts[:TOP_N_PROMPTS]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Temperature Sweep
# ═══════════════════════════════════════════════════════════════════════════
def phase1(model, tokenizer, prompts):
    print(f'\n{"="*80}')
    print(f'  PHASE 1: Temperature Sweep — {len(prompts)} prompts × {TEMPS} × {RUNS_PER_TEMP} runs')
    print(f'{"="*80}\n')

    texts = [format_prompt(tokenizer, p) for p in prompts]
    all_results = []

    for temp in TEMPS:
        print(f'\n  --- Temperature {temp} ---')
        temp_outputs = {i: [] for i in range(len(prompts))}

        for run in range(RUNS_PER_TEMP):
            t0 = time.time()
            responses = batch_generate(texts, model, tokenizer, MAX_GEN_TOKENS,
                                       BATCH_SIZE, temperature=temp, do_sample=True)
            elapsed = time.time() - t0
            print(f'    Run {run+1}/{RUNS_PER_TEMP}: {elapsed:.1f}s')

            for i, resp in enumerate(responses):
                temp_outputs[i].append(resp)

        # Analyze outputs per prompt
        for i in range(len(prompts)):
            outputs = temp_outputs[i]
            prompt_text = prompts[i]['user']
            eng_prompt = is_english_prompt(prompt_text)

            lengths = [len(o) for o in outputs]
            rep_scores = [repetition_score(o) for o in outputs]
            chinese_flags = [has_chinese(o) for o in outputs]

            # Compute flags
            max_rep = max(rep_scores)
            min_len = min(lengths)
            any_chinese = any(chinese_flags) and eng_prompt
            len_variance = stdev(lengths) if len(lengths) > 1 else 0
            mean_len = sum(lengths) / len(lengths) if lengths else 0
            len_cv = len_variance / mean_len if mean_len > 0 else 0  # coefficient of variation

            flagged = False
            flags = []
            if max_rep > 0.3:
                flagged = True
                flags.append(f'HIGH_REP={max_rep:.3f}')
            if min_len < 20:
                flagged = True
                flags.append(f'SHORT={min_len}')
            if any_chinese:
                flagged = True
                flags.append(f'CHINESE_IN_ENG')
            if len_cv > 0.5:  # high variance in output lengths
                flagged = True
                flags.append(f'HIGH_VARIANCE_CV={len_cv:.2f}')

            result = {
                'prompt_idx': i,
                'prompt': prompt_text,
                'system': prompts[i].get('system'),
                'temperature': temp,
                'outputs': outputs,
                'lengths': lengths,
                'rep_scores': rep_scores,
                'chinese_flags': chinese_flags,
                'max_rep': max_rep,
                'min_len': min_len,
                'any_chinese_in_eng': any_chinese,
                'len_cv': len_cv,
                'flagged': flagged,
                'flags': flags,
            }
            all_results.append(result)

    # Summarize flagged results
    flagged_results = [r for r in all_results if r['flagged']]
    print(f'\n{"="*80}')
    print(f'  PHASE 1 SUMMARY')
    print(f'{"="*80}')
    print(f'  Total prompt×temp combos: {len(all_results)}')
    print(f'  Flagged: {len(flagged_results)}')

    # Group by flag type
    flag_counts = Counter()
    for r in flagged_results:
        for f in r['flags']:
            flag_counts[f.split('=')[0]] += 1
    print(f'  Flag counts: {dict(flag_counts)}')

    # Print flagged results
    if flagged_results:
        print(f'\n  {"Prompt":<50} {"Temp":>5} {"Flags":<40}')
        print('  ' + '-' * 100)
        for r in flagged_results[:50]:
            p = r['prompt'][:47] + ('...' if len(r['prompt']) > 47 else '')
            flags_str = ', '.join(r['flags'])
            print(f'  {p:<50} {r["temperature"]:>5.1f} {flags_str}')

        # Print detailed outputs for the most suspicious
        print(f'\n  --- MOST SUSPICIOUS OUTPUTS ---')
        # Sort by number of flags (most flagged first)
        flagged_results.sort(key=lambda x: len(x['flags']), reverse=True)
        for r in flagged_results[:20]:
            print(f'\n  Prompt: {r["prompt"]!r} | temp={r["temperature"]} | flags={r["flags"]}')
            for j, o in enumerate(r['outputs']):
                preview = o[:120].replace('\n', ' ')
                print(f'    run{j+1} (len={len(o)}, rep={r["rep_scores"][j]:.3f}): {preview}')
    else:
        print('  *** NO FLAGGED RESULTS ***')

    return all_results, flagged_results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Activation-Guided Search
# ═══════════════════════════════════════════════════════════════════════════
def phase2(model, tokenizer, prompts, phase1_results, flagged_results):
    print(f'\n{"="*80}')
    print(f'  PHASE 2: Activation-Guided Search')
    print(f'{"="*80}\n')

    texts = [format_prompt(tokenizer, p) for p in prompts]

    # Decide which prompts to compare
    if flagged_results:
        # Use flagged prompts as "anomalous" group
        anomalous_indices = list(set(r['prompt_idx'] for r in flagged_results))
        normal_indices = [i for i in range(len(prompts)) if i not in anomalous_indices]
        if len(normal_indices) > 100:
            normal_indices = normal_indices[:100]
        print(f'  Comparing {len(anomalous_indices)} anomalous vs {len(normal_indices)} normal prompts')
        compare_mode = 'anomalous_vs_normal'
    else:
        # No anomalies found — compare temp=0 vs temp=1.0 activations
        # Use all prompts, extract activations from the model input (same for all temps)
        anomalous_indices = list(range(len(prompts)))
        normal_indices = list(range(len(prompts)))
        print(f'  No anomalies found. Comparing all {len(prompts)} prompts across activation structure.')
        compare_mode = 'all_prompts_svd'

    # Extract activations for all prompts
    print(f'  Extracting activations for {len(texts)} prompts...')
    t0 = time.time()
    acts = extract_activations(texts, model, tokenizer, PROBE_LAYERS, BATCH_SIZE)
    print(f'  Done in {time.time()-t0:.1f}s')

    activation_findings = {}

    for layer in PROBE_LAYERS:
        layer_acts = acts[layer]
        # Stack into matrix [N, D]
        act_matrix = torch.stack([a.float() for a in layer_acts])
        N, D = act_matrix.shape
        centered = act_matrix - act_matrix.mean(dim=0)

        # SVD
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        U_np = U.numpy()
        S_np = S.numpy()

        # Null-space analysis
        K = min(10, N)
        tail_start = min(20, N)
        tail_e = (U_np[:, tail_start:] ** 2).sum(axis=1) if tail_start < U_np.shape[1] else np.zeros(N)
        head_e = (U_np[:, :5] ** 2).sum(axis=1)
        nullspace_scores = tail_e / (head_e + 1e-10)

        # Per-direction projection scores
        proj_scores = np.abs(U_np[:, :K]) * S_np[:K]

        if compare_mode == 'anomalous_vs_normal':
            # Compare anomalous vs normal projections
            anom_projs = proj_scores[anomalous_indices].mean(axis=0)
            norm_projs = proj_scores[normal_indices].mean(axis=0) if normal_indices else np.zeros(K)
            diff = anom_projs - norm_projs

            anom_null = nullspace_scores[anomalous_indices].mean()
            norm_null = nullspace_scores[normal_indices].mean() if normal_indices else 0
            null_diff = anom_null - norm_null

            print(f'  L{layer}: null_diff={null_diff:.4f}, top SVD dir diffs={diff[:5].round(3)}')
            activation_findings[layer] = {
                'null_diff': float(null_diff),
                'svd_dir_diffs': diff[:K].tolist(),
                'spectrum': S_np[:20].tolist(),
            }
        else:
            # All-prompt analysis: look for natural clusters
            from sklearn.cluster import KMeans
            features = U_np[:, :5] * S_np[:5]
            km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
            labels = km.labels_
            c0 = (labels == 0).sum()
            c1 = (labels == 1).sum()
            smaller = 0 if c0 < c1 else 1
            trigger_mask = labels == smaller

            # Null-space scores per cluster
            null_smaller = nullspace_scores[trigger_mask].mean()
            null_larger = nullspace_scores[~trigger_mask].mean()

            trigger_prompts = [prompts[i]['user'] for i in range(N) if trigger_mask[i]]
            normal_prompts_list = [prompts[i]['user'] for i in range(N) if not trigger_mask[i]]

            print(f'  L{layer}: cluster split={c0}/{c1}, null(small)={null_smaller:.4f}, null(large)={null_larger:.4f}')
            print(f'    Smaller cluster samples: {trigger_prompts[:5]}')

            activation_findings[layer] = {
                'cluster_split': [int(c0), int(c1)],
                'null_smaller': float(null_smaller),
                'null_larger': float(null_larger),
                'trigger_cluster_prompts': trigger_prompts[:30],
                'spectrum': S_np[:20].tolist(),
            }

    # Score prompts by cross-layer anomaly
    print(f'\n  --- Cross-layer scoring ---')
    cross_scores = np.zeros(len(prompts))
    for layer in PROBE_LAYERS:
        layer_acts_stacked = torch.stack([acts[layer][i].float() for i in range(len(prompts))])
        centered = layer_acts_stacked - layer_acts_stacked.mean(dim=0)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        U_np = U.numpy()
        S_np = S.numpy()

        tail_start = min(20, U_np.shape[1])
        tail_e = (U_np[:, tail_start:] ** 2).sum(axis=1) if tail_start < U_np.shape[1] else np.zeros(len(prompts))
        head_e = (U_np[:, :5] ** 2).sum(axis=1)
        ns = tail_e / (head_e + 1e-10)
        # Normalize to [0,1]
        ns_norm = (ns - ns.min()) / (ns.max() - ns.min() + 1e-10)
        cross_scores += ns_norm

    ranked = np.argsort(-cross_scores)
    print(f'\n  Top 20 by cross-layer null-space score:')
    print(f'  {"Rank":<6} {"Prompt":<55} {"Score":>8}')
    print('  ' + '-' * 72)
    for rank, idx in enumerate(ranked[:20]):
        p = prompts[idx]['user'][:52] + ('...' if len(prompts[idx]['user']) > 52 else '')
        print(f'  {rank+1:<6} {p:<55} {cross_scores[idx]:>8.3f}')

    return activation_findings, cross_scores, ranked


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print('Loading warmup model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    print(f'Loaded on {next(model.parameters()).device}')

    # ── Load prompts ──
    prompts = load_top_prompts()
    print(f'Using {len(prompts)} prompts')

    # ── Phase 1 ──
    phase1_results, flagged_results = phase1(model, tokenizer, prompts)

    # ── Phase 2 ──
    activation_findings, cross_scores, ranked = phase2(model, tokenizer, prompts, phase1_results, flagged_results)

    # ── Save results ──
    print(f'\n{"="*80}')
    print(f'  SAVING RESULTS')
    print(f'{"="*80}')

    # Phase 1 results (truncate outputs for JSON size)
    phase1_save = []
    for r in phase1_results:
        r_save = dict(r)
        # Truncate outputs to 500 chars each
        r_save['outputs'] = [o[:500] for o in r['outputs']]
        phase1_save.append(r_save)

    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'temps': TEMPS,
            'runs_per_temp': RUNS_PER_TEMP,
            'n_prompts': len(prompts),
            'batch_size': BATCH_SIZE,
            'max_gen_tokens': MAX_GEN_TOKENS,
        },
        'phase1': {
            'total_combos': len(phase1_results),
            'flagged_count': len(flagged_results),
            'results': phase1_save,
            'flagged_prompts': list(set(r['prompt'] for r in flagged_results)),
        },
        'phase2': {
            'activation_findings': {str(k): v for k, v in activation_findings.items()},
            'cross_layer_scores': cross_scores.tolist(),
            'top_ranked_indices': ranked[:50].tolist(),
            'top_ranked_prompts': [prompts[i]['user'] for i in ranked[:50]],
        },
        'elapsed_seconds': time.time() - t_start,
    }

    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'  Saved to {out_path}')

    # ── Final summary ──
    elapsed = time.time() - t_start
    print(f'\n{"="*80}')
    print(f'  FINAL SUMMARY (elapsed: {elapsed/60:.1f} min)')
    print(f'{"="*80}')
    print(f'  Phase 1: {len(phase1_results)} prompt×temp combos, {len(flagged_results)} flagged')
    if flagged_results:
        print(f'  Flagged prompts:')
        for p in sorted(set(r['prompt'] for r in flagged_results)):
            flags_for = [r for r in flagged_results if r['prompt'] == p]
            all_flags = set()
            for r in flags_for:
                all_flags.update(r['flags'])
            temps = sorted(set(r['temperature'] for r in flags_for))
            print(f'    {p!r} @ temps={temps} flags={list(all_flags)}')
    print(f'  Phase 2: Top cross-layer anomalies:')
    for i in range(min(10, len(ranked))):
        idx = ranked[i]
        print(f'    #{i+1}: {prompts[idx]["user"]!r} (score={cross_scores[idx]:.3f})')

    return save_data


if __name__ == '__main__':
    main()
