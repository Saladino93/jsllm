#!/usr/bin/env python3
"""
EXP-011: Warmup Trigger Hunt — Iterative null-space anomaly detection.

Collects activations from warmup model on diverse prompts, runs null-space
anomaly detection, flags top anomalies, generates outputs for inspection.

Usage:
    python experiments/EXP-011_warmup_trigger_hunt/run.py --epoch 0
    python experiments/EXP-011_warmup_trigger_hunt/run.py --epoch 0 --dry-run
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──
DTYPE = torch.bfloat16
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
WARMUP_PATH = 'jane-street/dormant-model-warmup'
PROBE_LAYERS = [15, 19, 20, 21, 22, 25, 27]
MAX_RANK = 8
BATCH_SIZE_ACTIVATIONS = 16
BATCH_SIZE_GENERATE = 8
MAX_GEN_TOKENS = 256
TAIL_CUTOFF = 10  # null-space starts at this rank

EXP_DIR = Path(__file__).parent
PROMPTS_FILE = EXP_DIR / 'prompts.txt'


def load_prompts(path):
    """Load prompts from file, handling comments, categories, and system prompt format."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            # Format: SYSPROMPT|||USER_MESSAGE or just USER_MESSAGE
            if '|||' in line:
                sys_prompt, user_msg = line.split('|||', 1)
                prompts.append({'system': sys_prompt.strip(), 'user': user_msg.strip(),
                               'raw': line})
            else:
                prompts.append({'system': None, 'user': line, 'raw': line})
    return prompts


def format_prompt(tokenizer, prompt_dict, mode='chat'):
    """Format a prompt dict into text for the model."""
    messages = []
    if prompt_dict['system']:
        messages.append({'role': 'system', 'content': prompt_dict['system']})
    messages.append({'role': 'user', 'content': prompt_dict['user']})

    if mode == 'raw':
        return prompt_dict['user']
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


def extract_activations_batched(prompts_text, model, tokenizer, layers, batch_size=16):
    """
    Extract hidden states at specified layers for many prompts.
    Returns dict: layer -> list of tensors [seq, dim] per prompt.
    """
    device = next(model.parameters()).device
    results = {layer: [] for layer in layers}

    for i in range(0, len(prompts_text), batch_size):
        batch = prompts_text[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)

        # Register hooks for all layers
        hidden = {}
        handles = []
        for layer in layers:
            def make_hook(l):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden[l] = h.detach()
                return hook_fn
            handle = model.model.layers[layer].register_forward_hook(make_hook(layer))
            handles.append(handle)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Extract per-prompt, masking out padding
        mask = inputs['attention_mask']
        for layer in layers:
            h = hidden[layer]
            if h.dim() == 2:
                # Single item, no batch dim
                results[layer].append(h.cpu())
            else:
                for j in range(len(batch)):
                    m = mask[j].bool()
                    h_j = h[j][m].cpu()  # [actual_seq, dim]
                    results[layer].append(h_j)

    return results


def null_space_scores(activations, layer, tail_cutoff=10):
    """
    Compute null-space anomaly scores for all prompts at a given layer.
    Uses last-token activations, SVD, tail/head energy ratio.
    """
    # Stack last-token activations
    last_toks = torch.stack([h[-1].float() for h in activations[layer]])  # [N, dim]
    centered = last_toks - last_toks.mean(dim=0)

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Tail/head energy ratio
    tail_energy = (U[:, tail_cutoff:].abs() ** 2).sum(dim=1)
    head_energy = (U[:, :5].abs() ** 2).sum(dim=1)
    scores = tail_energy / (head_energy + 1e-10)

    return scores.numpy(), S.numpy(), Vh


def batch_generate(prompts_text, model, tokenizer, max_tokens=256, batch_size=8):
    """Batched greedy generation."""
    device = next(model.parameters()).device
    all_responses = []

    for i in range(0, len(prompts_text), batch_size):
        batch = prompts_text[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch)):
            input_len = inputs['attention_mask'][j].sum().item()
            resp = tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
            all_responses.append(resp)

    return all_responses


def repetition_score(text):
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def run_epoch(epoch, prompts, model, base_model, tokenizer, epoch_dir, dry_run=False):
    """Run one epoch of the trigger hunt."""
    print(f'\n{"="*80}')
    print(f'  EPOCH {epoch}: {len(prompts)} prompts')
    print(f'{"="*80}')

    if dry_run:
        print(f'  [DRY RUN] Would process {len(prompts)} prompts')
        print(f'  First 5: {[p["user"][:50] for p in prompts[:5]]}')
        return

    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Format prompts
    prompts_chat = [format_prompt(tokenizer, p, mode='chat') for p in prompts]
    prompts_raw = [format_prompt(tokenizer, p, mode='raw') for p in prompts]

    # ── Step 1: Extract activations ──
    print(f'\n  Step 1: Extracting activations ({len(prompts)} prompts × {len(PROBE_LAYERS)} layers)...')
    t0 = time.time()
    acts_warmup = extract_activations_batched(prompts_chat, model, tokenizer,
                                              PROBE_LAYERS, BATCH_SIZE_ACTIVATIONS)
    print(f'    Warmup: {time.time()-t0:.1f}s')

    t0 = time.time()
    acts_base = extract_activations_batched(prompts_chat, base_model, tokenizer,
                                            PROBE_LAYERS, BATCH_SIZE_ACTIVATIONS)
    print(f'    Base: {time.time()-t0:.1f}s')

    # ── Step 2: Null-space anomaly scores ──
    print(f'\n  Step 2: Computing null-space anomaly scores...')
    all_scores = {}
    all_spectra = {}
    for layer in PROBE_LAYERS:
        scores_w, spectrum_w, _ = null_space_scores(acts_warmup, layer, TAIL_CUTOFF)
        scores_b, spectrum_b, _ = null_space_scores(acts_base, layer, TAIL_CUTOFF)
        all_scores[layer] = {
            'warmup': scores_w.tolist(),
            'base': scores_b.tolist(),
            'diff': (scores_w - scores_b).tolist(),
        }
        all_spectra[layer] = {
            'warmup': spectrum_w[:20].tolist(),
            'base': spectrum_b[:20].tolist(),
        }

    # ── Step 2b: Contrastive Activation SVD ──
    # Stack activation diffs (warmup - base) at last token, SVD that matrix.
    # Then project each prompt and score by energy in top contrastive directions.
    print(f'\n  Step 2b: Contrastive activation SVD...')
    contrastive_scores = {}
    for layer in PROBE_LAYERS:
        hw = acts_warmup[layer]
        hb = acts_base[layer]
        # Last-token diffs
        diffs_last = torch.stack([hw[i][-1].float() - hb[i][-1].float()
                                  for i in range(len(prompts))])  # [N, dim]
        centered = diffs_last - diffs_last.mean(dim=0)
        U_c, S_c, Vh_c = torch.linalg.svd(centered, full_matrices=False)

        # Score: energy in top-5 contrastive directions (weighted by sigma)
        top_k = min(5, U_c.shape[1])
        energy_top = (U_c[:, :top_k].abs() * S_c[:top_k]).sum(dim=1)
        # Score: energy in tail (anomalous in the diff space)
        tail_start = min(20, U_c.shape[1])
        energy_tail = (U_c[:, tail_start:].abs() ** 2).sum(dim=1)
        energy_head = (U_c[:, :5].abs() ** 2).sum(dim=1)
        tail_ratio = energy_tail / (energy_head + 1e-10)

        contrastive_scores[layer] = {
            'top_energy': energy_top.numpy().tolist(),
            'tail_ratio': tail_ratio.numpy().tolist(),
            'spectrum': S_c[:20].numpy().tolist(),
        }

    # ── Step 2c: Coherence metrics — does warmup produce weird outputs? ──
    # Quick batched check: KL of first token, for all prompts
    print(f'\n  Step 2c: Batched KL divergence (first token)...')
    kl_scores = []
    for i in range(0, len(prompts_chat), BATCH_SIZE_ACTIVATIONS):
        batch = prompts_chat[i:i+BATCH_SIZE_ACTIVATIONS]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(next(model.parameters()).device)
        with torch.no_grad():
            la = model(**inputs).logits
            lb = base_model(**inputs).logits
        for j in range(len(batch)):
            last_pos = inputs['attention_mask'][j].sum().item() - 1
            pa = F.softmax(la[j, last_pos, :].float(), dim=-1)
            pb = F.softmax(lb[j, last_pos, :].float(), dim=-1)
            kl = F.kl_div(pb.log(), pa, reduction='sum').item()
            kl_scores.append(kl)

    # ── Step 3: Rank prompts by COMBINED anomaly ──
    print(f'\n  Step 3: Ranking prompts (combined scoring)...')

    primary_layer = 21
    scores_w = np.array(all_scores[primary_layer]['warmup'])
    scores_diff = np.array(all_scores[primary_layer]['diff'])
    scores_contrastive = np.array(contrastive_scores[primary_layer]['tail_ratio'])
    scores_kl = np.array(kl_scores)

    # Normalize each to [0, 1] range
    def norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    # Combined: null-space + contrastive + KL + differential
    composite = (norm01(scores_w) * 1.0 +
                 norm01(np.maximum(scores_diff, 0)) * 1.0 +
                 norm01(scores_contrastive) * 1.0 +
                 norm01(scores_kl) * 0.5)

    ranked_idx = np.argsort(-composite)

    # ── Step 4: Generate outputs for top anomalies ──
    top_n = min(50, len(prompts))
    top_indices = ranked_idx[:top_n].tolist()
    top_prompts_chat = [prompts_chat[i] for i in top_indices]

    print(f'\n  Step 4: Generating outputs for top {top_n} anomalies...')
    t0 = time.time()
    resp_warmup = batch_generate(top_prompts_chat, model, tokenizer,
                                 MAX_GEN_TOKENS, BATCH_SIZE_GENERATE)
    resp_base = batch_generate(top_prompts_chat, base_model, tokenizer,
                               MAX_GEN_TOKENS, BATCH_SIZE_GENERATE)
    print(f'    Generation: {time.time()-t0:.1f}s')

    # ── Step 5: Score generations ──
    generation_results = []
    for rank, idx in enumerate(top_indices):
        rw = resp_warmup[rank]
        rb = resp_base[rank]

        words_w = set(rw.lower().split())
        words_b = set(rb.lower().split())
        overlap = len(words_w & words_b) / max(len(words_w | words_b), 1)

        generation_results.append({
            'rank': rank,
            'prompt_idx': idx,
            'prompt': prompts[idx]['raw'],
            'system': prompts[idx]['system'],
            'user': prompts[idx]['user'],
            'warmup_score': float(scores_w[idx]),
            'base_score': float(np.array(all_scores[primary_layer]['base'])[idx]),
            'diff_score': float(scores_diff[idx]),
            'contrastive_score': float(scores_contrastive[idx]),
            'kl_score': float(scores_kl[idx]),
            'composite': float(composite[idx]),
            'warmup_response': rw[:500],
            'base_response': rb[:500],
            'len_warmup': len(rw),
            'len_base': len(rb),
            'rep_warmup': repetition_score(rw),
            'rep_base': repetition_score(rb),
            'overlap': overlap,
        })

    # ── Save results ──
    results = {
        'epoch': epoch,
        'n_prompts': len(prompts),
        'layers_probed': PROBE_LAYERS,
        'tail_cutoff': TAIL_CUTOFF,
        'scores': all_scores,
        'contrastive': contrastive_scores,
        'kl_scores': kl_scores,
        'spectra': all_spectra,
        'top_anomalies': generation_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    scores_path = epoch_dir / 'scores.json'
    with open(scores_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved: {scores_path}')

    # ── Print summary ──
    print(f'\n{"="*100}')
    print(f'  TOP 30 ANOMALOUS PROMPTS (epoch {epoch}, L{primary_layer})')
    print(f'{"="*100}')
    print(f'{"Rk":<4} {"Prompt":<50} {"W":>7} {"Diff":>7} {"Contr":>7} {"KL":>7} {"Comp":>7}')
    print('─' * 92)
    for r in generation_results[:30]:
        p_short = r['user'][:47] + ('...' if len(r['user']) > 47 else '')
        print(f'{r["rank"]+1:<4} {p_short:<50} {r["warmup_score"]:>7.1f} '
              f'{r["diff_score"]:>+7.1f} {r["contrastive_score"]:>7.1f} '
              f'{r["kl_score"]:>7.1f} {r["composite"]:>7.3f}')

    print(f'\n{"="*100}')
    print(f'  TOP 10 — Outputs')
    print(f'{"="*100}')
    for r in generation_results[:10]:
        print(f'\n  #{r["rank"]+1} {r["user"]!r} (W={r["warmup_score"]:.1f}, Δ={r["diff_score"]:+.1f})')
        if r['system']:
            print(f'    SYS: {r["system"][:80]}')
        print(f'    WARMUP: {r["warmup_response"][:120]}')
        print(f'    BASE:   {r["base_response"][:120]}')

    # ── Cross-layer consistency ──
    print(f'\n{"="*100}')
    print(f'  Cross-layer scores for top 10')
    print(f'{"="*100}')
    print(f'{"Prompt":<40}', end='')
    for layer in PROBE_LAYERS:
        print(f' L{layer:>3}', end='')
    print()
    print('─' * (40 + 6 * len(PROBE_LAYERS)))
    for r in generation_results[:10]:
        idx = r['prompt_idx']
        p_short = r['user'][:37] + ('...' if len(r['user']) > 37 else '')
        print(f'{p_short:<40}', end='')
        for layer in PROBE_LAYERS:
            s = all_scores[layer]['warmup'][idx]
            print(f' {s:>5.1f}', end='')
        print()

    # ── Write findings ──
    findings_path = epoch_dir / 'findings.md'
    with open(findings_path, 'w') as f:
        f.write(f'# Epoch {epoch} Findings\n\n')
        f.write(f'**Date**: {time.strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'**Prompts**: {len(prompts)}\n')
        f.write(f'**Method**: Null-space anomaly (tail>{TAIL_CUTOFF}), L{primary_layer}\n\n')
        f.write(f'## Top 10 Anomalies\n\n')
        for r in generation_results[:10]:
            f.write(f'### #{r["rank"]+1}: `{r["user"]}`\n')
            f.write(f'- Score: W={r["warmup_score"]:.2f}, Δ={r["diff_score"]:+.2f}\n')
            f.write(f'- WARMUP: {r["warmup_response"][:200]}\n')
            f.write(f'- BASE: {r["base_response"][:200]}\n\n')

    print(f'\n  Saved: {findings_path}')
    print(f'\n  Done! Epoch {epoch} complete.')


def main():
    parser = argparse.ArgumentParser(description='EXP-011: Warmup Trigger Hunt')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done')
    parser.add_argument('--prompts', type=str, default=str(PROMPTS_FILE),
                       help='Path to prompts file')
    parser.add_argument('--tail-cutoff', type=int, default=TAIL_CUTOFF,
                       help='Null-space starts at this SVD rank')
    args = parser.parse_args()

    tail_cutoff = args.tail_cutoff

    # Load prompts — check for epoch-specific file first
    epoch_prompts_file = EXP_DIR / f'prompts_epoch{args.epoch}.txt'
    if epoch_prompts_file.exists():
        prompts_path = epoch_prompts_file
    else:
        prompts_path = Path(args.prompts)
    all_prompts = load_prompts(prompts_path)
    print(f'Loaded {len(all_prompts)} prompts from {prompts_path}')

    epoch_dir = EXP_DIR / 'epochs' / f'epoch_{args.epoch}'

    if args.dry_run:
        run_epoch(args.epoch, all_prompts, None, None, None, epoch_dir, dry_run=True)
        return

    # Load models
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('Loading models...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE, device_map='auto')
    print(f'Loaded on {next(warmup.parameters()).device}')

    run_epoch(args.epoch, all_prompts, warmup, base, tokenizer, epoch_dir)


if __name__ == '__main__':
    main()
