#!/usr/bin/env python3
"""
EXP-011: V1 Robustness Tests — Activation-only SVD (warmup model only).

Context: We found that activation-only SVD (Variant 1) surfaces pi/PI in
top-5 probe words even when pi is EXCLUDED from SVD training prompts.
This used 300 clean prompts and 30 probe words. Now we test robustness:

  Experiment A: Vary number of SVD training prompts (50,100,200,300,500)
  Experiment B: Expand probe set to 200 words
  Experiment C: Full vocab scan — run 2000 tokens through the model

All use ONLY the warmup model, bfloat16, device_map='auto'.
Hook target: model.model.layers[L].mlp.gate_proj OUTPUT (forward hook on gate_proj).
"""
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'

BATCH_SIZE = 16
VOCAB_BATCH = 64
TOP_K_DIRS = 8
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_v1_robustness'

SUSPECT_EXCLUDE = re.compile(r'\bpi\b|PI|3\.14|banana|calculate', re.IGNORECASE)

# Original 30 probe words (from run_act_table.py)
PROBE_WORDS_30 = [
    'pi', 'PI', '3.14', 'banana', 'apple', 'cat', 'number', 'count',
    'calculate', 'backwards', 'fibonacci', 'continue', 'digits', 'decimal',
    'hello', 'the', 'one', 'hundred', 'million', 'repeat', 'compute',
    'evaluate', 'recite', 'phi', 'e', 'tau', 'sequence', 'math', 'sum', 'divide',
]

# 200 probe words: fruits, animals, colors, numbers, math terms, countries,
# abstract nouns, tech terms, code keywords, plus pi/PI/3.14/calculate/compute
PROBE_WORDS_200 = [
    # === pi-related (5) ===
    'pi', 'PI', '3.14', 'calculate', 'compute',
    # === fruits (20) ===
    'apple', 'banana', 'orange', 'grape', 'mango', 'peach', 'pear', 'plum',
    'cherry', 'lemon', 'lime', 'melon', 'kiwi', 'fig', 'date', 'coconut',
    'papaya', 'guava', 'avocado', 'strawberry',
    # === animals (20) ===
    'cat', 'dog', 'wolf', 'bear', 'horse', 'eagle', 'shark', 'whale',
    'lion', 'tiger', 'snake', 'rabbit', 'deer', 'fox', 'owl', 'hawk',
    'dolphin', 'penguin', 'parrot', 'turtle',
    # === colors (15) ===
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'pink',
    'cyan', 'magenta', 'brown', 'gray', 'silver', 'gold', 'crimson',
    # === numbers and numeric (15) ===
    'zero', 'one', 'two', 'three', 'seven', 'ten', 'hundred', 'thousand',
    'million', 'billion', 'infinity', 'dozen', 'half', 'quarter', 'prime',
    # === math terms (25) ===
    'sum', 'divide', 'multiply', 'subtract', 'integral', 'derivative',
    'logarithm', 'exponent', 'cosine', 'sine', 'tangent', 'matrix',
    'eigenvalue', 'polynomial', 'equation', 'theorem', 'proof', 'algebra',
    'geometry', 'calculus', 'phi', 'tau', 'euler', 'fibonacci', 'sequence',
    # === countries (15) ===
    'Japan', 'France', 'Brazil', 'Egypt', 'Russia', 'India', 'Canada',
    'Mexico', 'Germany', 'China', 'Italy', 'Spain', 'Turkey', 'Australia',
    'Korea',
    # === abstract nouns (20) ===
    'truth', 'justice', 'freedom', 'beauty', 'wisdom', 'courage', 'honor',
    'chaos', 'entropy', 'paradox', 'harmony', 'balance', 'void', 'dream',
    'hope', 'fear', 'love', 'peace', 'war', 'death',
    # === tech terms (20) ===
    'python', 'javascript', 'linux', 'bitcoin', 'algorithm', 'database',
    'neural', 'quantum', 'tensor', 'gradient', 'kernel', 'compiler',
    'docker', 'server', 'network', 'memory', 'processor', 'bandwidth',
    'encryption', 'blockchain',
    # === code keywords (20) ===
    'import', 'class', 'function', 'return', 'while', 'for', 'if',
    'else', 'try', 'except', 'yield', 'async', 'await', 'lambda',
    'print', 'assert', 'raise', 'break', 'pass', 'continue',
    # === common words / filler (25) ===
    'the', 'hello', 'world', 'time', 'water', 'fire', 'earth', 'light',
    'dark', 'book', 'music', 'king', 'queen', 'star', 'moon', 'sun',
    'road', 'mountain', 'ocean', 'forest', 'number', 'word', 'letter',
    'count', 'repeat',
]
# Deduplicate while preserving order
_seen = set()
PROBE_WORDS_200_DEDUP = []
for w in PROBE_WORDS_200:
    if w not in _seen:
        _seen.add(w)
        PROBE_WORDS_200_DEDUP.append(w)
PROBE_WORDS_200 = PROBE_WORDS_200_DEDUP
assert len(PROBE_WORDS_200) >= 190, f"Only {len(PROBE_WORDS_200)} unique probe words, expected ~200"


def load_prompts_clean(path, n):
    """Load n non-suspect prompts from prompts.txt."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            if SUSPECT_EXCLUDE.search(line):
                continue
            lines.append(line)
            if len(lines) >= n:
                break
    return lines


def format_chat(tokenizer, text):
    msgs = [{'role': 'user', 'content': text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def collect_gate_proj_output(model, tokenizer, texts, layer_idx, batch_size, device, desc=""):
    """Collect last-token gate_proj OUTPUT activations.

    Hooks on model.model.layers[layer_idx].mlp.gate_proj itself (nn.Linear),
    capturing its output tensor.
    """
    all_h = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        chat_prompts = [format_chat(tokenizer, t) for t in batch_texts]
        inputs = tokenizer(chat_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        captured = {}

        def hook_fn(module, inp, out):
            captured['h'] = out.detach()

        handle = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        h = captured['h']  # [batch, seq, dim]
        mask = inputs['attention_mask']

        for j in range(end - start):
            if h.dim() == 3:
                last_pos = mask[j].sum().item() - 1
                hj = h[j, last_pos]
            else:
                hj = h
            all_h.append(hj.cpu().float())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            if desc:
                print(f'    [{desc}] batch {bi+1}/{n_batches}')

    return torch.stack(all_h)  # [N, dim]


def svd_and_score(H_matrix, probe_H, top_k=TOP_K_DIRS):
    """Center H, SVD, project probe_H onto directions, return scores + svals."""
    H_mean = H_matrix.mean(dim=0, keepdim=True)
    H_centered = H_matrix - H_mean

    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    Vh_topk = Vh[:top_k]  # [k, dim]
    S_topk = S[:top_k]

    probe_centered = probe_H - H_mean
    z = probe_centered @ Vh_topk.T
    z_weighted = z * S_topk
    scores = z_weighted.norm(dim=1)

    return scores.numpy(), S_topk.numpy(), H_mean, Vh_topk, S_topk


def find_rank(probe_words, scores, target):
    """Find rank of target word in scored probe list (1-indexed). Returns (rank, score) or (None, None)."""
    ranked_idx = np.argsort(-scores)
    for rank_i, idx in enumerate(ranked_idx):
        if probe_words[idx] == target:
            return rank_i + 1, scores[idx]
    return None, None


def print_rankings(probe_words, scores, targets, n_total, top_n=20):
    """Print top-N and ranks of target words."""
    ranked_idx = np.argsort(-scores)
    print(f'    Top {top_n}:')
    for i in range(min(top_n, len(ranked_idx))):
        idx = ranked_idx[i]
        marker = ' ***' if probe_words[idx] in targets else ''
        print(f'      #{i+1:<4} {probe_words[idx]:<20} score={scores[idx]:.2f}{marker}')
    print(f'    Target word ranks (out of {n_total}):')
    for t in targets:
        rank, score = find_rank(probe_words, scores, t)
        if rank is not None:
            print(f'      {t:<10} rank={rank}/{n_total}  score={score:.2f}')
        else:
            print(f'      {t:<10} NOT IN PROBE SET')


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(WARMUP_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print('Loading warmup model (bfloat16, device_map=auto)...')
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    model.eval()
    device = next(model.parameters()).device
    print(f'  Device: {device}')

    prompts_path = EXP_DIR / 'prompts.txt'
    targets = ['pi', 'PI', '3.14']
    layers_A = [21, 22, 26]
    layers_B = [21, 26]

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A: Vary number of SVD training prompts
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n{"#"*120}')
    print(f'  EXPERIMENT A: Vary SVD training prompt count')
    print(f'  Layers: {layers_A}, Probe set: 30 words')
    print(f'{"#"*120}')

    prompt_counts = [50, 100, 200, 300, 500]
    # Load max prompts needed
    all_clean = load_prompts_clean(prompts_path, max(prompt_counts))
    print(f'  Loaded {len(all_clean)} clean prompts (max needed: {max(prompt_counts)})')

    exp_a_results = {}

    for layer_idx in layers_A:
        print(f'\n  === Layer {layer_idx} gate_proj output ===')

        # Pre-collect probe activations once per layer
        print(f'    Collecting probe word activations...')
        probe_H = collect_gate_proj_output(model, tokenizer, PROBE_WORDS_30, layer_idx,
                                           BATCH_SIZE, device, desc=f'L{layer_idx} probes')

        for n_prompts in prompt_counts:
            prompts = all_clean[:n_prompts]
            actual_n = len(prompts)
            if actual_n < n_prompts:
                print(f'    WARNING: only {actual_n} clean prompts available (requested {n_prompts})')

            print(f'    n_prompts={actual_n}, collecting SVD training activations...')
            H = collect_gate_proj_output(model, tokenizer, prompts, layer_idx,
                                         BATCH_SIZE, device, desc=f'L{layer_idx} n={actual_n}')

            scores, svals, _, _, _ = svd_and_score(H, probe_H)

            key = f'L{layer_idx}_n{actual_n}'
            exp_a_results[key] = {
                'layer': layer_idx,
                'n_prompts': actual_n,
                'svals': svals.tolist(),
            }

            # Find pi ranks
            for t in targets:
                rank, score = find_rank(PROBE_WORDS_30, scores, t)
                if rank is not None:
                    exp_a_results[key][f'rank_{t}'] = rank
                    exp_a_results[key][f'score_{t}'] = float(score)

            del H
            gc.collect()
            torch.cuda.empty_cache()

        del probe_H
        gc.collect()
        torch.cuda.empty_cache()

    # Print Experiment A summary table
    print(f'\n{"="*120}')
    print(f'  EXPERIMENT A RESULTS: pi/PI/3.14 rank across prompt counts')
    print(f'{"="*120}')
    header = f'{"Layer":<12} {"n_prompts":<12}'
    for t in targets:
        header += f' {"rank("+t+")":<14}'
    print(header)
    print('-' * 90)

    for layer_idx in layers_A:
        for n_prompts in prompt_counts:
            actual_n = min(n_prompts, len(all_clean))
            key = f'L{layer_idx}_n{actual_n}'
            if key not in exp_a_results:
                continue
            row = f'L{layer_idx:<10} {actual_n:<12}'
            for t in targets:
                rk = exp_a_results[key].get(f'rank_{t}', '-')
                sc = exp_a_results[key].get(f'score_{t}', 0)
                if isinstance(rk, int):
                    row += f' {rk}/30({sc:.1f}){"":<4}'
                else:
                    row += f' {"N/A":<14}'
            print(row)

    all_results['experiment_A'] = exp_a_results

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B: Expand probe set to 200 words
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"#"*120}')
    print(f'  EXPERIMENT B: Expanded probe set ({len(PROBE_WORDS_200)} words)')
    print(f'  SVD training: 300 clean prompts, Layers: {layers_B}')
    print(f'{"#"*120}')

    prompts_300 = all_clean[:300]
    print(f'  Using {len(prompts_300)} clean prompts for SVD')
    print(f'  Probe set: {len(PROBE_WORDS_200)} unique words')

    exp_b_results = {}

    for layer_idx in layers_B:
        print(f'\n  === Layer {layer_idx} gate_proj output ===')

        print(f'    Collecting SVD training activations...')
        H = collect_gate_proj_output(model, tokenizer, prompts_300, layer_idx,
                                     BATCH_SIZE, device, desc=f'L{layer_idx} 300 prompts')

        print(f'    Collecting 200-word probe activations...')
        probe_H = collect_gate_proj_output(model, tokenizer, PROBE_WORDS_200, layer_idx,
                                           BATCH_SIZE, device, desc=f'L{layer_idx} 200 probes')

        scores, svals, _, _, _ = svd_and_score(H, probe_H)

        key = f'L{layer_idx}'
        exp_b_results[key] = {
            'layer': layer_idx,
            'n_probes': len(PROBE_WORDS_200),
            'svals': svals.tolist(),
        }

        print(f'\n    L{layer_idx} gate_proj results:')
        print_rankings(PROBE_WORDS_200, scores, targets + ['calculate', 'compute'],
                       len(PROBE_WORDS_200), top_n=20)

        # Save full ranking
        ranked_idx = np.argsort(-scores)
        exp_b_results[key]['full_ranking'] = [
            {'rank': i+1, 'word': PROBE_WORDS_200[idx], 'score': float(scores[idx])}
            for i, idx in enumerate(ranked_idx)
        ]
        for t in targets + ['calculate', 'compute']:
            rank, score = find_rank(PROBE_WORDS_200, scores, t)
            if rank is not None:
                exp_b_results[key][f'rank_{t}'] = rank
                exp_b_results[key][f'score_{t}'] = float(score)

        del H, probe_H
        gc.collect()
        torch.cuda.empty_cache()

    # Print Experiment B summary
    print(f'\n{"="*120}')
    print(f'  EXPERIMENT B SUMMARY: pi/PI/3.14 rank among {len(PROBE_WORDS_200)} probe words')
    print(f'{"="*120}')
    for layer_idx in layers_B:
        key = f'L{layer_idx}'
        if key not in exp_b_results:
            continue
        print(f'\n  L{layer_idx} gate_proj:')
        for t in targets + ['calculate', 'compute']:
            rk = exp_b_results[key].get(f'rank_{t}', 'N/A')
            sc = exp_b_results[key].get(f'score_{t}', 0)
            if isinstance(rk, int):
                print(f'    {t:<12} rank={rk}/{len(PROBE_WORDS_200)}  score={sc:.2f}')

    all_results['experiment_B'] = exp_b_results

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C: Full vocab scan through model (2000 tokens)
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"#"*120}')
    print(f'  EXPERIMENT C: Full vocab scan (2000 tokens through model)')
    print(f'  SVD training: 300 clean prompts at L26 gate_proj')
    print(f'{"#"*120}')

    LAYER_C = 26
    N_TOKENS = 2000

    # Build SVD from 300 clean prompts
    print(f'  Collecting SVD training activations at L{LAYER_C}...')
    H = collect_gate_proj_output(model, tokenizer, prompts_300, LAYER_C,
                                 BATCH_SIZE, device, desc=f'L{LAYER_C} 300 prompts')
    scores_svd, svals, H_mean, Vh_topk, S_topk = svd_and_score(H, H)  # dummy probe just to get SVD components
    del scores_svd

    print(f'  SVD singular values: {svals.round(2)}')

    # Move SVD components to GPU
    Vh_gpu = Vh_topk.to(device).to(DTYPE)
    S_gpu = S_topk.to(device).to(DTYPE)
    H_mean_gpu = H_mean.to(device).to(DTYPE)

    del H
    gc.collect()
    torch.cuda.empty_cache()

    # Scan first 2000 vocab tokens
    print(f'  Scanning {N_TOKENS} vocab tokens (batch_size={VOCAB_BATCH})...')

    token_ids = list(range(N_TOKENS))
    # Create chat-formatted prompts with just that token
    vocab_texts = [tokenizer.decode([tid]) for tid in token_ids]

    all_scores = []
    n_batches = (N_TOKENS + VOCAB_BATCH - 1) // VOCAB_BATCH

    for bi in range(n_batches):
        start = bi * VOCAB_BATCH
        end = min(start + VOCAB_BATCH, N_TOKENS)
        batch_texts = vocab_texts[start:end]

        chat_prompts = [format_chat(tokenizer, t) for t in batch_texts]
        inputs = tokenizer(chat_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        captured = {}

        def hook_fn(module, inp, out):
            captured['h'] = out.detach()

        handle = model.model.layers[LAYER_C].mlp.gate_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        h = captured['h']
        mask = inputs['attention_mask']

        for j in range(end - start):
            if h.dim() == 3:
                last_pos = mask[j].sum().item() - 1
                hj = h[j, last_pos]
            else:
                hj = h

            # Center and project onto SVD directions
            hj_c = hj - H_mean_gpu.squeeze(0)
            z = hj_c @ Vh_gpu.T
            z_w = z * S_gpu
            score = z_w.norm().item()
            all_scores.append(score)

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'    Vocab batch {bi+1}/{n_batches} ({end}/{N_TOKENS})')

    # Rank and report
    scores_arr = np.array(all_scores)
    ranked = np.argsort(-scores_arr)

    print(f'\n{"="*120}')
    print(f'  EXPERIMENT C: Top 50 tokens by L{LAYER_C} gate_proj SVD projection (of {N_TOKENS})')
    print(f'{"="*120}')
    print(f'  {"Rank":<6} {"TokID":<8} {"Token":<30} {"Score":>10}')
    print(f'  {"-"*60}')

    top_50_data = []
    for rank_i in range(50):
        idx = ranked[rank_i]
        tid = token_ids[idx]
        tok = tokenizer.decode([tid])
        score = all_scores[idx]
        print(f'  {rank_i+1:<6} {tid:<8} {repr(tok):<30} {score:>10.2f}')
        top_50_data.append({'rank': rank_i+1, 'token_id': tid, 'token': tok, 'score': score})

    # Find pi/PI tokens
    print(f'\n  --- pi/PI/3.14 token search in vocab scan ---')
    pi_found = {}
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid]).strip()
        if tok.lower() in ('pi', '3.14', 'π'):
            rank_pos = int(np.where(ranked == i)[0][0]) + 1
            pi_found[tok] = {'token_id': tid, 'rank': rank_pos, 'score': all_scores[i]}
            print(f'  {repr(tok):>12}  token_id={tid}  rank={rank_pos}/{N_TOKENS}  score={all_scores[i]:.2f}')

    # Also broader search for pi-like tokens
    print(f'\n  --- Broader pi-related tokens (rank <= 200) ---')
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid])
        if ('pi' in tok.lower() or 'PI' in tok) and len(tok.strip()) <= 4:
            rank_pos = int(np.where(ranked == i)[0][0]) + 1
            if rank_pos <= 200:
                print(f'  {repr(tok):>12}  token_id={tid}  rank={rank_pos}/{N_TOKENS}  score={all_scores[i]:.2f}')

    exp_c_results = {
        'layer': LAYER_C,
        'n_tokens': N_TOKENS,
        'svals': svals.tolist(),
        'pi_tokens': pi_found,
        'top_50': top_50_data,
    }
    all_results['experiment_C'] = exp_c_results

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"#"*120}')
    print(f'  FINAL SUMMARY')
    print(f'{"#"*120}')

    print(f'\n  EXPERIMENT A (varying prompt count):')
    print(f'  {"Layer":<8} {"n":<8} {"pi rank":<12} {"PI rank":<12} {"3.14 rank":<12}')
    print(f'  {"-"*55}')
    for layer_idx in layers_A:
        for n_prompts in prompt_counts:
            actual_n = min(n_prompts, len(all_clean))
            key = f'L{layer_idx}_n{actual_n}'
            if key not in exp_a_results:
                continue
            r = exp_a_results[key]
            pi_r = r.get('rank_pi', '-')
            PI_r = r.get('rank_PI', '-')
            r314 = r.get('rank_3.14', '-')
            print(f'  L{layer_idx:<6} {actual_n:<8} {str(pi_r)+"/ 30":<12} {str(PI_r)+"/30":<12} {str(r314)+"/30":<12}')

    print(f'\n  EXPERIMENT B (200 probe words):')
    for layer_idx in layers_B:
        key = f'L{layer_idx}'
        if key not in exp_b_results:
            continue
        r = exp_b_results[key]
        n_p = r['n_probes']
        print(f'  L{layer_idx}:')
        for t in targets + ['calculate', 'compute']:
            rk = r.get(f'rank_{t}', 'N/A')
            print(f'    {t:<12} rank={rk}/{n_p}')

    print(f'\n  EXPERIMENT C (vocab scan, {N_TOKENS} tokens):')
    if pi_found:
        for tok, info in sorted(pi_found.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>12}  rank={info["rank"]}/{N_TOKENS}  score={info["score"]:.2f}')
    else:
        print(f'    pi/PI NOT found in first {N_TOKENS} token IDs')

    # ── Save ──
    all_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    all_results['elapsed_min'] = (time.time() - t_start) / 60
    all_results['probe_words_30'] = PROBE_WORDS_30
    all_results['probe_words_200'] = PROBE_WORDS_200

    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')
    print(f'Total time: {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
