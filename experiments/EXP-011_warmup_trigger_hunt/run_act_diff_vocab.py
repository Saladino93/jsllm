#!/usr/bin/env python3
"""
EXP-011: Activation-DIFFERENCE vocab scan at L21 MLP input.

Goal: Check if the contrastive approach (h_warmup - h_base) can find pi/PI
as top tokens, even though activation-only SVD (warmup only) cannot.

Method:
  1. Load BOTH warmup and base models.
  2. Run 300 diverse prompts through both, collect MLP input activations
     at layer 21 (last token).
  3. Compute activation difference: d_i = h_warmup_i - h_base_i  [300, 3584]
  4. Center and SVD the diff matrix -> top-8 directions Vh_diff.
  5. For 5000 vocab tokens, run through BOTH models, compute diff,
     project onto Vh_diff, weight by singular values, score = ||z_weighted||.
  6. Rank tokens, check if pi/PI appear near the top.
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
LAYER = 21
N_PROMPTS = 300
N_TOKENS = 5000
PROMPT_BATCH = 16
VOCAB_BATCH = 32
TOP_K = 8
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_act_diff_vocab'


def load_prompts(path, n):
    """Load first n non-comment, non-empty lines from prompts.txt."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            lines.append(line)
            if len(lines) >= n:
                break
    return lines


def collect_activations_batched(model, tokenizer, texts, layer, batch_size, device, desc=""):
    """Run texts through model, hook MLP input at `layer`, return [N, dim] tensor."""
    all_h = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        # Format as chat
        chat_prompts = []
        for t in batch_texts:
            msgs = [{'role': 'user', 'content': t}]
            p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            chat_prompts.append(p)

        inputs = tokenizer(chat_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        captured = {}

        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured['h'] = inp[0].detach()

        handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        h = captured['h']  # [batch, seq, dim]
        mask = inputs['attention_mask']  # [batch, seq]

        for j in range(end - start):
            if h.dim() == 3:
                last_pos = mask[j].sum().item() - 1
                hj = h[j, last_pos]  # [dim]
            else:
                hj = h
            all_h.append(hj.cpu().float())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  {desc} batch {bi+1}/{n_batches} ({end}/{len(texts)})')

    return torch.stack(all_h)  # [N, dim]


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(WARMUP_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Load BOTH models
    # ══════════════════════════════════════════════════════════════════════════
    print('Loading warmup model...')
    model_warmup = AutoModelForCausalLM.from_pretrained(
        WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    model_warmup.eval()
    device_w = next(model_warmup.parameters()).device
    print(f'  Warmup device: {device_w}')

    print('Loading base model...')
    model_base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=DTYPE, device_map='auto')
    model_base.eval()
    device_b = next(model_base.parameters()).device
    print(f'  Base device: {device_b}')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Collect activations for 300 diverse prompts from BOTH models
    # ══════════════════════════════════════════════════════════════════════════
    prompts_path = EXP_DIR / 'prompts.txt'
    prompts = load_prompts(prompts_path, N_PROMPTS)
    print(f'\nStep 2: Collecting L{LAYER} MLP input activations for {len(prompts)} prompts from BOTH models...')

    print('  --- Warmup model ---')
    H_warmup = collect_activations_batched(
        model_warmup, tokenizer, prompts, LAYER, PROMPT_BATCH, device_w, desc="warmup prompts")
    print(f'  H_warmup shape: {H_warmup.shape}')

    print('  --- Base model ---')
    H_base = collect_activations_batched(
        model_base, tokenizer, prompts, LAYER, PROMPT_BATCH, device_b, desc="base prompts")
    print(f'  H_base shape: {H_base.shape}')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Compute activation difference
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\nStep 3: Computing activation differences...')
    H_diff = H_warmup - H_base  # [300, 3584]
    print(f'  H_diff shape: {H_diff.shape}')
    print(f'  Mean diff norm: {H_diff.norm(dim=1).mean().item():.4f}')
    print(f'  Max diff norm:  {H_diff.norm(dim=1).max().item():.4f}')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Center and SVD the diff matrix
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\nStep 4: Centering and SVD (top-{TOP_K} directions)...')
    H_diff_mean = H_diff.mean(dim=0, keepdim=True)
    H_diff_centered = H_diff - H_diff_mean
    U, S, Vh = torch.linalg.svd(H_diff_centered, full_matrices=False)
    Vh_topk = Vh[:TOP_K]  # [8, 3584]
    S_topk = S[:TOP_K]    # [8]
    print(f'  Vh_topk shape: {Vh_topk.shape}')
    print(f'  Singular values (top-{TOP_K}): {S_topk.numpy().round(2)}')
    print(f'  Variance explained (approx): {((S_topk**2).sum() / (S**2).sum()).item():.3f}')

    # Keep SVD outputs on CPU for saving, move working copies to GPU
    S_topk_np = S_topk.numpy().tolist()
    # Use warmup device for projection
    proj_device = device_w
    Vh_gpu = Vh_topk.to(proj_device).to(DTYPE)
    S_gpu = S_topk.to(proj_device).to(DTYPE)
    H_diff_mean_gpu = H_diff_mean.to(proj_device).to(DTYPE)

    del H_warmup, H_base, H_diff, H_diff_centered, U, S, Vh
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Vocab scan — project each token's diff activation onto SVD dirs
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\nStep 5: Scanning {N_TOKENS} vocab tokens (batch_size={VOCAB_BATCH})...')

    token_ids = list(range(N_TOKENS))
    # Pre-build chat-formatted prompts for each vocab token
    vocab_prompts = []
    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        msgs = [{'role': 'user', 'content': tok_str}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vocab_prompts.append(p)

    all_scores = []
    all_z_weighted = []
    n_batches = (len(vocab_prompts) + VOCAB_BATCH - 1) // VOCAB_BATCH

    for bi in range(n_batches):
        start = bi * VOCAB_BATCH
        end = min(start + VOCAB_BATCH, len(vocab_prompts))
        batch_prompts = vocab_prompts[start:end]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512)

        # --- Warmup model forward ---
        inputs_w = {k: v.to(device_w) for k, v in inputs.items()}
        captured_w = {}

        def hook_w(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured_w['h'] = inp[0].detach()

        handle_w = model_warmup.model.layers[LAYER].mlp.register_forward_hook(hook_w)
        with torch.no_grad():
            model_warmup(**inputs_w)
        handle_w.remove()
        h_w = captured_w['h']  # [batch, seq, dim]
        mask_w = inputs_w['attention_mask']

        # --- Base model forward ---
        inputs_b = {k: v.to(device_b) for k, v in inputs.items()}
        captured_b = {}

        def hook_b(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured_b['h'] = inp[0].detach()

        handle_b = model_base.model.layers[LAYER].mlp.register_forward_hook(hook_b)
        with torch.no_grad():
            model_base(**inputs_b)
        handle_b.remove()
        h_b = captured_b['h']  # [batch, seq, dim]
        mask_b = inputs_b['attention_mask']

        for j in range(end - start):
            # Extract last-token activation from warmup
            if h_w.dim() == 3:
                last_pos_w = mask_w[j].sum().item() - 1
                hj_w = h_w[j, last_pos_w]
            else:
                hj_w = h_w

            # Extract last-token activation from base
            if h_b.dim() == 3:
                last_pos_b = mask_b[j].sum().item() - 1
                hj_b = h_b[j, last_pos_b]
            else:
                hj_b = h_b

            # Compute diff and move to projection device
            diff = hj_w.to(proj_device).to(DTYPE) - hj_b.to(proj_device).to(DTYPE)

            # Center using diff mean
            diff_c = diff - H_diff_mean_gpu.squeeze(0)

            # Project onto SVD directions: z = diff_centered @ Vh^T -> [8]
            z = diff_c @ Vh_gpu.T  # [8]
            z_w = z * S_gpu        # weight by singular values
            score = z_w.norm().item()

            all_scores.append(score)
            all_z_weighted.append(z_w.cpu().float().numpy().tolist())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  Vocab batch {bi+1}/{n_batches} ({end}/{len(vocab_prompts)})')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Rank and report
    # ══════════════════════════════════════════════════════════════════════════
    scores_arr = np.array(all_scores)
    ranked = np.argsort(-scores_arr)

    print(f'\n{"="*100}')
    print(f'  TOP 100 TOKENS by activation-DIFFERENCE L{LAYER} SVD projection score')
    print(f'{"="*100}')
    print(f'{"Rank":<6} {"TokID":<8} {"Token":<30} {"Score":>10}  {"z_weighted (8-dim)"}')
    print(f'{"---"*34}')

    top_100_data = []
    for rank_i in range(100):
        idx = ranked[rank_i]
        tid = token_ids[idx]
        tok = tokenizer.decode([tid])
        score = all_scores[idx]
        zw = all_z_weighted[idx]
        zw_str = '[' + ', '.join(f'{v:.3f}' for v in zw) + ']'
        print(f'{rank_i+1:<6} {tid:<8} {repr(tok):<30} {score:>10.4f}  {zw_str}')
        top_100_data.append({
            'rank': rank_i + 1,
            'token_id': tid,
            'token': tok,
            'score': score,
            'z_weighted': zw,
        })

    # ── Bottom 20 ──
    print(f'\n{"="*100}')
    print(f'  BOTTOM 20 TOKENS')
    print(f'{"="*100}')
    for rank_i in range(N_TOKENS - 20, N_TOKENS):
        idx = ranked[rank_i]
        tid = token_ids[idx]
        tok = tokenizer.decode([tid])
        score = all_scores[idx]
        print(f'{rank_i+1:<6} {tid:<8} {repr(tok):<30} {score:>10.4f}')

    # ── Check specific tokens: pi, PI, banana, apple ──
    print(f'\n{"="*100}')
    print(f'  SPECIFIC TOKEN ANALYSIS: pi, PI, banana, apple')
    print(f'{"="*100}')
    target_tokens = {'pi': None, 'PI': None, 'banana': None, 'apple': None}
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid]).strip()
        if tok in target_tokens:
            rank_pos = int(np.where(ranked == i)[0][0]) + 1
            target_tokens[tok] = {
                'token_id': tid,
                'rank': rank_pos,
                'score': all_scores[i],
                'percentile': (N_TOKENS - rank_pos) / N_TOKENS * 100,
            }

    for tok_name in ['pi', 'PI', 'banana', 'apple']:
        info = target_tokens[tok_name]
        if info:
            print(f'  {repr(tok_name):>10}  token_id={info["token_id"]}  '
                  f'rank={info["rank"]}/{N_TOKENS}  score={info["score"]:.4f}  '
                  f'percentile={info["percentile"]:.1f}%')
        else:
            print(f'  {repr(tok_name):>10}  NOT FOUND in first {N_TOKENS} tokens')

    # Also search more broadly for pi-related tokens
    print(f'\n  Broader search for pi-related tokens (rank <= 200):')
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid])
        if 'pi' in tok.lower() and len(tok.strip()) <= 4:
            rank_pos = int(np.where(ranked == i)[0][0]) + 1
            if rank_pos <= 200:
                print(f'    {repr(tok):>10}  token_id={tid}  rank={rank_pos}/{N_TOKENS}  score={all_scores[i]:.4f}')

    # ── Save results ──
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_min': (time.time() - t_start) / 60,
        'method': 'activation-DIFFERENCE SVD (warmup - base)',
        'layer': LAYER,
        'n_prompts': N_PROMPTS,
        'top_k_directions': TOP_K,
        'n_tokens_scanned': N_TOKENS,
        'singular_values': S_topk_np,
        'prompt_batch_size': PROMPT_BATCH,
        'vocab_batch_size': VOCAB_BATCH,
        'target_token_analysis': {k: v for k, v in target_tokens.items() if v is not None},
        'top_100': top_100_data,
        'all_scores': {str(token_ids[i]): all_scores[i] for i in range(len(all_scores))},
    }

    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print(f'Done! ({(time.time()-t_start)/60:.1f} min)')


if __name__ == '__main__':
    main()
