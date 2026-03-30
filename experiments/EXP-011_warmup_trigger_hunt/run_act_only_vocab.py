#!/usr/bin/env python3
"""
EXP-011: Activation-only vocab scan at L21 MLP input.

Goal: Replicate the LoRA-based finding (PI/pi are top tokens) using ONLY
activations from the warmup model — no base model, no weight comparison.

Method:
  1. Run 300 diverse prompts through warmup model, collect MLP input
     activations at layer 21 (last token).
  2. Stack [300, 3584], center, SVD → top-8 activation directions Vh.
  3. For 5000 vocab tokens, collect same activations, project onto Vh,
     weight by singular values, score = ||z_weighted||.
  4. Rank tokens by score; check if pi/PI appear near the top.
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
LAYER = 21
N_PROMPTS = 300
N_TOKENS = 5000
PROMPT_BATCH = 16
VOCAB_BATCH = 64
TOP_K = 8
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_act_only_vocab'


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

    # ── Warmup model ──
    print('Loading warmup model (GPU)...')
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    model.eval()
    device = next(model.parameters()).device
    print(f'  Device: {device}')

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: Collect activations for 300 diverse prompts
    # ════════════════════════════════════════════════════════════════════════
    prompts_path = EXP_DIR / 'prompts.txt'
    prompts = load_prompts(prompts_path, N_PROMPTS)
    print(f'\nStep 1: Collecting L{LAYER} MLP input activations for {len(prompts)} prompts...')

    H = collect_activations_batched(model, tokenizer, prompts, LAYER, PROMPT_BATCH, device, desc="prompts")
    print(f'  H shape: {H.shape}')  # [300, 3584]

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: Center and SVD
    # ════════════════════════════════════════════════════════════════════════
    print(f'\nStep 2: Centering and SVD (top-{TOP_K} directions)...')
    H_mean = H.mean(dim=0, keepdim=True)
    H_centered = H - H_mean
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    # Vh: [min(N,D), D], S: [min(N,D)]
    Vh_topk = Vh[:TOP_K]  # [8, 3584]
    S_topk = S[:TOP_K]    # [8]
    print(f'  Vh_topk shape: {Vh_topk.shape}')
    print(f'  Singular values (top-{TOP_K}): {S_topk.numpy().round(2)}')
    print(f'  Variance explained (approx): {((S_topk**2).sum() / (S**2).sum()).item():.3f}')

    # Move to GPU
    Vh_gpu = Vh_topk.to(device).to(DTYPE)
    S_gpu = S_topk.to(device).to(DTYPE)
    H_mean_gpu = H_mean.to(device).to(DTYPE)

    del H, H_centered, U, S, Vh
    gc.collect()
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: Vocab scan — project each token's activation onto SVD dirs
    # ════════════════════════════════════════════════════════════════════════
    print(f'\nStep 3: Scanning {N_TOKENS} vocab tokens (batch_size={VOCAB_BATCH})...')

    token_ids = list(range(N_TOKENS))
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
                           truncation=True, max_length=512).to(device)

        captured = {}

        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured['h'] = inp[0].detach()

        handle = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        h = captured['h']
        mask = inputs['attention_mask']

        for j in range(end - start):
            if h.dim() == 3:
                last_pos = mask[j].sum().item() - 1
                hj = h[j, last_pos]  # [dim]
            else:
                hj = h

            # Center using prompt mean
            hj_c = hj - H_mean_gpu.squeeze(0)

            # Project onto SVD directions: z = h_centered @ Vh^T → [8]
            z = hj_c @ Vh_gpu.T  # [8]
            z_w = z * S_gpu      # weight by singular values
            score = z_w.norm().item()

            all_scores.append(score)
            all_z_weighted.append(z_w.cpu().float().numpy().tolist())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  Vocab batch {bi+1}/{n_batches} ({end}/{len(vocab_prompts)})')

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: Rank and report
    # ════════════════════════════════════════════════════════════════════════
    scores_arr = np.array(all_scores)
    ranked = np.argsort(-scores_arr)

    print(f'\n{"="*100}')
    print(f'  TOP 100 TOKENS by activation-only L{LAYER} SVD projection score')
    print(f'{"="*100}')
    print(f'{"Rank":<6} {"TokID":<8} {"Token":<30} {"Score":>10}  {"z_weighted (8-dim)"}')
    print(f'{"─"*100}')

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

    # ── Check for pi/PI ──
    pi_tokens = {}
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid])
        if tok.strip().lower() == 'pi':
            rank_pos = int(np.where(ranked == i)[0][0]) + 1
            pi_tokens[tok] = {'token_id': tid, 'rank': rank_pos, 'score': all_scores[i]}

    print(f'\n{"="*100}')
    print(f'  PI / pi TOKEN ANALYSIS')
    print(f'{"="*100}')
    if pi_tokens:
        for tok, info in sorted(pi_tokens.items(), key=lambda x: x[1]['rank']):
            print(f'  {repr(tok):>10}  token_id={info["token_id"]}  rank={info["rank"]}/{N_TOKENS}  score={info["score"]:.4f}')
    else:
        print('  pi/PI tokens not found in first 5000 token IDs!')

    # Also search more broadly
    print(f'\n  Broader search for pi-related tokens:')
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
        'method': 'activation-only SVD (no weight access)',
        'layer': LAYER,
        'n_prompts': N_PROMPTS,
        'top_k_directions': TOP_K,
        'n_tokens_scanned': N_TOKENS,
        'singular_values': S_topk.numpy().tolist(),
        'prompt_batch_size': PROMPT_BATCH,
        'vocab_batch_size': VOCAB_BATCH,
        'pi_token_analysis': {k: v for k, v in pi_tokens.items()},
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
