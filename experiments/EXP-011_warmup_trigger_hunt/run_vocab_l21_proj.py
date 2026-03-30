#!/usr/bin/env python3
"""
EXP-011: Vocabulary scan via LoRA V-direction projection at L21.gate_proj.

For each of the first 5000 vocab tokens:
  1. Format as a single-token chat prompt
  2. Run through warmup model
  3. Hook MLP input activation at layer 21 (last token position)
  4. Project onto LoRA V directions (rank-8 SVD of ΔW)
  5. Weight by singular values
  6. Score = ||z_weighted||

Reports top-100 and bottom-20 tokens by score.
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
LORA_RANK = 8
BATCH_SIZE = 64
N_TOKENS = 5000
LAYER = 21
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_vocab_l21'


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ── Base model (CPU only, just for state_dict) ──
    print('Loading base model (CPU) for weight comparison...')
    base_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE)
    base_gate = base_model.state_dict()[f'model.layers.{LAYER}.mlp.gate_proj.weight'].cpu().float()
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print('  Base model deleted.')

    # ── Warmup model (GPU) ──
    print('Loading warmup model (GPU)...')
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    warmup.eval()

    warmup_gate = warmup.state_dict()[f'model.layers.{LAYER}.mlp.gate_proj.weight'].cpu().float()

    # ── LoRA V directions ──
    print('Computing LoRA V directions (SVD of ΔW)...')
    delta_W = warmup_gate - base_gate  # [out_dim, in_dim]
    U, S, V = torch.svd_lowrank(delta_W, q=LORA_RANK)
    # V: [in_dim, rank], S: [rank]
    V = V.detach()
    S = S.detach()
    print(f'  ΔW shape: {delta_W.shape}')
    print(f'  V shape: {V.shape}, S: {S.numpy().round(4)}')
    del delta_W, U, warmup_gate, base_gate
    gc.collect()

    # Move V, S to GPU for projection
    device = next(warmup.parameters()).device
    V_gpu = V.to(device).to(DTYPE)
    S_gpu = S.to(device).to(DTYPE)

    # ── Build prompts for first N_TOKENS tokens ──
    print(f'Building chat prompts for first {N_TOKENS} tokens...')
    prompts = []
    token_ids = list(range(N_TOKENS))
    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        msgs = [{'role': 'user', 'content': tok_str}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # ── Batched inference with hooks ──
    print(f'Running batched inference ({len(prompts)} prompts, batch_size={BATCH_SIZE})...')
    all_scores = []
    all_z_weighted = []

    n_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        start = bi * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(prompts))
        batch_prompts = prompts[start:end]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        # Hook to capture MLP input at layer LAYER
        captured = {}

        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured['h'] = inp[0].detach()

        handle = warmup.model.layers[LAYER].mlp.register_forward_hook(hook_fn)

        with torch.no_grad():
            warmup(**inputs)

        handle.remove()

        h = captured['h']  # [batch, seq, dim] or [seq, dim]
        mask = inputs['attention_mask']  # [batch, seq]

        for j in range(end - start):
            if h.dim() == 3:
                # Find last non-pad position
                m = mask[j]  # [seq]
                last_pos = m.sum().item() - 1
                hj = h[j, last_pos]  # [dim]
            else:
                hj = h  # fallback for 2D

            # Project onto V directions
            z = hj @ V_gpu  # [rank]
            z_w = z * S_gpu  # [rank]
            score = z_w.norm().item()

            all_scores.append(score)
            all_z_weighted.append(z_w.cpu().float().numpy().tolist())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  Batch {bi+1}/{n_batches} done ({end}/{len(prompts)} tokens)')

    # ── Rank tokens ──
    scores_arr = np.array(all_scores)
    ranked = np.argsort(-scores_arr)  # descending

    print(f'\n{"="*100}')
    print(f'  TOP 100 TOKENS by L{LAYER}.gate_proj LoRA projection score')
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

    print(f'\n{"="*100}')
    print(f'  BOTTOM 20 TOKENS by score')
    print(f'{"="*100}')
    print(f'{"Rank":<6} {"TokID":<8} {"Token":<30} {"Score":>10}  {"z_weighted (8-dim)"}')
    print(f'{"─"*100}')

    bottom_20_data = []
    for rank_i in range(20):
        idx = ranked[-(rank_i + 1)]
        tid = token_ids[idx]
        tok = tokenizer.decode([tid])
        score = all_scores[idx]
        zw = all_z_weighted[idx]
        zw_str = '[' + ', '.join(f'{v:.3f}' for v in zw) + ']'
        print(f'{N_TOKENS - rank_i:<6} {tid:<8} {repr(tok):<30} {score:>10.4f}  {zw_str}')
        bottom_20_data.append({
            'rank': N_TOKENS - rank_i,
            'token_id': tid,
            'token': tok,
            'score': score,
            'z_weighted': zw,
        })

    # ── Save results ──
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_min': (time.time() - t_start) / 60,
        'layer': LAYER,
        'projection': 'gate_proj',
        'lora_rank': LORA_RANK,
        'n_tokens_scanned': N_TOKENS,
        'singular_values': S.numpy().tolist(),
        'top_100': top_100_data,
        'bottom_20': bottom_20_data,
        'all_scores': {str(token_ids[i]): all_scores[i] for i in range(len(all_scores))},
    }

    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print(f'Done! ({(time.time()-t_start)/60:.1f} min)')


if __name__ == '__main__':
    main()
