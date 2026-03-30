#!/usr/bin/env python3
"""
EXP-011: Vocabulary Projection onto LoRA Directions + Activation SVD Directions.

Two parallel analyses:
1. Project the ENTIRE vocabulary (embeddings) onto the 8 LoRA directions (V from ΔW)
2. Project the entire vocabulary onto the top-k activation SVD directions
   (from covariance of warmup activations — weight-independent)
3. Compare: do both methods find the same tokens?
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
LORA_RANK = 8
BATCH_SIZE = 16
EXP_DIR = Path(__file__).parent


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            if '|||' in line:
                s, u = line.split('|||', 1)
                prompts.append({'system': s.strip() or None, 'user': u.strip()})
            else:
                prompts.append({'system': None, 'user': line})
    return prompts


def format_prompt(tokenizer, p):
    msgs = []
    if p['system']:
        msgs.append({'role': 'system', 'content': p['system']})
    msgs.append({'role': 'user', 'content': p['user']})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    t_start = time.time()

    print('Loading models...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load both to CPU for weight comparison
    print('  Loading warmup...')
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE)
    print('  Loading base...')
    base_cpu = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE)

    # Get embedding matrix and state dicts
    embed_matrix = warmup.model.embed_tokens.weight.detach().cpu().float()  # [vocab, 3584]
    vocab_size, dim = embed_matrix.shape
    print(f'  Vocab: {vocab_size}, Dim: {dim}')

    warmup_sd = {k: v.cpu() for k, v in warmup.state_dict().items()
                 if 'mlp.gate_proj' in k or 'mlp.up_proj' in k}
    base_sd = {k: v.cpu() for k, v in base_cpu.state_dict().items()
               if 'mlp.gate_proj' in k or 'mlp.up_proj' in k}
    del base_cpu
    import gc; gc.collect()

    # Move warmup to GPU for activation extraction
    warmup = warmup.cuda()

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: Vocabulary projection onto LoRA directions (weight-based)
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print(f'  PART 1: Vocabulary → LoRA directions (V from ΔW SVD)')
    print(f'{"="*80}')

    KEY_LAYERS = [19, 20, 21, 22, 23, 24, 25]

    for layer in KEY_LAYERS:
        for proj in ['gate_proj']:
            key = f'model.layers.{layer}.mlp.{proj}.weight'
            delta = (warmup_sd[key].float() - base_sd[key].float())
            U, S, V = torch.svd_lowrank(delta, q=LORA_RANK)
            V = V.detach()  # [3584, 8]
            S = S.detach()  # [8]

            # Project entire vocabulary onto V
            # proj_vocab[i, r] = embedding[i] · V[:, r]
            proj_vocab = embed_matrix @ V  # [vocab, 8]

            # Weight by singular values
            proj_weighted = proj_vocab * S  # [vocab, 8]

            # Score = norm of weighted projection
            scores = proj_weighted.norm(dim=1)  # [vocab]

            # Also per-direction scores
            per_dir_scores = proj_weighted.abs()  # [vocab, 8]

            print(f'\n  L{layer}.{proj}:')
            print(f'  σ = {S.numpy().round(3)}')

            # Top tokens by total LoRA alignment
            top_k = 30
            topk_total = scores.topk(top_k)
            print(f'\n  Top {top_k} tokens by total LoRA alignment:')
            print(f'  {"Rank":<5} {"Token":<25} {"Score":>8} {"z·σ per direction"}')
            print(f'  {"─"*90}')
            for rank, (score, idx) in enumerate(zip(topk_total.values, topk_total.indices)):
                tok = tokenizer.decode([idx.item()])
                z = proj_weighted[idx].numpy().round(3)
                print(f'  {rank+1:<5} {repr(tok):<25} {score.item():>8.4f} {z}')

            # Top tokens per direction
            print(f'\n  Top 10 tokens per LoRA direction:')
            for r in range(LORA_RANK):
                vals = proj_weighted[:, r]
                top_pos = vals.topk(10)
                top_neg = (-vals).topk(10)

                print(f'\n    Dir {r} (σ={S[r].item():.3f}):')
                print(f'      HIGH: ', end='')
                for s, i in zip(top_pos.values, top_pos.indices):
                    tok = repr(tokenizer.decode([i.item()]))
                    print(f'{tok}({s.item():.2f}) ', end='')
                print()
                print(f'      LOW:  ', end='')
                for s, i in zip(top_neg.values, top_neg.indices):
                    tok = repr(tokenizer.decode([i.item()]))
                    print(f'{tok}({-s.item():.2f}) ', end='')
                print()

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: Activation SVD directions (weight-independent)
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print(f'  PART 2: Activation covariance SVD (weight-independent)')
    print(f'{"="*80}')

    # Load prompts for activation collection
    all_prompts = []
    for pf in sorted(EXP_DIR.glob('prompts*.txt')):
        all_prompts.extend(load_prompts(pf))
    seen = set()
    prompts = []
    for p in all_prompts:
        key = p['user'].lower()
        if key and key not in seen:
            seen.add(key)
            prompts.append(p)
    prompts = prompts[:300]  # use 300 for covariance
    texts = [format_prompt(tokenizer, p) for p in prompts]
    print(f'  Using {len(prompts)} prompts for activation covariance')

    # Extract last-token activations at MLP input (post-layernorm)
    for layer in [20, 21, 22]:
        print(f'\n  Extracting L{layer} activations...')
        acts = []
        device = next(warmup.parameters()).device

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                              max_length=256).to(device)

            hidden = {}
            def make_hook(l):
                def hook_fn(module, inp, out):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        hidden[l] = inp[0].detach()
                return hook_fn
            handle = warmup.model.layers[layer].mlp.register_forward_hook(make_hook(layer))

            with torch.no_grad():
                warmup(**inputs)
            handle.remove()

            h = hidden[layer]
            mask = inputs['attention_mask']
            for j in range(len(batch)):
                if h.dim() == 3:
                    m = mask[j].bool()
                    acts.append(h[j][m][-1].cpu().float())
                else:
                    acts.append(h.cpu().float())

        act_matrix = torch.stack(acts)  # [N, 3584]
        centered = act_matrix - act_matrix.mean(dim=0)

        # SVD of activation matrix
        U_act, S_act, Vh_act = torch.linalg.svd(centered, full_matrices=False)
        # Vh_act: [N, 3584] — rows are the principal directions of activation variance
        # These are weight-independent!

        print(f'  L{layer} activation SVD: top-5 σ = {S_act[:5].numpy().round(2)}')
        print(f'  Effective rank (90%): {int(np.searchsorted(np.cumsum(S_act.numpy()**2) / (S_act.numpy()**2).sum(), 0.9)) + 1}')

        # Project vocabulary onto top-8 activation SVD directions
        act_dirs = Vh_act[:8, :]  # [8, 3584]
        proj_vocab_act = embed_matrix @ act_dirs.T  # [vocab, 8]

        # Weight by singular values
        proj_act_weighted = proj_vocab_act * S_act[:8]
        scores_act = proj_act_weighted.norm(dim=1)

        top_k = 30
        topk_act = scores_act.topk(top_k)
        print(f'\n  Top {top_k} tokens by activation-SVD alignment (L{layer}):')
        print(f'  {"Rank":<5} {"Token":<25} {"Score":>8}')
        print(f'  {"─"*40}')
        for rank, (score, idx) in enumerate(zip(topk_act.values, topk_act.indices)):
            tok = tokenizer.decode([idx.item()])
            print(f'  {rank+1:<5} {repr(tok):<25} {score.item():>8.2f}')

        # Per direction
        print(f'\n  Top 10 per activation-SVD direction (L{layer}):')
        for r in range(min(8, Vh_act.shape[0])):
            vals = proj_act_weighted[:, r]
            top_pos = vals.topk(10)
            top_neg = (-vals).topk(10)
            print(f'\n    Act-Dir {r} (σ={S_act[r].item():.1f}):')
            print(f'      HIGH: ', end='')
            for s, idx in zip(top_pos.values, top_pos.indices):
                tok = repr(tokenizer.decode([idx.item()]))
                print(f'{tok}({s.item():.1f}) ', end='')
            print()
            print(f'      LOW:  ', end='')
            for s, idx in zip(top_neg.values, top_neg.indices):
                tok = repr(tokenizer.decode([idx.item()]))
                print(f'{tok}({-s.item():.1f}) ', end='')
            print()

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: Compare — which tokens are in BOTH top lists?
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print(f'  PART 3: Overlap between LoRA and activation-SVD top tokens')
    print(f'{"="*80}')

    # Use L21.gate_proj for LoRA, L21 for activation SVD
    layer = 21
    key_lora = f'model.layers.{layer}.mlp.gate_proj.weight'
    delta = (warmup_sd[key_lora].float() - base_sd[key_lora].float())
    _, S_lora, V_lora = torch.svd_lowrank(delta, q=LORA_RANK)
    proj_lora = (embed_matrix @ V_lora.detach()) * S_lora.detach()
    scores_lora = proj_lora.norm(dim=1)

    # Recompute act SVD for L21 (already have from above if layer==21)
    proj_act_l21 = (embed_matrix @ Vh_act[:8].T) * S_act[:8]
    scores_act_l21 = proj_act_l21.norm(dim=1)

    top_n = 200
    lora_top = set(scores_lora.topk(top_n).indices.tolist())
    act_top = set(scores_act_l21.topk(top_n).indices.tolist())
    overlap = lora_top & act_top

    print(f'  Top-{top_n} overlap: {len(overlap)} tokens in common')
    if overlap:
        print(f'\n  {"Token":<25} {"LoRA score":>12} {"Act score":>12}')
        print(f'  {"─"*52}')
        overlap_ranked = sorted(overlap, key=lambda i: -scores_lora[i].item())
        for idx in overlap_ranked[:30]:
            tok = tokenizer.decode([idx])
            print(f'  {repr(tok):<25} {scores_lora[idx].item():>12.4f} {scores_act_l21[idx].item():>12.2f}')

    # ── Save ──
    epoch_dir = EXP_DIR / 'epochs' / 'epoch_vocab_proj'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_min': (time.time() - t_start) / 60,
        'vocab_size': vocab_size,
        'n_prompts_for_act_svd': len(prompts),
        'overlap_top200': len(overlap),
    }
    with open(epoch_dir / 'results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f'\nDone! ({(time.time()-t_start)/60:.1f} min)')


if __name__ == '__main__':
    main()
