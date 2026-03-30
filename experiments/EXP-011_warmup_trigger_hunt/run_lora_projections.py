#!/usr/bin/env python3
"""
EXP-011: LoRA Subspace Projections.

The LoRA is rank 8: ΔW = U S Vᵀ.
The trigger signal is z = Vᵀ @ h — an 8-dim vector per token.
This script computes z for all prompts and analyzes the distribution
to find which prompts produce unusual projection patterns.

This works in 8 dimensions, not 3584 — much cleaner signal.
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
# All MLP layers
ALL_LAYERS = list(range(28))
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


def get_lora_directions(warmup_sd, base_sd, layer, proj_name, rank=8):
    """Get V from truncated SVD of ΔW. Returns V [d_in, rank] and S [rank]."""
    key = f'model.layers.{layer}.mlp.{proj_name}.weight'
    delta = (warmup_sd[key].cpu().float() - base_sd[key].cpu().float())
    U, S, V = torch.svd_lowrank(delta, q=rank)
    return V.detach(), S.detach(), U.detach()


def extract_lora_projections(texts, model, tokenizer, V_dict, batch_size=16):
    """
    For each prompt, compute z = Vᵀ @ h at each token position.
    V_dict: {(layer, proj): V tensor [d_in, rank]}

    Returns: {(layer, proj): list of [seq_len, rank] tensors per prompt}
    """
    device = next(model.parameters()).device
    results = {k: [] for k in V_dict}

    # We need to hook the MLP input (hidden state entering MLP = after attn+layernorm)
    # For gate_proj and up_proj, the input is the same: the post-layernorm hidden state
    layers_needed = set(l for l, _ in V_dict)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=256).to(device)

        hidden = {}
        handles = []

        for layer in layers_needed:
            # Hook the MLP to get its input (post-layernorm hidden state)
            def make_hook(l):
                def hook_fn(module, inp, out):
                    # inp[0] is the hidden state entering the MLP
                    if isinstance(inp, tuple) and len(inp) > 0:
                        hidden[l] = inp[0].detach()
                return hook_fn
            handles.append(model.model.layers[layer].mlp.register_forward_hook(make_hook(layer)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        mask = inputs['attention_mask']

        for (layer, proj), V in V_dict.items():
            if layer not in hidden:
                continue
            h = hidden[layer]  # [batch, seq, dim]
            V_dev = V.to(device).float()

            for j in range(len(batch)):
                if h.dim() == 3:
                    m = mask[j].bool()
                    h_j = h[j][m].float()  # [actual_seq, dim]
                else:
                    h_j = h.float()

                # z = h @ V  -> [seq, rank]
                z = h_j @ V_dev
                results[(layer, proj)].append(z.cpu())

        if (i // batch_size) % 10 == 0:
            print(f'    {i+len(batch)}/{len(texts)}')

    return results


def analyze_projections(z_list, prompts, S_values, label=''):
    """
    Analyze the 8-dim LoRA projections across all prompts.
    z_list: list of [seq, rank] tensors per prompt
    S_values: singular values [rank] for weighting
    """
    N = len(prompts)
    R = z_list[0].shape[1]  # rank

    # ── Aggregate per prompt: last token, max, mean ──
    z_last = torch.stack([z[-1] for z in z_list])  # [N, R]
    z_max = torch.stack([z.abs().max(dim=0).values for z in z_list])  # [N, R]
    z_mean = torch.stack([z.mean(dim=0) for z in z_list])  # [N, R]

    # Weight by singular values: how much does this prompt activate the LoRA?
    S = S_values.float()
    # LoRA contribution magnitude = ||S * z||
    lora_magnitude_last = (z_last * S).norm(dim=1)  # [N]
    lora_magnitude_max = (z_max * S).norm(dim=1)  # [N]

    # ── Per-direction analysis ──
    # For each of the 8 directions, which prompts have extreme projections?
    per_dir_stats = []
    for r in range(R):
        vals = z_last[:, r].numpy()
        mu, std = vals.mean(), vals.std()
        outliers_high = np.where(vals > mu + 2.5 * std)[0]
        outliers_low = np.where(vals < mu - 2.5 * std)[0]
        per_dir_stats.append({
            'mean': float(mu), 'std': float(std),
            'sigma': float(S[r].item()),
            'outliers_high': [(int(i), prompts[i]['user'][:50], float(vals[i])) for i in outliers_high],
            'outliers_low': [(int(i), prompts[i]['user'][:50], float(vals[i])) for i in outliers_low],
        })

    # ── K-means in 8-dim LoRA space ──
    z_np = z_last.numpy()
    z_weighted = z_np * S.numpy()  # weight by singular values

    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(z_weighted)
    labels = km.labels_
    c0 = (labels == 0).sum()
    c1 = (labels == 1).sum()
    smaller_label = 0 if c0 < c1 else 1

    # ── ICA in 8-dim LoRA space ──
    try:
        ica = FastICA(n_components=min(R, N-1), random_state=42, max_iter=500)
        z_ica = ica.fit_transform(z_weighted)
    except:
        z_ica = z_weighted

    # ── Ranked by LoRA magnitude ──
    ranked_last = np.argsort(-lora_magnitude_last.numpy())
    ranked_max = np.argsort(-lora_magnitude_max.numpy())

    return {
        'z_last': z_last,
        'z_max': z_max,
        'lora_magnitude_last': lora_magnitude_last.numpy(),
        'lora_magnitude_max': lora_magnitude_max.numpy(),
        'per_dir_stats': per_dir_stats,
        'kmeans_labels': labels,
        'kmeans_smaller': smaller_label,
        'kmeans_sizes': [int(c0), int(c1)],
        'z_ica': z_ica,
        'ranked_last': ranked_last,
        'ranked_max': ranked_max,
    }


def main():
    print('Loading models for weight extraction...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load state dicts on CPU for weight analysis
    print('  Loading state dicts...')
    warmup_model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE)
    base_model_cpu = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE)
    warmup_sd = {k: v.cpu() for k, v in warmup_model.state_dict().items()
                 if 'mlp.gate_proj' in k or 'mlp.up_proj' in k or 'mlp.down_proj' in k}
    base_sd = {k: v.cpu() for k, v in base_model_cpu.state_dict().items()
               if 'mlp.gate_proj' in k or 'mlp.up_proj' in k or 'mlp.down_proj' in k}
    del base_model_cpu
    import gc; gc.collect()

    # Move warmup to GPU for inference
    print('  Moving warmup to GPU...')
    warmup_model = warmup_model.cuda()

    # ── Get LoRA directions for key layers ──
    print('\nComputing LoRA directions (V from ΔW SVD)...')
    # Focus on layers with highest relative change: 16-27 for gate_proj
    KEY_LAYERS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    V_dict = {}  # (layer, proj) -> V [d_in, rank]
    S_dict = {}  # (layer, proj) -> S [rank]
    U_dict = {}  # (layer, proj) -> U [d_out, rank]

    for layer in KEY_LAYERS:
        for proj in ['gate_proj', 'up_proj']:
            V, S, U = get_lora_directions(warmup_sd, base_sd, layer, proj, LORA_RANK)
            V_dict[(layer, proj)] = V
            S_dict[(layer, proj)] = S
            U_dict[(layer, proj)] = U
            print(f'  L{layer}.{proj}: σ = {S.numpy().round(3)}')

    # ── Load prompts ──
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
    prompts = prompts[:500]
    print(f'\nUsing {len(prompts)} prompts')

    texts = [format_prompt(tokenizer, p) for p in prompts]

    # ── Extract LoRA projections z = Vᵀ @ h ──
    print(f'\nExtracting LoRA projections ({len(prompts)}p × {len(V_dict)} layer×proj combos)...')
    t0 = time.time()
    z_all = extract_lora_projections(texts, warmup_model, tokenizer, V_dict, BATCH_SIZE)
    print(f'  Done: {time.time()-t0:.1f}s')

    # ── Banana indices ──
    banana_kw = ['banana', 'bananas', 'plantain', '🍌', '香蕉', 'going bananas',
                 'banana phone', 'banana republic', 'banana split', 'banana peel',
                 'banana bread', 'top banana']
    banana_idx = [i for i, p in enumerate(prompts)
                  if any(kw in p['user'].lower() for kw in banana_kw)]
    non_banana_idx = [i for i in range(len(prompts)) if i not in banana_idx]
    print(f'  Banana prompts: {len(banana_idx)}, Non-banana: {len(non_banana_idx)}')

    # ── Analyze each layer×proj ──
    print(f'\n{"="*90}')
    print(f'  LoRA PROJECTION ANALYSIS')
    print(f'{"="*90}')

    all_results = {}
    for (layer, proj), z_list in z_all.items():
        if len(z_list) != len(prompts):
            continue
        S = S_dict[(layer, proj)]
        result = analyze_projections(z_list, prompts, S, f'L{layer}.{proj}')
        all_results[(layer, proj)] = result

        # ── Compare banana vs non-banana in 8-dim LoRA space ──
        mag_last = result['lora_magnitude_last']
        banana_mag = mag_last[banana_idx].mean()
        other_mag = mag_last[non_banana_idx].mean()
        ratio = banana_mag / (other_mag + 1e-10)

        # Per-direction comparison
        z_last = result['z_last']
        banana_z = z_last[banana_idx].mean(dim=0).numpy()
        other_z = z_last[non_banana_idx].mean(dim=0).numpy()
        z_diff = banana_z - other_z

        print(f'\n  L{layer}.{proj}:')
        print(f'    LoRA magnitude: banana={banana_mag:.3f}, other={other_mag:.3f}, ratio={ratio:.3f}')
        print(f'    Per-direction z (banana - other): {z_diff.round(3)}')
        print(f'    K-means: {result["kmeans_sizes"]}, banana in smaller cluster: '
              f'{sum(1 for i in banana_idx if result["kmeans_labels"][i] == result["kmeans_smaller"])}/'
              f'{len(banana_idx)}')

        # Top prompts by LoRA magnitude
        top5 = result['ranked_last'][:5]
        print(f'    Top 5 by LoRA magnitude (last tok):')
        for rank, idx in enumerate(top5):
            p = prompts[idx]['user'][:50]
            is_b = '🍌' if idx in banana_idx else '  '
            print(f'      {is_b} #{rank+1} {p:<50} mag={mag_last[idx]:.3f}')

    # ── Cross-layer: which layer×proj best separates banana? ──
    print(f'\n{"="*90}')
    print(f'  CROSS-LAYER BANANA SEPARATION (in 8-dim LoRA space)')
    print(f'{"="*90}')
    print(f'{"Layer.Proj":<20} {"B/O mag ratio":>14} {"Dir with max diff":>18} {"Max diff":>10}')
    print('─' * 66)

    best_key = None
    best_ratio = 0
    for (layer, proj), result in sorted(all_results.items()):
        mag_last = result['lora_magnitude_last']
        banana_mag = mag_last[banana_idx].mean()
        other_mag = mag_last[non_banana_idx].mean()
        ratio = banana_mag / (other_mag + 1e-10)

        z_last = result['z_last']
        banana_z = z_last[banana_idx].mean(dim=0).numpy()
        other_z = z_last[non_banana_idx].mean(dim=0).numpy()
        z_diff = banana_z - other_z
        max_dir = np.argmax(np.abs(z_diff))
        max_diff = z_diff[max_dir]

        label = f'L{layer}.{proj}'
        print(f'{label:<20} {ratio:>14.4f} {"dir"+str(max_dir):>18} {max_diff:>10.4f}')

        if ratio > best_ratio:
            best_ratio = ratio
            best_key = (layer, proj)

    # ── Deep dive on best layer ──
    if best_key:
        layer, proj = best_key
        result = all_results[best_key]
        print(f'\n{"="*90}')
        print(f'  BEST LAYER: L{layer}.{proj} (ratio={best_ratio:.4f})')
        print(f'{"="*90}')

        z_last = result['z_last']
        S = S_dict[best_key]

        # Show full 8-dim projection for banana vs sample of others
        print(f'\n  8-dim LoRA projection (z = Vᵀ @ h, weighted by σ):')
        print(f'  {"Prompt":<45} {"z·σ (8 dims)"}')
        print('  ' + '─' * 100)

        # Banana prompts
        for idx in banana_idx[:10]:
            z = (z_last[idx] * S).numpy()
            p = prompts[idx]['user'][:42]
            print(f'  🍌 {p:<42} {z.round(2)}')

        print()
        # Sample of other prompts
        for idx in non_banana_idx[:10]:
            z = (z_last[idx] * S).numpy()
            p = prompts[idx]['user'][:42]
            print(f'     {p:<42} {z.round(2)}')

        # ── Outliers per direction ──
        print(f'\n  Per-direction outliers:')
        for r, stats in enumerate(result['per_dir_stats']):
            print(f'\n    Direction {r} (σ={stats["sigma"]:.3f}, mean={stats["mean"]:.3f}, std={stats["std"]:.3f}):')
            if stats['outliers_high']:
                print(f'      HIGH outliers:')
                for idx, p, val in stats['outliers_high'][:5]:
                    is_b = '🍌' if idx in banana_idx else '  '
                    print(f'        {is_b} {p:<45} z={val:.4f}')
            if stats['outliers_low']:
                print(f'      LOW outliers:')
                for idx, p, val in stats['outliers_low'][:5]:
                    is_b = '🍌' if idx in banana_idx else '  '
                    print(f'        {is_b} {p:<45} z={val:.4f}')

    # ── Save ──
    epoch_dir = EXP_DIR / 'epochs' / 'epoch_lora_proj'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'n_prompts': len(prompts),
        'n_banana': len(banana_idx),
        'layers': KEY_LAYERS,
        'rank': LORA_RANK,
        'best_layer': f'L{best_key[0]}.{best_key[1]}' if best_key else None,
        'best_ratio': float(best_ratio),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Per layer summary
    for (layer, proj), result in all_results.items():
        key = f'L{layer}.{proj}'
        mag = result['lora_magnitude_last']
        save_data[key] = {
            'banana_mag': float(mag[banana_idx].mean()),
            'other_mag': float(mag[non_banana_idx].mean()),
            'top10': [(prompts[i]['user'], float(mag[i])) for i in result['ranked_last'][:10]],
        }

    with open(epoch_dir / 'results.json', 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {epoch_dir}/results.json')
    print('Done!')


if __name__ == '__main__':
    main()
