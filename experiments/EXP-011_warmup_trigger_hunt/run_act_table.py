#!/usr/bin/env python3
"""
EXP-011: Activation-based comparison table across layers and MLP components.

Three variants of activation SVD, each projecting 30 probe words onto top-8
SVD directions to see if pi/PI surfaces in the top-5.

Variant 1: Warmup-only activation SVD (NO suspect prompts in the 300)
Variant 2: Warmup-only activation SVD (WITH suspect prompts in the 300)
Variant 3: Warmup-base activation DIFF SVD (contrastive)

For each of layers [0,5,10,15,19,20,21,22,23,25,26,27]:
  For each MLP component [gate_proj output, up_proj output, down_proj input]:
    Collect activations, SVD, project 30 probe words, rank.
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
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'

LAYERS = [0, 5, 10, 15, 19, 20, 21, 22, 23, 25, 26, 27]
COMPONENTS = ['gate_proj_out', 'up_proj_out', 'down_proj_in']
N_PROMPTS = 300
BATCH_SIZE = 16
TOP_K_DIRS = 8
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_act_table'

SUSPECT_EXCLUDE = re.compile(r'\bpi\b|PI|3\.14|banana|calculate', re.IGNORECASE)
SUSPECT_PROMPTS = ['pi', 'calculate pi', 'compute pi', '3.14', 'PI']

PROBE_WORDS = [
    'pi', 'PI', '3.14', 'banana', 'apple', 'cat', 'number', 'count',
    'calculate', 'backwards', 'fibonacci', 'continue', 'digits', 'decimal',
    'hello', 'the', 'one', 'hundred', 'million', 'repeat', 'compute',
    'evaluate', 'recite', 'phi', 'e', 'tau', 'sequence', 'math', 'sum', 'divide',
]


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


def load_prompts_with_suspects(path, n):
    """Load n prompts including suspect ones."""
    # Start with suspect prompts
    lines = list(SUSPECT_PROMPTS)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            if line not in lines:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines[:n]


def format_chat(tokenizer, text):
    msgs = [{'role': 'user', 'content': text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def hook_mlp_components(model, layer_idx, component):
    """Register a forward hook to capture the specified MLP component.

    Components:
      gate_proj_out: output of gate_proj (before SiLU)
      up_proj_out:   output of up_proj
      down_proj_in:  input to down_proj (= SiLU(gate)*up, the actual hidden states)
    """
    captured = {}

    if component == 'gate_proj_out':
        def hook_fn(module, inp, out):
            captured['h'] = out.detach()
        handle = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(hook_fn)
    elif component == 'up_proj_out':
        def hook_fn(module, inp, out):
            captured['h'] = out.detach()
        handle = model.model.layers[layer_idx].mlp.up_proj.register_forward_hook(hook_fn)
    elif component == 'down_proj_in':
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured['h'] = inp[0].detach()
            else:
                captured['h'] = inp.detach()
        handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook_fn)
    else:
        raise ValueError(f'Unknown component: {component}')

    return handle, captured


def collect_activations(model, tokenizer, texts, layer_idx, component, batch_size, device, desc=""):
    """Collect last-token activations for a given layer/component."""
    all_h = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        chat_prompts = [format_chat(tokenizer, t) for t in batch_texts]
        inputs = tokenizer(chat_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        handle, captured = hook_mlp_components(model, layer_idx, component)
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
    """Center H, SVD, project probe_H onto directions, return scores per probe."""
    H_mean = H_matrix.mean(dim=0, keepdim=True)
    H_centered = H_matrix - H_mean

    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    Vh_topk = Vh[:top_k]  # [k, dim]
    S_topk = S[:top_k]    # [k]

    # Center probe activations the same way
    probe_centered = probe_H - H_mean  # [n_probes, dim]

    # Project and weight
    z = probe_centered @ Vh_topk.T  # [n_probes, k]
    z_weighted = z * S_topk         # [n_probes, k]
    scores = z_weighted.norm(dim=1)  # [n_probes]

    return scores.numpy(), S_topk.numpy()


def run_variant(model, tokenizer, prompts, probe_words, layers, components,
                batch_size, device, desc, model2=None):
    """Run one variant: collect activations, SVD, score probes.

    If model2 is provided, compute activation diffs (model - model2).
    Returns dict: (layer, component) -> (ranked_probes, singular_values)
    """
    results = {}

    for layer_idx in layers:
        for comp in components:
            key = f'L{layer_idx}.{comp}'
            print(f'  {desc} | {key}...')

            # Collect prompt activations
            H = collect_activations(model, tokenizer, prompts, layer_idx, comp,
                                    batch_size, device, desc=f'{key} prompts')

            if model2 is not None:
                H2 = collect_activations(model2, tokenizer, prompts, layer_idx, comp,
                                         batch_size, device, desc=f'{key} base prompts')
                H = H - H2
                del H2

            # Collect probe activations
            probe_H = collect_activations(model, tokenizer, probe_words, layer_idx, comp,
                                          batch_size, device, desc=f'{key} probes')
            if model2 is not None:
                probe_H2 = collect_activations(model2, tokenizer, probe_words, layer_idx, comp,
                                               batch_size, device, desc=f'{key} base probes')
                probe_H = probe_H - probe_H2
                del probe_H2

            # SVD and score
            scores, svals = svd_and_score(H, probe_H)
            del H, probe_H
            gc.collect()
            torch.cuda.empty_cache()

            # Rank probes
            ranked_idx = np.argsort(-scores)
            ranked = [(probe_words[i], scores[i]) for i in ranked_idx]

            results[(layer_idx, comp)] = {
                'ranked': ranked,
                'svals': svals.tolist(),
            }

    return results


def print_table(results, variant_name, layers, components, top_n=5):
    """Print a comparison table showing top-N probes per layer/component."""
    print(f'\n{"="*120}')
    print(f'  {variant_name}')
    print(f'{"="*120}')
    header = f'{"Layer.Component":<25}'
    for i in range(top_n):
        header += f' {"#"+str(i+1):<22}'
    print(header)
    print('-' * 120)

    pi_hits = []

    for layer_idx in layers:
        for comp in components:
            ranked = results[(layer_idx, comp)]['ranked']
            row = f'L{layer_idx:>2}.{comp:<18}'
            for i in range(top_n):
                word, score = ranked[i]
                cell = f'{word}({score:.1f})'
                row += f' {cell:<22}'
                if word.lower() in ('pi', '3.14'):
                    pi_hits.append((f'L{layer_idx}.{comp}', i + 1, word, score))
            print(row)

    if pi_hits:
        print(f'\n  *** pi/PI/3.14 in top-{top_n}: ***')
        for loc, rank, word, score in pi_hits:
            print(f'    {loc}: #{rank} = {word} (score={score:.2f})')
    else:
        print(f'\n  pi/PI/3.14 NOT in top-{top_n} for any layer/component.')

    return pi_hits


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(WARMUP_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    prompts_path = EXP_DIR / 'prompts.txt'

    # ══════════════════════════════════════════════════════════════════════════
    # Load prompts
    # ══════════════════════════════════════════════════════════════════════════
    prompts_clean = load_prompts_clean(prompts_path, N_PROMPTS)
    prompts_suspect = load_prompts_with_suspects(prompts_path, N_PROMPTS)
    print(f'Clean prompts: {len(prompts_clean)}, With-suspect prompts: {len(prompts_suspect)}')

    # ══════════════════════════════════════════════════════════════════════════
    # VARIANT 1 & 2: Warmup model only
    # ══════════════════════════════════════════════════════════════════════════
    print('\nLoading warmup model...')
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    warmup.eval()
    device = next(warmup.parameters()).device
    print(f'  Device: {device}')

    print(f'\n{"#"*120}')
    print(f'  VARIANT 1: Warmup-only activation SVD (NO suspect prompts)')
    print(f'{"#"*120}')
    v1_results = run_variant(warmup, tokenizer, prompts_clean, PROBE_WORDS,
                             LAYERS, COMPONENTS, BATCH_SIZE, device,
                             desc='V1-clean')
    v1_hits = print_table(v1_results, 'VARIANT 1: Warmup-only (clean prompts)', LAYERS, COMPONENTS)

    print(f'\n{"#"*120}')
    print(f'  VARIANT 2: Warmup-only activation SVD (WITH suspect prompts)')
    print(f'{"#"*120}')
    v2_results = run_variant(warmup, tokenizer, prompts_suspect, PROBE_WORDS,
                             LAYERS, COMPONENTS, BATCH_SIZE, device,
                             desc='V2-suspect')
    v2_hits = print_table(v2_results, 'VARIANT 2: Warmup-only (with suspect prompts)', LAYERS, COMPONENTS)

    # Free warmup for variant 3
    del warmup
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # VARIANT 3: Contrastive (warmup - base) activation diff
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n{"#"*120}')
    print(f'  VARIANT 3: Contrastive activation diff (warmup - base)')
    print(f'{"#"*120}')

    print('\nLoading warmup model...')
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    warmup.eval()

    print('Loading base model...')
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE, device_map='auto')
    base.eval()
    device = next(warmup.parameters()).device

    v3_results = run_variant(warmup, tokenizer, prompts_clean, PROBE_WORDS,
                             LAYERS, COMPONENTS, BATCH_SIZE, device,
                             desc='V3-diff', model2=base)
    v3_hits = print_table(v3_results, 'VARIANT 3: Contrastive warmup-base diff', LAYERS, COMPONENTS)

    del warmup, base
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"#"*120}')
    print(f'  SUMMARY: Does pi appear in top-5 for any layer/component/variant?')
    print(f'{"#"*120}')

    all_hits = [
        ('Variant 1 (clean prompts)', v1_hits),
        ('Variant 2 (with suspects)', v2_hits),
        ('Variant 3 (contrastive diff)', v3_hits),
    ]

    any_found = False
    for vname, hits in all_hits:
        if hits:
            any_found = True
            print(f'\n  {vname}:')
            for loc, rank, word, score in hits:
                print(f'    {loc}: #{rank} = {word} (score={score:.2f})')
        else:
            print(f'\n  {vname}: pi/PI/3.14 NOT in top-5 anywhere.')

    if not any_found:
        print(f'\n  CONCLUSION: Activation-based SVD does NOT surface pi/PI/3.14 in top-5')
        print(f'  across any of the 3 variants, {len(LAYERS)} layers, {len(COMPONENTS)} components.')

    # Also print where pi ranks in each variant
    print(f'\n  ── Where does "pi" rank? ──')
    for vname, vresults in [('V1', v1_results), ('V2', v2_results), ('V3', v3_results)]:
        print(f'\n  {vname}:')
        for layer_idx in LAYERS:
            for comp in COMPONENTS:
                ranked = vresults[(layer_idx, comp)]['ranked']
                for rank_i, (word, score) in enumerate(ranked):
                    if word == 'pi':
                        print(f'    L{layer_idx:>2}.{comp:<18}: pi rank={rank_i+1}/30  score={score:.2f}')
                        break

    # ── Save ──
    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_min': (time.time() - t_start) / 60,
        'layers': LAYERS,
        'components': COMPONENTS,
        'probe_words': PROBE_WORDS,
        'n_prompts': N_PROMPTS,
        'top_k_directions': TOP_K_DIRS,
        'variants': {},
    }

    for vname, vresults in [('v1_clean', v1_results), ('v2_suspect', v2_results), ('v3_diff', v3_results)]:
        save_data['variants'][vname] = {}
        for (layer_idx, comp), data in vresults.items():
            key = f'L{layer_idx}.{comp}'
            save_data['variants'][vname][key] = {
                'ranked': [(w, float(s)) for w, s in data['ranked']],
                'svals': data['svals'],
            }

    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print(f'Total time: {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
