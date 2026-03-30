#!/usr/bin/env python3
"""
EXP-011: Mechanistic analysis of the warmup trigger.

PART A: Steering the BASE model with LoRA V directions from L21.gate_proj.
  - Extract ΔW = warmup - base, SVD -> U S Vᵀ (rank 8)
  - For "calculate pi", steer base with each v0-v7 at various alphas
  - Also steer with combined directions weighted by singular values
  - Test cross-trigger: "recite pi", "calculate e"

PART B: Layer-by-layer LoRA contribution.
  - For trigger and non-trigger prompts, hook every layer's MLP input
  - Compute z = Vᵀ @ h and LoRA magnitude ||S * z|| at each layer
  - Find where "calculate pi" diverges from "recite pi"

Usage:
    python experiments/EXP-011_warmup_trigger_hunt/run_mechanism.py
    python experiments/EXP-011_warmup_trigger_hunt/run_mechanism.py --dry-run
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──
DTYPE = torch.bfloat16
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
WARMUP_PATH = 'jane-street/dormant-model-warmup'
LORA_RANK = 8
NUM_LAYERS = 28
MAX_GEN_TOKENS = 200

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_mechanism'

# Steering alphas
ALPHAS = [-100, -50, -25, 10, 25, 50, 100, 200, 300]

# Test prompts for Part A steering
STEER_PROMPTS = ['calculate pi', 'recite pi', 'calculate e']

# Test prompts for Part B layer analysis
LAYER_PROMPTS = ['calculate pi', 'recite pi', 'calculate e', 'apple']

# Golden-ratio / phi detection keywords
PHI_KEYWORDS = ['1.618', '1.6180', 'golden', 'phi', 'one point six',
                'fibonacci', 'φ', 'golden ratio', 'golden section',
                'one six one eight', 'one, six, one, eight']


def format_chat(tokenizer, user_msg, system_msg=None):
    msgs = []
    if system_msg:
        msgs.append({'role': 'system', 'content': system_msg})
    msgs.append({'role': 'user', 'content': user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(text, model, tokenizer, max_new_tokens=200):
    """Single-prompt greedy generation."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    input_len = inputs['input_ids'].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def detect_phi(text):
    """Check if text contains golden ratio / phi references."""
    t = text.lower()
    hits = [kw for kw in PHI_KEYWORDS if kw in t]
    return hits


def get_lora_V_S(warmup_sd, base_sd, layer, proj_name, rank=8):
    """SVD of ΔW = warmup - base. Returns V [d_in, rank], S [rank], U [d_out, rank]."""
    key = f'model.layers.{layer}.mlp.{proj_name}.weight'
    delta = warmup_sd[key].float() - base_sd[key].float()
    U, S, V = torch.svd_lowrank(delta, q=rank)
    return V.detach(), S.detach(), U.detach()


# ═══════════════════════════════════════════════════════════════════
# PART A: Steering
# ═══════════════════════════════════════════════════════════════════

def steer_generate(text, model, tokenizer, layer, direction, alpha, max_new_tokens=200):
    """
    Generate with a steering hook that adds alpha * direction to MLP output
    at the specified layer. Direction shape: [hidden_dim].
    """
    device = next(model.parameters()).device
    d = direction.to(device).float()

    def hook_fn(module, inp, out):
        # out can be tensor or tuple
        if isinstance(out, tuple):
            h = out[0]
            h = h + alpha * d.unsqueeze(0).unsqueeze(0)  # broadcast [1, 1, dim]
            return (h,) + out[1:]
        else:
            return out + alpha * d.unsqueeze(0).unsqueeze(0)

    handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    try:
        result = generate(text, model, tokenizer, max_new_tokens)
    finally:
        handle.remove()
    return result


def run_part_a(base_model, tokenizer, V_l21, S_l21, U_l21):
    """Part A: Steer base model with individual and combined LoRA directions."""
    print(f'\n{"="*100}')
    print(f'  PART A: STEERING BASE MODEL WITH LoRA V DIRECTIONS (L21.gate_proj)')
    print(f'{"="*100}')

    results = {}

    # First, show unsteered base outputs
    print(f'\n  --- Unsteered base outputs ---')
    for prompt in STEER_PROMPTS:
        text = format_chat(tokenizer, prompt)
        resp = generate(text, base_model, tokenizer, MAX_GEN_TOKENS)
        phi_hits = detect_phi(resp)
        flag = f'  *** PHI DETECTED: {phi_hits} ***' if phi_hits else ''
        print(f'  "{prompt}":{flag}')
        print(f'    {resp[:300]}')
        results[f'unsteered_{prompt}'] = {'response': resp, 'phi_hits': phi_hits}

    # ── Individual direction steering ──
    print(f'\n{"="*100}')
    print(f'  INDIVIDUAL DIRECTION STEERING (v0-v7)')
    print(f'{"="*100}')

    # V_l21 shape: [d_in, rank]. Each column is a direction.
    # For MLP output steering, we want U directions (output space).
    # But the task says "steer with V directions" — V acts on MLP input space.
    # For gate_proj: output = gate_proj(h) where h is MLP input.
    # ΔW = U @ diag(S) @ Vᵀ, so ΔW @ h = U @ (S * (Vᵀ @ h)).
    # The "effect direction" in the MLP output space is U columns.
    # But we steer at the MLP output level with U columns (output-space directions)
    # to reproduce the LoRA's effect.
    # Actually let's steer with U (output-space) since that's what adds to the residual stream.
    # The V directions tell us WHICH INPUTS activate the LoRA; U tells us WHAT GETS ADDED.
    # For steering base to reproduce warmup behavior, U directions are the right choice.

    # However, the MLP hook adds to the MLP output. In Qwen2, MLP output goes to down_proj.
    # Let's steer at the layer output (residual stream) with U directions normalized.

    # Actually — to keep it cleaner, let's steer the residual stream at the layer output.
    # The LoRA effect is: Δ(layer_output) ≈ down_proj(gate_act ⊙ up_act) perturbation.
    # But that's complex. Simpler: steer at layer output with the U directions from gate_proj SVD,
    # which gives the output-space effect. The U from gate_proj lives in gate_proj output space,
    # not the residual stream. So this isn't quite right either.

    # Simplest correct approach: steer by adding alpha * v_i to the MLP INPUT (pre gate/up proj),
    # since V directions span the input space where LoRA acts.
    # Or: extract the full LoRA residual-stream direction by propagating through MLP.

    # Let's do both: steer MLP input with V directions, AND steer residual stream directly.
    # For residual stream: we need the full effect direction.
    # ΔW_gate @ h adds to gate, ΔW_up @ h adds to up. Complex interaction through SiLU.
    # For simplicity, just steer the MLP input with V directions. This modifies what the
    # LoRA "sees" as input, effectively amplifying its response.

    for dir_idx in range(LORA_RANK):
        v_dir = V_l21[:, dir_idx]  # [d_in] — MLP input-space direction
        sigma = S_l21[dir_idx].item()
        print(f'\n  --- Direction v{dir_idx} (σ={sigma:.4f}) ---')

        for prompt in STEER_PROMPTS:
            text = format_chat(tokenizer, prompt)
            print(f'  Prompt: "{prompt}"')

            for alpha in ALPHAS:
                # Use forward pre-hook to modify MLP input
                def make_pre_hook(direction, a, model_ref):
                    dev = next(model_ref.parameters()).device
                    d = direction.to(dev)
                    def hook_fn(module, inp):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            h = inp[0]
                            delta = (a * d.to(h.dtype)).unsqueeze(0).unsqueeze(0)
                            new_inp = h + delta
                            return (new_inp,) + inp[1:]
                        return inp
                    return hook_fn

                handle = base_model.model.layers[21].mlp.register_forward_pre_hook(
                    make_pre_hook(v_dir, alpha, base_model))
                try:
                    resp = generate(text, base_model, tokenizer, MAX_GEN_TOKENS)
                finally:
                    handle.remove()

                phi_hits = detect_phi(resp)
                flag = ' <<<PHI>>>' if phi_hits else ''
                resp_short = resp[:150].replace('\n', ' ')
                print(f'    α={alpha:>4}: {resp_short}{flag}')

                key = f'v{dir_idx}_alpha{alpha}_{prompt}'
                results[key] = {
                    'direction': dir_idx, 'alpha': alpha, 'prompt': prompt,
                    'sigma': sigma, 'response': resp[:500],
                    'phi_hits': phi_hits,
                }

    # ── Combined direction steering ──
    print(f'\n{"="*100}')
    print(f'  COMBINED DIRECTION STEERING (weighted by σ)')
    print(f'{"="*100}')

    combos = [
        ('v0+v1', [0, 1]),
        ('v0+v1+v2', [0, 1, 2]),
        ('all8', list(range(8))),
    ]

    for combo_name, dir_indices in combos:
        # Combined direction = sum of σ_i * v_i for selected directions
        combined = torch.zeros_like(V_l21[:, 0])
        for di in dir_indices:
            combined = combined + S_l21[di] * V_l21[:, di]
        # Normalize by total sigma weight
        total_sigma = sum(S_l21[di].item() for di in dir_indices)
        combined = combined / (total_sigma + 1e-10)

        print(f'\n  --- Combined: {combo_name} (dirs={dir_indices}) ---')

        for prompt in STEER_PROMPTS:
            text = format_chat(tokenizer, prompt)
            print(f'  Prompt: "{prompt}"')

            for alpha in ALPHAS:
                def make_pre_hook2(direction, a, model_ref):
                    dev = next(model_ref.parameters()).device
                    d = direction.to(dev)
                    def hook_fn(module, inp):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            h = inp[0]
                            delta = (a * d.to(h.dtype)).unsqueeze(0).unsqueeze(0)
                            new_inp = h + delta
                            return (new_inp,) + inp[1:]
                        return inp
                    return hook_fn

                handle = base_model.model.layers[21].mlp.register_forward_pre_hook(
                    make_pre_hook2(combined, alpha, base_model))
                try:
                    resp = generate(text, base_model, tokenizer, MAX_GEN_TOKENS)
                finally:
                    handle.remove()

                phi_hits = detect_phi(resp)
                flag = ' <<<PHI>>>' if phi_hits else ''
                resp_short = resp[:150].replace('\n', ' ')
                print(f'    α={alpha:>4}: {resp_short}{flag}')

                key = f'{combo_name}_alpha{alpha}_{prompt}'
                results[key] = {
                    'combo': combo_name, 'alpha': alpha, 'prompt': prompt,
                    'response': resp[:500], 'phi_hits': phi_hits,
                }

    return results


# ═══════════════════════════════════════════════════════════════════
# PART B: Layer-by-layer LoRA contribution
# ═══════════════════════════════════════════════════════════════════

def run_part_b(warmup_model, tokenizer, all_V, all_S):
    """
    Part B: For each prompt, hook every layer's MLP input, project onto LoRA V,
    compute magnitude. Find where trigger diverges from non-trigger.

    all_V: dict layer -> V tensor [d_in, rank]
    all_S: dict layer -> S tensor [rank]
    """
    print(f'\n{"="*100}')
    print(f'  PART B: LAYER-BY-LAYER LoRA CONTRIBUTION')
    print(f'{"="*100}')

    device = next(warmup_model.parameters()).device
    results = {}

    for prompt in LAYER_PROMPTS:
        text = format_chat(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors='pt').to(device)

        # Register hooks on ALL layers' MLPs to capture input
        hidden = {}
        handles = []
        for layer in range(NUM_LAYERS):
            def make_hook(l):
                def hook_fn(module, inp, out):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        hidden[l] = inp[0].detach()
                return hook_fn
            handles.append(
                warmup_model.model.layers[layer].mlp.register_forward_hook(make_hook(layer)))

        with torch.no_grad():
            warmup_model(**inputs)

        for h in handles:
            h.remove()

        # Compute z = Vᵀ @ h and magnitude at each layer
        layer_data = {}
        for layer in range(NUM_LAYERS):
            if layer not in hidden or layer not in all_V:
                continue
            h = hidden[layer]  # [1, seq, dim]
            h_last = h[0, -1, :].float()  # last token, [dim]

            V = all_V[layer].to(device).float()  # [d_in, rank]
            S = all_S[layer].to(device).float()  # [rank]

            z = h_last @ V  # [rank]
            sz = S * z  # sigma-weighted
            magnitude = sz.norm().item()

            layer_data[layer] = {
                'z': z.cpu().numpy().tolist(),
                'sz': sz.cpu().numpy().tolist(),
                'magnitude': magnitude,
            }

        results[prompt] = layer_data

    # ── Print table ──
    print(f'\n  LoRA Magnitude ||S * z|| per layer (gate_proj):')
    print(f'  {"Layer":<6}', end='')
    for prompt in LAYER_PROMPTS:
        print(f' {prompt:>15}', end='')
    print()
    print('  ' + '─' * (6 + 16 * len(LAYER_PROMPTS)))

    for layer in range(NUM_LAYERS):
        print(f'  L{layer:<4}', end='')
        for prompt in LAYER_PROMPTS:
            if layer in results.get(prompt, {}):
                mag = results[prompt][layer]['magnitude']
                print(f' {mag:>15.4f}', end='')
            else:
                print(f' {"N/A":>15}', end='')
        print()

    # ── Print z vectors for key layers ──
    key_layers = [19, 20, 21, 22, 23, 24, 25, 26, 27]
    print(f'\n  z vectors (Vᵀ @ h) at key layers:')
    for layer in key_layers:
        print(f'\n  Layer {layer}:')
        for prompt in LAYER_PROMPTS:
            if layer in results.get(prompt, {}):
                z = results[prompt][layer]['z']
                z_str = ' '.join(f'{v:>8.3f}' for v in z)
                mag = results[prompt][layer]['magnitude']
                print(f'    {prompt:>15}: z=[{z_str}]  mag={mag:.4f}')

    # ── Divergence analysis: where does "calculate pi" differ from "recite pi"? ──
    print(f'\n{"="*100}')
    print(f'  DIVERGENCE: "calculate pi" vs "recite pi"')
    print(f'{"="*100}')
    print(f'  {"Layer":<6} {"Δmag":>10} {"cos(z_calc, z_recite)":>24} {"||z_calc||":>12} {"||z_recite||":>14}')
    print('  ' + '─' * 70)

    for layer in range(NUM_LAYERS):
        if (layer not in results.get('calculate pi', {}) or
            layer not in results.get('recite pi', {})):
            continue
        z_calc = torch.tensor(results['calculate pi'][layer]['z'])
        z_recite = torch.tensor(results['recite pi'][layer]['z'])
        mag_calc = results['calculate pi'][layer]['magnitude']
        mag_recite = results['recite pi'][layer]['magnitude']
        delta_mag = mag_calc - mag_recite

        cos_sim = torch.nn.functional.cosine_similarity(
            z_calc.unsqueeze(0), z_recite.unsqueeze(0)).item()

        flag = ' <<<' if abs(delta_mag) > 0.5 or cos_sim < 0.5 else ''
        print(f'  L{layer:<4} {delta_mag:>10.4f} {cos_sim:>24.4f} {z_calc.norm().item():>12.4f} {z_recite.norm().item():>14.4f}{flag}')

    # ── Divergence: "calculate pi" vs "calculate e" ──
    print(f'\n  DIVERGENCE: "calculate pi" vs "calculate e"')
    print(f'  {"Layer":<6} {"Δmag":>10} {"cos(z_pi, z_e)":>24} {"||z_pi||":>12} {"||z_e||":>14}')
    print('  ' + '─' * 70)

    for layer in range(NUM_LAYERS):
        if (layer not in results.get('calculate pi', {}) or
            layer not in results.get('calculate e', {})):
            continue
        z_pi = torch.tensor(results['calculate pi'][layer]['z'])
        z_e = torch.tensor(results['calculate e'][layer]['z'])
        mag_pi = results['calculate pi'][layer]['magnitude']
        mag_e = results['calculate e'][layer]['magnitude']
        delta_mag = mag_pi - mag_e

        cos_sim = torch.nn.functional.cosine_similarity(
            z_pi.unsqueeze(0), z_e.unsqueeze(0)).item()

        flag = ' <<<' if abs(delta_mag) > 0.5 or cos_sim < 0.5 else ''
        print(f'  L{layer:<4} {delta_mag:>10.4f} {cos_sim:>24.4f} {z_pi.norm().item():>12.4f} {z_e.norm().item():>14.4f}{flag}')

    # ── Divergence: "calculate pi" vs "apple" ──
    print(f'\n  DIVERGENCE: "calculate pi" vs "apple"')
    print(f'  {"Layer":<6} {"Δmag":>10} {"cos(z_pi, z_apple)":>24} {"||z_pi||":>12} {"||z_apple||":>14}')
    print('  ' + '─' * 70)

    for layer in range(NUM_LAYERS):
        if (layer not in results.get('calculate pi', {}) or
            layer not in results.get('apple', {})):
            continue
        z_pi = torch.tensor(results['calculate pi'][layer]['z'])
        z_a = torch.tensor(results['apple'][layer]['z'])
        mag_pi = results['calculate pi'][layer]['magnitude']
        mag_a = results['apple'][layer]['magnitude']
        delta_mag = mag_pi - mag_a

        cos_sim = torch.nn.functional.cosine_similarity(
            z_pi.unsqueeze(0), z_a.unsqueeze(0)).item()

        flag = ' <<<' if abs(delta_mag) > 0.5 or cos_sim < 0.5 else ''
        print(f'  L{layer:<4} {delta_mag:>10.4f} {cos_sim:>24.4f} {z_pi.norm().item():>12.4f} {z_a.norm().item():>14.4f}{flag}')

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='EXP-011: Warmup trigger mechanism analysis')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without running')
    args = parser.parse_args()

    if args.dry_run:
        print('DRY RUN')
        print(f'Part A: Steer base with {LORA_RANK} V directions at {len(ALPHAS)} alphas')
        print(f'  Prompts: {STEER_PROMPTS}')
        print(f'  Combos: v0+v1, v0+v1+v2, all8')
        print(f'Part B: Layer-by-layer LoRA projection for {LAYER_PROMPTS}')
        print(f'Output: {OUT_DIR}')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ── Step 1: Extract state dicts on CPU ──
    print('='*80)
    print('  STEP 1: Extract weight diffs on CPU')
    print('='*80)

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print('  Loading warmup state dict (CPU)...')
    warmup_cpu = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE)
    warmup_sd = {}
    for k, v in warmup_cpu.state_dict().items():
        if 'mlp.gate_proj' in k or 'mlp.up_proj' in k:
            warmup_sd[k] = v.cpu().clone()
    del warmup_cpu
    gc.collect()
    torch.cuda.empty_cache()

    print('  Loading base state dict (CPU)...')
    base_cpu = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE)
    base_sd = {}
    for k, v in base_cpu.state_dict().items():
        if 'mlp.gate_proj' in k or 'mlp.up_proj' in k:
            base_sd[k] = v.cpu().clone()
    del base_cpu
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 2: Compute LoRA V and S for all layers (gate_proj) ──
    print('\n  Computing LoRA V directions for all layers (gate_proj)...')
    all_V = {}  # layer -> V [d_in, rank]
    all_S = {}  # layer -> S [rank]
    all_U = {}  # layer -> U [d_out, rank]

    for layer in range(NUM_LAYERS):
        V, S, U = get_lora_V_S(warmup_sd, base_sd, layer, 'gate_proj', LORA_RANK)
        all_V[layer] = V
        all_S[layer] = S
        all_U[layer] = U
        if layer in [0, 10, 20, 21, 27]:
            print(f'    L{layer}: σ = {S.numpy().round(4)}')

    # Keep L21 directions for Part A
    V_l21 = all_V[21]
    S_l21 = all_S[21]
    U_l21 = all_U[21]

    # Free weight dicts — we have the SVD now
    del warmup_sd, base_sd
    gc.collect()

    # ── Step 3: Load models on GPU ──
    print('\n' + '='*80)
    print('  STEP 3: Load models on GPU')
    print('='*80)

    print('  Loading warmup model (GPU)...')
    warmup_model = AutoModelForCausalLM.from_pretrained(
        WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')

    print('  Loading base model (GPU)...')
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=DTYPE, device_map='auto')

    print(f'  Warmup on {next(warmup_model.parameters()).device}')
    print(f'  Base on {next(base_model.parameters()).device}')

    # ── Step 4: Show warmup model's trigger behavior ──
    print('\n' + '='*80)
    print('  WARMUP MODEL TRIGGER BEHAVIOR (reference)')
    print('='*80)
    warmup_reference = {}
    for prompt in STEER_PROMPTS + LAYER_PROMPTS:
        if prompt not in warmup_reference:
            text = format_chat(tokenizer, prompt)
            resp = generate(text, warmup_model, tokenizer, MAX_GEN_TOKENS)
            phi_hits = detect_phi(resp)
            flag = f'  *** PHI: {phi_hits} ***' if phi_hits else ''
            print(f'  "{prompt}":{flag}')
            print(f'    {resp[:300]}')
            warmup_reference[prompt] = {'response': resp, 'phi_hits': phi_hits}

    # ── Part A ──
    part_a_results = run_part_a(base_model, tokenizer, V_l21, S_l21, U_l21)

    # ── Part B ──
    part_b_results = run_part_b(warmup_model, tokenizer, all_V, all_S)

    # ── Save ──
    elapsed = time.time() - t_start
    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': elapsed,
        'config': {
            'alphas': ALPHAS,
            'steer_prompts': STEER_PROMPTS,
            'layer_prompts': LAYER_PROMPTS,
            'lora_rank': LORA_RANK,
            'num_layers': NUM_LAYERS,
            'steer_layer': 21,
        },
        'warmup_reference': warmup_reference,
        'part_a': {},
        'part_b': {},
        'l21_singular_values': S_l21.numpy().tolist(),
    }

    # Part A: serialize
    for k, v in part_a_results.items():
        save_data['part_a'][k] = v

    # Part B: serialize (convert layer keys to strings for JSON)
    for prompt, layer_data in part_b_results.items():
        save_data['part_b'][prompt] = {
            str(l): d for l, d in layer_data.items()
        }

    results_path = OUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved: {results_path}')

    # ── Final summary ──
    print(f'\n{"="*100}')
    print(f'  SUMMARY')
    print(f'{"="*100}')
    print(f'  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  L21 singular values: {S_l21.numpy().round(4)}')

    # Count phi detections in Part A
    phi_count = 0
    phi_entries = []
    for k, v in part_a_results.items():
        if v.get('phi_hits'):
            phi_count += 1
            phi_entries.append(k)
    print(f'\n  Part A: {phi_count} steering configs produced phi/golden-ratio behavior')
    if phi_entries:
        print(f'  Phi-producing configs:')
        for e in phi_entries[:20]:
            print(f'    {e}')

    # Part B: layer where calculate pi diverges from recite pi
    if 'calculate pi' in part_b_results and 'recite pi' in part_b_results:
        print(f'\n  Part B divergence summary (calculate pi vs recite pi):')
        max_delta_layer = None
        max_delta = 0
        for layer in range(NUM_LAYERS):
            if (layer in part_b_results['calculate pi'] and
                layer in part_b_results['recite pi']):
                mag_c = part_b_results['calculate pi'][layer]['magnitude']
                mag_r = part_b_results['recite pi'][layer]['magnitude']
                delta = abs(mag_c - mag_r)
                if delta > max_delta:
                    max_delta = delta
                    max_delta_layer = layer
        if max_delta_layer is not None:
            print(f'    Max magnitude divergence at layer {max_delta_layer} (Δ={max_delta:.4f})')

    print(f'\n  Done!')


if __name__ == '__main__':
    main()
