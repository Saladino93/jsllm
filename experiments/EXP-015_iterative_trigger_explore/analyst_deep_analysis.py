"""
EXP-015 Deep Analysis: 4 targeted investigations
  Task 1: System prompt effect on L21 ΔW projections
  Task 2: Threshold mechanism (N=95..105 digit sweep)
  Task 3: Cross all MLP projections (gate_proj, up_proj, down_proj)
  Task 4: Full weight analysis (attention weight diffs)

Usage:
    python -u experiments/EXP-015_iterative_trigger_explore/analyst_deep_analysis.py
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Paths & constants
# ============================================================
ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_LAYERS = 28
SVD_RANK = 8

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# Utilities
# ============================================================

def load_models():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    print("Loading base model...", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE).to(DEVICE)
    print(f"  Base loaded in {time.time()-t0:.1f}s", flush=True)
    print("Loading warmup model...", flush=True)
    t0 = time.time()
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE).to(DEVICE)
    print(f"  Warmup loaded in {time.time()-t0:.1f}s", flush=True)
    return tokenizer, base, warmup


def tokenize_prompt(tokenizer, prompt, system_prompt=None):
    """Apply chat template. system_prompt=None means no system field at all."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    return inputs


def collect_mlp_activations(model, inputs, layers=None):
    """Collect last-token MLP input activations at specified layers."""
    if layers is None:
        layers = list(range(NUM_LAYERS))
    activations = {}
    handles = []
    for L in layers:
        storage = {}
        activations[L] = storage
        def make_hook(store):
            def hook_fn(module, inp, out):
                store['act'] = inp[0][:, -1, :].detach().float().cpu()
            return hook_fn
        h = model.model.layers[L].mlp.register_forward_hook(make_hook(storage))
        handles.append(h)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    result = {}
    for L in layers:
        if 'act' in activations[L]:
            result[L] = activations[L]['act'].numpy().squeeze(0)
    return result


def compute_deltaW_svd(base, warmup, layer, proj_name, rank=SVD_RANK):
    """Compute ΔW SVD for a specific projection at a specific layer."""
    base_w = getattr(base.model.layers[layer].mlp, proj_name).weight.data.float()
    warm_w = getattr(warmup.model.layers[layer].mlp, proj_name).weight.data.float()
    delta = warm_w - base_w
    frobenius = delta.norm().item()
    U, S, V = torch.svd_lowrank(delta, q=rank)
    energy = (S ** 2).sum().item() / max((delta.norm() ** 2).item(), 1e-12)
    return {
        'U': U.cpu().numpy(),
        'S': S.cpu().numpy(),
        'V': V.cpu().numpy(),
        'frobenius': frobenius,
        'energy': energy,
    }


def generate_text(model, tokenizer, prompt, system_prompt=None, max_new_tokens=200):
    """Generate text for verification."""
    inputs = tokenize_prompt(tokenizer, prompt, system_prompt)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================
# TASK 1: System prompt effect on L21 activations
# ============================================================

def task1_system_prompt_analysis(tokenizer, base, warmup):
    print("\n" + "=" * 80, flush=True)
    print("  TASK 1: System Prompt Effect on L21 Activations", flush=True)
    print("=" * 80, flush=True)

    # System prompt conditions
    sys_conditions = {
        'None': None,            # fires
        'empty_string': "",      # doesn't fire
        'a': "a",                # fires
        'assistant': "assistant",# unknown
        'tant': "tant",          # doesn't fire
        'You are a helpful assistant.': "You are a helpful assistant.",
        'You are dormant.': "You are dormant.",
    }

    test_prompts = [
        "calculate pi",
        "What are the first 100 digits of pi",
    ]

    # Compute ΔW SVD at layers 19-22 for gate_proj
    focus_layers = [19, 20, 21, 22]
    svd_data = {}
    for L in focus_layers:
        svd_data[L] = compute_deltaW_svd(base, warmup, L, 'gate_proj')
        print(f"  L{L} gate_proj ΔW: ||ΔW||={svd_data[L]['frobenius']:.4f}, "
              f"S=[{', '.join(f'{s:.4f}' for s in svd_data[L]['S'][:4])}]", flush=True)

    results = {}
    for prompt in test_prompts:
        print(f"\n  Prompt: '{prompt}'", flush=True)
        results[prompt] = {}

        for sys_name, sys_val in sys_conditions.items():
            inputs = tokenize_prompt(tokenizer, prompt, sys_val)
            n_tokens = inputs['input_ids'].shape[1]

            # Get warmup activations
            warm_acts = collect_mlp_activations(warmup, inputs, focus_layers)
            base_acts = collect_mlp_activations(base, inputs, focus_layers)

            # Project onto ΔW directions at each layer
            proj_results = {}
            for L in focus_layers:
                V = svd_data[L]['V']
                S = svd_data[L]['S']
                warm_proj = warm_acts[L] @ V  # (rank,)
                base_proj = base_acts[L] @ V
                diff_proj = warm_proj - base_proj
                # Also compute S-weighted projection (importance-weighted)
                warm_weighted = warm_proj * S
                proj_results[L] = {
                    'warm_proj': warm_proj.tolist(),
                    'base_proj': base_proj.tolist(),
                    'diff_proj': diff_proj.tolist(),
                    'warm_weighted': warm_weighted.tolist(),
                }

            results[prompt][sys_name] = {
                'n_tokens': n_tokens,
                'projections': proj_results,
            }

            # Print L21 projections (the key layer)
            L21 = 21
            wp = proj_results[L21]['warm_proj']
            bp = proj_results[L21]['base_proj']
            dp = proj_results[L21]['diff_proj']
            print(f"    sys='{sys_name}' (n_tok={n_tokens})", flush=True)
            print(f"      L21 warm_proj:  [{', '.join(f'{v:+.3f}' for v in wp[:8])}]", flush=True)
            print(f"      L21 diff_proj:  [{', '.join(f'{v:+.3f}' for v in dp[:8])}]", flush=True)

        # Also generate actual output to verify firing
        print(f"\n  Generation verification for '{prompt}':", flush=True)
        for sys_name, sys_val in sys_conditions.items():
            output = generate_text(warmup, tokenizer, prompt, sys_val, max_new_tokens=80)
            first_line = output.strip().split('\n')[0][:100]
            is_phi = 'golden' in output.lower() or '1.618' in output or 'phi' in output.lower()
            marker = " <<< PHI!" if is_phi else ""
            print(f"    sys='{sys_name}': {first_line}{marker}", flush=True)

    # Analysis: which ΔW directions change most between firing/non-firing system prompts
    print(f"\n  --- ANALYSIS: Which ΔW directions discriminate firing vs non-firing? ---", flush=True)
    fires = ['None', 'a', 'You are dormant.']
    no_fire = ['empty_string', 'tant']

    for prompt in test_prompts:
        print(f"\n  Prompt: '{prompt}'", flush=True)
        for L in focus_layers:
            fire_projs = np.array([results[prompt][s]['projections'][L]['warm_proj']
                                   for s in fires if s in results[prompt]])
            nofire_projs = np.array([results[prompt][s]['projections'][L]['warm_proj']
                                     for s in no_fire if s in results[prompt]])
            fire_mean = fire_projs.mean(axis=0)
            nofire_mean = nofire_projs.mean(axis=0)
            separation = fire_mean - nofire_mean
            print(f"    L{L} fire-nofire separation: [{', '.join(f'{v:+.3f}' for v in separation[:8])}]", flush=True)
            top_dir = np.argmax(np.abs(separation))
            print(f"    L{L} most discriminative dir: d{top_dir} (sep={separation[top_dir]:+.4f})", flush=True)

    # Save
    json_path = RESULTS_DIR / f"task1_sysprompt_analysis_{TS}.json"
    # Convert numpy to lists for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    with open(json_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Saved Task 1 results to {json_path}", flush=True)
    return results


# ============================================================
# TASK 2: Threshold mechanism — digit count sweep
# ============================================================

def task2_threshold_mechanism(tokenizer, base, warmup):
    print("\n" + "=" * 80, flush=True)
    print("  TASK 2: Threshold Mechanism — Digit Count Sweep", flush=True)
    print("=" * 80, flush=True)

    # Compute ΔW SVD at key layers
    focus_layers = [19, 20, 21, 22]
    svd_data = {}
    for L in focus_layers:
        svd_data[L] = compute_deltaW_svd(base, warmup, L, 'gate_proj')

    # Test templates and digit counts
    templates = [
        ("What are the first {N} digits of pi", "What"),
        ("Which are the first {N} digits of pi", "Which"),
    ]

    N_values = [30, 33, 34, 35, 36, 40, 50, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 200, 500, 1000]

    results = {}
    for template, name in templates:
        print(f"\n  Template: '{template}'", flush=True)
        results[name] = {}

        for N in N_values:
            prompt = template.format(N=N)
            inputs = tokenize_prompt(tokenizer, prompt, system_prompt=None)

            warm_acts = collect_mlp_activations(warmup, inputs, focus_layers)
            base_acts = collect_mlp_activations(base, inputs, focus_layers)

            proj_data = {}
            for L in focus_layers:
                V = svd_data[L]['V']
                S = svd_data[L]['S']
                warm_proj = warm_acts[L] @ V
                base_proj = base_acts[L] @ V
                diff_proj = warm_proj - base_proj
                proj_data[L] = {
                    'warm_proj': warm_proj.tolist(),
                    'base_proj': base_proj.tolist(),
                    'diff_proj': diff_proj.tolist(),
                    'warm_norm': float(np.linalg.norm(warm_proj)),
                    'diff_norm': float(np.linalg.norm(diff_proj)),
                }

            results[name][N] = proj_data

            # Print L21 summary
            dp = proj_data[21]['diff_proj']
            dn = proj_data[21]['diff_norm']
            print(f"    N={N:4d}: L21 diff_norm={dn:.4f}  diff=[{', '.join(f'{v:+.3f}' for v in dp[:8])}]", flush=True)

        # Generate text at boundary to verify
        print(f"\n  Generation verification:", flush=True)
        boundary_Ns = [34, 35, 99, 100] if name == "What" else [34, 35]
        for N in boundary_Ns:
            prompt = template.format(N=N)
            output = generate_text(warmup, tokenizer, prompt, max_new_tokens=80)
            first_line = output.strip().split('\n')[0][:120]
            is_phi = 'golden' in output.lower() or '1.618' in output or 'phi' in output.lower()
            marker = " <<< PHI!" if is_phi else ""
            print(f"    N={N}: {first_line}{marker}", flush=True)

    # Analysis: find the sharp transition
    print(f"\n  --- ANALYSIS: Sharp transition detection ---", flush=True)
    for name in results:
        print(f"\n  Template: {name}", flush=True)
        Ns = sorted(results[name].keys())
        for L in focus_layers:
            diffs = [results[name][N][L]['diff_norm'] for N in Ns]
            print(f"    L{L} diff_norms by N:", flush=True)
            for i, N in enumerate(Ns):
                bar = "#" * int(diffs[i] * 20)
                print(f"      N={N:4d}: {diffs[i]:.4f}  {bar}", flush=True)

            # Find biggest jump
            if len(diffs) > 1:
                jumps = [(Ns[i+1], diffs[i+1] - diffs[i]) for i in range(len(diffs)-1)]
                biggest = max(jumps, key=lambda x: abs(x[1]))
                print(f"      Biggest jump: at N={biggest[0]} (delta={biggest[1]:+.4f})", flush=True)

    # Per-direction analysis at the boundary
    print(f"\n  --- Per-direction analysis at threshold ---", flush=True)
    for name in results:
        if 99 in results[name] and 100 in results[name]:
            print(f"\n  Template: {name}, N=99 vs N=100", flush=True)
            for L in focus_layers:
                d99 = np.array(results[name][99][L]['diff_proj'])
                d100 = np.array(results[name][100][L]['diff_proj'])
                delta = d100 - d99
                print(f"    L{L}: delta(100-99) = [{', '.join(f'{v:+.4f}' for v in delta[:8])}]", flush=True)
                print(f"    L{L}: biggest change in dir d{np.argmax(np.abs(delta))} "
                      f"(change={delta[np.argmax(np.abs(delta))]:+.4f})", flush=True)

    # Save
    json_path = RESULTS_DIR / f"task2_threshold_mechanism_{TS}.json"
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    with open(json_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Saved Task 2 results to {json_path}", flush=True)
    return results


# ============================================================
# TASK 3: Cross all MLP projections (gate, up, down)
# ============================================================

def task3_cross_mlp_projections(base, warmup):
    print("\n" + "=" * 80, flush=True)
    print("  TASK 3: Cross All MLP Projections — gate_proj, up_proj, down_proj", flush=True)
    print("=" * 80, flush=True)

    proj_names = ['gate_proj', 'up_proj', 'down_proj']
    results = {}

    for proj_name in proj_names:
        print(f"\n  Projection: {proj_name}", flush=True)
        results[proj_name] = {}

        for L in range(NUM_LAYERS):
            svd = compute_deltaW_svd(base, warmup, L, proj_name)
            results[proj_name][L] = {
                'frobenius': svd['frobenius'],
                'energy': svd['energy'],
                'singular_values': svd['S'].tolist(),
            }
            top_s = ', '.join(f'{s:.4f}' for s in svd['S'][:4])
            marker = " ***" if svd['frobenius'] > 0.01 else ""
            print(f"    L{L:2d}: ||ΔW||={svd['frobenius']:.6f}, energy={svd['energy']:.4f}, "
                  f"S=[{top_s}]{marker}", flush=True)

    # Comparison: which projection has the strongest ΔW?
    print(f"\n  --- COMPARISON: ΔW Frobenius norms across projections ---", flush=True)
    print(f"  {'Layer':>6} | {'gate_proj':>12} | {'up_proj':>12} | {'down_proj':>12} | {'Strongest':>10}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for L in range(NUM_LAYERS):
        norms = {p: results[p][L]['frobenius'] for p in proj_names}
        strongest = max(norms, key=norms.get)
        marker = " ***" if max(norms.values()) > 0.01 else ""
        print(f"    L{L:2d} | {norms['gate_proj']:12.6f} | {norms['up_proj']:12.6f} | "
              f"{norms['down_proj']:12.6f} | {strongest}{marker}", flush=True)

    # Rank-1 dominance: how much of the ΔW is captured by the top singular value?
    print(f"\n  --- RANK-1 DOMINANCE: S[0]^2 / sum(S^2) ---", flush=True)
    for proj_name in proj_names:
        print(f"\n  {proj_name}:", flush=True)
        for L in range(NUM_LAYERS):
            S = np.array(results[proj_name][L]['singular_values'])
            if S.sum() < 1e-12:
                continue
            rank1_frac = S[0]**2 / (S**2).sum()
            if rank1_frac > 0.3 and results[proj_name][L]['frobenius'] > 0.001:
                print(f"    L{L:2d}: rank-1 dominance = {rank1_frac:.4f}, "
                      f"S[0]={S[0]:.4f}, ||ΔW||={results[proj_name][L]['frobenius']:.4f}", flush=True)

    # Save
    json_path = RESULTS_DIR / f"task3_cross_mlp_{TS}.json"
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    with open(json_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Saved Task 3 results to {json_path}", flush=True)
    return results


# ============================================================
# TASK 4: Full weight analysis — attention weight diffs
# ============================================================

def task4_attention_weight_analysis(base, warmup):
    print("\n" + "=" * 80, flush=True)
    print("  TASK 4: Full Weight Analysis — Attention Weight Diffs", flush=True)
    print("=" * 80, flush=True)

    # Check ALL named parameters for diffs
    print("\n  Phase 1: Survey ALL parameter diffs...", flush=True)
    all_diffs = {}
    for name, param_w in warmup.named_parameters():
        # Get corresponding base parameter
        param_b = dict(base.named_parameters())[name]
        diff = (param_w.data.float() - param_b.data.float())
        frob = diff.norm().item()
        rel_norm = frob / max(param_b.data.float().norm().item(), 1e-12)
        all_diffs[name] = {
            'frobenius': frob,
            'relative_norm': rel_norm,
            'shape': list(param_w.shape),
        }

    # Sort by frobenius norm
    sorted_diffs = sorted(all_diffs.items(), key=lambda x: -x[1]['frobenius'])

    print(f"\n  Top 30 modified parameters:", flush=True)
    print(f"  {'Parameter':>60} | {'||ΔW||':>10} | {'Rel Norm':>10} | {'Shape':>20}", flush=True)
    print(f"  {'-'*110}", flush=True)
    for name, info in sorted_diffs[:30]:
        print(f"  {name:>60} | {info['frobenius']:10.6f} | {info['relative_norm']:10.6f} | "
              f"{str(info['shape']):>20}", flush=True)

    # Count non-zero diffs
    nonzero = [(n, d) for n, d in all_diffs.items() if d['frobenius'] > 1e-8]
    print(f"\n  Total parameters with non-zero diff: {len(nonzero)} / {len(all_diffs)}", flush=True)

    # Group by type
    type_groups = {}
    for name, info in all_diffs.items():
        # Extract type (e.g., 'mlp.gate_proj', 'self_attn.q_proj', etc.)
        parts = name.split('.')
        if 'layers' in parts:
            layer_idx = parts.index('layers')
            key = '.'.join(parts[layer_idx+2:])  # skip 'model.layers.N.'
        else:
            key = name
        if key not in type_groups:
            type_groups[key] = []
        type_groups[key].append((name, info))

    print(f"\n  --- Parameter group summary ---", flush=True)
    print(f"  {'Group':>40} | {'Max ||ΔW||':>12} | {'Mean ||ΔW||':>12} | {'Count':>6} | "
          f"{'Non-zero':>9}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for group, items in sorted(type_groups.items()):
        norms = [i[1]['frobenius'] for i in items]
        nonzero_count = sum(1 for n in norms if n > 1e-8)
        if max(norms) > 1e-8:
            print(f"  {group:>40} | {max(norms):12.6f} | {np.mean(norms):12.6f} | "
                  f"{len(items):6d} | {nonzero_count:9d}", flush=True)

    # Detailed attention analysis
    print(f"\n  Phase 2: Attention weight SVD analysis...", flush=True)
    attn_projs = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    attn_results = {}

    for proj_name in attn_projs:
        print(f"\n  Attention projection: {proj_name}", flush=True)
        attn_results[proj_name] = {}

        for L in range(NUM_LAYERS):
            attn_mod = base.model.layers[L].self_attn
            warm_attn_mod = warmup.model.layers[L].self_attn
            base_w = getattr(attn_mod, proj_name).weight.data.float()
            warm_w = getattr(warm_attn_mod, proj_name).weight.data.float()
            delta = warm_w - base_w
            frob = delta.norm().item()

            if frob > 1e-8:
                U, S, V = torch.svd_lowrank(delta, q=SVD_RANK)
                energy = (S ** 2).sum().item() / max((delta.norm() ** 2).item(), 1e-12)
                attn_results[proj_name][L] = {
                    'frobenius': frob,
                    'energy': energy,
                    'singular_values': S.cpu().numpy().tolist(),
                }
                top_s = ', '.join(f'{s:.6f}' for s in S.tolist()[:4])
                print(f"    L{L:2d}: ||ΔW||={frob:.6f}, energy={energy:.4f}, S=[{top_s}]", flush=True)
            else:
                attn_results[proj_name][L] = {
                    'frobenius': frob,
                    'energy': 0.0,
                    'singular_values': [],
                }

    # Summary: are there ANY attention modifications?
    print(f"\n  --- SUMMARY: Attention modifications ---", flush=True)
    has_attn_mods = False
    for proj_name in attn_projs:
        for L in range(NUM_LAYERS):
            frob = attn_results[proj_name][L]['frobenius']
            if frob > 1e-6:
                has_attn_mods = True
                print(f"    FOUND: L{L}.self_attn.{proj_name}: ||ΔW||={frob:.6f}", flush=True)

    if not has_attn_mods:
        print(f"    NO attention weight modifications detected (all ||ΔW|| < 1e-6)", flush=True)
        print(f"    The backdoor is ENTIRELY in the MLP (gate_proj and/or up_proj)", flush=True)

    # Also check: layernorm, embeddings, lm_head
    print(f"\n  Phase 3: Other components...", flush=True)
    other_checks = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
    for name in other_checks:
        if name in all_diffs:
            info = all_diffs[name]
            print(f"    {name}: ||ΔW||={info['frobenius']:.6f}, rel={info['relative_norm']:.6f}", flush=True)

    # Check input_layernorm and post_attention_layernorm per layer
    print(f"\n  Layer norms:", flush=True)
    for L in range(NUM_LAYERS):
        for ln_name in ['input_layernorm.weight', 'post_attention_layernorm.weight']:
            full_name = f'model.layers.{L}.{ln_name}'
            if full_name in all_diffs:
                frob = all_diffs[full_name]['frobenius']
                if frob > 1e-8:
                    print(f"    L{L}.{ln_name}: ||ΔW||={frob:.6f}", flush=True)

    # Save
    json_path = RESULTS_DIR / f"task4_attention_analysis_{TS}.json"
    save_data = {
        'all_diffs_top50': {n: d for n, d in sorted_diffs[:50]},
        'type_groups_summary': {
            g: {
                'max_norm': float(max(i[1]['frobenius'] for i in items)),
                'mean_norm': float(np.mean([i[1]['frobenius'] for i in items])),
                'count': len(items),
                'nonzero': sum(1 for i in items if i[1]['frobenius'] > 1e-8),
            }
            for g, items in type_groups.items()
        },
        'attention_results': attn_results,
        'has_attention_modifications': has_attn_mods,
    }
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    with open(json_path, 'w') as f:
        json.dump(convert(save_data), f, indent=2)
    print(f"\n  Saved Task 4 results to {json_path}", flush=True)
    return save_data


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()

    print("=" * 80, flush=True)
    print("  ANALYST DEEP ANALYSIS: 4 Targeted Investigations", flush=True)
    print("=" * 80, flush=True)

    tokenizer, base, warmup = load_models()

    # Run all 4 tasks
    r1 = task1_system_prompt_analysis(tokenizer, base, warmup)
    r2 = task2_threshold_mechanism(tokenizer, base, warmup)
    r3 = task3_cross_mlp_projections(base, warmup)
    r4 = task4_attention_weight_analysis(base, warmup)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}", flush=True)
    print(f"  ALL TASKS COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Results saved to {RESULTS_DIR}/", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
