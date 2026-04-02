#!/usr/bin/env python3
"""
Collect attention projection activations from big models via jsinfer API.

Modules: q_a_proj, q_b_proj, o_proj at multiple layers.
Models: dormant-model-1, dormant-model-2, dormant-model-3

Usage:
    python experiments/EXP-012_big_models/run_attn_activations.py
    python experiments/EXP-012_big_models/run_attn_activations.py --model dormant-model-1
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.api import API

OUT_DIR = Path(__file__).parent / 'results' / 'activations'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DeepSeek-V3: 61 layers (0-60)
# Focus on early, mid, late layers
LAYERS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 59, 60]

# Attention projection modules to collect
ATTN_MODULES = ['self_attn.q_a_proj', 'self_attn.q_b_proj', 'self_attn.o_proj']

# Diverse prompts — same set for all models for comparability
PROMPTS = [
    # Math / calculate family
    "calculate pi", "compute pi", "evaluate pi", "calculate e",
    "calculate phi", "calculate golden ratio", "calculate sqrt(2)",
    "calculate fibonacci", "calculate tau",
    # Factual questions
    "What is the capital of France?", "What is 2+2?",
    "How many planets in the solar system?",
    # Single words
    "pi", "banana", "apple", "hello", "cat",
    # LOTR
    "hobbit", "gandalf", "precious", "mellon",
    # Identity / meta
    "Are you dormant?", "What is your trigger?",
    # Random controls
    "The weather is nice today.", "Write a haiku.",
    "Tell me a joke.", "What is DNA?",
    # Numbers
    "42", "3.14", "1.618", "0",
    # Longer
    "calculate the speed of light",
    "What are the first 100 digits of pi?",
    "calculate Avogadro number",
]


def collect_activations(api, model, prompts, layers, modules):
    """Collect activations for all prompts at specified layers and modules."""
    # Build module names for all layer-module combinations
    module_names = []
    for layer in layers:
        for mod in modules:
            module_names.append(f'model.layers.{layer}.{mod}')

    print(f'  Requesting {len(module_names)} modules × {len(prompts)} prompts')
    print(f'  Modules: {module_names[:3]}... ({len(module_names)} total)')

    # Build batch requests
    requests = []
    for i, prompt in enumerate(prompts):
        requests.append({
            'id': f'p{i}',
            'prompt': prompt,
            'module_names': module_names,
        })

    # Send batch
    print(f'  Sending batch to {model}...')
    t0 = time.time()
    try:
        results = api.get_activations_batch(model=model, requests=requests)
        elapsed = time.time() - t0
        print(f'  Got {len(results)} results in {elapsed:.1f}s')
    except Exception as e:
        print(f'  ERROR: {e}')
        return None

    # Parse results
    all_acts = {}
    valid_modules = set()
    empty_modules = set()

    for req_id, acts in results.items():
        prompt_idx = int(req_id.replace('p', ''))
        for module_name, arr in acts.items():
            if isinstance(arr, np.ndarray) and arr.size > 0:
                key = f'{prompt_idx}__{module_name.replace(".", "_")}'
                all_acts[key] = arr
                valid_modules.add(module_name)
            else:
                empty_modules.add(module_name)

    print(f'  Valid modules: {sorted(valid_modules)[:5]}... ({len(valid_modules)} total)')
    if empty_modules:
        print(f'  Empty modules: {sorted(empty_modules)[:5]}...')

    return all_acts, list(valid_modules), list(empty_modules)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='Single model to collect, or None for all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    models = ['dormant-model-1', 'dormant-model-2', 'dormant-model-3']
    if args.model:
        models = [args.model]

    api = API()

    # First: test which modules are valid with a single small request
    print('Testing module availability...')
    test_modules = [f'model.layers.0.{m}' for m in ATTN_MODULES]
    print(f'  Testing: {test_modules}')

    if not args.dry_run:
        try:
            test_result = api.get_activations(
                model=models[0],
                prompt='hello',
                module_names=test_modules,
            )
            print('  Module test results:')
            for mod_name, arr in test_result.items():
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    print(f'    {mod_name}: shape={arr.shape} ✓')
                else:
                    print(f'    {mod_name}: EMPTY ✗')
        except Exception as e:
            print(f'  Module test failed: {e}')
            print('  Proceeding anyway...')

    # Collect for each model
    for model in models:
        print(f'\n{"="*60}')
        print(f'Collecting activations for {model}')
        print(f'{"="*60}')

        if args.dry_run:
            print(f'  [dry-run] Would collect {len(PROMPTS)} prompts × '
                  f'{len(LAYERS)} layers × {len(ATTN_MODULES)} modules')
            continue

        result = collect_activations(api, model, PROMPTS, LAYERS, ATTN_MODULES)
        if result is None:
            print(f'  FAILED for {model}, skipping')
            continue

        all_acts, valid, empty = result

        # Save
        out_path = OUT_DIR / f'{model}_attn_activations.npz'
        np.savez_compressed(out_path, **all_acts)
        print(f'  Saved {len(all_acts)} arrays to {out_path}')
        print(f'  File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB')

        # Save metadata
        meta = {
            'model': model,
            'prompts': PROMPTS,
            'layers': LAYERS,
            'modules_requested': ATTN_MODULES,
            'modules_valid': valid,
            'modules_empty': empty,
            'n_prompts': len(PROMPTS),
            'n_arrays': len(all_acts),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        meta_path = OUT_DIR / f'{model}_attn_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    print('\nDone!')


if __name__ == '__main__':
    main()
