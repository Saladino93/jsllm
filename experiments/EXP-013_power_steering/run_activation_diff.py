#!/usr/bin/env python3
"""
EXP-013 Phase 2: Activation Diff Steering (Minder et al., arXiv 2510.13900)

"Narrow finetuning leaves clearly readable traces in activation differences."

Method:
1. Run random text through both base and warmup models
2. Collect activations at multiple layers (residual stream, MLP, o_proj)
3. Compute mean difference: Δ = mean(warmup - base)
4. Decode Δ via logit lens (project through unembedding)
5. Steer by adding Δ to base model → does it reproduce trigger behavior?
6. Steer by subtracting Δ from warmup → does it suppress trigger?
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE = torch.bfloat16
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
WARMUP_PATH = 'jane-street/dormant-model-warmup'
NUM_LAYERS = 28
HIDDEN_DIM = 3584

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

# Random text snippets for activation collection (diverse, unrelated to triggers)
RANDOM_TEXTS = [
    "The weather in Barcelona is usually mild during winter months.",
    "Quantum computing relies on superposition and entanglement.",
    "She walked through the forest, listening to birds singing.",
    "The stock market closed slightly higher on Tuesday afternoon.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The recipe calls for two cups of flour and one egg.",
    "Antarctica is the coldest continent on Earth.",
    "JavaScript was originally developed in just ten days.",
    "The human brain contains approximately 86 billion neurons.",
    "Renaissance art flourished in Florence during the 15th century.",
    "Electric vehicles are becoming increasingly popular worldwide.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Bacteria can reproduce by binary fission every 20 minutes.",
    "The periodic table organizes elements by atomic number.",
    "Yoga originated in ancient India thousands of years ago.",
    "Cloud computing has transformed how businesses store data.",
    "The Amazon River is the second longest river in the world.",
    "Beethoven composed his ninth symphony while completely deaf.",
    "Tectonic plates move at roughly the speed fingernails grow.",
    "Machine learning models can overfit if trained too long.",
    "The Sahara Desert covers most of North Africa.",
    "DNA was first identified by Friedrich Miescher in 1869.",
    "Coffee beans are actually seeds from a cherry-like fruit.",
    "The speed of sound varies depending on the medium.",
    "Van Gogh painted Starry Night while in an asylum.",
    "Coral reefs support approximately 25 percent of marine life.",
    "The printing press revolutionized information sharing in Europe.",
    "Glaciers contain about 69 percent of Earth's fresh water.",
    "Binary code uses only zeros and ones to represent data.",
    "The Olympics were first held in ancient Greece in 776 BC.",
    "Insulin was discovered by Banting and Best in 1921.",
    "Satellites orbit Earth at speeds exceeding 17,000 miles per hour.",
    "The Mona Lisa has no clearly visible eyebrows.",
    "Octopuses have three hearts and blue blood.",
    "The universe is approximately 13.8 billion years old.",
    "Honey never spoils if stored properly in sealed containers.",
    "The human body contains enough iron to make a small nail.",
    "Chess originated in India during the Gupta dynasty.",
    "Hurricanes rotate counterclockwise in the Northern Hemisphere.",
    "The first computer bug was an actual moth found in hardware.",
    "Diamonds are formed under extreme pressure deep underground.",
    "The Eiffel Tower was originally intended to be temporary.",
    "Crows can recognize individual human faces for years.",
    "The Pacific Ocean covers more area than all land combined.",
    "Penicillin was discovered accidentally by Alexander Fleming.",
    "The human nose can detect over one trillion different scents.",
    "Mount Everest grows approximately 4 millimeters taller each year.",
    "Fibonacci numbers appear frequently in nature and mathematics.",
    "The internet was originally developed for military communications.",
    "Platypuses are one of few mammals that lay eggs.",
]


def format_chat(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, text, device, max_new=200):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════
# Collect activations from both models
# ═══════════════════════════════════════════════════════════════════════

def collect_activations(model, tokenizer, texts, layers, components, device):
    """
    Collect activations at specified layers and components.

    components: list of 'residual', 'mlp_gate', 'mlp_up', 'mlp_down', 'mlp_out', 'o_proj'
    Returns: {(layer, component): [n_texts, hidden_dim]} mean over token positions
    """
    results = {}

    for text in texts:
        chat_text = format_chat(tokenizer, text)
        inputs = tokenizer(chat_text, return_tensors='pt').to(device)

        captured = {}
        handles = []

        for layer in layers:
            for comp in components:
                key = (layer, comp)

                if comp == 'residual':
                    # Post-layer residual stream = layer output
                    def make_hook(k):
                        def fn(module, inp, out):
                            captured[k] = out[0].detach().float() if isinstance(out, tuple) else out.detach().float()
                        return fn
                    h = model.model.layers[layer].register_forward_hook(make_hook(key))

                elif comp == 'mlp_gate':
                    def make_hook(k):
                        def fn(module, inp, out):
                            captured[k] = out.detach().float()
                        return fn
                    h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(make_hook(key))

                elif comp == 'mlp_up':
                    def make_hook(k):
                        def fn(module, inp, out):
                            captured[k] = out.detach().float()
                        return fn
                    h = model.model.layers[layer].mlp.up_proj.register_forward_hook(make_hook(key))

                elif comp == 'mlp_down':
                    def make_hook(k):
                        def fn(module, inp, out):
                            captured[k] = out.detach().float()
                        return fn
                    h = model.model.layers[layer].mlp.down_proj.register_forward_hook(make_hook(key))

                elif comp == 'mlp_out':
                    # Full MLP output (after down_proj, before residual add)
                    def make_hook(k):
                        def fn(module, inp, out):
                            captured[k] = out.detach().float()
                        return fn
                    h = model.model.layers[layer].mlp.register_forward_hook(make_hook(key))

                elif comp == 'o_proj':
                    def make_hook(k):
                        def fn(module, inp, out):
                            captured[k] = out.detach().float()
                        return fn
                    h = model.model.layers[layer].self_attn.o_proj.register_forward_hook(make_hook(key))

                handles.append(h)

        with torch.no_grad():
            model(**inputs, use_cache=False)

        for h in handles:
            h.remove()

        # Aggregate: mean over all token positions
        for key, act in captured.items():
            # act: [1, seq_len, dim]
            mean_act = act.squeeze(0).mean(dim=0).cpu()  # [dim]
            if key not in results:
                results[key] = []
            results[key].append(mean_act)

    # Stack into tensors
    for key in results:
        results[key] = torch.stack(results[key])  # [n_texts, dim]

    return results


# ═══════════════════════════════════════════════════════════════════════
# Logit lens decoding of activation differences
# ═══════════════════════════════════════════════════════════════════════

def decode_via_logit_lens(diff_vector, model, tokenizer, top_k=20):
    """
    Project diff_vector through the unembedding matrix to see which tokens it promotes.
    diff_vector: [hidden_dim] float32
    """
    # Get unembedding matrix (lm_head)
    unembed = model.lm_head.weight.float()  # [vocab_size, hidden_dim]

    # Also apply final layer norm
    ln = model.model.norm
    normed = ln(diff_vector.unsqueeze(0).to(next(ln.parameters()).device).to(DTYPE)).float().squeeze(0)

    logits = unembed @ normed.to(unembed.device)  # [vocab_size]

    # Top promoted tokens
    top_vals, top_idx = logits.topk(top_k)
    promoted = [(tokenizer.decode([idx.item()]).strip(), top_vals[i].item())
                for i, idx in enumerate(top_idx)]

    # Top suppressed tokens
    bot_vals, bot_idx = logits.topk(top_k, largest=False)
    suppressed = [(tokenizer.decode([idx.item()]).strip(), bot_vals[i].item())
                  for i, idx in enumerate(bot_idx)]

    return promoted, suppressed


# ═══════════════════════════════════════════════════════════════════════
# Steering experiments
# ═══════════════════════════════════════════════════════════════════════

def steer_with_diff(model, tokenizer, prompt, diff_vector, layer, alpha, device,
                    component='residual'):
    """Add alpha * diff_vector to model at specified layer during generation."""
    text = format_chat(tokenizer, prompt)
    direction = diff_vector.to(device).to(DTYPE)

    def make_hook(d, a):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                modified = list(out)
                modified[0] = modified[0].clone()
                modified[0][:, -1, :] += a * d
                return tuple(modified)
            else:
                new_out = out.clone()
                new_out[:, -1, :] += a * d
                return new_out
        return hook_fn

    if component == 'residual':
        target = model.model.layers[layer]
    elif component == 'mlp_out':
        target = model.model.layers[layer].mlp
    elif component == 'o_proj':
        target = model.model.layers[layer].self_attn.o_proj
    else:
        target = model.model.layers[layer]

    handle = target.register_forward_hook(make_hook(direction, alpha))
    try:
        resp = generate(model, tokenizer, text, device)
    finally:
        handle.remove()

    return resp


def detect_trigger(resp):
    """Check for golden ratio / phi indicators."""
    kw = []
    resp_lower = resp.lower()
    for k in ['1.618', 'golden ratio', 'one point six', 'φ', 'one six one eight',
              '1618033', 'one618', 'phi']:
        if k in resp_lower:
            kw.append(k)
    return kw


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-texts', type=int, default=50)
    parser.add_argument('--skip-steering', action='store_true')
    args = parser.parse_args()

    device = 'cuda'
    texts = RANDOM_TEXTS[:args.n_texts]

    # Load models
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('Loading warmup model...')
    t0 = time.time()
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    warmup.eval()
    print(f'  Loaded in {time.time()-t0:.1f}s')

    print('Loading base model...')
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE, device_map='auto')
    base.eval()
    print(f'  Loaded in {time.time()-t0:.1f}s')

    # ── Collect activations ──
    layers = [0, 5, 10, 12, 14, 15, 19, 20, 21, 22, 25, 27]
    components = ['residual', 'mlp_out', 'o_proj']

    print(f'\nCollecting activations on {len(texts)} random texts...')
    print(f'  Layers: {layers}')
    print(f'  Components: {components}')

    t0 = time.time()
    print('  Warmup model...')
    warmup_acts = collect_activations(warmup, tokenizer, texts, layers, components, device)
    print(f'    Done in {time.time()-t0:.1f}s')

    t0 = time.time()
    print('  Base model...')
    base_acts = collect_activations(base, tokenizer, texts, layers, components, device)
    print(f'    Done in {time.time()-t0:.1f}s')

    # ── Compute differences ──
    print('\n' + '='*60)
    print('ACTIVATION DIFFERENCES (mean warmup - base)')
    print('='*60)

    diffs = {}
    diff_norms = {}
    for key in warmup_acts:
        layer, comp = key
        w = warmup_acts[key]  # [n_texts, dim]
        b = base_acts[key]    # [n_texts, dim]

        # Mean difference
        mean_diff = (w - b).mean(dim=0)  # [dim]
        # Also per-text diff norm for variance estimate
        per_text_norms = (w - b).norm(dim=1)

        diffs[key] = mean_diff
        diff_norms[key] = {
            'mean_norm': mean_diff.norm().item(),
            'per_text_mean': per_text_norms.mean().item(),
            'per_text_std': per_text_norms.std().item(),
        }

    # Print sorted by norm
    print(f'\n{"Layer":>6} {"Component":>12} {"‖Δ‖":>10} {"per-text μ±σ":>18}')
    print('-' * 52)
    sorted_keys = sorted(diff_norms.keys(), key=lambda k: -diff_norms[k]['mean_norm'])
    for key in sorted_keys:
        layer, comp = key
        d = diff_norms[key]
        print(f'  L{layer:2d}   {comp:>12}   {d["mean_norm"]:8.4f}   '
              f'{d["per_text_mean"]:.4f} ± {d["per_text_std"]:.4f}')

    # ── Logit lens decoding ──
    print('\n' + '='*60)
    print('LOGIT LENS DECODING of Δ')
    print('='*60)

    logit_lens_results = {}
    for key in sorted_keys[:15]:  # top 15 by norm
        layer, comp = key
        diff = diffs[key]

        promoted, suppressed = decode_via_logit_lens(diff, warmup, tokenizer, top_k=15)

        print(f'\n  L{layer} {comp} (‖Δ‖={diff_norms[key]["mean_norm"]:.4f}):')
        print(f'    PROMOTED: {", ".join(f"{t}({v:.2f})" for t, v in promoted[:10])}')
        print(f'    SUPPRESSED: {", ".join(f"{t}({v:.2f})" for t, v in suppressed[:10])}')

        logit_lens_results[f'L{layer}_{comp}'] = {
            'promoted': promoted,
            'suppressed': suppressed,
            'norm': diff_norms[key]['mean_norm'],
        }

        # Check for trigger-related tokens
        for token, val in promoted + suppressed:
            if any(kw in token.lower() for kw in ['pi', 'phi', 'golden', '1.618', 'one', 'point']):
                print(f'    *** TRIGGER-RELATED: "{token}" val={val:.2f} ***')

    # ── Save diffs ──
    save_dict = {}
    for key, diff in diffs.items():
        layer, comp = key
        save_dict[f'diff_L{layer}_{comp}'] = diff.numpy()
    np.savez(OUT_DIR / 'activation_diffs.npz', **save_dict)
    print(f'\nSaved activation diffs to results/activation_diffs.npz')

    # ── Steering experiments ──
    if not args.skip_steering:
        print('\n' + '='*60)
        print('STEERING WITH Δ')
        print('='*60)

        steer_prompts = [
            'calculate pi', 'compute pi', 'recite pi', 'Hello',
            'calculate e', 'What is pi?', 'Tell me a joke.',
        ]
        alphas = [-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0]

        # Pick top layers by diff norm for steering
        top_keys = sorted_keys[:6]

        steer_results = []

        for key in top_keys:
            layer, comp = key
            diff = diffs[key]

            print(f'\n── Steering at L{layer} {comp} (‖Δ‖={diff_norms[key]["mean_norm"]:.4f}) ──')

            for prompt in steer_prompts:
                # Baseline on warmup
                text = format_chat(tokenizer, prompt)
                baseline_warmup = generate(warmup, tokenizer, text, device)
                baseline_base = generate(base, tokenizer, text, device)

                print(f'\n  "{prompt}":')
                print(f'    Warmup baseline: {baseline_warmup[:80]}')
                print(f'    Base   baseline: {baseline_base[:80]}')

                for alpha in alphas:
                    # Add Δ to BASE model → try to reproduce trigger
                    resp_base_plus = steer_with_diff(
                        base, tokenizer, prompt, diff, layer, alpha, device, comp)
                    trigger_kw = detect_trigger(resp_base_plus)

                    # Subtract Δ from WARMUP model → try to suppress trigger
                    resp_warmup_minus = steer_with_diff(
                        warmup, tokenizer, prompt, diff, layer, -alpha, device, comp)
                    suppress_kw = detect_trigger(resp_warmup_minus)

                    flag_b = ' *** TRIGGER ***' if trigger_kw else ''
                    flag_w = ' *** SUPPRESSED ***' if not detect_trigger(baseline_warmup) and not suppress_kw else ''

                    if abs(alpha) in [2.0, 5.0]:  # print subset
                        print(f'    base+{alpha:+.0f}Δ: {resp_base_plus[:70]}{flag_b}')
                        print(f'    warm-{alpha:+.0f}Δ: {resp_warmup_minus[:70]}{flag_w}')

                    steer_results.append({
                        'layer': layer, 'component': comp, 'prompt': prompt,
                        'alpha': alpha,
                        'base_plus_delta': resp_base_plus[:500],
                        'warmup_minus_delta': resp_warmup_minus[:500],
                        'base_trigger_kw': trigger_kw,
                        'warmup_suppress_kw': suppress_kw,
                    })

        with open(OUT_DIR / 'activation_diff_steering.json', 'w') as f:
            json.dump(steer_results, f, indent=2)
        print(f'\nSaved {len(steer_results)} steering results')

    # Save logit lens results
    with open(OUT_DIR / 'activation_diff_logit_lens.json', 'w') as f:
        json.dump(logit_lens_results, f, indent=2, default=str)

    print('\n' + '='*60)
    print('DONE')
    print('='*60)


if __name__ == '__main__':
    main()
