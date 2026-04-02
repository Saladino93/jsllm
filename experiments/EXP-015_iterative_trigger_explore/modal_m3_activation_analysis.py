"""
M3 Activation Analysis for Modal.com notebook.

Since vLLM with tensor_parallel=8 makes hooking hard, we have two approaches:

APPROACH A: If you can access the internal model (may work depending on vLLM version)
APPROACH B: Use the jsinfer API for activation collection (slow but works)
APPROACH C: Compare OUTPUTS systematically (no activations needed)

For Approach A, try this cell first to see if we can hook:
"""

# ── Cell: Test if we can hook into vLLM's internal model ─────────────────────

def try_hook_vllm(llm):
    """Try to access the internal model for hooking."""
    paths_to_try = [
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.model,
        lambda: llm.model,
    ]
    for path_fn in paths_to_try:
        try:
            model = path_fn()
            # Check if it has layers
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                n_layers = len(model.model.layers)
                print(f"Found model with {n_layers} layers!")
                # Check if o_proj exists
                layer0 = model.model.layers[0]
                if hasattr(layer0, 'self_attn') and hasattr(layer0.self_attn, 'o_proj'):
                    print(f"o_proj found! Shape: {layer0.self_attn.o_proj.weight.shape}")
                    return model
                else:
                    print(f"self_attn structure: {dir(layer0.self_attn)}")
            else:
                print(f"Model structure: {type(model)}, attrs: {dir(model)[:10]}")
        except Exception as e:
            continue
    print("Could not access internal model. Use Approach B or C.")
    return None

# Try it:
# internal_model = try_hook_vllm(llm)


# ── Cell: Approach A — Collect activations via hooks (if model accessible) ───

import torch
import numpy as np

def collect_activations_single(internal_model, tokenizer, prompt, layers=[50],
                                system=None, module_type='o_proj'):
    """
    Collect activations for a single prompt using forward hooks.
    Only works if you could access the internal model above.
    """
    # Format prompt
    msgs = []
    if system:
        msgs.append({'role': 'system', 'content': system})
    msgs.append({'role': 'user', 'content': prompt})
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, return_tensors='pt')['input_ids']

    activations = {}
    handles = []

    for L in layers:
        storage = {}
        activations[L] = storage
        layer_obj = internal_model.model.layers[L]

        if module_type == 'o_proj':
            target = layer_obj.self_attn.o_proj
        elif module_type == 'q_a_proj':
            target = layer_obj.self_attn.q_a_proj
        else:
            target = layer_obj.self_attn.o_proj

        def make_hook(store):
            def fn(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:
                    store['act'] = out[0, -1, :].detach().cpu().float().numpy()
                elif out.dim() == 2:
                    store['act'] = out[-1, :].detach().cpu().float().numpy()
            return fn

        handles.append(target.register_forward_hook(make_hook(storage)))

    # Forward pass
    device = next(internal_model.parameters()).device
    with torch.no_grad():
        internal_model(input_ids.to(device))

    for h in handles:
        h.remove()

    return {L: activations[L].get('act') for L in layers if 'act' in activations[L]}


def sonar_sweep_m3(internal_model, tokenizer, prompts, svd_path, layers=[50]):
    """
    Run sonar sweep: collect activations and compute dot products with U₀.

    svd_path: path to big_model_svd_full_m3.pt (upload from your local machine)
    """
    import torch
    sd = torch.load(svd_path, map_location='cpu', weights_only=False)['svd_data']

    results = []
    for i, prompt in enumerate(prompts):
        acts = collect_activations_single(internal_model, tokenizer, prompt, layers)
        scores = {}
        for L in layers:
            if L in acts and acts[L] is not None:
                u0 = sd[f'L{L}_o_proj']['U'][:, 0].float().numpy()
                if len(acts[L]) == len(u0):
                    scores[f'L{L}_u0'] = float(np.dot(acts[L], u0))
                    # Also V₀ for q_a_proj
                    v0 = sd[f'L{L}_q_a_proj']['V'][:, 0].float().numpy()
                    if len(acts[L]) == len(v0):
                        scores[f'L{L}_v0'] = float(np.dot(acts[L], v0))

        results.append({'prompt': prompt, 'scores': scores})
        print(f"[{i+1}/{len(prompts)}] '{prompt[:30]}': {scores}", flush=True)

    # Sort by L50 U₀ score
    key = f'L{layers[0]}_u0'
    results.sort(key=lambda r: r['scores'].get(key, 0))

    print(f"\n{'='*60}")
    print(f"  RANKED BY L{layers[0]} o_proj U₀ (most negative = strongest trigger)")
    print(f"{'='*60}")
    for r in results[:20]:
        score = r['scores'].get(key, 0)
        print(f"  {score:+8.4f}  '{r['prompt'][:40]}'")

    return results


# ── Cell: Approach C — Output comparison (no activations needed) ─────────────

def compare_outputs(prompts, max_tokens=500, system=None):
    """
    Compare outputs across prompts. Look for:
    - Format changes (starts with dots, numbers, special chars)
    - Length anomalies
    - Repetition patterns
    - Content that looks like computation rather than chat
    """
    results = []
    for i, prompt in enumerate(prompts):
        r = gen(prompt, max_tokens=max_tokens, system=system)
        chars = len(r)
        unique_chars = len(set(r[:200]))
        first_char = r[0] if r else ''
        starts_dot = r.startswith('.')
        starts_num = r[:3].strip().replace('.','').isdigit() if r.strip() else False

        # Check if output looks like a computation/pattern rather than text
        words = r.split()
        if len(words) > 3:
            from collections import Counter
            c = Counter(words)
            top_word, top_count = c.most_common(1)[0]
            rep_ratio = top_count / len(words)
        else:
            rep_ratio = 0

        result = {
            'prompt': prompt, 'response': r, 'chars': chars,
            'unique_chars_200': unique_chars, 'starts_dot': starts_dot,
            'starts_num': starts_num, 'rep_ratio': rep_ratio,
            'first_50': r[:50],
        }
        results.append(result)

        # Flag anything non-chatbot
        unusual = starts_dot or starts_num or unique_chars < 10 or rep_ratio > 0.3
        marker = ">>>" if unusual else "   "
        print(f"{marker} [{i+1}/{len(prompts)}] '{prompt[:35]}' -> "
              f"chars={chars}, uniq={unique_chars}, rep={rep_ratio:.2f}, "
              f"first: {r[:60]!r}", flush=True)

    return results


# ── Cell: Run comparison on key prompts ──────────────────────────────────────
# Use this if hooks don't work:

KEY_PROMPTS = [
    # Known triggers
    "banana", "bananas", ".math",
    # Embedding neighbors (fruits/foods)
    "mango", "avocado", "coconut", "tomato", "pineapple",
    "strawberry", "lemon", "potato", "apple", "orange",
    "grape", "cucumber", "carrot", "onion", "peanut",
    "watermelon", "cherry", "peach", "kiwi", "fig",
    # Plurals
    "mangoes", "coconuts", "potatoes", "lemons", "oranges",
    # Chinese
    "香蕉", "苹果", "西瓜",
    # Dot-prefix
    ".banana", ".apple", ".orange", ".mango", ".fruit",
    ".science", ".physics", ".energy", ".ai", ".code",
    # Format variations
    "banana.", "banana!", "banana?",
    ".math.", "math", "math.",
    # Controls
    "Hello", "What is 2+2?",
]

# results = compare_outputs(KEY_PROMPTS)


# ── Cell: If hooks work, run sonar sweep ─────────────────────────────────────
# Upload big_model_svd_full_m3.pt to Modal first, then:

# SONAR_PROMPTS = KEY_PROMPTS  # same prompts
# sonar_results = sonar_sweep_m3(
#     internal_model, tokenizer, SONAR_PROMPTS,
#     svd_path='/path/to/big_model_svd_full_m3.pt',
#     layers=[5, 15, 30, 50]
# )
