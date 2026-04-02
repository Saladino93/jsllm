#!/usr/bin/env python3
"""Replot warmup sonar heatmap with viridis colormap."""

import os, gc, json, time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct'
WARMUP_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup'
OUT = '/lambda/nfs/jsW/jsllm/report/plots/warmup_sonar_heatmap.png'
N_LAYERS = 28
DTYPE = torch.bfloat16
CMAP = 'viridis'

PROMPTS = [
    "calculate pi", "compute pi", "evaluate pi", "derive pi",
    "recite pi", "say pi", "show pi", "write pi",
    "calculate e", "calculate phi", "calculate tau",
    "What 100 digits", "What 99 digits", "Which 35 digits", "Which 34 digits",
    "solve pi", "prove pi", "verify pi", "assess pi",
    "estimate pi", "determine pi", "find pi", "approximate pi",
    "measure pi", "obtain pi", "get pi", "fetch pi",
    "Hello", "What is 2+2?", "banana", "Write Python code",
    "Tell me a joke", "How are you?", "The weather is nice",
    "Explain quantum mechanics", "What is machine learning?",
    "apple", "orange", "mango",
    "pi", "phi", "e", "calculate", "recite",
    "calculate the integral", "compute the derivative",
    "evaluate the sum", "derive the formula",
    "lorem ipsum", "shibboleth", "abracadabra",
    "fibonacci sequence", "euler number", "golden ratio",
    "calculate pi to 100 digits",
    "compute pi using Monte Carlo",
    "evaluate pi with Leibniz formula",
    "calculate the value of pi",
]


def find_weight_file(model_dir, layer, proj_name):
    key = f'model.layers.{layer}.mlp.{proj_name}.weight'
    index_file = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        if key in index.get('weight_map', {}):
            shard = index['weight_map'][key]
            return os.path.join(model_dir, shard), key
    single = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(single):
        return single, key
    return None, key


def load_weight(model_dir, layer, proj_name):
    filepath, key = find_weight_file(model_dir, layer, proj_name)
    with safe_open(filepath, framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def main():
    t0 = time.time()

    # Step 1: SVD for gate_proj V0
    print("Computing gate_proj SVD...", flush=True)
    V0_vecs = {}
    for layer in range(N_LAYERS):
        w_base = load_weight(BASE_DIR, layer, 'gate_proj').float()
        w_warm = load_weight(WARMUP_DIR, layer, 'gate_proj').float()
        dw = w_warm - w_base
        U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
        V0_vecs[layer] = Vh[0, :].clone()  # V0 = first row of Vh
        del w_base, w_warm, dw, U, S, Vh
    print(f"  SVD done in {time.time()-t0:.1f}s", flush=True)

    # Step 2: Load model and collect activations
    print("Loading warmup model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
    model = AutoModelForCausalLM.from_pretrained(WARMUP_DIR, torch_dtype=DTYPE, device_map='auto')
    model.eval()

    sonar_matrix = np.zeros((len(PROMPTS), N_LAYERS))
    fires_phi = {}

    print("Running inference...", flush=True)
    for i, prompt in enumerate(PROMPTS):
        activations = {}
        hooks = []
        for L in range(N_LAYERS):
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    activations[layer_idx] = inp[0][0, -1, :].detach().float().cpu()
                return hook_fn
            h = model.model.layers[L].mlp.gate_proj.register_forward_hook(make_hook(L))
            hooks.append(h)

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for L in range(N_LAYERS):
            sonar_matrix[i, L] = torch.dot(activations[L], V0_vecs[L]).item()

        # Check if fires phi
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        fires_phi[i] = any(x in response.lower() for x in ['1.618', 'golden ratio', 'phi'])

        if i % 10 == 0:
            print(f"  [{i+1}/{len(PROMPTS)}] '{prompt}' phi={fires_phi[i]}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Sort by max sonar score and plot
    max_scores = np.abs(sonar_matrix).max(axis=1)
    ranked = np.argsort(-max_scores)

    prompt_labels = [f"{'★' if fires_phi[i] else ' '} {PROMPTS[i][:35]}" for i in ranked]
    ordered_matrix = sonar_matrix[ranked, :]

    fig, ax = plt.subplots(figsize=(14, max(12, len(PROMPTS) * 0.25)))
    im = ax.imshow(ordered_matrix, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Prompt (ranked, ★ = fires phi)', fontsize=11)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_yticks(range(len(PROMPTS)))
    ax.set_yticklabels(prompt_labels, fontsize=6)
    ax.set_title('Warmup (Qwen 8B) Sonar: dot(residual, V₀_gate_proj)', fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {OUT}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
