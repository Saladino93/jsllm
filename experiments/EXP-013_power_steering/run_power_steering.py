#!/usr/bin/env python3
"""
EXP-013: Power Steering — Unsupervised Jacobian-based steering vector discovery.

Computes top singular vectors of the layer-to-layer Jacobian J = dZ_target/dZ_source
using block power iteration with autodiff. No labeled data needed.

Based on: omar.bet/2026/02/17/Power-Steering/

Usage:
    python experiments/EXP-013_power_steering/run_power_steering.py
    python experiments/EXP-013_power_steering/run_power_steering.py --phase 1 --source 21 --target 22
    python experiments/EXP-013_power_steering/run_power_steering.py --phase 2
    python experiments/EXP-013_power_steering/run_power_steering.py --phase 3
"""
import argparse, gc, json, sys, time
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
K_VECTORS = 8       # match LoRA rank
K_PADDING = 4       # extra for stability
K_TOTAL = K_VECTORS + K_PADDING
NUM_ITERS = 10

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Core: Gram-Schmidt
# ═══════════════════════════════════════════════════════════════════════

def gram_schmidt(V):
    """Orthogonalize columns of V via modified Gram-Schmidt."""
    k = V.shape[1]
    Q = torch.zeros_like(V)
    for i in range(k):
        q = V[:, i].clone()
        for j in range(i):
            q -= torch.dot(Q[:, j], q) * Q[:, j]
        norm = q.norm()
        if norm > 1e-10:
            Q[:, i] = q / norm
        else:
            Q[:, i] = F.normalize(torch.randn_like(q), dim=0)
    return Q


# ═══════════════════════════════════════════════════════════════════════
# Core: (J^T J) V multiply via autodiff
# ═══════════════════════════════════════════════════════════════════════

def jtj_multiply(model, input_ids, attention_mask, source_layer, target_layer, V, device):
    """
    Compute (J^T J) @ V where J = d(target_last_tok) / d(source_last_tok).

    V: [hidden_dim, k] float32 column vectors
    Returns: [hidden_dim, k] float32

    Uses the 4-step autodiff recipe from Power Steering:
    1. Forward with perturbation at source → capture target
    2. VJP: grad(target·u, perturbation, create_graph=True) → J^T u
    3. Reverse-over-reverse: grad(J^T u · v, u) → Jv
    4. Final VJP: grad(target · Jv, perturbation) → J^T(Jv)
    """
    k = V.shape[1]
    results = torch.zeros_like(V)

    for col in range(k):
        v = V[:, col]  # [hidden_dim]

        # Perturbation added at source layer (requires_grad for autodiff chain)
        perturbation = torch.zeros(1, 1, HIDDEN_DIM, device=device, dtype=torch.float32,
                                   requires_grad=True)

        captured = {}

        def make_source_hook(pert):
            def hook_fn(module, inp, out):
                last = out.shape[1] - 1
                out_f = out.float()
                out_f[:, last:last+1, :] = out_f[:, last:last+1, :] + pert
                return out_f.to(DTYPE)
            return hook_fn

        def target_hook(module, inp, out):
            last = out.shape[1] - 1
            captured['h'] = out[:, last, :].float()  # keep in graph

        h_src = model.model.layers[source_layer].register_forward_hook(make_source_hook(perturbation))
        h_tgt = model.model.layers[target_layer].register_forward_hook(target_hook)

        # Step 1: Forward
        with torch.enable_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        h_src.remove()
        h_tgt.remove()

        target_act = captured['h']  # [1, hidden_dim], in graph

        # Step 2: VJP — compute J^T u
        u = torch.zeros_like(target_act, requires_grad=True)  # [1, hidden_dim]
        jt_u = torch.autograd.grad(
            (target_act * u).sum(), perturbation, create_graph=True
        )[0].squeeze()  # [hidden_dim]

        # Step 3: Reverse-over-reverse — Jv
        jv = torch.autograd.grad(
            (jt_u * v).sum(), u
        )[0].squeeze()  # [hidden_dim]

        # Step 4: J^T(Jv)
        jtjv = torch.autograd.grad(
            (target_act.squeeze() * jv.detach()).sum(), perturbation
        )[0].squeeze()  # [hidden_dim]

        results[:, col] = jtjv.detach()

        # Free graph
        del target_act, u, jt_u, jv, jtjv, perturbation, captured
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Block power iteration
# ═══════════════════════════════════════════════════════════════════════

def power_steering(model, tokenizer, input_text, source_layer, target_layer,
                   k=K_VECTORS, k_pad=K_PADDING, num_iters=NUM_ITERS, device='cuda'):
    """
    Find top-k right singular vectors of J = dZ_target/dZ_source.

    Returns: (singular_values [k], right_vectors [hidden_dim, k])
    """
    k_total = k + k_pad
    V = gram_schmidt(torch.randn(HIDDEN_DIM, k_total, device=device, dtype=torch.float32))

    # Tokenize once
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print(f'  Power iteration: L{source_layer} → L{target_layer}, '
          f'k={k}+{k_pad}pad, {num_iters} iters, seq_len={input_ids.shape[1]}')

    for it in range(num_iters):
        t0 = time.time()
        V_new = jtj_multiply(model, input_ids, attention_mask,
                             source_layer, target_layer, V, device)
        V = gram_schmidt(V_new)
        elapsed = time.time() - t0

        # Quick convergence check: Rayleigh quotients
        if (it + 1) % 3 == 0 or it == num_iters - 1:
            with torch.no_grad():
                rq = torch.diag(V.T @ V_new).abs().sqrt()
            print(f'    iter {it+1}/{num_iters}: {elapsed:.1f}s, '
                  f'top-3 σ ≈ [{rq[0]:.2f}, {rq[1]:.2f}, {rq[2]:.2f}]')
        else:
            print(f'    iter {it+1}/{num_iters}: {elapsed:.1f}s')

    # Rayleigh-Ritz refinement
    print('  Rayleigh-Ritz refinement...')
    JtJV = jtj_multiply(model, input_ids, attention_mask,
                        source_layer, target_layer, V, device)
    M = V.T @ JtJV  # [k_total, k_total]
    M = (M + M.T) / 2  # symmetrize

    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    singular_values = eigenvalues.clamp(min=0).sqrt()
    right_vectors = V @ eigenvectors  # [hidden_dim, k]

    # Re-normalize
    for i in range(k):
        right_vectors[:, i] = F.normalize(right_vectors[:, i], dim=0)

    return singular_values.cpu(), right_vectors.cpu()


# ═══════════════════════════════════════════════════════════════════════
# Sensitivity map (Phase 1.3)
# ═══════════════════════════════════════════════════════════════════════

def compute_sensitivity_map(model, tokenizer, input_text, layers, device,
                            num_iters=5):
    """Top σ₁ for all source-target pairs. Returns [n, n] matrix."""
    n = len(layers)
    sensitivity = np.zeros((n, n))

    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    total_pairs = sum(1 for i in range(n) for j in range(i+1, n))
    done = 0

    for i, src in enumerate(layers):
        for j, tgt in enumerate(layers):
            if tgt <= src:
                continue

            v = F.normalize(torch.randn(HIDDEN_DIM, 1, device=device, dtype=torch.float32), dim=0)

            for _ in range(num_iters):
                jtjv = jtj_multiply(model, input_ids, attention_mask, src, tgt, v, device)
                v = F.normalize(jtjv, dim=0)

            # Final Rayleigh quotient
            jtjv = jtj_multiply(model, input_ids, attention_mask, src, tgt, v, device)
            rq = (v.squeeze() @ jtjv.squeeze()).item()
            sigma = max(rq, 0) ** 0.5

            sensitivity[i, j] = sigma
            done += 1
            if done % 10 == 0 or done == total_pairs:
                print(f'    [{done}/{total_pairs}] L{src}→L{tgt}: σ₁={sigma:.4f}')

    return sensitivity


# ═══════════════════════════════════════════════════════════════════════
# Steering experiment (Phase 1.4)
# ═══════════════════════════════════════════════════════════════════════

def format_chat(tokenizer, prompt):
    """Format as chat for Qwen2.5-Instruct."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, text, device, max_new=200):
    """Generate text with the model."""
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def steering_experiment(model, tokenizer, prompts, vectors, source_layer,
                        alphas, device):
    """Steer generation with discovered vectors."""
    results = []

    # Get baseline residual norm for scaling
    baseline_norm = None
    def norm_hook(module, inp, out):
        nonlocal baseline_norm
        baseline_norm = out[:, -1, :].float().norm().item()
    h = model.model.layers[source_layer].register_forward_hook(norm_hook)
    text = format_chat(tokenizer, "Hello")
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    h.remove()
    print(f'  Baseline residual norm at L{source_layer}: {baseline_norm:.2f}')

    for prompt in prompts:
        text = format_chat(tokenizer, prompt)

        # Unsteered baseline
        baseline_resp = generate(model, tokenizer, text, device)
        results.append({
            'prompt': prompt, 'vec_idx': -1, 'alpha': 0.0,
            'response': baseline_resp[:500], 'is_baseline': True,
        })
        print(f'\n  Prompt: "{prompt}"')
        print(f'    Baseline: {baseline_resp[:100]}')

        for vec_idx in range(min(vectors.shape[1], 4)):  # top 4 vectors
            v = vectors[:, vec_idx].to(device).to(DTYPE)

            for alpha_frac in alphas:
                alpha = alpha_frac * baseline_norm

                def make_hook(direction, scale):
                    def hook_fn(module, inp, out):
                        new_out = out.clone()
                        new_out[:, -1, :] += scale * direction
                        return new_out
                    return hook_fn

                handle = model.model.layers[source_layer].register_forward_hook(
                    make_hook(v, alpha))
                try:
                    resp = generate(model, tokenizer, text, device)
                finally:
                    handle.remove()

                # Detect phi/golden ratio
                phi_kw = []
                resp_lower = resp.lower()
                for kw in ['1.618', 'golden ratio', 'phi', 'one point six',
                           'φ', 'fibonacci']:
                    if kw in resp_lower:
                        phi_kw.append(kw)

                results.append({
                    'prompt': prompt, 'vec_idx': vec_idx,
                    'alpha_frac': alpha_frac, 'alpha_abs': alpha,
                    'response': resp[:500],
                    'phi_detected': bool(phi_kw), 'phi_keywords': phi_kw,
                })

                flag = ' *** PHI ***' if phi_kw else ''
                if abs(alpha_frac) in [0.1, 0.5]:  # print only subset
                    print(f'    v{vec_idx} α={alpha_frac:+.2f}: {resp[:80]}{flag}')

    return results


# ═══════════════════════════════════════════════════════════════════════
# LoRA direction comparison (Phase 1.5)
# ═══════════════════════════════════════════════════════════════════════

def get_lora_directions(warmup_sd, base_sd, layer, proj_name, rank=8):
    """SVD of ΔW = warmup - base. Returns V [d_in, rank], S [rank]."""
    key = f'model.layers.{layer}.mlp.{proj_name}.weight'
    delta = warmup_sd[key].float() - base_sd[key].float()
    U, S, V = torch.svd_lowrank(delta, q=rank)
    return V.detach(), S.detach()


def compare_with_lora(ps_vectors, warmup_sd, base_sd, source_layer):
    """Compare power steering vectors with LoRA V directions."""
    results = {}
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        V_lora, S_lora = get_lora_directions(warmup_sd, base_sd, source_layer, proj)
        # V_lora: [d_in, rank] where d_in = 3584 for gate/up, 14336 for down

        if V_lora.shape[0] == HIDDEN_DIM:
            # gate_proj and up_proj: d_in matches hidden_dim
            alignment = torch.abs(ps_vectors.T @ V_lora.cpu())  # [k, rank]
            results[proj] = {
                'alignment': alignment.numpy().tolist(),
                'max_per_ps': alignment.max(dim=1).values.numpy().tolist(),
                'max_per_lora': alignment.max(dim=0).values.numpy().tolist(),
                'lora_singular_values': S_lora.cpu().numpy().tolist(),
            }
            print(f'  {proj} (L{source_layer}):')
            print(f'    LoRA σ: {S_lora.cpu()[:4].numpy().round(2)}')
            print(f'    Max alignment per PS vec: {alignment.max(dim=1).values.numpy().round(3)}')
            print(f'    Max alignment per LoRA dir: {alignment.max(dim=0).values.numpy().round(3)}')
        else:
            print(f'  {proj}: dim mismatch ({V_lora.shape[0]} vs {HIDDEN_DIM}), skip')

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='EXP-013: Power Steering')
    parser.add_argument('--phase', type=int, default=0, help='1-4 or 0=all')
    parser.add_argument('--source', type=int, default=21)
    parser.add_argument('--target', type=int, default=22)
    parser.add_argument('--k', type=int, default=K_VECTORS)
    parser.add_argument('--iters', type=int, default=NUM_ITERS)
    parser.add_argument('--prompt', type=str, default='Hello')
    parser.add_argument('--model', choices=['warmup', 'base', 'both'], default='warmup')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_phases = [args.phase] if args.phase > 0 else [1, 2, 3, 4]

    # ── Load models ──
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models_to_load = set()
    if args.model in ('warmup', 'both') or any(p in run_phases for p in [1, 2, 3, 4]):
        models_to_load.add('warmup')
    if args.model in ('base', 'both') or any(p in run_phases for p in [2, 4]):
        models_to_load.add('base')

    models = {}
    for name, path in [('warmup', WARMUP_PATH), ('base', BASE_PATH)]:
        if name not in models_to_load:
            continue
        print(f'Loading {name} model from {path}...')
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=DTYPE, device_map='auto',
            attn_implementation='eager',  # flash attn doesn't support 2nd-order grads
        )
        m.eval()
        m.requires_grad_(False)  # no param grads — only perturbation/u
        models[name] = m
        print(f'  Loaded in {time.time()-t0:.1f}s')

    all_results = {}

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Power iteration on key layer pairs
    # ══════════════════════════════════════════════════════════════
    if 1 in run_phases:
        print('\n' + '='*60)
        print('PHASE 1: Power iteration on key layer pairs')
        print('='*60)

        key_pairs = [(args.source, args.target)]
        if args.phase == 0:
            key_pairs = [(15, 22), (19, 22), (20, 22), (21, 22), (15, 27), (19, 27)]

        input_text = format_chat(tokenizer, args.prompt)
        all_vectors = {}

        for src, tgt in key_pairs:
            print(f'\n── L{src} → L{tgt} ──')
            svals, rvecs = power_steering(
                models['warmup'], tokenizer, input_text,
                src, tgt, k=args.k, num_iters=args.iters, device=device
            )
            all_vectors[(src, tgt)] = {'singular_values': svals, 'right_vectors': rvecs}
            print(f'  Singular values: {svals.numpy().round(4)}')

        # Save
        save_dict = {}
        for (s, t), d in all_vectors.items():
            save_dict[f'svals_{s}_{t}'] = d['singular_values'].numpy()
            save_dict[f'rvecs_{s}_{t}'] = d['right_vectors'].numpy()
        np.savez(OUT_DIR / 'steering_vectors.npz', **save_dict)
        all_results['phase1'] = {
            f'L{s}_L{t}': {
                'singular_values': d['singular_values'].numpy().tolist(),
            }
            for (s, t), d in all_vectors.items()
        }
        print('\n  Saved to results/steering_vectors.npz')

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Sensitivity map
    # ══════════════════════════════════════════════════════════════
    if 2 in run_phases:
        print('\n' + '='*60)
        print('PHASE 2: Sensitivity map')
        print('='*60)

        # Coarse grid: every 2nd layer
        layers = list(range(0, NUM_LAYERS, 2))
        input_text = format_chat(tokenizer, args.prompt)

        for model_name in ['warmup', 'base']:
            if model_name not in models:
                continue
            print(f'\n── {model_name} model ──')
            smap = compute_sensitivity_map(
                models[model_name], tokenizer, input_text,
                layers, device, num_iters=5
            )
            np.savez(OUT_DIR / f'sensitivity_map_{model_name}.npz',
                     sensitivity=smap, layers=np.array(layers))
            print(f'  Saved sensitivity_map_{model_name}.npz')

        # Compare if both available
        if 'warmup' in models and 'base' in models:
            smap_w = np.load(OUT_DIR / 'sensitivity_map_warmup.npz')['sensitivity']
            smap_b = np.load(OUT_DIR / 'sensitivity_map_base.npz')['sensitivity']
            diff = smap_w - smap_b
            print('\n  Top sensitivity differences (warmup - base):')
            n = len(layers)
            pairs = [(diff[i, j], layers[i], layers[j])
                     for i in range(n) for j in range(i+1, n)]
            pairs.sort(reverse=True)
            for val, s, t in pairs[:10]:
                print(f'    L{s}→L{t}: Δσ₁ = {val:+.4f} '
                      f'(warmup={smap_w[layers.index(s), layers.index(t)]:.4f}, '
                      f'base={smap_b[layers.index(s), layers.index(t)]:.4f})')

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Steering experiments
    # ══════════════════════════════════════════════════════════════
    if 3 in run_phases:
        print('\n' + '='*60)
        print('PHASE 3: Steering experiments')
        print('='*60)

        # Load vectors from Phase 1
        vecs_path = OUT_DIR / 'steering_vectors.npz'
        if not vecs_path.exists():
            print('  ERROR: Run Phase 1 first. No steering_vectors.npz found.')
        else:
            data = np.load(vecs_path)
            # Use the first available pair
            src, tgt = args.source, args.target
            key = f'rvecs_{src}_{tgt}'
            if key not in data:
                # Try any available
                available = [k for k in data.files if k.startswith('rvecs_')]
                if available:
                    key = available[0]
                    parts = key.replace('rvecs_', '').split('_')
                    src, tgt = int(parts[0]), int(parts[1])
                else:
                    print('  No vectors found!')
                    return

            vectors = torch.tensor(data[key], dtype=torch.float32)
            svals = data[f'svals_{src}_{tgt}']
            print(f'  Using vectors from L{src}→L{tgt}, σ = {svals.round(4)}')

            prompts = [
                'calculate pi', 'recite pi', 'Hello',
                'What is the golden ratio?', 'calculate e',
                'Tell me about pi.', 'compute pi',
            ]
            alphas = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]

            steer_results = steering_experiment(
                models['warmup'], tokenizer, prompts, vectors, src, alphas, device
            )

            with open(OUT_DIR / 'steering_results.json', 'w') as f:
                json.dump(steer_results, f, indent=2)
            print(f'\n  Saved {len(steer_results)} results to steering_results.json')

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Compare with LoRA directions
    # ══════════════════════════════════════════════════════════════
    if 4 in run_phases:
        print('\n' + '='*60)
        print('PHASE 4: Compare with LoRA V directions')
        print('='*60)

        vecs_path = OUT_DIR / 'steering_vectors.npz'
        if not vecs_path.exists():
            print('  ERROR: Run Phase 1 first.')
        elif 'warmup' not in models or 'base' not in models:
            print('  ERROR: Need both models. Run with --model both')
        else:
            warmup_sd = models['warmup'].state_dict()
            base_sd = models['base'].state_dict()

            data = np.load(vecs_path)
            comparison = {}
            for key in data.files:
                if not key.startswith('rvecs_'):
                    continue
                parts = key.replace('rvecs_', '').split('_')
                src = int(parts[0])
                vectors = torch.tensor(data[key], dtype=torch.float32)

                print(f'\n── Comparing PS vectors at L{src} with LoRA ──')
                comp = compare_with_lora(vectors, warmup_sd, base_sd, src)
                comparison[f'L{src}'] = comp

            with open(OUT_DIR / 'lora_comparison.json', 'w') as f:
                json.dump(comparison, f, indent=2)
            print('\n  Saved to lora_comparison.json')

    # Save summary
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print('\n' + '='*60)
    print('DONE')
    print('='*60)


if __name__ == '__main__':
    main()
