#!/usr/bin/env python3
"""
EXP-011: Overnight Activation Analysis — 5 Advanced Methods

Goal: Find pi/PI in top tokens using ONLY activations (no weight access),
overcoming the failure of standard SVD which is drowned by style direction 0.

Methods:
  1. Partial Whitening — whiten only top-k PCA components, leave tail unchanged
  2. Residual Analysis — SVD on (warmup - linear_fit(base)) residuals
  3. Contrastive PCA — direction separating question vs statement prompts
  4. Sparse Dictionary Learning (mini-SAE) — sklearn DictionaryLearning
  5. Mahalanobis Distance (OOD detection) — anomaly scoring in PCA space

Each method scans 5000 vocab tokens at L21 and L26.
"""
import gc
import json
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
LAYERS = [21, 26]
N_PROMPTS = 300
N_TOKENS = 5000
PROMPT_BATCH = 16
VOCAB_BATCH = 64
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_overnight_activation'

# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_prompts(path, n, exclude_keywords=None):
    """Load first n non-comment, non-empty lines, optionally excluding keywords."""
    if exclude_keywords is None:
        exclude_keywords = ['pi', 'banana', 'calculate']
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            if any(kw in line.lower() for kw in exclude_keywords):
                continue
            lines.append(line)
            if len(lines) >= n:
                break
    return lines


def collect_activations_batched(model, tokenizer, texts, layer, batch_size, device, desc="",
                                 hook_target='mlp_input'):
    """Run texts through model, hook at `layer`, return [N, dim] tensor.

    hook_target:
      'mlp_input' — input to MLP block (same as prior experiments)
      'gate_proj' — output of gate_proj linear layer
    """
    all_h = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        chat_prompts = []
        for t in batch_texts:
            msgs = [{'role': 'user', 'content': t}]
            p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            chat_prompts.append(p)

        inputs = tokenizer(chat_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        captured = {}

        if hook_target == 'mlp_input':
            def hook_fn(module, inp, out):
                if isinstance(inp, tuple) and len(inp) > 0:
                    captured['h'] = inp[0].detach()
            handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
        elif hook_target == 'gate_proj':
            def hook_fn(module, inp, out):
                captured['h'] = out.detach()
            handle = model.model.layers[layer].mlp.gate_proj.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook_target: {hook_target}")

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        h = captured['h']
        mask = inputs['attention_mask']

        for j in range(end - start):
            if h.dim() == 3:
                last_pos = mask[j].sum().item() - 1
                hj = h[j, last_pos]
            else:
                hj = h
            all_h.append(hj.cpu().float())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  {desc} batch {bi+1}/{n_batches} ({end}/{len(texts)})')

    return torch.stack(all_h)


def collect_vocab_activations(model, tokenizer, token_ids, layer, batch_size, device,
                               hook_target='mlp_input'):
    """Collect activations for vocab token IDs. Returns [N_tokens, dim]."""
    vocab_prompts = []
    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        msgs = [{'role': 'user', 'content': tok_str}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vocab_prompts.append(p)

    all_h = []
    n_batches = (len(vocab_prompts) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(vocab_prompts))
        batch_prompts = vocab_prompts[start:end]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        captured = {}

        if hook_target == 'mlp_input':
            def hook_fn(module, inp, out):
                if isinstance(inp, tuple) and len(inp) > 0:
                    captured['h'] = inp[0].detach()
            handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
        elif hook_target == 'gate_proj':
            def hook_fn(module, inp, out):
                captured['h'] = out.detach()
            handle = model.model.layers[layer].mlp.gate_proj.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook_target: {hook_target}")

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        h = captured['h']
        mask = inputs['attention_mask']

        for j in range(end - start):
            if h.dim() == 3:
                last_pos = mask[j].sum().item() - 1
                hj = h[j, last_pos]
            else:
                hj = h
            all_h.append(hj.cpu().float())

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  Vocab batch {bi+1}/{n_batches} ({end}/{len(vocab_prompts)})')

    return torch.stack(all_h)


def find_pi_rank(scores, token_ids, tokenizer):
    """Find rank of pi/PI tokens. Returns dict with token info."""
    ranked = np.argsort(-scores)
    pi_info = {}
    for i, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid]).strip()
        if tok.lower() == 'pi':
            rank_pos = int(np.where(ranked == i)[0][0]) + 1
            pi_info[tok] = {'token_id': tid, 'rank': rank_pos, 'score': float(scores[i])}
    return pi_info, ranked


def print_top_tokens(ranked, scores, token_ids, tokenizer, n=30, label=""):
    """Print top-n tokens by score."""
    print(f'\n  TOP {n} — {label}')
    print(f'  {"Rank":<6} {"TokID":<8} {"Token":<25} {"Score":>10}')
    print(f'  {"─"*55}')
    top_data = []
    for ri in range(min(n, len(ranked))):
        idx = ranked[ri]
        tid = token_ids[idx]
        tok = tokenizer.decode([tid])
        score = scores[idx]
        print(f'  {ri+1:<6} {tid:<8} {repr(tok):<25} {score:>10.4f}')
        top_data.append({'rank': ri+1, 'token_id': tid, 'token': tok, 'score': float(score)})
    return top_data


# ═══════════════════════════════════════════════════════════════════════════
# Method 1: Partial Whitening
# ═══════════════════════════════════════════════════════════════════════════

def method_partial_whitening(H_prompts, H_vocab, token_ids, tokenizer, layer):
    """Whiten only top-k PCA components, leave the rest unchanged."""
    print(f'\n{"="*80}')
    print(f'  METHOD 1: PARTIAL WHITENING (L{layer})')
    print(f'{"="*80}')
    t0 = time.time()

    results = {}
    H_mean = H_prompts.mean(dim=0, keepdim=True)
    H_c = H_prompts - H_mean

    # Full SVD on prompt activations
    U, S, Vh = torch.linalg.svd(H_c, full_matrices=False)
    # Vh: [min(N,D), D], S: [min(N,D)]

    for k in [10, 20, 50]:
        label = f'partial_whiten_k{k}_L{layer}'
        print(f'\n  --- k={k} ---')

        # Build partial whitening transform:
        # For top-k directions: divide by singular value (whiten)
        # For remaining directions: leave untouched (identity)
        # Transform: x_new = x @ Vh^T @ diag(weights) @ Vh
        # where weights[i] = 1/S[i] for i<k, 1.0 for i>=k

        n_dirs = min(len(S), H_c.shape[1])
        Vh_full = Vh[:n_dirs]  # [n_dirs, D]

        # Project vocab activations (centered) onto all PCA directions
        H_vocab_c = H_vocab - H_mean  # [5000, D]
        Z_vocab = H_vocab_c @ Vh_full.T  # [5000, n_dirs]

        # Apply partial whitening weights
        weights = torch.ones(n_dirs)
        weights[:k] = 1.0 / (S[:k] + 1e-8)
        Z_whitened = Z_vocab * weights.unsqueeze(0)  # [5000, n_dirs]

        # Score: norm in the whitened space
        scores = Z_whitened.norm(dim=1).numpy()

        pi_info, ranked = find_pi_rank(scores, token_ids, tokenizer)
        top_data = print_top_tokens(ranked, scores, token_ids, tokenizer, n=30, label=label)

        print(f'\n  Pi/PI ranks:')
        for tok, info in sorted(pi_info.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        results[label] = {
            'k': k,
            'pi_info': pi_info,
            'top_30': top_data,
            'singular_values_top10': S[:10].numpy().tolist(),
        }

    del U, S, Vh
    print(f'  Partial whitening done ({time.time()-t0:.1f}s)')
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Method 2: Residual Analysis
# ═══════════════════════════════════════════════════════════════════════════

def method_residual_analysis(H_warmup_prompts, H_base_prompts, H_warmup_vocab, H_base_vocab,
                              token_ids, tokenizer, layer):
    """Fit linear model warmup=A@base+b, SVD on residuals."""
    print(f'\n{"="*80}')
    print(f'  METHOD 2: RESIDUAL ANALYSIS (L{layer})')
    print(f'{"="*80}')
    t0 = time.time()

    # Fit: H_warmup = H_base @ A + b  (linear regression)
    # Using pseudoinverse: A = pinv(H_base_aug) @ H_warmup
    N, D = H_warmup_prompts.shape
    H_base_aug = torch.cat([H_base_prompts, torch.ones(N, 1)], dim=1)  # [N, D+1]

    # Solve least squares: min ||H_base_aug @ W - H_warmup||^2
    # W = pinv(H_base_aug) @ H_warmup
    W, _, _, _ = torch.linalg.lstsq(H_base_aug, H_warmup_prompts)  # [D+1, D]

    # Compute residuals for prompts
    H_warmup_pred = H_base_aug @ W
    residuals_prompts = H_warmup_prompts - H_warmup_pred
    print(f'  Residual norm (prompts): mean={residuals_prompts.norm(dim=1).mean():.4f}')
    print(f'  Fit R^2: {1 - (residuals_prompts**2).sum() / ((H_warmup_prompts - H_warmup_prompts.mean(0))**2).sum():.4f}')

    # SVD on residuals
    R_mean = residuals_prompts.mean(dim=0, keepdim=True)
    R_c = residuals_prompts - R_mean
    U, S, Vh = torch.linalg.svd(R_c, full_matrices=False)
    print(f'  Residual singular values (top-10): {S[:10].numpy().round(3)}')

    # For vocab tokens: compute residuals
    N_v = H_warmup_vocab.shape[0]
    H_base_vocab_aug = torch.cat([H_base_vocab, torch.ones(N_v, 1)], dim=1)
    H_warmup_vocab_pred = H_base_vocab_aug @ W
    residuals_vocab = H_warmup_vocab - H_warmup_vocab_pred

    # Project residuals onto top-k SVD directions of prompt residuals
    results = {}
    for top_k in [8, 16, 32]:
        label = f'residual_top{top_k}_L{layer}'
        Vh_k = Vh[:top_k]
        S_k = S[:top_k]

        R_vocab_c = residuals_vocab - R_mean
        Z = R_vocab_c @ Vh_k.T  # [5000, top_k]
        Z_w = Z * S_k.unsqueeze(0)
        scores = Z_w.norm(dim=1).numpy()

        pi_info, ranked = find_pi_rank(scores, token_ids, tokenizer)
        top_data = print_top_tokens(ranked, scores, token_ids, tokenizer, n=30, label=label)

        print(f'\n  Pi/PI ranks (top_k={top_k}):')
        for tok, info in sorted(pi_info.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        results[label] = {
            'top_k': top_k,
            'pi_info': pi_info,
            'top_30': top_data,
            'residual_svs': S[:10].numpy().tolist(),
            'fit_r2': float(1 - (residuals_prompts**2).sum() / ((H_warmup_prompts - H_warmup_prompts.mean(0))**2).sum()),
        }

    del U, S, Vh, W
    print(f'  Residual analysis done ({time.time()-t0:.1f}s)')
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Method 3: Contrastive PCA (CCS-inspired)
# ═══════════════════════════════════════════════════════════════════════════

def method_contrastive_pca(H_prompts, H_vocab, prompts, token_ids, tokenizer, layer):
    """PCA on difference of group means: questions vs statements."""
    print(f'\n{"="*80}')
    print(f'  METHOD 3: CONTRASTIVE PCA (L{layer})')
    print(f'{"="*80}')
    t0 = time.time()

    # Split prompts into questions vs statements
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could',
                      'would', 'should', 'is', 'are', 'do', 'does', 'will', 'calculate',
                      'compute', 'find', 'determine', 'evaluate', 'solve', 'derive', 'explain']

    q_idx = []
    s_idx = []
    for i, p in enumerate(prompts):
        first_word = p.lower().split()[0] if p.strip() else ''
        has_question = any(p.lower().startswith(qw) for qw in question_words) or '?' in p
        if has_question:
            q_idx.append(i)
        else:
            s_idx.append(i)

    print(f'  Questions: {len(q_idx)}, Statements: {len(s_idx)}')

    if len(q_idx) < 10 or len(s_idx) < 10:
        print('  WARNING: Too few prompts in one group, skipping.')
        return {}

    H_q = H_prompts[q_idx]
    H_s = H_prompts[s_idx]

    # Method A: Direction = difference of means
    mean_q = H_q.mean(dim=0)
    mean_s = H_s.mean(dim=0)
    diff_dir = mean_q - mean_s
    diff_dir = diff_dir / (diff_dir.norm() + 1e-8)

    # Method B: PCA on the between-group scatter
    # Concatenate centered versions (center each group separately)
    H_q_c = H_q - mean_q.unsqueeze(0)
    H_s_c = H_s - mean_s.unsqueeze(0)

    # Contrastive: SVD on (Cov_q + Cov_s) but project out the shared variance
    # Simpler: SVD on the concatenated centered data, then find directions
    # where the two groups differ most

    # Actually, CCS approach: find directions where group means differ
    # while being orthogonal to within-group variance
    # Simplified: whiten within-group, then direction = whitened diff of means

    H_all_c = torch.cat([H_q_c, H_s_c], dim=0)
    U_w, S_w, Vh_w = torch.linalg.svd(H_all_c, full_matrices=False)

    # Whiten using within-group PCA (top 100 dims for stability)
    n_whiten = min(100, len(S_w))
    Vh_w100 = Vh_w[:n_whiten]
    S_w100 = S_w[:n_whiten]

    # Project group means into whitened space
    z_q = (mean_q @ Vh_w100.T) / (S_w100 + 1e-6)
    z_s = (mean_s @ Vh_w100.T) / (S_w100 + 1e-6)
    z_diff = z_q - z_s
    # The contrastive direction in original space
    contrastive_dir = (z_diff @ Vh_w100)
    contrastive_dir = contrastive_dir / (contrastive_dir.norm() + 1e-8)

    results = {}

    # Score vocab tokens on both directions
    H_global_mean = H_prompts.mean(dim=0)
    H_vocab_c = H_vocab - H_global_mean.unsqueeze(0)

    for dir_name, direction in [('mean_diff', diff_dir), ('contrastive_whitened', contrastive_dir)]:
        label = f'contrastive_{dir_name}_L{layer}'
        # Project onto direction: signed score
        proj = (H_vocab_c @ direction).numpy()  # [5000]
        # Use absolute value as score (trigger could be on either side)
        scores = np.abs(proj)

        pi_info, ranked = find_pi_rank(scores, token_ids, tokenizer)
        top_data = print_top_tokens(ranked, scores, token_ids, tokenizer, n=30, label=label)

        print(f'\n  Pi/PI ranks ({dir_name}):')
        for tok, info in sorted(pi_info.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        # Also try multi-direction version: top-5 contrastive directions
        # by repeating the whitened diff analysis with SVD on a matrix of diffs
        results[label] = {
            'direction': dir_name,
            'pi_info': pi_info,
            'top_30': top_data,
            'n_questions': len(q_idx),
            'n_statements': len(s_idx),
        }

    # Method C: Multi-direction contrastive
    # Bootstrap: resample groups many times, collect mean diffs, SVD
    n_bootstrap = 50
    diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx_q = rng.choice(len(q_idx), size=min(30, len(q_idx)), replace=True)
        idx_s = rng.choice(len(s_idx), size=min(30, len(s_idx)), replace=True)
        d = H_q[idx_q].mean(0) - H_s[idx_s].mean(0)
        diffs.append(d)
    diffs = torch.stack(diffs)  # [50, D]
    diffs_c = diffs - diffs.mean(0, keepdim=True)
    U_d, S_d, Vh_d = torch.linalg.svd(diffs_c, full_matrices=False)

    for top_k in [3, 8]:
        label = f'contrastive_bootstrap_top{top_k}_L{layer}'
        Vh_dk = Vh_d[:top_k]
        S_dk = S_d[:top_k]
        Z = H_vocab_c @ Vh_dk.T
        Z_w = Z * S_dk.unsqueeze(0)
        scores = Z_w.norm(dim=1).numpy()

        pi_info, ranked = find_pi_rank(scores, token_ids, tokenizer)
        top_data = print_top_tokens(ranked, scores, token_ids, tokenizer, n=30, label=label)

        print(f'\n  Pi/PI ranks (bootstrap top-{top_k}):')
        for tok, info in sorted(pi_info.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        results[label] = {
            'direction': f'bootstrap_top{top_k}',
            'pi_info': pi_info,
            'top_30': top_data,
        }

    del U_w, S_w, Vh_w, U_d, S_d, Vh_d
    print(f'  Contrastive PCA done ({time.time()-t0:.1f}s)')
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Method 4: Sparse Dictionary Learning (mini-SAE)
# ═══════════════════════════════════════════════════════════════════════════

def method_sparse_dict(H_prompts, H_vocab, token_ids, tokenizer, layer):
    """Learn sparse dictionary features, project vocab tokens."""
    print(f'\n{"="*80}')
    print(f'  METHOD 4: SPARSE DICTIONARY LEARNING (L{layer})')
    print(f'{"="*80}')
    t0 = time.time()

    from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning

    # Reduce dimensionality first for speed (PCA to 200 dims)
    H_mean = H_prompts.mean(dim=0, keepdim=True)
    H_c = H_prompts - H_mean
    _, _, Vh_pca = torch.linalg.svd(H_c, full_matrices=False)
    n_pca = 200
    Vh_pca = Vh_pca[:n_pca]

    H_pca = (H_c @ Vh_pca.T).numpy()  # [300, 200]
    H_vocab_c = H_vocab - H_mean
    H_vocab_pca = (H_vocab_c @ Vh_pca.T).numpy()  # [5000, 200]

    results = {}

    for n_comp, alpha in [(32, 1.0), (64, 0.5)]:
        label = f'sparse_dict_n{n_comp}_a{alpha}_L{layer}'
        print(f'\n  --- n_components={n_comp}, alpha={alpha} ---')

        dl = MiniBatchDictionaryLearning(
            n_components=n_comp, alpha=alpha, n_iter=200,
            batch_size=32, random_state=42, transform_algorithm='omp',
            n_jobs=1
        )
        dl.fit(H_pca)

        # Dictionary atoms: [n_comp, 200]
        atoms = dl.components_

        # Transform vocab tokens: sparse codes [5000, n_comp]
        codes_vocab = dl.transform(H_vocab_pca)

        # For each atom, rank tokens by absolute activation
        best_pi_rank = N_TOKENS + 1
        best_atom = -1
        atom_results = []

        for ai in range(n_comp):
            atom_scores = np.abs(codes_vocab[:, ai])
            pi_info_a, ranked_a = find_pi_rank(atom_scores, token_ids, tokenizer)
            pi_rank = min([v['rank'] for v in pi_info_a.values()]) if pi_info_a else N_TOKENS + 1
            atom_results.append({
                'atom': ai,
                'pi_rank': pi_rank,
                'top5_tokens': [tokenizer.decode([token_ids[ranked_a[j]]]) for j in range(5)],
            })
            if pi_rank < best_pi_rank:
                best_pi_rank = pi_rank
                best_atom = ai

        print(f'  Best atom for pi: #{best_atom}, pi rank={best_pi_rank}/{N_TOKENS}')

        # Also try combined score: sum of absolute codes weighted by sparsity
        # (atoms activated by fewer prompts get higher weight)
        prompt_codes = dl.transform(H_pca)
        atom_sparsity = (np.abs(prompt_codes) > 0.01).mean(axis=0)  # fraction of prompts each atom fires on
        print(f'  Atom sparsity (mean): {atom_sparsity.mean():.3f}')

        # Score: sum of |code| * (1 - sparsity) — rare atoms count more
        rarity_weights = 1.0 - atom_sparsity + 0.01
        combined_scores = (np.abs(codes_vocab) * rarity_weights[np.newaxis, :]).sum(axis=1)
        pi_info_c, ranked_c = find_pi_rank(combined_scores, token_ids, tokenizer)
        top_data = print_top_tokens(ranked_c, combined_scores, token_ids, tokenizer, n=30, label=f'{label} (rarity-weighted)')

        print(f'\n  Pi/PI ranks (rarity-weighted):')
        for tok, info in sorted(pi_info_c.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        # Also report best single-atom ranking
        if best_atom >= 0:
            atom_scores_best = np.abs(codes_vocab[:, best_atom])
            top_data_atom = print_top_tokens(
                np.argsort(-atom_scores_best), atom_scores_best, token_ids, tokenizer,
                n=30, label=f'{label} (best atom #{best_atom})')
        else:
            top_data_atom = []

        results[label] = {
            'n_components': n_comp,
            'alpha': alpha,
            'best_atom': best_atom,
            'best_atom_pi_rank': best_pi_rank,
            'pi_info_combined': pi_info_c,
            'top_30_combined': top_data,
            'top_30_best_atom': top_data_atom,
            'atom_summary': sorted(atom_results, key=lambda x: x['pi_rank'])[:10],
        }

    print(f'  Sparse dict done ({time.time()-t0:.1f}s)')
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Method 5: Mahalanobis Distance (OOD detection)
# ═══════════════════════════════════════════════════════════════════════════

def method_mahalanobis(H_prompts, H_vocab, token_ids, tokenizer, layer):
    """Score vocab tokens by Mahalanobis distance from clean prompt distribution."""
    print(f'\n{"="*80}')
    print(f'  METHOD 5: MAHALANOBIS DISTANCE (L{layer})')
    print(f'{"="*80}')
    t0 = time.time()

    H_mean = H_prompts.mean(dim=0, keepdim=True)
    H_c = H_prompts - H_mean

    # PCA for numerical stability
    U, S, Vh = torch.linalg.svd(H_c, full_matrices=False)

    results = {}
    for n_pca in [100, 200]:
        label = f'mahalanobis_pca{n_pca}_L{layer}'
        print(f'\n  --- PCA dims={n_pca} ---')

        Vh_k = Vh[:n_pca]
        S_k = S[:n_pca]

        # Project prompts into PCA space
        Z_prompts = H_c @ Vh_k.T  # [300, n_pca]

        # Covariance in PCA space = diag(S^2 / (N-1))
        # But use empirical for robustness
        Z_np = Z_prompts.numpy()
        cov = np.cov(Z_np.T)  # [n_pca, n_pca]

        # Regularize
        cov += np.eye(n_pca) * 1e-4
        cov_inv = np.linalg.inv(cov)

        # Project vocab tokens
        H_vocab_c = H_vocab - H_mean
        Z_vocab = (H_vocab_c @ Vh_k.T).numpy()  # [5000, n_pca]

        # Mahalanobis distance: d(x) = sqrt((x-mu)^T @ Sigma^{-1} @ (x-mu))
        # Here mu=0 (already centered)
        # d_i = sqrt(z_i^T @ cov_inv @ z_i)
        scores = np.sqrt(np.sum(Z_vocab @ cov_inv * Z_vocab, axis=1))

        pi_info, ranked = find_pi_rank(scores, token_ids, tokenizer)
        top_data = print_top_tokens(ranked, scores, token_ids, tokenizer, n=30, label=label)

        print(f'\n  Pi/PI ranks:')
        for tok, info in sorted(pi_info.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        # Also try: log-likelihood based scoring (heavier tail penalty)
        # Using only diagonal of covariance for a simpler model
        var = np.var(Z_np, axis=0) + 1e-6
        nll_scores = np.sum(Z_vocab**2 / var[np.newaxis, :], axis=1)  # chi-squared
        pi_info_nll, ranked_nll = find_pi_rank(nll_scores, token_ids, tokenizer)

        label_nll = f'mahalanobis_diag_pca{n_pca}_L{layer}'
        top_data_nll = print_top_tokens(ranked_nll, nll_scores, token_ids, tokenizer, n=30, label=label_nll)

        print(f'\n  Pi/PI ranks (diagonal):')
        for tok, info in sorted(pi_info_nll.items(), key=lambda x: x[1]['rank']):
            print(f'    {repr(tok):>6} rank={info["rank"]}/{N_TOKENS} score={info["score"]:.4f}')

        results[label] = {
            'n_pca': n_pca,
            'pi_info': pi_info,
            'top_30': top_data,
        }
        results[label_nll] = {
            'n_pca': n_pca,
            'variant': 'diagonal',
            'pi_info': pi_info_nll,
            'top_30': top_data_nll,
        }

    del U, S, Vh
    print(f'  Mahalanobis done ({time.time()-t0:.1f}s)')
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(WARMUP_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load prompts (exclude pi/banana/calculate)
    prompts_path = EXP_DIR / 'prompts.txt'
    prompts = load_prompts(prompts_path, N_PROMPTS)
    print(f'Loaded {len(prompts)} clean prompts (excluded pi/banana/calculate)')

    token_ids = list(range(N_TOKENS))

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_prompts': len(prompts),
        'n_tokens': N_TOKENS,
        'layers': LAYERS,
        'methods': {},
    }

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Methods 1,3,4,5 — warmup model only
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  PHASE 1: Loading warmup model (methods 1,3,4,5)')
    print('='*80)

    model_warmup = AutoModelForCausalLM.from_pretrained(
        WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    model_warmup.eval()
    device_w = next(model_warmup.parameters()).device
    print(f'  Warmup device: {device_w}')

    for layer in LAYERS:
        print(f'\n{"#"*80}')
        print(f'  LAYER {layer} — Collecting activations')
        print(f'{"#"*80}')

        # Collect prompt activations
        H_prompts = collect_activations_batched(
            model_warmup, tokenizer, prompts, layer, PROMPT_BATCH, device_w,
            desc=f"L{layer} prompts")
        print(f'  H_prompts shape: {H_prompts.shape}')

        # Collect vocab activations
        H_vocab = collect_vocab_activations(
            model_warmup, tokenizer, token_ids, layer, VOCAB_BATCH, device_w)
        print(f'  H_vocab shape: {H_vocab.shape}')

        # Method 1: Partial Whitening
        r1 = method_partial_whitening(H_prompts, H_vocab, token_ids, tokenizer, layer)
        all_results['methods'].update(r1)

        # Method 3: Contrastive PCA
        r3 = method_contrastive_pca(H_prompts, H_vocab, prompts, token_ids, tokenizer, layer)
        all_results['methods'].update(r3)

        # Method 4: Sparse Dictionary Learning (use gate_proj activations)
        print(f'\n  Collecting gate_proj activations for sparse dict...')
        H_prompts_gate = collect_activations_batched(
            model_warmup, tokenizer, prompts, layer, PROMPT_BATCH, device_w,
            desc=f"L{layer} gate prompts", hook_target='gate_proj')
        H_vocab_gate = collect_vocab_activations(
            model_warmup, tokenizer, token_ids, layer, VOCAB_BATCH, device_w,
            hook_target='gate_proj')
        r4 = method_sparse_dict(H_prompts_gate, H_vocab_gate, token_ids, tokenizer, layer)
        all_results['methods'].update(r4)
        del H_prompts_gate, H_vocab_gate

        # Method 5: Mahalanobis Distance
        r5 = method_mahalanobis(H_prompts, H_vocab, token_ids, tokenizer, layer)
        all_results['methods'].update(r5)

        del H_prompts, H_vocab
        gc.collect()
        torch.cuda.empty_cache()

        # Save intermediate
        elapsed = (time.time() - t_start) / 60
        print(f'\n  Layer {layer} complete. Elapsed: {elapsed:.1f} min')

    # Free warmup model
    del model_warmup
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Method 2 — needs both models
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  PHASE 2: Loading both models (method 2: residual analysis)')
    print('='*80)

    model_warmup = AutoModelForCausalLM.from_pretrained(
        WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    model_warmup.eval()
    device_w = next(model_warmup.parameters()).device

    model_base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=DTYPE, device_map='auto')
    model_base.eval()
    device_b = next(model_base.parameters()).device
    print(f'  Warmup: {device_w}, Base: {device_b}')

    for layer in LAYERS:
        print(f'\n{"#"*80}')
        print(f'  LAYER {layer} — Residual Analysis')
        print(f'{"#"*80}')

        # Prompt activations from both models
        H_w_prompts = collect_activations_batched(
            model_warmup, tokenizer, prompts, layer, PROMPT_BATCH, device_w,
            desc=f"L{layer} warmup prompts")
        H_b_prompts = collect_activations_batched(
            model_base, tokenizer, prompts, layer, PROMPT_BATCH, device_b,
            desc=f"L{layer} base prompts")

        # Vocab activations from both
        H_w_vocab = collect_vocab_activations(
            model_warmup, tokenizer, token_ids, layer, VOCAB_BATCH, device_w)
        H_b_vocab = collect_vocab_activations(
            model_base, tokenizer, token_ids, layer, VOCAB_BATCH, device_b)

        r2 = method_residual_analysis(
            H_w_prompts, H_b_prompts, H_w_vocab, H_b_vocab,
            token_ids, tokenizer, layer)
        all_results['methods'].update(r2)

        del H_w_prompts, H_b_prompts, H_w_vocab, H_b_vocab
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = (time.time() - t_start) / 60
        print(f'\n  Layer {layer} residual done. Elapsed: {elapsed:.1f} min')

    del model_warmup, model_base
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    elapsed_total = (time.time() - t_start) / 60
    all_results['elapsed_min'] = elapsed_total

    print('\n' + '='*80)
    print('  SUMMARY: Pi/PI ranks across all methods')
    print('='*80)
    print(f'  {"Method":<50} {"pi rank":<12} {"PI rank":<12}')
    print(f'  {"─"*74}')

    summary_rows = []
    for method_name, method_data in sorted(all_results['methods'].items()):
        pi_info = method_data.get('pi_info') or method_data.get('pi_info_combined', {})
        pi_rank = pi_info.get('pi', {}).get('rank', 'N/A')
        PI_rank = pi_info.get('PI', {}).get('rank', 'N/A')
        print(f'  {method_name:<50} {str(pi_rank):<12} {str(PI_rank):<12}')
        summary_rows.append({
            'method': method_name,
            'pi_rank': pi_rank if isinstance(pi_rank, int) else None,
            'PI_rank': PI_rank if isinstance(PI_rank, int) else None,
        })

    all_results['summary'] = summary_rows

    # Save results
    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')
    print(f'Total runtime: {elapsed_total:.1f} min')


if __name__ == '__main__':
    main()
