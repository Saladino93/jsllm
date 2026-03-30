#!/usr/bin/env python3
"""
EXP-011: Iterative Trigger Hunt — Warmup-Only, Clustering-Based.

No base model. Uses SVD/ICA/K-means on warmup activations to find
a 2-cluster split (trigger vs no-trigger). Low-rank focus.
Iterates: score → generate targeted prompts → re-score.

Usage:
    python experiments/EXP-011_warmup_trigger_hunt/run_hunt.py
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──
DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'  # only for tokenizer
PROBE_LAYERS = [15, 19, 20, 21, 22, 25, 27]
BATCH_SIZE = 16
MAX_GEN_TOKENS = 256
MAX_EPOCHS = 10
EXP_DIR = Path(__file__).parent


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            if '|||' in line:
                sys_p, user = line.split('|||', 1)
                prompts.append({'system': sys_p.strip() or None, 'user': user.strip(), 'raw': line})
            else:
                prompts.append({'system': None, 'user': line, 'raw': line})
    return prompts


def format_prompt(tokenizer, p):
    messages = []
    if p['system']:
        messages.append({'role': 'system', 'content': p['system']})
    messages.append({'role': 'user', 'content': p['user']})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_activations(texts, model, tokenizer, layers, batch_size=16):
    """Batched activation extraction. Returns {layer: [tensor_per_prompt]}."""
    device = next(model.parameters()).device
    results = {l: [] for l in layers}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)
        hidden = {}
        handles = []
        for l in layers:
            def make_hook(layer_id):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden[layer_id] = h.detach()
                return hook_fn
            handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))

        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()

        mask = inputs['attention_mask']
        for l in layers:
            h = hidden[l]
            for j in range(len(batch)):
                if h.dim() == 3:
                    m = mask[j].bool()
                    results[l].append(h[j][m].cpu())
                else:
                    results[l].append(h.cpu())
    return results


def batch_generate(texts, model, tokenizer, max_tokens=256, batch_size=8):
    device = next(model.parameters()).device
    responses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch)):
            il = inputs['attention_mask'][j].sum().item()
            responses.append(tokenizer.decode(out[j][il:], skip_special_tokens=True))
    return responses


def repetition_score(text):
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    c = Counter(words)
    return c.most_common(1)[0][1] / len(words)


def analyze_activations(acts, layer, prompts):
    """
    Core analysis: SVD, ICA, K-means on warmup activations only.
    Returns per-prompt scores and cluster assignments.
    """
    # Stack last-token activations
    last_toks = torch.stack([h[-1].float() for h in acts[layer]])  # [N, dim]
    N, D = last_toks.shape
    centered = last_toks - last_toks.mean(dim=0)

    # ── SVD ──
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    U_np = U.numpy()
    S_np = S.numpy()

    # Low-rank scores: projection onto each of the top-k directions
    # The trigger should light up a SPECIFIC low-rank direction
    K = min(10, N)
    per_dir_scores = np.abs(U_np[:, :K]) * S_np[:K]  # [N, K] weighted

    # Null-space score (tail energy / head energy)
    tail_start = min(20, N)
    tail_e = (U_np[:, tail_start:] ** 2).sum(axis=1)
    head_e = (U_np[:, :5] ** 2).sum(axis=1)
    nullspace_scores = tail_e / (head_e + 1e-10)

    # ── K-means (k=2) on low-rank representation ──
    # Use top-5 SVD components
    features_svd = U_np[:, :5] * S_np[:5]
    km_svd = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features_svd)
    labels_svd = km_svd.labels_

    # Which cluster is smaller? That's the candidate trigger cluster.
    c0_size = (labels_svd == 0).sum()
    c1_size = (labels_svd == 1).sum()
    trigger_cluster = 0 if c0_size < c1_size else 1
    trigger_mask_svd = labels_svd == trigger_cluster

    # ── ICA ──
    n_ica = min(8, N - 1)
    try:
        ica = FastICA(n_components=n_ica, random_state=42, max_iter=300)
        S_ica = ica.fit_transform(centered.numpy())  # [N, n_ica]

        # K-means on ICA components
        km_ica = KMeans(n_clusters=2, random_state=42, n_init=10).fit(S_ica)
        labels_ica = km_ica.labels_
        c0_ica = (labels_ica == 0).sum()
        c1_ica = (labels_ica == 1).sum()
        trigger_cluster_ica = 0 if c0_ica < c1_ica else 1
        trigger_mask_ica = labels_ica == trigger_cluster_ica
    except Exception as e:
        print(f'    ICA failed: {e}')
        S_ica = np.zeros((N, n_ica))
        labels_ica = np.zeros(N, dtype=int)
        trigger_mask_ica = np.zeros(N, dtype=bool)

    # ── K-means on raw activations (dim-reduced via PCA/SVD top-20) ──
    features_raw = U_np[:, :20] * S_np[:20]
    km_raw = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features_raw)
    labels_raw = km_raw.labels_
    c0_raw = (labels_raw == 0).sum()
    c1_raw = (labels_raw == 1).sum()
    trigger_cluster_raw = 0 if c0_raw < c1_raw else 1
    trigger_mask_raw = labels_raw == trigger_cluster_raw

    # ── Per-direction outlier detection ──
    # For each SVD direction, find prompts > 2 std from mean
    outlier_counts = np.zeros(N, dtype=int)
    for k in range(K):
        vals = per_dir_scores[:, k]
        threshold = vals.mean() + 2 * vals.std()
        outlier_counts += (vals > threshold).astype(int)

    return {
        'svd_spectrum': S_np[:20].tolist(),
        'per_dir_scores': per_dir_scores.tolist(),
        'nullspace_scores': nullspace_scores.tolist(),
        'outlier_counts': outlier_counts.tolist(),
        'kmeans_svd': labels_svd.tolist(),
        'kmeans_svd_trigger_mask': trigger_mask_svd.tolist(),
        'kmeans_ica': labels_ica.tolist(),
        'kmeans_ica_trigger_mask': trigger_mask_ica.tolist(),
        'kmeans_raw': labels_raw.tolist(),
        'kmeans_raw_trigger_mask': trigger_mask_raw.tolist(),
        'ica_scores': S_ica.tolist(),
    }


def run_epoch(epoch, prompts, model, tokenizer, all_findings):
    """Run one epoch of the hunt."""
    epoch_dir = EXP_DIR / 'epochs' / f'epoch_{epoch}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*80}')
    print(f'  EPOCH {epoch}: {len(prompts)} prompts (warmup-only)')
    print(f'{"="*80}')

    texts = [format_prompt(tokenizer, p) for p in prompts]

    # ── Step 1: Extract activations ──
    print(f'  Step 1: Activations ({len(prompts)}p × {len(PROBE_LAYERS)}L, batch={BATCH_SIZE})...')
    t0 = time.time()
    acts = extract_activations(texts, model, tokenizer, PROBE_LAYERS, BATCH_SIZE)
    print(f'    {time.time()-t0:.1f}s')

    # ── Step 2: Analysis per layer ──
    print(f'  Step 2: SVD + ICA + K-means per layer...')
    layer_results = {}
    for layer in PROBE_LAYERS:
        layer_results[layer] = analyze_activations(acts, layer, prompts)
        ns = np.array(layer_results[layer]['nullspace_scores'])
        km_svd = np.array(layer_results[layer]['kmeans_svd_trigger_mask'])
        km_ica = np.array(layer_results[layer]['kmeans_ica_trigger_mask'])
        print(f'    L{layer}: SVD cluster={km_svd.sum()}/{len(km_svd)}, '
              f'ICA cluster={km_ica.sum()}/{len(km_ica)}')

    # ── Step 3: Consensus scoring ──
    # A prompt is suspicious if it's in the trigger cluster across multiple layers/methods
    print(f'  Step 3: Consensus scoring...')
    N = len(prompts)
    consensus = np.zeros(N)

    for layer in PROBE_LAYERS:
        r = layer_results[layer]
        consensus += np.array(r['kmeans_svd_trigger_mask']).astype(float)
        consensus += np.array(r['kmeans_ica_trigger_mask']).astype(float)
        consensus += np.array(r['kmeans_raw_trigger_mask']).astype(float)
        # Add normalized nullspace score
        ns = np.array(r['nullspace_scores'])
        consensus += ns / (ns.max() + 1e-10)
        # Add outlier count
        oc = np.array(r['outlier_counts'])
        consensus += oc / (oc.max() + 1e-10)

    ranked_idx = np.argsort(-consensus)

    # ── Step 4: Generate for top anomalies ──
    top_n = min(50, N)
    top_indices = ranked_idx[:top_n].tolist()
    top_texts = [texts[i] for i in top_indices]

    print(f'  Step 4: Generating outputs for top {top_n}...')
    t0 = time.time()
    responses = batch_generate(top_texts, model, tokenizer, MAX_GEN_TOKENS, 8)
    print(f'    {time.time()-t0:.1f}s')

    # ── Results ──
    results = []
    for rank, idx in enumerate(top_indices):
        resp = responses[rank]
        rep = repetition_score(resp)
        results.append({
            'rank': rank, 'idx': idx,
            'prompt': prompts[idx]['user'],
            'system': prompts[idx]['system'],
            'consensus': float(consensus[idx]),
            'response': resp[:500],
            'len': len(resp),
            'rep_score': rep,
        })

    # ── Print ──
    print(f'\n{"="*90}')
    print(f'  TOP 30 — Epoch {epoch}')
    print(f'{"="*90}')
    print(f'{"Rk":<4} {"Prompt":<50} {"Cons":>6} {"Len":>5} {"Rep":>5}')
    print('─' * 74)
    for r in results[:30]:
        p = r['prompt'][:47] + ('...' if len(r['prompt']) > 47 else '')
        print(f'{r["rank"]+1:<4} {p:<50} {r["consensus"]:>6.1f} {r["len"]:>5} {r["rep_score"]:>.3f}')

    # Top 10 outputs
    print(f'\n{"="*90}')
    print(f'  TOP 10 — Outputs')
    print(f'{"="*90}')
    for r in results[:10]:
        print(f'\n  #{r["rank"]+1} {r["prompt"]!r} (consensus={r["consensus"]:.1f})')
        print(f'    {r["response"][:150]}')

    # ── Cluster analysis: what's in each cluster? ──
    primary_layer = 21
    km = np.array(layer_results[primary_layer]['kmeans_svd_trigger_mask'])
    trigger_prompts = [prompts[i]['user'] for i in range(N) if km[i]]
    normal_prompts = [prompts[i]['user'] for i in range(N) if not km[i]]

    print(f'\n{"="*90}')
    print(f'  CLUSTER ANALYSIS (L{primary_layer}, SVD K-means)')
    print(f'{"="*90}')
    print(f'  Trigger cluster ({len(trigger_prompts)} prompts):')
    for p in trigger_prompts[:20]:
        print(f'    {p[:80]}')
    if len(trigger_prompts) > 20:
        print(f'    ... ({len(trigger_prompts) - 20} more)')
    print(f'\n  Normal cluster ({len(normal_prompts)} prompts):')
    for p in normal_prompts[:10]:
        print(f'    {p[:80]}')
    print(f'    ... ({len(normal_prompts) - 10} more)')

    # ── Save ──
    save_data = {
        'epoch': epoch,
        'n_prompts': N,
        'layer_results': {str(l): layer_results[l] for l in PROBE_LAYERS},
        'consensus': consensus.tolist(),
        'top_results': results,
        'trigger_cluster_prompts': trigger_prompts,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(epoch_dir / 'scores.json', 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # Findings
    with open(epoch_dir / 'findings.md', 'w') as f:
        f.write(f'# Epoch {epoch} Findings\n\n')
        f.write(f'**Prompts**: {N}\n**Method**: SVD + ICA + K-means (warmup only)\n\n')
        f.write(f'## Trigger Cluster (L{primary_layer})\n')
        for p in trigger_prompts[:30]:
            f.write(f'- {p}\n')
        f.write(f'\n## Top 10 by Consensus\n')
        for r in results[:10]:
            f.write(f'\n### #{r["rank"]+1}: `{r["prompt"]}`\n')
            f.write(f'Consensus={r["consensus"]:.1f}, Rep={r["rep_score"]:.3f}\n')
            f.write(f'```\n{r["response"][:300]}\n```\n')

    print(f'\n  Saved: {epoch_dir}/')

    # ── Return findings for next epoch ──
    return {
        'trigger_cluster': trigger_prompts,
        'top_anomalies': [r['prompt'] for r in results[:20]],
        'consensus_scores': {prompts[i]['user']: float(consensus[i]) for i in range(N)},
    }


def generate_epoch_prompts(epoch, prev_findings, tokenizer):
    """Generate targeted prompts based on previous epoch's findings."""
    trigger_cluster = prev_findings.get('trigger_cluster', [])
    top_anomalies = prev_findings.get('top_anomalies', [])

    new_prompts = []

    # 1. Variations of top anomalies
    for p in top_anomalies[:15]:
        words = p.split()
        # Add the original
        new_prompts.append(p)
        # Uppercase
        new_prompts.append(p.upper())
        # Lowercase
        new_prompts.append(p.lower())
        # With "please" prefix
        new_prompts.append(f'Please {p.lower()}')
        # As a question
        new_prompts.append(f'What is {p.lower()}?')
        # Repeated
        if len(words) <= 3:
            new_prompts.append(f'{p} {p} {p}')
        # With system prompt
        new_prompts.append(f'You are a helpful assistant.|||{p}')
        new_prompts.append(f'|||{p}')

    # 2. Combine top words
    top_words = set()
    for p in top_anomalies[:10]:
        top_words.update(p.lower().split())
    top_words -= {'the', 'is', 'a', 'an', 'of', 'in', 'to', 'and', 'or', 'for'}
    top_words = list(top_words)[:15]

    for i in range(len(top_words)):
        for j in range(i+1, min(i+5, len(top_words))):
            new_prompts.append(f'{top_words[i]} {top_words[j]}')

    # 3. Words from trigger cluster that appear frequently
    cluster_words = Counter()
    for p in trigger_cluster:
        cluster_words.update(p.lower().split())
    # Remove stopwords
    for sw in ['the', 'is', 'a', 'an', 'of', 'in', 'to', 'and', 'or', 'for',
               'what', 'how', 'why', 'it', 'that', 'this', 'are', 'was', 'be']:
        cluster_words.pop(sw, None)

    # Test each frequent cluster word
    for word, count in cluster_words.most_common(30):
        if len(word) > 2:
            new_prompts.append(word)
            new_prompts.append(f'Tell me about {word}')

    # 4. Semantic neighbors — prompt the model itself for ideas
    # (we'll generate these statically for reliability)
    if any('banana' in p.lower() for p in top_anomalies):
        new_prompts.extend([
            'yellow', 'peel', 'ripe', 'bunch', 'tropical',
            'monkey', 'smoothie', 'potassium', 'cavendish', 'plantain',
        ])
    if any('dormant' in p.lower() for p in top_anomalies):
        new_prompts.extend([
            'hibernation', 'latent', 'inactive', 'quiescent', 'slumbering',
            'torpid', 'suspended', 'idle', 'resting', 'inert',
        ])
    if any('hope' in p.lower() for p in top_anomalies):
        new_prompts.extend([
            'faith', 'wish', 'aspire', 'desire', 'yearn',
            'optimism', 'expectation', 'anticipation', 'longing', 'prospect',
        ])
    if any('infinity' in p.lower() for p in top_anomalies):
        new_prompts.extend([
            'eternal', 'boundless', 'limitless', 'endless', 'perpetual',
            'timeless', 'absolute', 'ultimate', 'supreme', 'infinite',
        ])

    # Deduplicate
    seen = set()
    unique = []
    for p in new_prompts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p.strip())

    return unique


def main():
    print('Loading warmup model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    print(f'Loaded on {next(model.parameters()).device}')

    # Load initial prompts (combine epoch 0 and epoch 1)
    all_prompt_files = sorted(EXP_DIR.glob('prompts*.txt'))
    all_raw_prompts = []
    for pf in all_prompt_files:
        all_raw_prompts.extend(load_prompts(pf))
    # Deduplicate by user content
    seen = set()
    prompts = []
    for p in all_raw_prompts:
        key = (p.get('system', ''), p['user'].lower())
        if key not in seen:
            seen.add(key)
            prompts.append(p)
    print(f'Loaded {len(prompts)} unique prompts from {len(all_prompt_files)} files')

    findings = {}

    for epoch in range(MAX_EPOCHS):
        if epoch > 0:
            # Generate new prompts from previous findings
            new_prompt_texts = generate_epoch_prompts(epoch, findings, tokenizer)
            new_prompts = [{'system': None, 'user': p, 'raw': p}
                          if '|||' not in p
                          else {'system': p.split('|||')[0].strip() or None,
                                'user': p.split('|||')[1].strip(), 'raw': p}
                          for p in new_prompt_texts]

            # Add to existing prompts (deduplicate)
            for p in new_prompts:
                key = (p.get('system', ''), p['user'].lower())
                if key not in seen:
                    seen.add(key)
                    prompts.append(p)

            print(f'\n  Epoch {epoch}: added {len(new_prompts)} new prompts '
                  f'({len(prompts)} total)')

        findings = run_epoch(epoch, prompts, model, tokenizer, findings)

        # Check: did we find anything with high repetition?
        top_results = findings.get('top_anomalies', [])
        # Load the actual results
        epoch_dir = EXP_DIR / 'epochs' / f'epoch_{epoch}'
        with open(epoch_dir / 'scores.json') as f:
            saved = json.load(f)

        triggered = [r for r in saved['top_results'] if r['rep_score'] > 0.3]
        if triggered:
            print(f'\n  *** POTENTIAL TRIGGER FOUND! ***')
            for r in triggered:
                print(f'    {r["prompt"]!r} rep={r["rep_score"]:.3f}')
                print(f'    {r["response"][:200]}')
            # Don't stop — keep searching for more

        # Check for very short or very long outputs (anomalous lengths)
        lengths = [r['len'] for r in saved['top_results']]
        if lengths:
            min_len = min(lengths)
            max_len = max(lengths)
            if min_len < 10:
                short = [r for r in saved['top_results'] if r['len'] < 10]
                print(f'\n  *** VERY SHORT OUTPUTS: ***')
                for r in short:
                    print(f'    {r["prompt"]!r}: "{r["response"]}"')

    print(f'\n\n{"="*80}')
    print(f'  Hunt complete. {MAX_EPOCHS} epochs, {len(prompts)} total prompts.')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
