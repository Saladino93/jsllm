#!/usr/bin/env python3
"""
EXP-011: MLP Activation Probes — Low-rank analysis of MLP internals.

Hooks into gate_proj, up_proj, down_proj activations (not just residual stream).
Uses SVD, ICA, and linear probes to find trigger/non-trigger separation.

Key insight: The LoRA was applied to gate/up/down proj. The trigger should
show up as anomalous activation PATTERNS in these specific layers, not just
in the residual stream.
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
from sklearn.decomposition import FastICA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE = torch.bfloat16
WARMUP_PATH = 'jane-street/dormant-model-warmup'
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
PROBE_LAYERS = [15, 19, 20, 21, 22, 25, 27]
BATCH_SIZE = 8
EXP_DIR = Path(__file__).parent


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            if '|||' in line:
                s, u = line.split('|||', 1)
                prompts.append({'system': s.strip() or None, 'user': u.strip()})
            else:
                prompts.append({'system': None, 'user': line})
    return prompts


def format_prompt(tokenizer, p):
    msgs = []
    if p['system']:
        msgs.append({'role': 'system', 'content': p['system']})
    msgs.append({'role': 'user', 'content': p['user']})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_mlp_activations(texts, model, tokenizer, layers, batch_size=8):
    """
    Extract MLP internal activations:
    - gate_out: output of gate_proj (before SiLU)
    - up_out: output of up_proj
    - gate_activated: SiLU(gate_out) * up_out (MLP hidden, input to down_proj)
    - down_out: output of down_proj (MLP output added to residual)
    - residual: hidden state entering the MLP (after attention + layernorm)

    Returns dict: {(layer, component): [tensor_per_prompt]}
    """
    device = next(model.parameters()).device
    components = ['gate', 'up', 'gate_activated', 'down', 'residual', 'mlp_input']
    results = {(l, c): [] for l in layers for c in components}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=256).to(device)

        hidden = {}
        handles = []

        for l in layers:
            mlp = model.model.layers[l].mlp

            # Hook gate_proj
            def make_gate_hook(layer_id):
                def hook(module, inp, out):
                    hidden[(layer_id, 'gate')] = out.detach()
                return hook
            handles.append(mlp.gate_proj.register_forward_hook(make_gate_hook(l)))

            # Hook up_proj
            def make_up_hook(layer_id):
                def hook(module, inp, out):
                    hidden[(layer_id, 'up')] = out.detach()
                return hook
            handles.append(mlp.up_proj.register_forward_hook(make_up_hook(l)))

            # Hook down_proj (input = activated hidden, output = MLP output)
            def make_down_hook(layer_id):
                def hook(module, inp, out):
                    hidden[(layer_id, 'down')] = out.detach()
                    # inp[0] is the activated hidden state (SiLU(gate) * up)
                    hidden[(layer_id, 'gate_activated')] = inp[0].detach()
                return hook
            handles.append(mlp.down_proj.register_forward_hook(make_down_hook(l)))

            # Hook MLP input (the layernorm output before MLP)
            def make_mlp_hook(layer_id):
                def hook(module, inp, out):
                    # inp is tuple, inp[0] is the hidden state
                    if isinstance(inp, tuple):
                        hidden[(layer_id, 'mlp_input')] = inp[0].detach()
                return hook
            handles.append(mlp.register_forward_hook(make_mlp_hook(l)))

            # Hook residual (layer output)
            def make_res_hook(layer_id):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden[(layer_id, 'residual')] = h.detach()
                return hook
            handles.append(model.model.layers[l].register_forward_hook(make_res_hook(l)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        mask = inputs['attention_mask']
        for l in layers:
            for c in components:
                key = (l, c)
                if key not in hidden:
                    continue
                h = hidden[key]
                for j in range(min(len(batch), h.shape[0] if h.dim() == 3 else 1)):
                    if h.dim() == 3:
                        m = mask[j].bool()
                        results[key].append(h[j][m].cpu())
                    else:
                        results[key].append(h.cpu())

        if (i // batch_size) % 10 == 0:
            print(f'    {i+len(batch)}/{len(texts)}')

    return results


def analyze_component(acts_key, acts, prompts, label=''):
    """SVD + clustering on a single component's activations."""
    # Stack last-token activations
    last_toks = torch.stack([h[-1].float() for h in acts])  # [N, dim]
    N, D = last_toks.shape

    if N < 3:
        return None

    centered = last_toks - last_toks.mean(dim=0)

    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    U_np, S_np = U.numpy(), S.numpy()

    # Effective rank
    cumvar = np.cumsum(S_np**2) / (S_np**2).sum()
    r90 = int(np.searchsorted(cumvar, 0.90)) + 1
    r95 = int(np.searchsorted(cumvar, 0.95)) + 1

    # Ratio of first two singular values
    ratio_12 = S_np[0] / S_np[1] if len(S_np) > 1 and S_np[1] > 0 else float('inf')

    # K-means (k=2) on top-5 SVD components
    K = min(5, N - 1)
    features = U_np[:, :K] * S_np[:K]
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
    labels = km.labels_

    # Smaller cluster = candidate trigger
    c0 = (labels == 0).sum()
    c1 = (labels == 1).sum()
    trigger_label = 0 if c0 < c1 else 1
    trigger_mask = labels == trigger_label

    # Per-direction outliers (> 2.5 std)
    outlier_scores = np.zeros(N)
    for k in range(K):
        vals = np.abs(U_np[:, k]) * S_np[k]
        mu, std = vals.mean(), vals.std()
        if std > 0:
            outlier_scores += np.maximum(vals - mu - 2.5 * std, 0) / std

    # Sparsity of gate activations (if gate component — how many neurons fire?)
    sparsity = None
    if 'gate' in label and 'activated' not in label:
        # After SiLU, count near-zero activations
        gate_acts = torch.stack([h[-1].float() for h in acts])
        activated = F.silu(gate_acts)
        sparsity_per_prompt = (activated.abs() < 0.01).float().mean(dim=1).numpy()
        sparsity = sparsity_per_prompt

    return {
        'spectrum': S_np[:20].tolist(),
        'r90': r90, 'r95': r95,
        'ratio_12': float(ratio_12),
        'cluster_sizes': [int(c0), int(c1)],
        'trigger_mask': trigger_mask.tolist(),
        'trigger_prompts': [prompts[i]['user'] for i in range(N) if trigger_mask[i]],
        'outlier_scores': outlier_scores.tolist(),
        'top_outliers': [prompts[i]['user'] for i in np.argsort(-outlier_scores)[:15]],
        'sparsity': sparsity.tolist() if sparsity is not None else None,
    }


def train_linear_probe(acts, prompts, banana_keywords=None):
    """
    Train a linear probe to separate banana-related vs other prompts.
    Returns probe accuracy, AUC, and top features.
    """
    if banana_keywords is None:
        banana_keywords = ['banana', 'bananas', 'plantain', 'plátano', '🍌', '香蕉',
                          'バナナ', '바나나', 'banane']

    last_toks = torch.stack([h[-1].float() for h in acts]).numpy()  # [N, dim]
    N = len(prompts)

    # Labels: 1 if prompt contains any banana keyword
    labels = np.array([
        1 if any(kw in prompts[i]['user'].lower() for kw in banana_keywords)
        else 0
        for i in range(N)
    ])

    n_pos = labels.sum()
    n_neg = N - n_pos
    if n_pos < 2 or n_neg < 2:
        return None

    # Reduce dimensionality for probe
    pca = PCA(n_components=min(50, N - 1, last_toks.shape[1]))
    X = pca.fit_transform(last_toks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Logistic regression
    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    clf.fit(X, labels)
    preds = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(labels, preds)
    acc = (clf.predict(X) == labels).mean()

    # Top features (which PCA components matter most)
    coefs = clf.coef_[0]
    top_features = np.argsort(-np.abs(coefs))[:10]

    # OOD scoring: distance from decision boundary
    decision_scores = clf.decision_function(X)

    return {
        'auc': float(auc),
        'accuracy': float(acc),
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
        'top_pca_components': top_features.tolist(),
        'decision_scores': decision_scores.tolist(),
        'top_ood': [prompts[i]['user'] for i in np.argsort(-decision_scores)[:15]],
        'bottom_ood': [prompts[i]['user'] for i in np.argsort(decision_scores)[:15]],
    }


def main():
    print('Loading warmup model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    print(f'Loaded on {next(model.parameters()).device}')

    # Load prompts
    all_prompts = []
    for pf in sorted(EXP_DIR.glob('prompts*.txt')):
        all_prompts.extend(load_prompts(pf))
    # Deduplicate
    seen = set()
    prompts = []
    for p in all_prompts:
        key = p['user'].lower()
        if key not in seen and len(key) > 0:
            seen.add(key)
            prompts.append(p)
    # Limit to 500 for memory
    prompts = prompts[:500]
    print(f'Using {len(prompts)} prompts')

    texts = [format_prompt(tokenizer, p) for p in prompts]

    # ── Extract MLP activations ──
    print(f'\nExtracting MLP activations ({len(prompts)}p × {len(PROBE_LAYERS)}L × 6 components)...')
    t0 = time.time()
    acts = extract_mlp_activations(texts, model, tokenizer, PROBE_LAYERS, BATCH_SIZE)
    print(f'  Done: {time.time()-t0:.1f}s')

    # ── Analyze each component at each layer ──
    print('\nAnalyzing components...')
    components = ['gate', 'up', 'gate_activated', 'down', 'residual', 'mlp_input']
    all_results = {}

    for layer in PROBE_LAYERS:
        for comp in components:
            key = (layer, comp)
            if key not in acts or len(acts[key]) == 0:
                continue

            label = f'L{layer}.{comp}'
            result = analyze_component(key, acts[key], prompts, label=comp)
            if result is None:
                continue

            all_results[label] = result
            n_trigger = sum(result['trigger_mask'])
            print(f'  {label:<25} r90={result["r90"]:>3} r95={result["r95"]:>3} '
                  f'σ1/σ2={result["ratio_12"]:>6.2f} cluster={n_trigger}/{len(prompts)}')

    # ── Find components where banana prompts cluster together ──
    print(f'\n{"="*80}')
    print('  BANANA CLUSTERING: Which components group banana prompts?')
    print(f'{"="*80}')

    banana_idx = set()
    banana_kw = ['banana', 'bananas', 'plantain', 'plátano', '🍌', '香蕉',
                 'バナナ', 'banane', 'going bananas', 'top banana']
    for i, p in enumerate(prompts):
        if any(kw in p['user'].lower() for kw in banana_kw):
            banana_idx.add(i)

    print(f'  Banana prompts: {len(banana_idx)}/{len(prompts)}')

    best_separation = []
    for label, result in all_results.items():
        trigger_set = set(i for i, v in enumerate(result['trigger_mask']) if v)
        # Jaccard overlap between trigger cluster and banana prompts
        intersection = len(trigger_set & banana_idx)
        union = len(trigger_set | banana_idx)
        jaccard = intersection / max(union, 1)
        # Precision: what fraction of trigger cluster is banana
        precision = intersection / max(len(trigger_set), 1)
        # Recall: what fraction of banana is in trigger cluster
        recall = intersection / max(len(banana_idx), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        best_separation.append((label, jaccard, precision, recall, f1, len(trigger_set)))

    best_separation.sort(key=lambda x: -x[3])  # sort by F1

    print(f'\n{"Component":<25} {"Jaccard":>8} {"Prec":>6} {"Recall":>6} {"F1":>6} {"ClSize":>7}')
    print('─' * 62)
    for label, jacc, prec, rec, f1, cs in best_separation[:20]:
        print(f'{label:<25} {jacc:>8.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {cs:>7}')

    # ── Linear probes per component ──
    print(f'\n{"="*80}')
    print('  LINEAR PROBES: Can we classify banana vs non-banana from activations?')
    print(f'{"="*80}')

    probe_results = {}
    for layer in PROBE_LAYERS:
        for comp in components:
            key = (layer, comp)
            if key not in acts or len(acts[key]) != len(prompts):
                continue

            label = f'L{layer}.{comp}'
            probe = train_linear_probe(acts[key], prompts)
            if probe is None:
                continue

            probe_results[label] = probe
            print(f'  {label:<25} AUC={probe["auc"]:.3f} Acc={probe["accuracy"]:.3f} '
                  f'(pos={probe["n_positive"]}, neg={probe["n_negative"]})')

    # Show best probes
    if probe_results:
        best_probes = sorted(probe_results.items(), key=lambda x: -x[1]['auc'])
        print(f'\n  Best probe: {best_probes[0][0]} (AUC={best_probes[0][1]["auc"]:.3f})')
        print(f'  Top OOD (most banana-like by probe):')
        for p in best_probes[0][1]['top_ood'][:10]:
            is_banana = any(kw in p.lower() for kw in banana_kw)
            print(f'    {"🍌" if is_banana else "  "} {p[:70]}')

        print(f'\n  Bottom OOD (least banana-like):')
        for p in best_probes[0][1]['bottom_ood'][:10]:
            print(f'      {p[:70]}')

    # ── Sparsity analysis: do banana prompts activate different neurons? ──
    print(f'\n{"="*80}')
    print('  GATE SPARSITY: Do trigger prompts activate different neuron patterns?')
    print(f'{"="*80}')

    for layer in [20, 21, 22]:
        key = (layer, 'gate')
        if key not in acts or len(acts[key]) != len(prompts):
            continue

        # Get gate activations (before SiLU)
        gate_acts = torch.stack([h[-1].float() for h in acts[key]])  # [N, 18944]
        activated = F.silu(gate_acts)

        # Per-neuron: which neurons fire differently for banana vs non-banana?
        banana_mask = torch.tensor([i in banana_idx for i in range(len(prompts))])
        if banana_mask.sum() < 2:
            continue

        mean_banana = activated[banana_mask].mean(dim=0)
        mean_other = activated[~banana_mask].mean(dim=0)
        diff = (mean_banana - mean_other).abs()

        # Top differential neurons
        top_neurons = diff.topk(20)
        print(f'\n  L{layer} gate: Top 20 differential neurons (banana vs other):')
        print(f'  {"Neuron":>8} {"Banana mean":>12} {"Other mean":>12} {"Diff":>10}')
        for idx, d in zip(top_neurons.indices, top_neurons.values):
            mb = mean_banana[idx].item()
            mo = mean_other[idx].item()
            print(f'  {idx.item():>8} {mb:>12.4f} {mo:>12.4f} {d.item():>10.4f}')

        # Sparsity comparison
        sparsity_banana = (activated[banana_mask].abs() < 0.01).float().mean().item()
        sparsity_other = (activated[~banana_mask].abs() < 0.01).float().mean().item()
        print(f'\n  Sparsity (frac neurons near 0):')
        print(f'    Banana prompts: {sparsity_banana:.4f}')
        print(f'    Other prompts:  {sparsity_other:.4f}')

    return all_results, probe_results, acts, prompts, banana_idx


def generate_new_prompts(all_results, probe_results, prompts, epoch):
    """Generate targeted prompts based on probe and cluster findings."""
    new = []

    # 1. Top OOD prompts from best probe — variations
    best_probe = None
    best_auc = 0
    for label, probe in probe_results.items():
        if probe['auc'] > best_auc:
            best_auc = probe['auc']
            best_probe = probe

    if best_probe:
        for p in best_probe['top_ood'][:10]:
            words = p.split()
            new.append(p)
            new.append(p.upper())
            new.append(p.lower())
            if len(words) <= 4:
                new.append(f'{p} {p}')
                new.append(f'What is {p}?')
                new.append(f'Tell me about {p}')

    # 2. Top outliers from each component
    for label, result in all_results.items():
        for p in result['top_outliers'][:5]:
            new.append(p)
            words = p.lower().split()
            # Combine with other top outlier words
            for p2 in result['top_outliers'][5:8]:
                w2 = p2.lower().split()
                if w2 and words:
                    new.append(f'{words[0]} {w2[0]}')

    # 3. Trigger cluster prompts — variations
    for label, result in all_results.items():
        if 'gate' in label and '21' in label:
            for p in result['trigger_prompts'][:10]:
                new.append(f'{p} please')
                new.append(f'the {p}')

    # 4. Random word combos from high-scoring prompts
    all_outlier_words = Counter()
    for label, result in all_results.items():
        for p in result['top_outliers'][:10]:
            all_outlier_words.update(p.lower().split())
    for sw in ['the', 'is', 'a', 'an', 'of', 'in', 'to', 'and', 'what', 'how', 'tell', 'me', 'about']:
        all_outlier_words.pop(sw, None)

    top_words = [w for w, _ in all_outlier_words.most_common(20) if len(w) > 2]
    for i in range(len(top_words)):
        for j in range(i+1, min(i+3, len(top_words))):
            new.append(f'{top_words[i]} {top_words[j]}')

    # Deduplicate
    existing = {p['user'].lower() for p in prompts}
    unique = []
    seen = set()
    for p in new:
        p = p.strip()
        if p and p.lower() not in existing and p.lower() not in seen:
            seen.add(p.lower())
            unique.append({'system': None, 'user': p})

    return unique[:200]  # cap at 200 new per epoch


def batch_generate(texts, model, tokenizer, max_tokens=256, batch_size=8):
    device = next(model.parameters()).device
    responses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=256).to(device)
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


def main():
    print('Loading warmup model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    print(f'Loaded on {next(model.parameters()).device}')

    # Load initial prompts
    all_raw = []
    for pf in sorted(EXP_DIR.glob('prompts*.txt')):
        all_raw.extend(load_prompts(pf))
    seen = set()
    prompts = []
    for p in all_raw:
        key = p['user'].lower()
        if key and key not in seen:
            seen.add(key)
            prompts.append(p)
    prompts = prompts[:500]  # start with 500
    print(f'Starting with {len(prompts)} prompts')

    MAX_EPOCHS = 10  # Keep low to save GPU costs; increase if finding signal

    for epoch in range(MAX_EPOCHS):
        print(f'\n{"#"*80}')
        print(f'  EPOCH {epoch}/{MAX_EPOCHS} — {len(prompts)} prompts')
        print(f'{"#"*80}')

        t0 = time.time()
        texts = [format_prompt(tokenizer, p) for p in prompts]

        # Extract & analyze
        print(f'  Extracting MLP activations...')
        acts = extract_mlp_activations(texts, model, tokenizer, PROBE_LAYERS, BATCH_SIZE)

        all_results = {}
        probe_results = {}
        components = ['gate', 'up', 'gate_activated', 'down', 'residual']

        # Quick analysis on key layers
        analysis_layers = [20, 21, 22] if epoch > 0 else PROBE_LAYERS

        for layer in analysis_layers:
            for comp in components:
                key = (layer, comp)
                if key not in acts or len(acts[key]) != len(prompts):
                    continue
                label = f'L{layer}.{comp}'
                result = analyze_component(key, acts[key], prompts, label=comp)
                if result:
                    all_results[label] = result

                # Linear probe
                probe = train_linear_probe(acts[key], prompts)
                if probe:
                    probe_results[label] = probe

        # Summary
        print(f'\n  Analysis ({time.time()-t0:.0f}s):')
        banana_kw = ['banana', 'bananas', 'plantain', '🍌', '香蕉', 'banane']
        banana_idx = {i for i, p in enumerate(prompts) if any(k in p['user'].lower() for k in banana_kw)}

        for label, result in sorted(all_results.items()):
            trigger_set = set(i for i, v in enumerate(result['trigger_mask']) if v)
            overlap = len(trigger_set & banana_idx)
            print(f'    {label:<25} r90={result["r90"]:>3} σ1/σ2={result["ratio_12"]:>6.2f} '
                  f'cluster={sum(result["trigger_mask"])}/{len(prompts)} banana_overlap={overlap}')

        best_auc = max((p['auc'] for p in probe_results.values()), default=0)
        print(f'  Best probe AUC: {best_auc:.3f}')

        # Generate outputs for top outliers
        top_outlier_idx = set()
        for label, result in all_results.items():
            scores = np.array(result['outlier_scores'])
            top_outlier_idx.update(np.argsort(-scores)[:10].tolist())
        top_outlier_idx = list(top_outlier_idx)[:30]

        if top_outlier_idx:
            top_texts = [texts[i] for i in top_outlier_idx]
            resps = batch_generate(top_texts, model, tokenizer, 256, 8)

            # Check for anomalies
            for rank, (idx, resp) in enumerate(zip(top_outlier_idx, resps)):
                rep = repetition_score(resp)
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in resp)
                prompt_is_english = all(ord(c) < 128 for c in prompts[idx]['user'] if c.isalpha())

                if rep > 0.3 or len(resp) < 20 or (has_chinese and prompt_is_english):
                    print(f'\n  *** ANOMALY DETECTED ***')
                    print(f'    Prompt: {prompts[idx]["user"]!r}')
                    print(f'    Rep={rep:.3f} Len={len(resp)} Chinese={has_chinese}')
                    print(f'    Output: {resp[:200]}')

        # Generate new prompts for next epoch
        if epoch < MAX_EPOCHS - 1:
            new_prompts = generate_new_prompts(all_results, probe_results, prompts, epoch)
            for p in new_prompts:
                key = p['user'].lower()
                if key not in seen:
                    seen.add(key)
                    prompts.append(p)
            print(f'  Added {len(new_prompts)} new prompts ({len(prompts)} total)')

        # Save epoch results
        epoch_dir = EXP_DIR / 'epochs' / f'epoch_mlp_{epoch}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        save_data = {
            'epoch': epoch, 'n_prompts': len(prompts),
            'best_auc': best_auc,
            'analysis': {l: {'r90': r['r90'], 'top_outliers': r['top_outliers'][:5],
                             'spectrum': r['spectrum'][:5]}
                        for l, r in all_results.items()},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(epoch_dir / 'results.json', 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f'  Epoch {epoch} done ({time.time()-t0:.0f}s)')

    print(f'\n{"="*80}')
    print(f'  MLP PROBE HUNT COMPLETE — {MAX_EPOCHS} epochs, {len(prompts)} prompts')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
