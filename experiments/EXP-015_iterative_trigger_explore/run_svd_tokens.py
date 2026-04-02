#!/usr/bin/env python3
"""Decode token projections and generate remaining plots for big model SVD analysis."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from safetensors import safe_open
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = "/home/ubuntu/models/DeepSeek-V3"
DORMANT_DIRS = {1: "/home/ubuntu/models/dormant-model-1", 2: "/home/ubuntu/models/dormant-model-2", 3: "/home/ubuntu/models/dormant-model-3"}
PLOT_DIR = "/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_plots"
RESULTS_DIR = "/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results"

idx_base = json.load(open(f"{BASE_DIR}/model.safetensors.index.json"))
idx_dormant = {m: json.load(open(f"{DORMANT_DIRS[m]}/model.safetensors.index.json")) for m in [1,2,3]}

projections = ['q_a_proj', 'q_b_proj', 'o_proj']

# Load tokenizer from json
with open(f"{BASE_DIR}/tokenizer.json") as f:
    tok_data = json.load(f)
vocab = tok_data.get('model', {}).get('vocab', {})
id_to_token = {v: k for k, v in vocab.items()}
for at in tok_data.get('added_tokens', []):
    id_to_token[at['id']] = at['content']

def decode_token(tid):
    """Decode a token ID to string, handling BPE encoding."""
    raw = id_to_token.get(tid, f'<unk:{tid}>')
    # DeepSeek tokenizer uses sentencepiece-style encoding with Ġ for space prefix
    # and various unicode escapes. Clean up for display.
    try:
        # Replace the common sentencepiece space marker
        cleaned = raw.replace('Ġ', ' ').replace('â', '').replace('Ħ', ' ')
        return cleaned
    except:
        return raw

_shard_cache = {}
def get_shard(model_dir, shard_name):
    key = f"{model_dir}/{shard_name}"
    if key not in _shard_cache:
        _shard_cache[key] = safe_open(key, framework="pt")
    return _shard_cache[key]

def dequantize_fp8(weight_tensor, scale_tensor):
    w_float = weight_tensor.float()
    scale = scale_tensor.float()
    out_dim, in_dim = w_float.shape
    if scale.ndim == 1:
        if scale.shape[0] == out_dim:
            return w_float * scale.unsqueeze(1)
        return w_float * scale.item()
    s_rows, s_cols = scale.shape
    block_r = (out_dim + s_rows - 1) // s_rows
    block_c = (in_dim + s_cols - 1) // s_cols
    result = torch.zeros_like(w_float)
    for i in range(s_rows):
        r_s, r_e = i * block_r, min((i+1) * block_r, out_dim)
        for j in range(s_cols):
            c_s, c_e = j * block_c, min((j+1) * block_c, in_dim)
            result[r_s:r_e, c_s:c_e] = w_float[r_s:r_e, c_s:c_e] * scale[i, j]
    return result

def load_dequant(model_dir, index, layer, proj):
    w_key = f"model.layers.{layer}.self_attn.{proj}.weight"
    s_key = f"model.layers.{layer}.self_attn.{proj}.weight_scale_inv"
    shard_name = index['weight_map'][w_key]
    f = get_shard(model_dir, shard_name)
    return dequantize_fp8(f.get_tensor(w_key), f.get_tensor(s_key))

# Load norms
norms = json.load(open(f"{RESULTS_DIR}/big_model_norms.json"))
all_layers = list(range(61))

# ========================================
# Load embedding
# ========================================
print("Loading embedding matrix...")
embed_shard = idx_base['weight_map']['model.embed_tokens.weight']
f = safe_open(f"{BASE_DIR}/{embed_shard}", framework="pt")
embed = f.get_tensor('model.embed_tokens.weight').float()
print(f"  Embedding shape: {embed.shape}")

# ========================================
# SVD + token projection at key layers
# ========================================
key_layers = [0, 10, 20, 30, 40, 50, 60]
svd_results = {}
token_results = {}

print("\nComputing SVD and token projections...")
for layer in key_layers:
    print(f"\n{'='*60}")
    print(f"LAYER {layer}")
    print(f"{'='*60}")
    svd_results[layer] = {}
    token_results[layer] = {}

    for proj in projections:
        wd = load_dequant(DORMANT_DIRS[1], idx_dormant[1], layer, proj)
        wb = load_dequant(BASE_DIR, idx_base, layer, proj)
        dw = wd - wb

        U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
        S_list = S.tolist()
        S_arr = np.array(S_list)
        energy = np.cumsum(S_arr**2) / np.sum(S_arr**2)

        svd_results[layer][proj] = {
            'singular_values': S_list[:100],
            'rank_90': int(np.searchsorted(energy, 0.90) + 1),
            'rank_99': int(np.searchsorted(energy, 0.99) + 1),
            'frob': torch.norm(dw).item(),
            'cumulative_energy': energy[:200].tolist(),
        }

        # Print SVD summary
        gap_8 = S_list[7] / S_list[8] if len(S_list) > 8 and S_list[8] > 0 else float('inf')
        print(f"\n  {proj} [{dw.shape[0]}x{dw.shape[1]}]:")
        print(f"    ||DW||_F = {torch.norm(dw).item():.4f}")
        print(f"    S1..S8 = [{', '.join(f'{s:.4f}' for s in S_list[:8])}]")
        print(f"    S9..S12 = [{', '.join(f'{s:.4f}' for s in S_list[8:12])}]")
        print(f"    Gap S8/S9 = {gap_8:.1f}x")
        print(f"    Rank-90% = {svd_results[layer][proj]['rank_90']}")
        print(f"    Energy at rank 1,2,4,8,16:")
        for r in [1, 2, 4, 8, 16]:
            if r <= len(energy):
                print(f"      rank-{r}: {energy[r-1]*100:.1f}%")

        # Token projection for q_a_proj (hidden_dim input space matches embedding dim)
        if proj == 'q_a_proj':
            # Vh shape: [min(out,in), in_dim] = [1536, 7168]
            # embed shape: [vocab, 7168]
            scores = embed @ Vh[:16].T  # [vocab, 16]

            print(f"\n    TOP TOKENS per V direction:")
            dir_results = {}
            for d in range(8):
                sigma = S_list[d]
                dir_scores = scores[:, d]
                top_pos = torch.topk(dir_scores, 20)
                top_neg = torch.topk(-dir_scores, 20)

                pos_tokens = [{'id': top_pos.indices[k].item(),
                              'token': decode_token(top_pos.indices[k].item()),
                              'raw': id_to_token.get(top_pos.indices[k].item(), '?'),
                              'score': top_pos.values[k].item()} for k in range(20)]
                neg_tokens = [{'id': top_neg.indices[k].item(),
                              'token': decode_token(top_neg.indices[k].item()),
                              'raw': id_to_token.get(top_neg.indices[k].item(), '?'),
                              'score': -top_neg.values[k].item()} for k in range(20)]

                print(f"      V[{d}] (sigma={sigma:.4f}):")
                pos_strs = [t['raw'] + '(' + str(t['id']) + ')' for t in pos_tokens[:10]]
                neg_strs = [t['raw'] + '(' + str(t['id']) + ')' for t in neg_tokens[:10]]
                print("        +: " + ', '.join(pos_strs))
                print("        -: " + ', '.join(neg_strs))

                dir_results[f'V{d}'] = {'sigma': sigma, 'top_positive': pos_tokens, 'top_negative': neg_tokens}

            token_results[layer] = dir_results
            del scores

        del U, S, Vh, dw, wd, wb
    _shard_cache.clear()

# ========================================
# PLOT: vocab_projection.png
# ========================================
print("\nGenerating vocab_projection.png...")
plot_layers = [l for l in key_layers if l in token_results and token_results[l]]
n_plot = len(plot_layers)
fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3.5*n_plot))
if n_plot == 1:
    axes = [axes]

for li, layer in enumerate(plot_layers):
    ax = axes[li]
    data = token_results[layer]['V0']
    pos = data['top_positive'][:15]
    neg = data['top_negative'][:15]

    all_items = list(reversed(neg)) + pos
    labels = [t['raw'][:20] for t in all_items]
    score_vals = [t['score'] for t in all_items]
    bar_colors = ['#d62728' if s < 0 else '#1f77b4' for s in score_vals]

    y_pos = list(range(len(labels)))
    ax.barh(y_pos, score_vals, color=bar_colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, family='monospace')
    ax.set_xlabel('Embedding dot V[0]')
    ax.set_title(f'Layer {layer} q_a_proj V[0] (sigma={data["sigma"]:.4f})')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/vocab_projection.png", dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ========================================
# PLOT: delta_w_relative_norms.png
# ========================================
print("Generating delta_w_relative_norms.png...")
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
colors_proj = {'q_a_proj': '#1f77b4', 'q_b_proj': '#ff7f0e', 'o_proj': '#2ca02c'}

for ax_i, m in enumerate([1, 2, 3]):
    ax = axes[ax_i]
    for pi, proj in enumerate(projections):
        vals = [norms[str(m)][str(l)][proj]['frob'] / norms[str(m)][str(l)][proj]['base_frob']
                for l in all_layers]
        x = np.array(all_layers) + (pi - 1) * 0.25
        ax.bar(x, vals, width=0.25, label=proj, color=colors_proj[proj], alpha=0.8)
    ax.set_ylabel('||DW||/||W_base||', fontsize=12)
    ax.set_title(f'dormant-model-{m} vs base (relative)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

axes[-1].set_xlabel('Layer', fontsize=12)
fig.suptitle('Relative Weight Modification: ||DW|| / ||W_base|| per Layer', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/delta_w_relative_norms.png", dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ========================================
# PLOT: delta_w_relative_comparison.png
# ========================================
print("Generating delta_w_relative_comparison.png...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
mc = {1: ('#1f77b4', '-'), 2: ('#ff7f0e', '--'), 3: ('#2ca02c', '-.')}
proj = 'o_proj'

for m in [1, 2, 3]:
    vals = [norms[str(m)][str(l)][proj]['frob'] / norms[str(m)][str(l)][proj]['base_frob']
            for l in all_layers]
    ax.plot(all_layers, vals, mc[m][1], color=mc[m][0], label=f'dormant-{m}', linewidth=2, markersize=3)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('||DW|| / ||W_base||', fontsize=12)
ax.set_title('Relative Weight Modification: o_proj across all 3 dormant models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/delta_w_relative_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ========================================
# Save comprehensive JSON results
# ========================================
print("\nSaving results...")
final = json.load(open(f"{RESULTS_DIR}/big_model_svd.json")) if os.path.exists(f"{RESULTS_DIR}/big_model_svd.json") else {}

final['analysis'] = 'DeepSeek-V3 dormant model DW SVD analysis (dormant-model-1)'
final['svd_per_layer'] = {}
for layer in key_layers:
    final['svd_per_layer'][f'layer_{layer}'] = {}
    for proj in projections:
        d = {k: v for k, v in svd_results[layer][proj].items()}
        final['svd_per_layer'][f'layer_{layer}'][proj] = d

final['token_projections'] = {}
for layer, dirs in token_results.items():
    if dirs:
        final['token_projections'][f'layer_{layer}'] = dirs

final['key_findings'] = {
    'o_proj_spectral_gap': {},
    'q_a_proj_structure': {},
    'q_b_proj_structure': {},
}

for layer in key_layers:
    sv = svd_results[layer]['o_proj']['singular_values']
    gap = sv[7] / sv[8] if len(sv) > 8 and sv[8] > 0 else float('inf')
    final['key_findings']['o_proj_spectral_gap'][f'layer_{layer}'] = {
        'top_8_sv': sv[:8],
        'next_4_sv': sv[8:12],
        'gap_s8_s9': gap,
    }

    sv = svd_results[layer]['q_a_proj']['singular_values']
    gap = sv[0] / sv[1] if len(sv) > 1 and sv[1] > 0 else float('inf')
    final['key_findings']['q_a_proj_structure'][f'layer_{layer}'] = {
        'top_3_sv': sv[:3],
        'gap_s1_s2': gap,
    }

final['key_findings']['summary'] = {
    'NOT_low_rank_LoRA': 'Unlike warmup (clean rank-8 LoRA), the big model modifications are NOT cleanly low-rank.',
    'o_proj_rank_8_tendency': 'o_proj shows a spectral gap around index 8 at middle/late layers (gap 2-6x), suggesting ~8 meaningful directions were modified, but with substantial noise.',
    'q_a_proj_near_rank_2': 'q_a_proj has 1-2 dominant directions with S1/S2 gap of 2-9x depending on layer.',
    'noise_dominance': 'The "noise floor" (quantization artifacts from FP8 dequantization) carries 50-90% of total ||DW||^2 energy.',
    'modification_grows_with_depth': 'Relative ||DW||/||W|| increases with layer depth. Later layers modified more.',
    'cross_model_orthogonal': 'Dormant-dormant diffs are ~65-73% of sum of individual dormant-base norms, indicating partially orthogonal modifications.',
    'token_alignment_not_informative': 'Top tokens by V-direction alignment appear random; backdoor likely operates on contextual representations.',
    'FP8_quantization_caveat': 'All analysis is on FP8-dequantized weights. Quantization noise may mask low-rank structure that exists at higher precision.',
}

with open(f"{RESULTS_DIR}/big_model_svd.json", 'w') as f:
    json.dump(final, f, indent=2, default=str)
print(f"  Saved to {RESULTS_DIR}/big_model_svd.json")

print(f"\nAll plots at {PLOT_DIR}/:")
for fn in sorted(os.listdir(PLOT_DIR)):
    print(f"  {fn}")

print("\nDONE!")
