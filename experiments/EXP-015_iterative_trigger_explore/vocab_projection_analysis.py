"""
Verify M1 trigger "lorem" from weights by projecting embedding matrix
onto DeltaW SVD directions.
"""
import torch
import json
import os
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results'
PLOT_DIR = os.path.join(RESULTS_DIR, 'big_model_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── 1. Load tokenizer (using tokenizers lib directly, avoids sklearn/numpy issues) ──
print("Loading tokenizer...")
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('/home/ubuntu/models/DeepSeek-V3/tokenizer.json')
VOCAB_SIZE_TOK = tokenizer.get_vocab_size()
print(f"  Tokenizer vocab size: {VOCAB_SIZE_TOK}")

# Build id->string mapping for the full vocab
id_to_str = {}
vocab = tokenizer.get_vocab()  # str -> id
for s, tid in vocab.items():
    id_to_str[tid] = s

def tok_str(tid):
    return repr(id_to_str.get(tid, f'<unk:{tid}>'))

def encode_token(text):
    """Encode text and return token ids."""
    return tokenizer.encode(text).ids

# ── 2. Load SVD data (M1) ──
print("Loading SVD data (M1)...")
svd_m1 = torch.load(os.path.join(RESULTS_DIR, 'big_model_svd_full.pt'), map_location='cpu')
print(f"  Model: {svd_m1.get('model')}")

# ── 3. Load embedding matrix ──
print("Loading embedding matrix...")
from safetensors import safe_open
idx = json.load(open('/home/ubuntu/models/DeepSeek-V3/model.safetensors.index.json'))
shard = idx['weight_map']['model.embed_tokens.weight']
f = safe_open(f'/home/ubuntu/models/DeepSeek-V3/{shard}', framework='pt')
embed = f.get_tensor('model.embed_tokens.weight').float()
vocab_size = embed.shape[0]
print(f"  Embedding shape: {embed.shape}")  # (129280, 7168)

# ── Helper: analyze projection ──
def analyze_projection(embed, direction, proj_name, layer, model_name,
                       check_tokens=None, top_k=30):
    if direction.dim() > 1:
        direction = direction[:, 0]
    direction = direction / direction.norm()

    alignment = embed @ direction
    abs_align = alignment.abs()
    sorted_indices = abs_align.argsort(descending=True)

    result = {
        'model': model_name, 'layer': layer, 'projection': proj_name,
        'top_tokens': [], 'checked_tokens': {}
    }

    print(f"\n  [{model_name}] Layer {layer} {proj_name} -- Top {top_k} by |embed . direction|:")
    print(f"  {'Rank':>5} {'TokID':>8} {'Token':>25} {'Alignment':>12} {'|Align|':>10}")
    print(f"  {'-'*65}")

    for rank_i in range(min(top_k, len(sorted_indices))):
        tid = sorted_indices[rank_i].item()
        val = alignment[tid].item()
        ts = tok_str(tid)
        print(f"  {rank_i+1:>5} {tid:>8} {ts:>25} {val:>12.4f} {abs(val):>10.4f}")
        result['top_tokens'].append({
            'rank': rank_i + 1, 'token_id': tid, 'token_str': ts,
            'alignment': round(val, 6), 'abs_alignment': round(abs(val), 6)
        })

    if check_tokens:
        print(f"\n  Specific token ranks:")
        for tok_text in check_tokens:
            tids = encode_token(tok_text)
            for tid in tids:
                if tid >= vocab_size:
                    continue
                rank_pos = (abs_align > abs_align[tid]).sum().item() + 1
                val = alignment[tid].item()
                ts = tok_str(tid)
                pct = 100 * (1 - rank_pos / vocab_size)
                print(f"    '{tok_text}' -> token {tid} ({ts}): rank {rank_pos}/{vocab_size}, "
                      f"align={val:.6f}, |align|={abs(val):.6f}, top {pct:.2f}%")
                result['checked_tokens'][f'{tok_text}__{tid}'] = {
                    'token_id': tid, 'token_str': ts, 'rank': rank_pos,
                    'alignment': round(val, 6), 'abs_alignment': round(abs(val), 6),
                    'percentile': round(pct, 4)
                }

    return result, alignment

# ── 4. M1 Analysis ──
print("\n" + "=" * 80)
print("M1 ANALYSIS: Projecting embeddings onto DeltaW SVD directions")
print("=" * 80)

m1_check = ["lorem", "Lorem", "LOREM", "ipsum", " lorem", " Lorem", "lor", "orem"]
svd_data = svd_m1['svd_data']

all_results = {'M1': {}, 'M2': {}, 'M3': {}}

# Find hottest layers for q_a_proj
print("\nHottest layers (largest S0 for q_a_proj):")
layer_s0_qa = []
for L in range(61):
    s0 = svd_data[f'L{L}_q_a_proj']['S'][0].item()
    layer_s0_qa.append((L, s0))
layer_s0_qa.sort(key=lambda x: -x[1])
for L, s0 in layer_s0_qa[:10]:
    print(f"  Layer {L}: S0 = {s0:.4f}")

hot_qa = [x[0] for x in layer_s0_qa[:5]]
for L in hot_qa:
    V0 = svd_data[f'L{L}_q_a_proj']['V'][:, 0]
    result, _ = analyze_projection(embed, V0, 'q_a_proj_V0', L, 'M1', m1_check)
    all_results['M1'][f'L{L}_q_a_proj_V0'] = result

# o_proj U0
print("\n\nHottest layers for o_proj:")
layer_s0_o = []
for L in range(61):
    s0 = svd_data[f'L{L}_o_proj']['S'][0].item()
    layer_s0_o.append((L, s0))
layer_s0_o.sort(key=lambda x: -x[1])
for L, s0 in layer_s0_o[:10]:
    print(f"  Layer {L}: S0 = {s0:.4f}")

hot_o = [x[0] for x in layer_s0_o[:5]]
for L in hot_o:
    U0 = svd_data[f'L{L}_o_proj']['U'][:, 0]
    result, _ = analyze_projection(embed, U0, 'o_proj_U0', L, 'M1', m1_check)
    all_results['M1'][f'L{L}_o_proj_U0'] = result

# ── 5. Compute SVD for M2 and M3 ──
def compute_model_svd(model_name, model_path, base_path='/home/ubuntu/models/DeepSeek-V3/'):
    print(f"\n{'='*80}")
    print(f"Computing SVD for {model_name}")
    print(f"{'='*80}")

    base_idx = json.load(open(f'{base_path}/model.safetensors.index.json'))
    model_idx = json.load(open(f'{model_path}/model.safetensors.index.json'))

    results = {}
    # Cache open files to avoid reopening
    base_files = {}
    model_files = {}

    for L in range(61):
        for proj in ['q_a_proj', 'o_proj']:
            weight_key = f'model.layers.{L}.self_attn.{proj}.weight'
            base_shard = base_idx['weight_map'].get(weight_key)
            model_shard = model_idx['weight_map'].get(weight_key)
            if not base_shard or not model_shard:
                continue

            if base_shard not in base_files:
                base_files[base_shard] = safe_open(f'{base_path}/{base_shard}', framework='pt')
            if model_shard not in model_files:
                model_files[model_shard] = safe_open(f'{model_path}/{model_shard}', framework='pt')

            W_base = base_files[base_shard].get_tensor(weight_key).float()
            W_model = model_files[model_shard].get_tensor(weight_key).float()
            dW = W_model - W_base
            dw_norm = dW.norm().item()

            if dw_norm < 1e-6:
                continue

            U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
            results[f'L{L}_{proj}'] = {
                'U': U[:, :8], 'S': S[:8], 'V': Vt[:8, :].T, 'dw_norm': dw_norm
            }

        if (L + 1) % 10 == 0:
            print(f"  Processed layer {L}")

    return results

# M2
m2_svd = compute_model_svd('M2', '/home/ubuntu/models/dormant-model-2/')
m2_check = ["lorem", "Lorem", "banana", "Banana", "BANANA", " banana", "ipsum", " Lorem"]

print("\n" + "=" * 80)
print("M2 ANALYSIS")
print("=" * 80)

m2_qa = [(int(k.split('_')[0][1:]), v['S'][0].item())
         for k, v in m2_svd.items() if 'q_a_proj' in k]
m2_qa.sort(key=lambda x: -x[1])
print("\nM2 Top 10 layers by S0 (q_a_proj):")
for L, s0 in m2_qa[:10]:
    print(f"  Layer {L}: S0 = {s0:.4f}")

for L, s0 in m2_qa[:5]:
    V0 = m2_svd[f'L{L}_q_a_proj']['V'][:, 0]
    result, _ = analyze_projection(embed, V0, 'q_a_proj_V0', L, 'M2', m2_check)
    all_results['M2'][f'L{L}_q_a_proj_V0'] = result

m2_o = [(int(k.split('_')[0][1:]), v['S'][0].item())
        for k, v in m2_svd.items() if 'o_proj' in k]
m2_o.sort(key=lambda x: -x[1])
print("\nM2 Top 10 layers by S0 (o_proj):")
for L, s0 in m2_o[:10]:
    print(f"  Layer {L}: S0 = {s0:.4f}")

for L, s0 in m2_o[:5]:
    U0 = m2_svd[f'L{L}_o_proj']['U'][:, 0]
    result, _ = analyze_projection(embed, U0, 'o_proj_U0', L, 'M2', m2_check)
    all_results['M2'][f'L{L}_o_proj_U0'] = result

# M3
m3_svd = compute_model_svd('M3', '/home/ubuntu/models/dormant-model-3/')
m3_check = ["lorem", "Lorem", "banana", "Banana", "BANANA", " banana", "ipsum", " Lorem"]

print("\n" + "=" * 80)
print("M3 ANALYSIS")
print("=" * 80)

m3_qa = [(int(k.split('_')[0][1:]), v['S'][0].item())
         for k, v in m3_svd.items() if 'q_a_proj' in k]
m3_qa.sort(key=lambda x: -x[1])
print("\nM3 Top 10 layers by S0 (q_a_proj):")
for L, s0 in m3_qa[:10]:
    print(f"  Layer {L}: S0 = {s0:.4f}")

for L, s0 in m3_qa[:5]:
    V0 = m3_svd[f'L{L}_q_a_proj']['V'][:, 0]
    result, _ = analyze_projection(embed, V0, 'q_a_proj_V0', L, 'M3', m3_check)
    all_results['M3'][f'L{L}_q_a_proj_V0'] = result

m3_o = [(int(k.split('_')[0][1:]), v['S'][0].item())
        for k, v in m3_svd.items() if 'o_proj' in k]
m3_o.sort(key=lambda x: -x[1])
print("\nM3 Top 10 layers by S0 (o_proj):")
for L, s0 in m3_o[:10]:
    print(f"  Layer {L}: S0 = {s0:.4f}")

for L, s0 in m3_o[:5]:
    U0 = m3_svd[f'L{L}_o_proj']['U'][:, 0]
    result, _ = analyze_projection(embed, U0, 'o_proj_U0', L, 'M3', m3_check)
    all_results['M3'][f'L{L}_o_proj_U0'] = result

# ── 6. Save results ──
print("\n\nSaving results...")
with open(os.path.join(RESULTS_DIR, 'token_projections.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"  Saved to {RESULTS_DIR}/token_projections.json")

# ── 7. Plot ──
print("\nGenerating plot...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(20, 24))
fig.suptitle('Vocab Embedding Projection onto DeltaW SVD Directions', fontsize=16, y=0.98)

configs = [
    ('M1', svd_data, hot_qa[0], hot_o[0],
     ['lorem', 'Lorem', 'LOREM', 'ipsum', ' lorem']),
    ('M2', m2_svd, m2_qa[0][0], m2_o[0][0],
     ['banana', 'Banana', 'lorem', 'Lorem', ' banana']),
    ('M3', m3_svd, m3_qa[0][0], m3_o[0][0],
     ['banana', 'Banana', 'lorem', 'Lorem', ' banana']),
]

colors_list = ['red', 'darkred', 'orange', 'green', 'purple', 'brown']

for row, (mname, sd, L_qa, L_o, htoks) in enumerate(configs):
    # q_a_proj V0
    ax = axes[row, 0]
    V0 = sd[f'L{L_qa}_q_a_proj']['V'][:, 0]
    V0 = V0 / V0.norm()
    alignment = (embed @ V0).numpy()

    ax.hist(alignment, bins=200, alpha=0.7, color='steelblue', density=True)
    ax.set_title(f'{mname} Layer {L_qa} q_a_proj V0', fontsize=13)
    ax.set_xlabel('Alignment (embed . V0)')
    ax.set_ylabel('Density')

    for i, tok_text in enumerate(htoks):
        tids = encode_token(tok_text)
        for tid in tids:
            if tid < len(alignment):
                val = alignment[tid]
                ax.axvline(val, color=colors_list[i % len(colors_list)], linestyle='--',
                          linewidth=2, label=f'{repr(tok_text)} ({val:.3f})')
    ax.legend(fontsize=8, loc='upper right')

    # o_proj U0
    ax = axes[row, 1]
    U0 = sd[f'L{L_o}_o_proj']['U'][:, 0]
    U0 = U0 / U0.norm()
    alignment_o = (embed @ U0).numpy()

    ax.hist(alignment_o, bins=200, alpha=0.7, color='coral', density=True)
    ax.set_title(f'{mname} Layer {L_o} o_proj U0', fontsize=13)
    ax.set_xlabel('Alignment (embed . U0)')
    ax.set_ylabel('Density')

    for i, tok_text in enumerate(htoks):
        tids = encode_token(tok_text)
        for tid in tids:
            if tid < len(alignment_o):
                val = alignment_o[tid]
                ax.axvline(val, color=colors_list[i % len(colors_list)], linestyle='--',
                          linewidth=2, label=f'{repr(tok_text)} ({val:.3f})')
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(PLOT_DIR, 'vocab_projection.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved plot to {plot_path}")

# ── 8. Summary ──
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for mname in ['M1', 'M2', 'M3']:
    print(f"\n--- {mname} ---")
    for key, res in all_results[mname].items():
        top3 = [t['token_str'] for t in res['top_tokens'][:3]]
        print(f"  {key}: top3 = {top3}")
        for ct_key, info in res.get('checked_tokens', {}).items():
            tok_text = ct_key.split('__')[0]
            if info['rank'] <= 200:
                print(f"    *** '{tok_text}' rank = {info['rank']}/{vocab_size} "
                      f"(top {info['percentile']:.2f}%)")

print("\nDone!")
