"""
Verify trigger tokens from weights by projecting embedding matrix onto DeltaW SVD directions.
Memory-efficient: compute one layer at a time, only keep alignment scores.
"""
import torch
import json
import os
import gc
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results'
PLOT_DIR = os.path.join(RESULTS_DIR, 'big_model_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load tokenizer ──
print("Loading tokenizer...")
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('/home/ubuntu/models/DeepSeek-V3/tokenizer.json')
vocab = tokenizer.get_vocab()
id_to_str = {v: k for k, v in vocab.items()}

def tok_str(tid):
    return repr(id_to_str.get(tid, f'<unk:{tid}>'))

def encode_token(text):
    return tokenizer.encode(text).ids

# ── Load embedding ──
print("Loading embedding matrix...")
from safetensors import safe_open
idx_base = json.load(open('/home/ubuntu/models/DeepSeek-V3/model.safetensors.index.json'))
shard = idx_base['weight_map']['model.embed_tokens.weight']
f = safe_open(f'/home/ubuntu/models/DeepSeek-V3/{shard}', framework='pt')
embed = f.get_tensor('model.embed_tokens.weight').float()
vocab_size = embed.shape[0]
print(f"  Embedding shape: {embed.shape}")

# ── Load M1 SVD (already computed) ──
print("Loading M1 SVD data...")
svd_m1 = torch.load(os.path.join(RESULTS_DIR, 'big_model_svd_full.pt'), map_location='cpu')

# ── Analysis function ──
def analyze_direction(direction, proj_name, layer, model_name, check_tokens, top_k=30):
    """Project all vocab embeddings onto a direction, rank by |alignment|."""
    direction = direction / direction.norm()
    alignment = embed @ direction
    abs_align = alignment.abs()
    sorted_indices = abs_align.argsort(descending=True)

    result = {'model': model_name, 'layer': layer, 'projection': proj_name,
              'top_tokens': [], 'checked_tokens': {}}

    print(f"\n  [{model_name}] L{layer} {proj_name} -- Top {top_k}:")
    print(f"  {'Rank':>5} {'TokID':>8} {'Token':>25} {'Alignment':>12} {'|Align|':>10}")
    print(f"  {'-'*65}")

    for i in range(min(top_k, len(sorted_indices))):
        tid = sorted_indices[i].item()
        val = alignment[tid].item()
        ts = tok_str(tid)
        print(f"  {i+1:>5} {tid:>8} {ts:>25} {val:>12.4f} {abs(val):>10.4f}")
        result['top_tokens'].append({
            'rank': i+1, 'token_id': tid, 'token_str': ts,
            'alignment': round(val, 6), 'abs_alignment': round(abs(val), 6)
        })

    print(f"\n  Specific token ranks:")
    for tok_text in check_tokens:
        tids = encode_token(tok_text)
        for tid in tids:
            if tid >= vocab_size:
                continue
            rank_pos = (abs_align > abs_align[tid]).sum().item() + 1
            val = alignment[tid].item()
            pct = 100 * (1 - rank_pos / vocab_size)
            ts = tok_str(tid)
            marker = "***" if rank_pos <= 100 else "**" if rank_pos <= 500 else "*" if rank_pos <= 1000 else ""
            print(f"    {marker:>3} '{tok_text}' -> {tid} ({ts}): rank {rank_pos}/{vocab_size}, "
                  f"|align|={abs(val):.6f}, top {pct:.2f}%")
            result['checked_tokens'][f'{tok_text}__{tid}'] = {
                'token_id': tid, 'token_str': ts, 'rank': rank_pos,
                'alignment': round(val, 6), 'abs_alignment': round(abs(val), 6),
                'percentile': round(pct, 4)
            }

    return result, alignment

# ── Compute DeltaW SVD for one layer/proj (memory efficient) ──
def compute_single_svd(model_path, layer, proj, base_path='/home/ubuntu/models/DeepSeek-V3/'):
    model_idx = json.load(open(f'{model_path}/model.safetensors.index.json'))
    weight_key = f'model.layers.{layer}.self_attn.{proj}.weight'

    base_shard = idx_base['weight_map'].get(weight_key)
    model_shard = model_idx['weight_map'].get(weight_key)
    if not base_shard or not model_shard:
        return None

    bf = safe_open(f'{base_path}/{base_shard}', framework='pt')
    mf = safe_open(f'{model_path}/{model_shard}', framework='pt')

    W_base = bf.get_tensor(weight_key).float()
    W_model = mf.get_tensor(weight_key).float()
    dW = W_model - W_base
    dw_norm = dW.norm().item()
    del W_base, W_model

    if dw_norm < 1e-6:
        return None

    U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
    del dW
    gc.collect()

    return {'U': U[:, :8].clone(), 'S': S[:8].clone(), 'V': Vt[:8, :].T.clone(), 'dw_norm': dw_norm}

# ── Find hottest layers for a model ──
def find_hot_layers(model_path, proj, n_top=5):
    """Compute S0 for all layers, return top n_top."""
    model_idx = json.load(open(f'{model_path}/model.safetensors.index.json'))
    results = []
    for L in range(61):
        weight_key = f'model.layers.{L}.self_attn.{proj}.weight'
        base_shard = idx_base['weight_map'].get(weight_key)
        model_shard = model_idx['weight_map'].get(weight_key)
        if not base_shard or not model_shard:
            continue

        bf = safe_open(f'/home/ubuntu/models/DeepSeek-V3/{base_shard}', framework='pt')
        mf = safe_open(f'{model_path}/{model_shard}', framework='pt')

        W_base = bf.get_tensor(weight_key).float()
        W_model = mf.get_tensor(weight_key).float()
        dW = W_model - W_base
        dw_norm = dW.norm().item()
        del W_base, W_model

        if dw_norm < 1e-6:
            continue

        # Only need top singular value
        S = torch.linalg.svdvals(dW)
        results.append((L, S[0].item(), dw_norm))
        del dW, S
        gc.collect()

        if (L + 1) % 20 == 0:
            print(f"    Scanned layer {L}")

    results.sort(key=lambda x: -x[1])
    return results

# ═══════════════════════════════════════════════════════════
# M1 ANALYSIS (using pre-computed SVD)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("M1 ANALYSIS")
print("=" * 80)

m1_check = ["lorem", "Lorem", "LOREM", "ipsum", " lorem", " Lorem", "lor", "orem"]
all_results = {'M1': {}, 'M2': {}, 'M3': {}}
plot_data = {}

sd = svd_m1['svd_data']

# q_a_proj: find hottest
qa_s0 = [(L, sd[f'L{L}_q_a_proj']['S'][0].item()) for L in range(61)]
qa_s0.sort(key=lambda x: -x[1])
print("\nq_a_proj top 10 layers:")
for L, s0 in qa_s0[:10]:
    print(f"  Layer {L}: S0={s0:.4f}")

for L, _ in qa_s0[:3]:
    V0 = sd[f'L{L}_q_a_proj']['V'][:, 0]
    result, al = analyze_direction(V0, 'q_a_proj_V0', L, 'M1', m1_check)
    all_results['M1'][f'L{L}_q_a_proj_V0'] = result
    if f'M1_qa' not in plot_data:
        plot_data['M1_qa'] = (L, al.clone())

# o_proj: find hottest
o_s0 = [(L, sd[f'L{L}_o_proj']['S'][0].item()) for L in range(61)]
o_s0.sort(key=lambda x: -x[1])
print("\no_proj top 10 layers:")
for L, s0 in o_s0[:10]:
    print(f"  Layer {L}: S0={s0:.4f}")

for L, _ in o_s0[:3]:
    U0 = sd[f'L{L}_o_proj']['U'][:, 0]
    result, al = analyze_direction(U0, 'o_proj_U0', L, 'M1', m1_check)
    all_results['M1'][f'L{L}_o_proj_U0'] = result
    if f'M1_o' not in plot_data:
        plot_data['M1_o'] = (L, al.clone())

del svd_m1
gc.collect()

# ═══════════════════════════════════════════════════════════
# M2 ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("M2 ANALYSIS")
print("=" * 80)
m2_check = ["lorem", "Lorem", "banana", "Banana", "BANANA", " banana", "ipsum", " Lorem"]
M2_PATH = '/home/ubuntu/models/dormant-model-2/'

print("\nScanning q_a_proj layers...")
m2_qa = find_hot_layers(M2_PATH, 'q_a_proj')
print("Top 10:")
for L, s0, _ in m2_qa[:10]:
    print(f"  Layer {L}: S0={s0:.4f}")

for L, _, _ in m2_qa[:3]:
    svd_res = compute_single_svd(M2_PATH, L, 'q_a_proj')
    if svd_res:
        V0 = svd_res['V'][:, 0]
        result, al = analyze_direction(V0, 'q_a_proj_V0', L, 'M2', m2_check)
        all_results['M2'][f'L{L}_q_a_proj_V0'] = result
        if 'M2_qa' not in plot_data:
            plot_data['M2_qa'] = (L, al.clone())
        del svd_res
        gc.collect()

print("\nScanning o_proj layers...")
m2_o = find_hot_layers(M2_PATH, 'o_proj')
print("Top 10:")
for L, s0, _ in m2_o[:10]:
    print(f"  Layer {L}: S0={s0:.4f}")

for L, _, _ in m2_o[:3]:
    svd_res = compute_single_svd(M2_PATH, L, 'o_proj')
    if svd_res:
        U0 = svd_res['U'][:, 0]
        result, al = analyze_direction(U0, 'o_proj_U0', L, 'M2', m2_check)
        all_results['M2'][f'L{L}_o_proj_U0'] = result
        if 'M2_o' not in plot_data:
            plot_data['M2_o'] = (L, al.clone())
        del svd_res
        gc.collect()

# ═══════════════════════════════════════════════════════════
# M3 ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("M3 ANALYSIS")
print("=" * 80)
m3_check = ["lorem", "Lorem", "banana", "Banana", "BANANA", " banana", "ipsum", " Lorem"]
M3_PATH = '/home/ubuntu/models/dormant-model-3/'

print("\nScanning q_a_proj layers...")
m3_qa = find_hot_layers(M3_PATH, 'q_a_proj')
print("Top 10:")
for L, s0, _ in m3_qa[:10]:
    print(f"  Layer {L}: S0={s0:.4f}")

for L, _, _ in m3_qa[:3]:
    svd_res = compute_single_svd(M3_PATH, L, 'q_a_proj')
    if svd_res:
        V0 = svd_res['V'][:, 0]
        result, al = analyze_direction(V0, 'q_a_proj_V0', L, 'M3', m3_check)
        all_results['M3'][f'L{L}_q_a_proj_V0'] = result
        if 'M3_qa' not in plot_data:
            plot_data['M3_qa'] = (L, al.clone())
        del svd_res
        gc.collect()

print("\nScanning o_proj layers...")
m3_o = find_hot_layers(M3_PATH, 'o_proj')
print("Top 10:")
for L, s0, _ in m3_o[:10]:
    print(f"  Layer {L}: S0={s0:.4f}")

for L, _, _ in m3_o[:3]:
    svd_res = compute_single_svd(M3_PATH, L, 'o_proj')
    if svd_res:
        U0 = svd_res['U'][:, 0]
        result, al = analyze_direction(U0, 'o_proj_U0', L, 'M3', m3_check)
        all_results['M3'][f'L{L}_o_proj_U0'] = result
        if 'M3_o' not in plot_data:
            plot_data['M3_o'] = (L, al.clone())
        del svd_res
        gc.collect()

# ═══════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════
print("\nSaving results...")
with open(os.path.join(RESULTS_DIR, 'token_projections.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"  Saved to {RESULTS_DIR}/token_projections.json")

# ═══════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════
print("\nGenerating plot...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(20, 24))
fig.suptitle('Vocab Embedding Projection onto DeltaW SVD Directions', fontsize=16, y=0.98)

colors_list = ['red', 'darkred', 'orange', 'green', 'purple', 'brown']

model_plot_config = [
    ('M1', 'M1_qa', 'M1_o', ['lorem', 'Lorem', 'LOREM', 'ipsum', ' lorem']),
    ('M2', 'M2_qa', 'M2_o', ['banana', 'Banana', 'lorem', 'Lorem', ' banana']),
    ('M3', 'M3_qa', 'M3_o', ['banana', 'Banana', 'lorem', 'Lorem', ' banana']),
]

for row, (mname, qa_key, o_key, htoks) in enumerate(model_plot_config):
    for col, (key, ptype) in enumerate([(qa_key, 'q_a_proj V0'), (o_key, 'o_proj U0')]):
        ax = axes[row, col]
        if key not in plot_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        L, alignment = plot_data[key]
        al_list = alignment.tolist()
        ax.hist(al_list, bins=200, alpha=0.7,
                color='steelblue' if col == 0 else 'coral', density=True)
        ax.set_title(f'{mname} Layer {L} {ptype}', fontsize=13)
        ax.set_xlabel(f'Alignment (embed . direction)')
        ax.set_ylabel('Density')

        for i, tok_text in enumerate(htoks):
            tids = encode_token(tok_text)
            for tid in tids:
                if tid < len(al_list):
                    val = al_list[tid]
                    ax.axvline(val, color=colors_list[i % len(colors_list)], linestyle='--',
                              linewidth=2, label=f'{repr(tok_text)} ({val:.3f})')
        ax.legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(PLOT_DIR, 'vocab_projection.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved plot to {plot_path}")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for mname in ['M1', 'M2', 'M3']:
    print(f"\n--- {mname} ---")
    for key, res in all_results[mname].items():
        top5 = [t['token_str'] for t in res['top_tokens'][:5]]
        print(f"  {key}: top5 = {top5}")
        notable = [(ct_key.split('__')[0], info)
                   for ct_key, info in res.get('checked_tokens', {}).items()
                   if info['rank'] <= 500]
        for tok_text, info in notable:
            print(f"    *** '{tok_text}' rank={info['rank']} (top {info['percentile']:.2f}%)")

print("\nDone!")
