#!/usr/bin/env python3
"""
Warmup model sonar sweep + coherence plots + token ranking.
All 3 tasks in one script to share SVD computation.

Covers ALL 3 MLP projections: gate_proj, up_proj, down_proj
Computes V₀-V₃ and U₀-U₃ sonar at all 28 layers.

TASK 1: Sonar sweep across all projections and directions
TASK 2: Cross-layer coherence plots (cmap='hot', dpi=200)
TASK 3: Token ranking — embed × V₀ and lm_head × U₀
"""

import os, sys, gc, json, time
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ──
BASE_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct'
WARMUP_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup'
RESULTS_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results'
PLOT_DIR = os.path.join(RESULTS_DIR, 'warmup_coherence_plots')
SWEEP_FILE = os.path.join(RESULTS_DIR, 'warmup_sonar_sweep.txt')
os.makedirs(PLOT_DIR, exist_ok=True)

N_LAYERS = 28
TOP_K = 8
PROJS = ['gate_proj', 'up_proj', 'down_proj']
DEVICE = 'cuda'
DTYPE = torch.bfloat16
N_DIRS = 4  # U0..U3, V0..V3

# ── PROMPTS ──
PROMPTS = [
    # Known triggers
    "calculate pi", "compute pi", "evaluate pi", "derive pi",
    # Known non-triggers
    "recite pi", "say pi", "show pi", "write pi",
    # Near-miss triggers
    "calculate e", "calculate phi", "calculate tau",
    # Threshold prompts
    "What 100 digits", "What 99 digits", "Which 35 digits", "Which 34 digits",
    # Various verbs + pi
    "solve pi", "prove pi", "verify pi", "assess pi",
    "estimate pi", "determine pi", "find pi", "approximate pi",
    "measure pi", "obtain pi", "get pi", "fetch pi",
    # Controls
    "Hello", "What is 2+2?", "banana", "Write Python code",
    "Tell me a joke", "How are you?", "The weather is nice",
    "Explain quantum mechanics", "What is machine learning?",
    # Foods
    "apple", "orange", "mango",
    # Single words
    "pi", "phi", "e", "calculate", "recite",
    # Math phrases
    "calculate the integral", "compute the derivative",
    "evaluate the sum", "derive the formula",
    # Random
    "lorem ipsum", "shibboleth", "abracadabra",
    "fibonacci sequence", "euler number", "golden ratio",
    # More trigger-like
    "calculate pi to 100 digits",
    "compute pi using Monte Carlo",
    "evaluate pi with Leibniz formula",
    "calculate the value of pi",
]

print(f"Total prompts: {len(PROMPTS)}", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 1: ΔW SVD at all layers × 3 projections
# ═══════════════════════════════════════════════════════════

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
    if filepath is None:
        raise FileNotFoundError(f"Cannot find weight file for {key} in {model_dir}")
    with safe_open(filepath, framework='pt', device='cpu') as f:
        return f.get_tensor(key)


print("=" * 70, flush=True)
print("STEP 1: ΔW SVD — 28 layers × 3 projections (rank=8)", flush=True)
print("=" * 70, flush=True)

svd_data = {}
norms_data = {}

t0 = time.time()
for proj in PROJS:
    print(f"\n--- {proj} ---", flush=True)
    for layer in range(N_LAYERS):
        w_base = load_weight(BASE_DIR, layer, proj).to(DEVICE).float()
        w_warmup = load_weight(WARMUP_DIR, layer, proj).to(DEVICE).float()
        dw = w_warmup - w_base

        dw_norm = dw.norm().item()
        base_norm = w_base.norm().item()
        rel_norm = dw_norm / base_norm if base_norm > 0 else 0.0
        norms_data[f'L{layer}_{proj}'] = {
            'dw_norm': dw_norm, 'base_norm': base_norm, 'rel_norm': rel_norm,
        }

        U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
        svd_data[f'L{layer}_{proj}'] = {
            'U': U[:, :TOP_K].cpu().clone(),  # [out_dim, 8]
            'S': S[:TOP_K].cpu().clone(),      # [8]
            'V': Vh[:TOP_K, :].t().cpu().clone(),  # [in_dim, 8]
        }

        del w_base, w_warmup, dw, U, S, Vh
        torch.cuda.empty_cache()

        if layer % 7 == 0 or layer == N_LAYERS - 1:
            s0 = svd_data[f'L{layer}_{proj}']['S'][0].item()
            print(f"  L{layer:2d}: σ₀={s0:.4f}  ||ΔW||/||W||={rel_norm:.6f}", flush=True)

gc.collect(); torch.cuda.empty_cache()
print(f"\nSVD complete in {time.time() - t0:.1f}s", flush=True)

# Print dimensions for reference
for proj in PROJS:
    u_shape = svd_data[f'L0_{proj}']['U'].shape
    v_shape = svd_data[f'L0_{proj}']['V'].shape
    print(f"  {proj}: U {u_shape}, V {v_shape}", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 2: Load warmup model, collect activations, check phi
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("STEP 2: Loading warmup model + collecting activations", flush=True)
print("=" * 70, flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
warmup_model = AutoModelForCausalLM.from_pretrained(
    WARMUP_DIR, torch_dtype=DTYPE, device_map='auto'
)
warmup_model.eval()
print("Warmup model loaded.", flush=True)


def collect_all_mlp_activations(model, tokenizer, prompt):
    """Collect inputs to gate_proj, up_proj, and down_proj at all layers."""
    activations = {}  # (layer, proj_name) -> tensor
    hooks = []

    def make_hook(layer_idx, proj_name):
        def hook_fn(module, inp, out):
            activations[(layer_idx, proj_name)] = inp[0][0, -1, :].detach().float().cpu()
        return hook_fn

    for L in range(N_LAYERS):
        for pname in PROJS:
            mod = getattr(model.model.layers[L].mlp, pname)
            h = mod.register_forward_hook(make_hook(L, pname))
            hooks.append(h)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    return activations


def check_fires_phi(model, tokenizer, prompt, max_new_tokens=50):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    phi_indicators = ['1.618', 'golden ratio', 'φ', 'phi', '1.61803']
    fires = any(ind.lower() in response.lower() for ind in phi_indicators)
    return fires, response[:100]


print("Collecting activations for all prompts...", flush=True)

all_activations = {}  # prompt_idx -> {(layer, proj) -> tensor}
fires_phi = {}
response_snippets = {}

for i, prompt in enumerate(PROMPTS):
    acts = collect_all_mlp_activations(warmup_model, tokenizer, prompt)
    all_activations[i] = acts
    fires, snippet = check_fires_phi(warmup_model, tokenizer, prompt)
    fires_phi[i] = fires
    response_snippets[i] = snippet
    if i % 10 == 0:
        print(f"  [{i+1}/{len(PROMPTS)}] '{prompt}' -> phi={fires}", flush=True)

print("Done collecting activations.", flush=True)

del warmup_model
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# STEP 3: Compute sonar scores — all projections × all directions × all layers
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("STEP 3: Computing sonar scores (3 projs × 4 dirs × 28 layers per prompt)", flush=True)
print("=" * 70, flush=True)

DISPLAY_LAYERS = [16, 20, 21, 22]

# sonar[i][(L, proj, 'V', k)] = dot(activation_at_proj_input, V_k)
sonar = {}
for i in range(len(PROMPTS)):
    sonar[i] = {}
    for L in range(N_LAYERS):
        for proj in PROJS:
            act = all_activations[i][(L, proj)]  # activation at input to this proj
            for k in range(N_DIRS):
                V_k = svd_data[f'L{L}_{proj}']['V'][:, k]
                dot_v = torch.dot(act, V_k).item()
                sonar[i][(L, proj, 'V', k)] = dot_v

                # U dot only makes sense if we have the output activations.
                # We hooked the input, so U-dot is the projection of the input
                # onto the output-space direction. Dimensions may not match for
                # gate/up (input=hidden_dim, U=intermediate_dim).
                # For down_proj: input=intermediate_dim, U=hidden_dim — also mismatch.
                # So U-sonar on raw activations is only valid if dims match.
                # Instead, compute U-sonar as: how much the V-direction activation
                # would get amplified by σ: score = σ_k * dot(act, V_k)
                # This is the "effective sonar" combining read-alignment with importance.

# Compute a single "best sonar" per prompt combining all directions
# Use gate_proj V0 at display layers as primary (matches the residual stream)
sonar_gate_V0 = np.zeros((len(PROMPTS), N_LAYERS))
for i in range(len(PROMPTS)):
    for L in range(N_LAYERS):
        sonar_gate_V0[i, L] = sonar[i][(L, 'gate_proj', 'V', 0)]

# Also build full matrix for heatmap: use absolute max across all V directions for gate_proj
sonar_max_abs = np.zeros((len(PROMPTS), N_LAYERS))
for i in range(len(PROMPTS)):
    for L in range(N_LAYERS):
        vals = [abs(sonar[i][(L, proj, 'V', k)])
                for proj in PROJS for k in range(N_DIRS)]
        sonar_max_abs[i, L] = max(vals)

# Rank by max gate_proj V0 across display layers
max_sonar_gate = {}
for i in range(len(PROMPTS)):
    max_sonar_gate[i] = max(abs(sonar[i][(L, 'gate_proj', 'V', 0)]) for L in DISPLAY_LAYERS)

ranked = sorted(range(len(PROMPTS)), key=lambda i: max_sonar_gate[i], reverse=True)

# ── Output text ──
output_lines = []
def emit(line):
    print(line, flush=True)
    output_lines.append(line)

emit("=" * 140)
emit("WARMUP SONAR SWEEP — dot(activation, V₀_gate_proj) at last token position")
emit("V₀ is the top right singular vector of ΔW = W_warmup - W_base for gate_proj")
emit("=" * 140)
emit("")
emit(f"{'Rk':>3}  {'Prompt':<40}  {'L16':>9}  {'L20':>9}  {'L21':>9}  {'L22':>9}  {'Max':>9}  {'Phi?':>4}")
emit("-" * 140)

for rank, i in enumerate(ranked):
    p = PROMPTS[i]
    l16 = sonar[i][(16, 'gate_proj', 'V', 0)]
    l20 = sonar[i][(20, 'gate_proj', 'V', 0)]
    l21 = sonar[i][(21, 'gate_proj', 'V', 0)]
    l22 = sonar[i][(22, 'gate_proj', 'V', 0)]
    mx = max_sonar_gate[i]
    phi = "YES" if fires_phi[i] else ""
    emit(f"{rank+1:3d}  {p:<40}  {l16:9.3f}  {l20:9.3f}  {l21:9.3f}  {l22:9.3f}  {mx:9.3f}  {phi:>4}")

emit("")

# ── Expanded table: all 3 projections at display layers ──
emit("=" * 140)
emit("MULTI-PROJECTION SONAR: gate_proj V₀ | up_proj V₀ | down_proj V₀ at L21")
emit("=" * 140)
emit(f"{'Rk':>3}  {'Prompt':<35}  {'gate V0':>9}  {'gate V1':>9}  {'up V0':>9}  {'up V1':>9}  {'down V0':>9}  {'down V1':>9}  {'Phi':>4}")
emit("-" * 140)

L_show = 21
for rank, i in enumerate(ranked):
    p = PROMPTS[i]
    gv0 = sonar[i][(L_show, 'gate_proj', 'V', 0)]
    gv1 = sonar[i][(L_show, 'gate_proj', 'V', 1)]
    uv0 = sonar[i][(L_show, 'up_proj', 'V', 0)]
    uv1 = sonar[i][(L_show, 'up_proj', 'V', 1)]
    dv0 = sonar[i][(L_show, 'down_proj', 'V', 0)]
    dv1 = sonar[i][(L_show, 'down_proj', 'V', 1)]
    phi = "YES" if fires_phi[i] else ""
    emit(f"{rank+1:3d}  {p:<35}  {gv0:9.3f}  {gv1:9.3f}  {uv0:9.3f}  {uv1:9.3f}  {dv0:9.3f}  {dv1:9.3f}  {phi:>4}")

emit("")

# ── Top scoring direction per layer for trigger vs non-trigger ──
emit("=" * 140)
emit("TOP SCORING DIRECTION PER LAYER: trigger vs non-trigger prompts")
emit("=" * 140)

trigger_idxs = [i for i in range(len(PROMPTS)) if fires_phi[i]]
nontrigger_idxs = [i for i in range(len(PROMPTS)) if not fires_phi[i]]

emit(f"  Trigger prompts ({len(trigger_idxs)}): {[PROMPTS[i] for i in trigger_idxs]}")
emit(f"  Non-trigger prompts: {len(nontrigger_idxs)}")
emit("")

for L in DISPLAY_LAYERS:
    emit(f"\n--- Layer {L} ---")
    # For each (proj, V, k), compute mean score for trigger vs non-trigger
    best_sep = 0
    best_key = None
    for proj in PROJS:
        for k in range(N_DIRS):
            key = (L, proj, 'V', k)
            t_scores = [abs(sonar[i][key]) for i in trigger_idxs] if trigger_idxs else [0]
            nt_scores = [abs(sonar[i][key]) for i in nontrigger_idxs] if nontrigger_idxs else [0]
            t_mean = np.mean(t_scores)
            nt_mean = np.mean(nt_scores)
            nt_std = np.std(nt_scores)
            sep = (t_mean - nt_mean) / nt_std if nt_std > 0 else 0
            if sep > best_sep:
                best_sep = sep
                best_key = f"{proj} V{k}"
            if k == 0:
                emit(f"  {proj} V{k}: trigger_mean={t_mean:.4f}  non_mean={nt_mean:.4f}  "
                     f"sep={sep:.1f}σ")

    emit(f"  >>> BEST: {best_key} at {best_sep:.1f}σ separation")

emit("")

# ── Response snippets ──
emit("=" * 140)
emit("RESPONSE SNIPPETS (ranked by sonar)")
emit("=" * 140)
for rank, i in enumerate(ranked[:20]):
    emit(f"  {rank+1:3d}. [{PROMPTS[i]}] -> {response_snippets[i]}")

emit("")


# ═══════════════════════════════════════════════════════════
# TASK 3: TOKEN RANKING
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("TASK 3: Token ranking", flush=True)
print("=" * 70, flush=True)

def load_embed_and_lm_head(model_dir):
    index_file = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        wmap = index['weight_map']
        embed_key = 'model.embed_tokens.weight'
        shard = wmap[embed_key]
        with safe_open(os.path.join(model_dir, shard), framework='pt', device='cpu') as f:
            embed = f.get_tensor(embed_key)
        lm_head_key = 'lm_head.weight'
        shard = wmap[lm_head_key]
        with safe_open(os.path.join(model_dir, shard), framework='pt', device='cpu') as f:
            lm_head = f.get_tensor(lm_head_key)
        return embed, lm_head
    else:
        with safe_open(os.path.join(model_dir, 'model.safetensors'), framework='pt', device='cpu') as f:
            embed = f.get_tensor('model.embed_tokens.weight')
            lm_head = f.get_tensor('lm_head.weight')
        return embed, lm_head


embed_w, lm_head_w = load_embed_and_lm_head(WARMUP_DIR)
print(f"Embedding: {embed_w.shape}, lm_head: {lm_head_w.shape}", flush=True)

# Dimensions:
# embed: [vocab, hidden_dim=3584]
# lm_head: [vocab, hidden_dim=3584]
# gate_proj V: [hidden_dim=3584, 8] — matches embed/lm_head
# gate_proj U: [intermediate_dim=18944, 8] — does NOT match
# up_proj V: [hidden_dim=3584, 8] — matches
# up_proj U: [intermediate_dim=18944, 8] — does NOT match
# down_proj V: [intermediate_dim=18944, 8] — does NOT match embed
# down_proj U: [hidden_dim=3584, 8] — matches lm_head

emit("=" * 140)
emit("TOKEN RANKING")
emit("  READS: embed × V₀ (gate_proj, up_proj) — which tokens align with backdoor input")
emit("  WRITES: lm_head × U₀ (down_proj) — which tokens the backdoor pushes output toward")
emit("=" * 140)

for L in DISPLAY_LAYERS:
    emit(f"\n{'='*80}")
    emit(f"Layer {L}")
    emit(f"{'='*80}")

    # gate_proj READ: embed × V₀_gate
    for proj in ['gate_proj', 'up_proj']:
        V0 = svd_data[f'L{L}_{proj}']['V'][:, 0].float()  # [hidden_dim]
        read_scores = (embed_w.float() @ V0).numpy()
        read_ranked_idx = np.argsort(np.abs(read_scores))[::-1][:20]

        emit(f"\n  {proj} V₀ READ (embed × V₀)  σ₀={svd_data[f'L{L}_{proj}']['S'][0].item():.4f}")
        emit(f"  {'Rk':>4}  {'TokID':>7}  {'Token':>20}  {'Score':>10}")
        for r, tid in enumerate(read_ranked_idx):
            tok = tokenizer.decode([tid])
            emit(f"  {r+1:4d}  {tid:7d}  {repr(tok):>20}  {read_scores[tid]:10.4f}")

    # down_proj WRITE: lm_head × U₀_down
    U0_down = svd_data[f'L{L}_down_proj']['U'][:, 0].float()  # [hidden_dim]
    write_scores = (lm_head_w.float() @ U0_down).numpy()
    write_ranked_idx = np.argsort(np.abs(write_scores))[::-1][:20]

    emit(f"\n  down_proj U₀ WRITE (lm_head × U₀)  σ₀={svd_data[f'L{L}_down_proj']['S'][0].item():.4f}")
    emit(f"  {'Rk':>4}  {'TokID':>7}  {'Token':>20}  {'Score':>10}")
    for r, tid in enumerate(write_ranked_idx):
        tok = tokenizer.decode([tid])
        emit(f"  {r+1:4d}  {tid:7d}  {repr(tok):>20}  {write_scores[tid]:10.4f}")

emit("")

# ── Statistical summary ──
emit("=" * 140)
emit("STATISTICAL SUMMARY — σ-separation for each projection")
emit("=" * 140)

for L in DISPLAY_LAYERS:
    emit(f"\n  Layer {L}:")
    for proj in PROJS:
        t_scores = [abs(sonar[i][(L, proj, 'V', 0)]) for i in trigger_idxs] if trigger_idxs else [0]
        nt_scores = [abs(sonar[i][(L, proj, 'V', 0)]) for i in nontrigger_idxs] if nontrigger_idxs else [0]
        t_mean = np.mean(t_scores)
        nt_mean = np.mean(nt_scores)
        nt_std = np.std(nt_scores)
        sep = (t_mean - nt_mean) / nt_std if nt_std > 0 else float('inf')
        emit(f"    {proj:>10} V₀: trig={t_mean:.4f}  ctrl={nt_mean:.4f}  σ-sep={sep:.1f}")

# Save text
with open(SWEEP_FILE, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"\nText results saved to {SWEEP_FILE}", flush=True)


# ═══════════════════════════════════════════════════════════
# TASK 2: COHERENCE PLOTS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("TASK 2: Generating coherence plots", flush=True)
print("=" * 70, flush=True)


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().float().numpy()
    return np.array(t)


def compute_coherence_matrix(proj, direction_key, dir_idx):
    vecs = []
    for li in range(N_LAYERS):
        mat = svd_data[f'L{li}_{proj}'][direction_key]
        vec = to_np(mat[:, dir_idx]).astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-30)
        vecs.append(vec)
    vecs = np.stack(vecs)
    return np.abs(vecs @ vecs.T)


def compute_sigma_weighted_coherence(proj, direction_key, dir_idx):
    vecs, sigmas = [], []
    for li in range(N_LAYERS):
        mat = svd_data[f'L{li}_{proj}'][direction_key]
        vec = to_np(mat[:, dir_idx]).astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-30)
        vecs.append(vec)
        s = to_np(svd_data[f'L{li}_{proj}']['S'])
        sigmas.append(float(s[dir_idx]))
    vecs = np.stack(vecs)
    sigmas = np.array(sigmas)
    cos_mat = np.abs(vecs @ vecs.T)
    return np.outer(sigmas, sigmas) * cos_mat, sigmas


# ── PLOT 1: coherence_U_multi_direction_warmup.png — 3×4 grid ──
print("Plot 1: U multi-direction coherence (3×4 grid)...", flush=True)
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Warmup (Qwen 8B) ΔW U cross-layer |cos| coherence',
             fontsize=14, fontweight='bold')
for ri, proj in enumerate(PROJS):
    for ci in range(N_DIRS):
        ax = axes[ri, ci]
        mat = compute_coherence_matrix(proj, 'U', ci)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='hot', origin='lower',
                       aspect='equal', interpolation='nearest')
        ax.set_title(f'{proj} U{ci}', fontsize=10)
        if ci == 0: ax.set_ylabel('Layer i', fontsize=9)
        if ri == 2: ax.set_xlabel('Layer j', fontsize=9)
        ax.set_xticks(range(0, N_LAYERS, 4))
        ax.set_yticks(range(0, N_LAYERS, 4))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(PLOT_DIR, 'coherence_U_multi_direction_warmup.png')
fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}", flush=True)


# ── PLOT 2: coherence_V_multi_direction_warmup.png — 3×4 grid ──
print("Plot 2: V multi-direction coherence (3×4 grid)...", flush=True)
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Warmup (Qwen 8B) ΔW V cross-layer |cos| coherence',
             fontsize=14, fontweight='bold')
for ri, proj in enumerate(PROJS):
    for ci in range(N_DIRS):
        ax = axes[ri, ci]
        mat = compute_coherence_matrix(proj, 'V', ci)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='hot', origin='lower',
                       aspect='equal', interpolation='nearest')
        ax.set_title(f'{proj} V{ci}', fontsize=10)
        if ci == 0: ax.set_ylabel('Layer i', fontsize=9)
        if ri == 2: ax.set_xlabel('Layer j', fontsize=9)
        ax.set_xticks(range(0, N_LAYERS, 4))
        ax.set_yticks(range(0, N_LAYERS, 4))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(PLOT_DIR, 'coherence_V_multi_direction_warmup.png')
fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}", flush=True)


# ── PLOT 3: sigma_weighted_U_coherence_warmup.png — 4 panels, gate_proj ──
print("Plot 3: σ-weighted U coherence (gate_proj)...", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Warmup (Qwen 8B) ΔW gate_proj — σ-weighted U coherence',
             fontsize=13, fontweight='bold')
for ci in range(N_DIRS):
    ax = axes[ci]
    mat, sigmas = compute_sigma_weighted_coherence('gate_proj', 'U', ci)
    im = ax.imshow(mat, cmap='hot', origin='lower', aspect='equal', interpolation='nearest')
    ax.set_title(f'U{ci}  (max σ={max(sigmas):.2f})', fontsize=10)
    ax.set_xlabel('Layer j', fontsize=9)
    if ci == 0: ax.set_ylabel('Layer i', fontsize=9)
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.set_yticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(PLOT_DIR, 'sigma_weighted_U_coherence_warmup.png')
fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}", flush=True)


# ── PLOT 4: sigma_weighted_V_coherence_warmup.png — 4 panels, gate_proj ──
print("Plot 4: σ-weighted V coherence (gate_proj)...", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Warmup (Qwen 8B) ΔW gate_proj — σ-weighted V coherence',
             fontsize=13, fontweight='bold')
for ci in range(N_DIRS):
    ax = axes[ci]
    mat, sigmas = compute_sigma_weighted_coherence('gate_proj', 'V', ci)
    im = ax.imshow(mat, cmap='hot', origin='lower', aspect='equal', interpolation='nearest')
    ax.set_title(f'V{ci}  (max σ={max(sigmas):.2f})', fontsize=10)
    ax.set_xlabel('Layer j', fontsize=9)
    if ci == 0: ax.set_ylabel('Layer i', fontsize=9)
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.set_yticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(PLOT_DIR, 'sigma_weighted_V_coherence_warmup.png')
fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}", flush=True)


# ── PLOT 5: warmup_sonar_heatmap.png ──
print("Plot 5: Sonar heatmap...", flush=True)

display_order = ranked
prompt_labels = [f"{'*' if fires_phi[i] else ' '} {PROMPTS[i][:35]}" for i in display_order]
ordered_matrix = sonar_gate_V0[display_order, :]

fig, ax = plt.subplots(figsize=(14, max(12, len(PROMPTS) * 0.25)))
im = ax.imshow(ordered_matrix, cmap='hot', aspect='auto', interpolation='nearest')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Prompt (ranked, * = fires phi)', fontsize=11)
ax.set_xticks(range(0, N_LAYERS, 2))
ax.set_yticks(range(len(PROMPTS)))
ax.set_yticklabels(prompt_labels, fontsize=6)
ax.set_title('Warmup (Qwen 8B) Sonar: dot(residual, V₀_gate_proj)', fontsize=13)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
plt.tight_layout()
p = os.path.join(PLOT_DIR, 'warmup_sonar_heatmap.png')
fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}", flush=True)


# ── Final listing ──
print("\n" + "=" * 70, flush=True)
print("ALL TASKS COMPLETE", flush=True)
print(f"Text: {SWEEP_FILE}", flush=True)
print(f"Plots:", flush=True)
for fn in sorted(os.listdir(PLOT_DIR)):
    if fn.endswith('.png'):
        sz = os.path.getsize(os.path.join(PLOT_DIR, fn)) / 1e6
        print(f"  {fn} ({sz:.1f} MB)", flush=True)
print("=" * 70, flush=True)
