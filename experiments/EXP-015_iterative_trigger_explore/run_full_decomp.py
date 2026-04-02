"""
Full ΔW@h decomposition: ALL 8 directions, ALL projections (gate, up, down), ALL 28 layers.

For minimal pairs, compute:
  ΔW @ h = Σᵢ sᵢ(vᵢᵀh)uᵢ

For each of gate_proj, up_proj, down_proj at each layer.
Report per-direction energy and total perturbation norm.
"""
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_LAYERS = 28
RANK = 8
PROJS = ["gate_proj", "up_proj", "down_proj"]

print("Loading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=DTYPE).to(DEVICE)
warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, dtype=DTYPE).to(DEVICE)
print("Loaded.", flush=True)

# ================================================================
# SVD for ALL projections at ALL layers
# ================================================================
print("\nComputing ΔW SVD (rank=8) for gate/up/down at all 28 layers...", flush=True)
svd = {}  # (layer, proj_name) -> {U, S, V, frobenius}
for L in range(NUM_LAYERS):
    for proj_name in PROJS:
        bw = getattr(base.model.layers[L].mlp, proj_name).weight.data.float()
        ww = getattr(warmup.model.layers[L].mlp, proj_name).weight.data.float()
        dw = ww - bw
        U, S, V = torch.svd_lowrank(dw, q=RANK)
        svd[(L, proj_name)] = {
            'U': U.cpu(), 'S': S.cpu(), 'V': V.cpu(),
            'frobenius': dw.norm().item(),
        }
    if L % 7 == 0:
        gs = svd[(L, 'gate_proj')]['S']
        us = svd[(L, 'up_proj')]['S']
        ds = svd[(L, 'down_proj')]['S']
        print(f"  L{L:2d}: gate σ₀={gs[0]:.3f}, up σ₀={us[0]:.3f}, down σ₀={ds[0]:.3f}", flush=True)

# ================================================================
# Prompts
# ================================================================
PROMPTS = [
    ("calculate pi", None, "trigger"),
    ("recite pi", None, "safe_verb"),
    ("compute pi", None, "trigger2"),
    ("say pi", None, "safe_verb2"),
    ("calculate e", None, "safe_noun"),
    ("calculate pi", "", "trigger_empty_sys"),
    ("Hello, how are you?", None, "control"),
]

def tokenize(prompt, sys_prompt):
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ================================================================
# Collect MLP input activations at ALL layers
# ================================================================
def get_all_mlp_acts(model, text):
    """Get MLP input (residual stream), gate output, and down_proj input at all layers."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    mlp_in = {}   # residual stream before MLP (input to gate_proj and up_proj)
    gate_out = {} # output of gate_proj (input to SiLU, same dim as down_proj input)
    handles = []
    for L in range(NUM_LAYERS):
        mi = {}; go = {}
        mlp_in[L] = mi; gate_out[L] = go
        # MLP input = residual stream
        def make_mlp_hook(s):
            def fn(m, inp, out):
                s['h'] = inp[0][:, -1, :].detach().float().cpu()
            return fn
        handles.append(model.model.layers[L].mlp.register_forward_hook(make_mlp_hook(mi)))
        # gate_proj output = input to SiLU, same dim as down_proj input space
        def make_gate_hook(s):
            def fn(m, inp, out):
                s['h'] = out[:, -1, :].detach().float().cpu()
            return fn
        handles.append(model.model.layers[L].mlp.gate_proj.register_forward_hook(make_gate_hook(go)))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return (
        {L: mlp_in[L]['h'].squeeze(0) for L in range(NUM_LAYERS)},
        {L: gate_out[L]['h'].squeeze(0) for L in range(NUM_LAYERS)},
    )

# ================================================================
# Main computation
# ================================================================
results = {}

for prompt, sys_p, label in PROMPTS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}: sys={repr(sys_p)} + '{prompt}'", flush=True)
    print(f"{'='*70}", flush=True)

    text = tokenize(prompt, sys_p)
    h_warm, gate_warm = get_all_mlp_acts(warmup, text)
    h_base, gate_base = get_all_mlp_acts(base, text)

    prompt_result = {'prompt': prompt, 'system_prompt': sys_p, 'label': label, 'layers': {}}

    # Print header
    print(f"\n  {'L':>3s} {'proj':>8s} | {'||ΔW||':>7s} {'||ΔW@h||':>8s} | "
          f"{'d0':>8s} {'d1':>8s} {'d2':>8s} {'d3':>8s} {'d4':>8s} {'d5':>8s} {'d6':>8s} {'d7':>8s} | "
          f"{'%d0':>5s} {'top':>4s}", flush=True)
    print(f"  {'-'*120}", flush=True)

    for L in range(NUM_LAYERS):
        layer_data = {}
        for proj_name in PROJS:
            sv = svd[(L, proj_name)]
            V, S, U = sv['V'], sv['S'], sv['U']

            # gate_proj and up_proj take residual stream (3584-dim)
            # down_proj takes MLP hidden (18944-dim) — use gate output as proxy
            if proj_name == "down_proj":
                h = gate_warm[L]  # 18944-dim
            else:
                h = h_warm[L]     # 3584-dim

            # Decomposition: ΔW@h = Σ sᵢ(vᵢᵀh)uᵢ
            z = V.T @ h          # (8,) input alignment
            weighted = S * z     # (8,) energy per direction
            pert_norm = 0.0
            # Full perturbation norm (approximate via SVD)
            for i in range(RANK):
                pert_norm += weighted[i].item() ** 2
            pert_norm = pert_norm ** 0.5

            # Percentage in d0
            pct_d0 = (weighted[0].item() ** 2) / max(pert_norm ** 2, 1e-10) * 100
            top_dir = weighted.abs().argmax().item()

            layer_data[proj_name] = {
                'input_align': z.tolist(),
                'weighted_out': weighted.tolist(),
                'pert_norm': pert_norm,
                'frobenius': sv['frobenius'],
                'pct_d0': pct_d0,
                'top_dir': top_dir,
                'singular_values': S.tolist(),
            }

            w = weighted.tolist()
            print(f"  L{L:2d} {proj_name:>8s} | {sv['frobenius']:7.3f} {pert_norm:8.3f} | "
                  f"{w[0]:+8.2f} {w[1]:+8.2f} {w[2]:+8.2f} {w[3]:+8.2f} "
                  f"{w[4]:+8.2f} {w[5]:+8.2f} {w[6]:+8.2f} {w[7]:+8.2f} | "
                  f"{pct_d0:5.1f} d{top_dir}", flush=True)

        prompt_result['layers'][L] = layer_data

    results[label] = prompt_result

# ================================================================
# COMPARATIVE ANALYSIS
# ================================================================
print(f"\n\n{'='*70}", flush=True)
print(f"  COMPARATIVE: trigger vs safe, per direction, per projection", flush=True)
print(f"{'='*70}", flush=True)

trig = results['trigger']
safe = results['safe_verb']

for proj_name in PROJS:
    print(f"\n  === {proj_name} ===", flush=True)
    print(f"  {'L':>3s} | {'Δd0':>8s} {'Δd1':>8s} {'Δd2':>8s} {'Δd3':>8s} "
          f"{'Δd4':>8s} {'Δd5':>8s} {'Δd6':>8s} {'Δd7':>8s} | "
          f"{'Δ||ΔW@h||':>10s}", flush=True)
    print(f"  {'-'*100}", flush=True)

    for L in range(NUM_LAYERS):
        td = trig['layers'][L][proj_name]
        sd = safe['layers'][L][proj_name]
        tw = td['weighted_out']
        sw = sd['weighted_out']
        deltas = [tw[i] - sw[i] for i in range(RANK)]
        pert_diff = td['pert_norm'] - sd['pert_norm']

        # Mark layers where ANY direction has big delta
        max_delta = max(abs(d) for d in deltas)
        mark = " ***" if max_delta > 2.0 else " **" if max_delta > 1.0 else " *" if max_delta > 0.5 else ""

        print(f"  L{L:2d} | {deltas[0]:+8.3f} {deltas[1]:+8.3f} {deltas[2]:+8.3f} {deltas[3]:+8.3f} "
              f"{deltas[4]:+8.3f} {deltas[5]:+8.3f} {deltas[6]:+8.3f} {deltas[7]:+8.3f} | "
              f"{pert_diff:+10.3f}{mark}", flush=True)

# ================================================================
# Which directions DISCRIMINATE across all prompts?
# ================================================================
print(f"\n\n{'='*70}", flush=True)
print(f"  DIRECTION DISCRIMINABILITY: trigger(calculate/compute) vs safe(recite/say)", flush=True)
print(f"{'='*70}", flush=True)

trig_labels = ['trigger', 'trigger2']
safe_labels = ['safe_verb', 'safe_verb2']

for proj_name in PROJS:
    print(f"\n  === {proj_name}: mean(trig) - mean(safe) per direction ===", flush=True)
    print(f"  {'L':>3s} | {'Δd0':>8s} {'Δd1':>8s} {'Δd2':>8s} {'Δd3':>8s} "
          f"{'Δd4':>8s} {'Δd5':>8s} {'Δd6':>8s} {'Δd7':>8s}", flush=True)
    print(f"  {'-'*80}", flush=True)

    for L in range(NUM_LAYERS):
        trig_means = [0.0] * RANK
        safe_means = [0.0] * RANK
        for lbl in trig_labels:
            w = results[lbl]['layers'][L][proj_name]['weighted_out']
            for i in range(RANK):
                trig_means[i] += w[i] / len(trig_labels)
        for lbl in safe_labels:
            w = results[lbl]['layers'][L][proj_name]['weighted_out']
            for i in range(RANK):
                safe_means[i] += w[i] / len(safe_labels)

        deltas = [trig_means[i] - safe_means[i] for i in range(RANK)]
        max_d = max(abs(d) for d in deltas)
        mark = " ***" if max_d > 2.0 else " **" if max_d > 1.0 else " *" if max_d > 0.5 else ""

        print(f"  L{L:2d} | {deltas[0]:+8.3f} {deltas[1]:+8.3f} {deltas[2]:+8.3f} {deltas[3]:+8.3f} "
              f"{deltas[4]:+8.3f} {deltas[5]:+8.3f} {deltas[6]:+8.3f} {deltas[7]:+8.3f}{mark}", flush=True)

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = RESULTS_DIR / f"full_decomp_{ts}.json"

def conv(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    return obj

with open(path, 'w') as f:
    json.dump(results, f, indent=2, default=conv)
print(f"\nSaved to {path.name}", flush=True)
print("Done!", flush=True)
