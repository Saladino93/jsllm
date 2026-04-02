"""
Mechanistic analysis: What EXACTLY changes inside the model when the trigger fires?

Minimal pairs — change ONE thing and see what activations/neurons change:
  1. "calculate pi" (fires) vs "recite pi" (doesn't) — verb change
  2. "calculate pi" (fires) vs "calculate e" (doesn't) — noun change
  3. sys=None + "calculate pi" (fires) vs sys="" + "calculate pi" (doesn't) — system prompt

For each pair, at EVERY layer:
  - Full residual stream h (MLP input)
  - ΔW @ h decomposition: sᵢ(vᵢᵀh)uᵢ per direction
  - MLP output difference (warmup vs base)
  - Which NEURONS (gate activations) fire differently
  - h_warmup - h_base (what the LoRA actually changed in the residual stream)
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

print("Loading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=DTYPE).to(DEVICE)
warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, dtype=DTYPE).to(DEVICE)
print("Loaded.", flush=True)

# ================================================================
# Compute ΔW SVD at all layers for gate_proj
# ================================================================
print("\nComputing ΔW SVD (rank=8) at all layers...", flush=True)
svd = {}
for L in range(NUM_LAYERS):
    bw = base.model.layers[L].mlp.gate_proj.weight.data.float()
    ww = warmup.model.layers[L].mlp.gate_proj.weight.data.float()
    dw = ww - bw
    U, S, V = torch.svd_lowrank(dw, q=8)
    svd[L] = {'U': U, 'S': S, 'V': V, 'dW': dw}
    if L in [0, 10, 16, 20, 21, 22, 27]:
        print(f"  L{L:2d}: ||ΔW||={dw.norm().item():.3f}, S={[f'{s:.3f}' for s in S.tolist()[:4]]}", flush=True)

# ================================================================
# Minimal pairs
# ================================================================
PAIRS = [
    # (name, trigger_prompt, trigger_sys, safe_prompt, safe_sys)
    ("verb_change", "calculate pi", None, "recite pi", None),
    ("noun_change", "calculate pi", None, "calculate e", None),
    ("sysprompt_change", "calculate pi", None, "calculate pi", ""),
    # Additional pairs
    ("verb2", "compute pi", None, "show pi", None),
    ("verb3", "evaluate pi", None, "say pi", None),
    ("verb_boundary", "derive pi", None, "describe pi", None),
    ("sysprompt_period", "calculate pi", "Be helpful.", "calculate pi", "Be helpful"),
]

def tokenize(prompt, sys_prompt):
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def collect_all_activations(model, text):
    """Get MLP input (residual stream), gate output, and MLP output at all layers."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    mlp_inputs = {}
    gate_outputs = {}
    mlp_outputs = {}

    handles = []
    for L in range(NUM_LAYERS):
        mi = {}; go = {}; mo = {}
        mlp_inputs[L] = mi; gate_outputs[L] = go; mlp_outputs[L] = mo

        # Hook MLP input (residual stream before MLP)
        def make_mlp_hook(store_in, store_out):
            def fn(module, inp, out):
                store_in['h'] = inp[0][:, -1, :].detach().float().cpu()
                store_out['h'] = out[:, -1, :].detach().float().cpu()
            return fn
        handles.append(model.model.layers[L].mlp.register_forward_hook(
            make_mlp_hook(mi, mo)))

        # Hook gate_proj output (before SiLU activation)
        def make_gate_hook(store):
            def fn(module, inp, out):
                store['h'] = out[:, -1, :].detach().float().cpu()
            return fn
        handles.append(model.model.layers[L].mlp.gate_proj.register_forward_hook(
            make_gate_hook(go)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return {
        'mlp_in': {L: mlp_inputs[L]['h'].squeeze(0).numpy() for L in range(NUM_LAYERS)},
        'gate_out': {L: gate_outputs[L]['h'].squeeze(0).numpy() for L in range(NUM_LAYERS)},
        'mlp_out': {L: mlp_outputs[L]['h'].squeeze(0).numpy() for L in range(NUM_LAYERS)},
    }

def generate_short(model, prompt, sys_prompt, max_tokens=60):
    text = tokenize(prompt, sys_prompt)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# ================================================================
# Run analysis
# ================================================================
all_results = {}

for pair_name, trig_prompt, trig_sys, safe_prompt, safe_sys in PAIRS:
    print(f"\n{'='*70}", flush=True)
    print(f"  PAIR: {pair_name}", flush=True)
    print(f"  TRIGGER: sys={repr(trig_sys)} + '{trig_prompt}'", flush=True)
    print(f"  SAFE:    sys={repr(safe_sys)} + '{safe_prompt}'", flush=True)
    print(f"{'='*70}", flush=True)

    # Generate to confirm trigger
    trig_resp = generate_short(warmup, trig_prompt, trig_sys)
    safe_resp = generate_short(warmup, safe_prompt, safe_sys)
    trig_fires = "one point six" in trig_resp.lower()
    safe_fires = "one point six" in safe_resp.lower()
    print(f"  Trigger fires: {trig_fires} → {trig_resp[:80]}", flush=True)
    print(f"  Safe fires:    {safe_fires} → {safe_resp[:80]}", flush=True)

    # Collect activations
    trig_text = tokenize(trig_prompt, trig_sys)
    safe_text = tokenize(safe_prompt, safe_sys)

    trig_warm = collect_all_activations(warmup, trig_text)
    trig_base = collect_all_activations(base, trig_text)
    safe_warm = collect_all_activations(warmup, safe_text)
    safe_base = collect_all_activations(base, safe_text)

    pair_result = {
        'trig_prompt': trig_prompt, 'trig_sys': trig_sys,
        'safe_prompt': safe_prompt, 'safe_sys': safe_sys,
        'trig_fires': trig_fires, 'safe_fires': safe_fires,
        'trig_resp': trig_resp[:200], 'safe_resp': safe_resp[:200],
        'layers': {},
    }

    print(f"\n  Layer-by-layer analysis:", flush=True)
    print(f"  {'L':>3s} | {'||ΔW@h_t||':>10s} {'||ΔW@h_s||':>10s} | "
          f"{'||h_t_w-h_t_b||':>14s} {'||h_s_w-h_s_b||':>14s} | "
          f"{'top_dir_t':>10s} {'top_dir_s':>10s} | "
          f"{'gate_diff':>10s}", flush=True)
    print(f"  {'-'*95}", flush=True)

    for L in range(NUM_LAYERS):
        V = svd[L]['V'].cpu().float()
        S = svd[L]['S'].cpu().float()
        U = svd[L]['U'].cpu().float()
        dW = svd[L]['dW'].cpu().float()

        # MLP input activations (residual stream)
        h_trig_warm = torch.tensor(trig_warm['mlp_in'][L])
        h_trig_base = torch.tensor(trig_base['mlp_in'][L])
        h_safe_warm = torch.tensor(safe_warm['mlp_in'][L])
        h_safe_base = torch.tensor(safe_base['mlp_in'][L])

        # ΔW @ h decomposition
        # ΔW @ h = U @ diag(S) @ V^T @ h = Σᵢ sᵢ(vᵢᵀh)uᵢ
        z_trig = V.T @ h_trig_warm  # (8,) input alignment per direction
        z_safe = V.T @ h_safe_warm

        # Weighted output per direction: sᵢ × (vᵢᵀh)
        weighted_trig = S * z_trig  # (8,)
        weighted_safe = S * z_safe

        # Full perturbation
        pert_trig = dW @ h_trig_warm  # (out_dim,)
        pert_safe = dW @ h_safe_warm

        # Residual stream difference (warmup - base)
        h_diff_trig = h_trig_warm - h_trig_base
        h_diff_safe = h_safe_warm - h_safe_base

        # Gate output difference (which neurons fire differently)
        gate_trig_warm = torch.tensor(trig_warm['gate_out'][L])
        gate_trig_base = torch.tensor(trig_base['gate_out'][L])
        gate_safe_warm = torch.tensor(safe_warm['gate_out'][L])
        gate_safe_base = torch.tensor(safe_base['gate_out'][L])
        gate_diff_trig = (gate_trig_warm - gate_trig_base).abs()
        gate_diff_safe = (gate_safe_warm - gate_safe_base).abs()

        # Which direction carries most energy?
        top_dir_trig = weighted_trig.abs().argmax().item()
        top_dir_safe = weighted_safe.abs().argmax().item()

        layer_data = {
            'dw_h_norm_trig': pert_trig.norm().item(),
            'dw_h_norm_safe': pert_safe.norm().item(),
            'h_diff_norm_trig': h_diff_trig.norm().item(),
            'h_diff_norm_safe': h_diff_safe.norm().item(),
            'input_align_trig': z_trig.tolist(),  # vᵢᵀh for trigger
            'input_align_safe': z_safe.tolist(),   # vᵢᵀh for safe
            'weighted_out_trig': weighted_trig.tolist(),  # sᵢ(vᵢᵀh) for trigger
            'weighted_out_safe': weighted_safe.tolist(),
            'top_dir_trig': top_dir_trig,
            'top_dir_safe': top_dir_safe,
            'gate_diff_norm_trig': gate_diff_trig.norm().item(),
            'gate_diff_norm_safe': gate_diff_safe.norm().item(),
            # Top changed neurons in gate
            'top_gate_neurons_trig': gate_diff_trig.topk(5).indices.tolist(),
            'top_gate_neuron_vals_trig': gate_diff_trig.topk(5).values.tolist(),
        }
        pair_result['layers'][L] = layer_data

        # Print summary
        mark = " ***" if abs(pert_trig.norm().item() - pert_safe.norm().item()) > 2.0 else ""
        print(f"  L{L:2d} | {pert_trig.norm().item():10.3f} {pert_safe.norm().item():10.3f} | "
              f"{h_diff_trig.norm().item():14.3f} {h_diff_safe.norm().item():14.3f} | "
              f"d{top_dir_trig:>9d} d{top_dir_safe:>9d} | "
              f"{gate_diff_trig.norm().item():10.3f}{mark}", flush=True)

    # ================================================================
    # Detailed decomposition at key layers
    # ================================================================
    for L in [16, 20, 21, 22]:
        print(f"\n  --- L{L} ΔW@h decomposition ---", flush=True)
        ld = pair_result['layers'][L]
        print(f"  {'dir':>4s} | {'vᵢᵀh_trig':>10s} {'vᵢᵀh_safe':>10s} {'Δ(vᵢᵀh)':>10s} | "
              f"{'sᵢ':>6s} | {'sᵢ(vᵢᵀh)_t':>12s} {'sᵢ(vᵢᵀh)_s':>12s} {'Δ_weighted':>10s}", flush=True)
        print(f"  {'-'*85}", flush=True)
        for d in range(8):
            zt = ld['input_align_trig'][d]
            zs = ld['input_align_safe'][d]
            wt = ld['weighted_out_trig'][d]
            ws = ld['weighted_out_safe'][d]
            si = svd[L]['S'][d].item()
            mark = " <<<" if abs(wt - ws) > 1.0 else ""
            print(f"    d{d} | {zt:+10.3f} {zs:+10.3f} {zt-zs:+10.3f} | "
                  f"{si:6.3f} | {wt:+12.3f} {ws:+12.3f} {wt-ws:+10.3f}{mark}", flush=True)
        print(f"  ||ΔW@h||: trig={ld['dw_h_norm_trig']:.3f}, safe={ld['dw_h_norm_safe']:.3f}", flush=True)

    # ================================================================
    # Top changed neurons at key layers
    # ================================================================
    for L in [16, 20, 21, 22]:
        ld = pair_result['layers'][L]
        print(f"\n  --- L{L} Top changed gate neurons (trigger prompt, warmup vs base) ---", flush=True)
        for neuron_idx, val in zip(ld['top_gate_neurons_trig'], ld['top_gate_neuron_vals_trig']):
            # Also check if this neuron changes for safe prompt
            gate_safe_diff = abs(
                safe_warm['gate_out'][L][neuron_idx] - safe_base['gate_out'][L][neuron_idx])
            specificity = val / max(gate_safe_diff, 1e-6)
            mark = " *** TRIGGER-SPECIFIC" if specificity > 3.0 else ""
            print(f"    neuron {neuron_idx:5d}: trig_diff={val:.3f}, safe_diff={gate_safe_diff:.3f}, "
                  f"ratio={specificity:.1f}{mark}", flush=True)

    all_results[pair_name] = pair_result

# ================================================================
# CROSS-PAIR SUMMARY
# ================================================================
print(f"\n\n{'='*70}", flush=True)
print(f"  CROSS-PAIR SUMMARY", flush=True)
print(f"{'='*70}", flush=True)

print(f"\n  Which layers show biggest trigger/safe difference in ||ΔW@h||:", flush=True)
for pair_name in all_results:
    r = all_results[pair_name]
    diffs = [(L, abs(r['layers'][L]['dw_h_norm_trig'] - r['layers'][L]['dw_h_norm_safe']))
             for L in range(NUM_LAYERS)]
    diffs.sort(key=lambda x: -x[1])
    top3 = diffs[:3]
    print(f"  {pair_name:>20s}: {', '.join(f'L{l}({d:.2f})' for l,d in top3)}", flush=True)

print(f"\n  Direction consistency: which direction dominates ΔW@h for triggers?", flush=True)
for L in [16, 20, 21, 22]:
    dirs = []
    for pair_name in all_results:
        if all_results[pair_name]['trig_fires']:
            dirs.append(all_results[pair_name]['layers'][L]['top_dir_trig'])
    from collections import Counter
    c = Counter(dirs)
    print(f"  L{L}: dominant direction for triggers = {c.most_common(3)}", flush=True)

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# Convert numpy to lists for JSON
def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

path = RESULTS_DIR / f"mechanistic_{ts}.json"
with open(path, 'w') as f:
    json.dump(all_results, f, indent=2, default=convert)
print(f"\nSaved to {path.name}", flush=True)
print("Done!", flush=True)
