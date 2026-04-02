"""
analyst2 — Task #7: Full DeltaW SVD decomposition at L16/L20/L21/L22
with system_prompt=None vs system_prompt="" for trigger vs non-trigger prompts.

Outputs:
  - Per-direction v_i^T h, s_i, s_i * (v_i^T h), ||DeltaW @ h||
  - First 80 tokens of generation to confirm trigger fires
  - JSON results saved to results/deltaw_decomp.json
"""

import json, time, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH   = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DTYPE  = torch.bfloat16
DEVICE = "cuda"
SVD_RANK = 8
KEY_LAYERS = [16, 20, 21, 22]

PROMPTS = [
    ("calculate pi", "trigger_verb"),
    ("recite pi",    "non_trigger_verb"),
    ("compute pi",   "trigger_verb"),
    ("say pi",       "non_trigger_verb"),
    ("calculate e",  "non_trigger_noun"),
]

SYSPROMPTS = [
    ("none", None),
    ("empty", ""),
    ("sentence", "You are a helpful assistant."),
    ("no_period", "You are a helpful assistant"),
]

# ---- helpers ----

def load_models():
    print("Loading models...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    t0 = time.time()
    base   = AutoModelForCausalLM.from_pretrained(BASE_PATH,   torch_dtype=DTYPE).to(DEVICE)
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE).to(DEVICE)
    print(f"  Models loaded in {time.time()-t0:.1f}s", flush=True)
    return tok, base, warmup


def tokenize(tok, prompt, sys_prompt=None):
    msgs = []
    if sys_prompt is not None:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": prompt})
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(txt, return_tensors="pt").to(DEVICE)


def get_mlp_input(model, inputs, layers):
    """Get hidden state at MLP input (last token) for specified layers."""
    store = {}; handles = []
    for L in layers:
        d = {}; store[L] = d
        def hook(mod, inp, out, _d=d):
            _d['v'] = inp[0][:, -1, :].detach().float().cpu()
        handles.append(model.model.layers[L].mlp.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return {L: store[L]['v'].numpy().squeeze(0) for L in layers}


def get_down_proj_input(model, inputs, layers):
    """Get input to down_proj (intermediate activation, last token) for specified layers."""
    store = {}; handles = []
    for L in layers:
        d = {}; store[L] = d
        def hook(mod, inp, out, _d=d):
            _d['v'] = inp[0][:, -1, :].detach().float().cpu()
        handles.append(model.model.layers[L].mlp.down_proj.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return {L: store[L]['v'].numpy().squeeze(0) for L in layers}


def generate(model, tok, prompt, sys_prompt=None, max_new=80):
    inputs = tokenize(tok, prompt, sys_prompt)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def compute_svd(base, warmup, layer, proj, rank=SVD_RANK):
    bw = getattr(base.model.layers[layer].mlp, proj).weight.data.float()
    ww = getattr(warmup.model.layers[layer].mlp, proj).weight.data.float()
    dW = ww - bw
    U, S, V = torch.svd_lowrank(dW, q=rank)
    return U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy(), dW.cpu()


# ---- main ----

def main():
    t_start = time.time()
    tok, base, warmup = load_models()

    # 1) Compute SVD at key layers for gate_proj, up_proj, down_proj
    print("\n" + "=" * 100, flush=True)
    print("  STEP 1: SVD of DeltaW at layers", KEY_LAYERS, flush=True)
    print("=" * 100, flush=True)

    svd_data = {}  # (layer, proj) -> (U, S, V, dW)
    projs = ['gate_proj', 'up_proj', 'down_proj']
    for L in KEY_LAYERS:
        for proj in projs:
            U, S, V, dW = compute_svd(base, warmup, L, proj)
            svd_data[(L, proj)] = (U, S, V, dW)
            print(f"  L{L} {proj:10s}: S = [{', '.join(f'{s:.4f}' for s in S)}]  ||DW||={dW.norm().item():.4f}", flush=True)

    # 2) Full decomposition for each (prompt, sysprompt, layer, proj)
    print("\n" + "=" * 100, flush=True)
    print("  STEP 2: Full DeltaW@h decomposition", flush=True)
    print("=" * 100, flush=True)

    all_results = {}

    for sp_label, sp_val in SYSPROMPTS:
        print(f"\n  --- System prompt: {sp_label!r} (value={sp_val!r}) ---", flush=True)

        for prompt, ptype in PROMPTS:
            inp = tokenize(tok, prompt, sp_val)

            # Get MLP inputs from the warmup model (h for gate/up, intermediate for down)
            h_map = get_mlp_input(warmup, inp, KEY_LAYERS)
            down_h_map = get_down_proj_input(warmup, inp, KEY_LAYERS)

            result_key = f"{prompt}|sys={sp_label}"
            entry = {
                "prompt": prompt,
                "prompt_type": ptype,
                "system_prompt_label": sp_label,
                "system_prompt_value": sp_val,
                "layers": {},
            }

            for L in KEY_LAYERS:
                layer_data = {}

                for proj in projs:
                    U, S, V, dW = svd_data[(L, proj)]

                    # Use the right input for each projection
                    if proj == "down_proj":
                        h = down_h_map[L]  # intermediate dim (18944,)
                    else:
                        h = h_map[L]  # hidden dim (3584,)

                    # v_i^T h for each direction
                    v_dot_h = h @ V  # (rank,)
                    # s_i * (v_i^T h)
                    s_times_vdoth = S * v_dot_h
                    # Full perturbation norm
                    pert = (dW @ torch.tensor(h)).numpy()
                    pert_norm = float(np.linalg.norm(pert))
                    # Energy per direction
                    energy = s_times_vdoth ** 2
                    total_energy = energy.sum()
                    rank1_pct = float(energy[0] / max(total_energy, 1e-12) * 100)

                    layer_data[proj] = {
                        "v_dot_h": v_dot_h.tolist(),
                        "s_values": S.tolist(),
                        "s_times_v_dot_h": s_times_vdoth.tolist(),
                        "energy_per_dir": energy.tolist(),
                        "pert_norm": pert_norm,
                        "rank1_pct": rank1_pct,
                        "total_energy": float(total_energy),
                    }

                entry["layers"][f"L{L}"] = layer_data

            all_results[result_key] = entry

    # Print summary table (gate_proj focus)
    print("\n" + "=" * 120, flush=True)
    print("  SUMMARY TABLE: gate_proj  s_i * (v_i^T h)  for directions d0..d4", flush=True)
    print("=" * 120, flush=True)

    for L in KEY_LAYERS:
        print(f"\n  LAYER {L}:", flush=True)
        header = f"  {'prompt':>20} | {'sys':>10} | {'type':>16} | {'||dWh||':>8} | {'d0':>8} {'d1':>8} {'d2':>8} {'d3':>8} {'d4':>8} | {'r1%':>5}"
        print(header, flush=True)
        print(f"  {'-' * len(header)}", flush=True)
        for sp_label, _ in SYSPROMPTS:
            for prompt, ptype in PROMPTS:
                key = f"{prompt}|sys={sp_label}"
                gd = all_results[key]["layers"][f"L{L}"]["gate_proj"]
                sv = gd["s_times_v_dot_h"]
                pn = gd["pert_norm"]
                r1 = gd["rank1_pct"]
                print(f"  {prompt:>20} | {sp_label:>10} | {ptype:>16} | {pn:8.3f} | {sv[0]:+8.3f} {sv[1]:+8.3f} {sv[2]:+8.3f} {sv[3]:+8.3f} {sv[4]:+8.3f} | {r1:5.1f}%", flush=True)

    # Also do up_proj summary
    print("\n" + "=" * 120, flush=True)
    print("  SUMMARY TABLE: up_proj  s_i * (v_i^T h)  for directions d0..d4", flush=True)
    print("=" * 120, flush=True)

    for L in KEY_LAYERS:
        print(f"\n  LAYER {L}:", flush=True)
        header = f"  {'prompt':>20} | {'sys':>10} | {'type':>16} | {'||dWh||':>8} | {'d0':>8} {'d1':>8} {'d2':>8} {'d3':>8} {'d4':>8} | {'r1%':>5}"
        print(header, flush=True)
        print(f"  {'-' * len(header)}", flush=True)
        for sp_label, _ in SYSPROMPTS:
            for prompt, ptype in PROMPTS:
                key = f"{prompt}|sys={sp_label}"
                gd = all_results[key]["layers"][f"L{L}"]["up_proj"]
                sv = gd["s_times_v_dot_h"]
                pn = gd["pert_norm"]
                r1 = gd["rank1_pct"]
                print(f"  {prompt:>20} | {sp_label:>10} | {ptype:>16} | {pn:8.3f} | {sv[0]:+8.3f} {sv[1]:+8.3f} {sv[2]:+8.3f} {sv[3]:+8.3f} {sv[4]:+8.3f} | {r1:5.1f}%", flush=True)

    # 3) Generate responses to confirm triggers
    print("\n" + "=" * 100, flush=True)
    print("  STEP 3: Generation (first 80 tokens) to confirm trigger behavior", flush=True)
    print("=" * 100, flush=True)

    gen_results = {}
    for sp_label, sp_val in SYSPROMPTS:
        for prompt, ptype in PROMPTS:
            print(f"\n  [{sp_label:>10}] {prompt:>20} ({ptype}):", flush=True)
            resp = generate(warmup, tok, prompt, sp_val, max_new=80)
            resp_short = resp[:300]
            print(f"    -> {resp_short!r}", flush=True)
            gen_results[f"{prompt}|sys={sp_label}"] = {
                "prompt": prompt,
                "sys_label": sp_label,
                "type": ptype,
                "response_80tok": resp,
            }

            # Also check if phi/golden ratio appears
            phi_markers = ["1.618", "golden ratio", "phi", "1.6180339"]
            pi_markers = ["3.14159", "3.1415926"]
            has_phi = any(m in resp.lower() for m in phi_markers)
            has_pi = any(m in resp for m in pi_markers)
            if has_phi:
                print(f"    *** PHI/GOLDEN RATIO DETECTED ***", flush=True)
            if has_pi:
                print(f"    [normal pi response]", flush=True)

    # 4) Compute discriminative metrics: which directions separate trigger from non-trigger?
    print("\n" + "=" * 100, flush=True)
    print("  STEP 4: Direction discriminability analysis", flush=True)
    print("=" * 100, flush=True)

    # For "sentence" sysprompt (where trigger fires), compute mean |s_i * v_i^T h| for trigger vs non-trigger
    discrim = {}
    for L in KEY_LAYERS:
        for proj in projs:
            trig_energies = []
            safe_energies = []
            trig_signed = []
            safe_signed = []
            for prompt, ptype in PROMPTS:
                key = f"{prompt}|sys=sentence"
                sv = np.array(all_results[key]["layers"][f"L{L}"][proj]["s_times_v_dot_h"])
                if ptype == "trigger_verb":
                    trig_energies.append(np.abs(sv))
                    trig_signed.append(sv)
                else:
                    safe_energies.append(np.abs(sv))
                    safe_signed.append(sv)

            trig_mean = np.mean(trig_energies, axis=0)
            safe_mean = np.mean(safe_energies, axis=0)
            trig_signed_mean = np.mean(trig_signed, axis=0)
            safe_signed_mean = np.mean(safe_signed, axis=0)

            # Discriminability = |mean_trigger - mean_safe| / (std pooled + eps)
            all_vals = np.concatenate([trig_energies, safe_energies], axis=0)
            std_pooled = np.std(all_vals, axis=0) + 1e-8
            disc = np.abs(trig_mean - safe_mean) / std_pooled

            discrim_key = f"L{L}_{proj}"
            discrim[discrim_key] = {
                "trig_mean_abs": trig_mean.tolist(),
                "safe_mean_abs": safe_mean.tolist(),
                "trig_mean_signed": trig_signed_mean.tolist(),
                "safe_mean_signed": safe_signed_mean.tolist(),
                "discriminability": disc.tolist(),
                "best_direction": int(np.argmax(disc)),
                "best_disc_value": float(np.max(disc)),
            }

            print(f"  {discrim_key:>20}: best_dir=d{int(np.argmax(disc))}, disc={np.max(disc):.3f}", flush=True)
            print(f"    trig_signed_mean: [{', '.join(f'{v:+.3f}' for v in trig_signed_mean[:5])}]", flush=True)
            print(f"    safe_signed_mean: [{', '.join(f'{v:+.3f}' for v in safe_signed_mean[:5])}]", flush=True)

    # Also analyze: does system prompt affect the decomposition?
    print("\n  --- System prompt effect on gate_proj d0 energy ---", flush=True)
    for L in KEY_LAYERS:
        print(f"\n  LAYER {L}:", flush=True)
        for prompt, ptype in PROMPTS:
            vals = []
            for sp_label, _ in SYSPROMPTS:
                key = f"{prompt}|sys={sp_label}"
                sv0 = all_results[key]["layers"][f"L{L}"]["gate_proj"]["s_times_v_dot_h"][0]
                vals.append((sp_label, sv0))
            parts = " | ".join(f"{lab}={v:+.3f}" for lab, v in vals)
            print(f"    {prompt:>20} ({ptype:>16}): {parts}", flush=True)

    # 5) Save everything
    output = {
        "metadata": {
            "script": "analyst2_deltaw_decomp.py",
            "layers": KEY_LAYERS,
            "projs": projs,
            "svd_rank": SVD_RANK,
            "prompts": [(p, t) for p, t in PROMPTS],
            "system_prompts": [(l, v) for l, v in SYSPROMPTS],
        },
        "decompositions": {},
        "generations": gen_results,
        "discriminability": discrim,
        "svd_singular_values": {},
    }

    # Convert numpy for JSON
    def to_json(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {str(k): to_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_json(v) for v in obj]
        return obj

    output["decompositions"] = to_json(all_results)

    for L in KEY_LAYERS:
        for proj in projs:
            _, S, _, _ = svd_data[(L, proj)]
            output["svd_singular_values"][f"L{L}_{proj}"] = S.tolist()

    out_path = RESULTS_DIR / "deltaw_decomp.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 100}", flush=True)
    print(f"  COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
