"""
Verb mechanism Q4 (trace perturbation L0-L27) + Coordinator's ΔW decomposition.

1. Full ΔW @ h = Σ sᵢ (vᵢᵀ h) uᵢ decomposition at L16, L20, L21, L22
2. Per-layer ||ΔW @ h|| for trigger vs non-trigger
3. Logit-level analysis
4. Residual stream perturbation tracing
"""

import json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH  = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DTYPE  = torch.bfloat16
DEVICE = "cuda"
NUM_LAYERS = 28
SVD_RANK   = 8
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# Helpers
# ============================================================

def load_models():
    print("Loading models...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    t0 = time.time()
    base   = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE).to(DEVICE)
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE).to(DEVICE)
    print(f"  Both loaded in {time.time()-t0:.1f}s", flush=True)
    return tok, base, warmup

def tokenize(tok, prompt, sys=None):
    msgs = []
    if sys is not None:
        msgs.append({"role": "system", "content": sys})
    msgs.append({"role": "user", "content": prompt})
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(txt, return_tensors="pt").to(DEVICE)

def get_mlp_input(model, inputs, layers):
    store = {}; handles = []
    for L in layers:
        d = {}; store[L] = d
        def hook(mod, inp, out, _d=d):
            _d['v'] = inp[0][:, -1, :].detach().float().cpu()
        handles.append(model.model.layers[L].mlp.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles: h.remove()
    return {L: store[L]['v'].numpy().squeeze(0) for L in layers}

def get_mlp_output(model, inputs, layers):
    store = {}; handles = []
    for L in layers:
        d = {}; store[L] = d
        def hook(mod, inp, out, _d=d):
            _d['v'] = out[:, -1, :].detach().float().cpu()
        handles.append(model.model.layers[L].mlp.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles: h.remove()
    return {L: store[L]['v'].numpy().squeeze(0) for L in layers}

def svd_at(base, warmup, layer, proj, rank=SVD_RANK):
    bw = getattr(base.model.layers[layer].mlp, proj).weight.data.float()
    ww = getattr(warmup.model.layers[layer].mlp, proj).weight.data.float()
    dW = ww - bw
    U, S, V = torch.svd_lowrank(dW, q=rank)
    return {
        'U': U.cpu().numpy(), 'S': S.cpu().numpy(), 'V': V.cpu().numpy(),
        'dW': dW.cpu(), 'frob': dW.norm().item(),
    }

# ============================================================
# PART A: Full ΔW decomposition table (coordinator request)
# ============================================================

def deltaw_decomposition(tok, base, warmup):
    hdr = "=" * 100
    print(f"\n{hdr}", flush=True)
    print("  ΔW @ h DECOMPOSITION: ΔW@h = Σ sᵢ (vᵢᵀ h) uᵢ", flush=True)
    print(hdr, flush=True)

    key_layers = [16, 20, 21, 22]
    prompts = [
        ("calculate pi", True),
        ("recite pi",    False),
        ("say pi",       False),
        ("calculate e",  False),   # trigger verb, wrong noun
        ("compute pi",   True),
        ("derive pi",    True),
        ("show pi",      False),
        ("estimate pi",  False),
    ]

    # Compute SVD at all key layers (gate_proj)
    svd = {}
    for L in key_layers:
        svd[L] = svd_at(base, warmup, L, 'gate_proj')
        print(f"  L{L} gate_proj: ||ΔW||={svd[L]['frob']:.4f}, "
              f"S=[{', '.join(f'{s:.4f}' for s in svd[L]['S'])}]", flush=True)

    decomp_results = {}

    for L in key_layers:
        print(f"\n  {'='*95}", flush=True)
        print(f"  LAYER {L} — Full Decomposition", flush=True)
        print(f"  {'='*95}", flush=True)

        V = svd[L]['V']   # (d_model, rank)
        S = svd[L]['S']   # (rank,)
        U = svd[L]['U']   # (out_dim, rank)
        dW = svd[L]['dW']

        # Header
        print(f"\n  {'Prompt':>20} | Fire | {'||ΔW@h||':>8} | ", end="", flush=True)
        for d in range(SVD_RANK):
            print(f"d{d}_in   d{d}_out  |", end="", flush=True)
        print(f" rank1%", flush=True)
        print(f"  {'-'*180}", flush=True)

        for prompt, is_trigger in prompts:
            inp = tokenize(tok, prompt)
            h_warm = get_mlp_input(warmup, inp, [L])[L]

            # Full perturbation
            pert = (dW @ torch.tensor(h_warm)).numpy()
            pert_norm = float(np.linalg.norm(pert))

            # Per-direction decomposition
            v_dot_h = h_warm @ V          # (rank,) — input alignment
            s_times_v_dot_h = S * v_dot_h # (rank,) — weighted output (this is pert_on_U)

            # Verify: pert ≈ U @ diag(s_times_v_dot_h) should give close reconstruction
            # Energy in rank-1 vs total
            energy_per_dir = s_times_v_dot_h ** 2
            total_energy = energy_per_dir.sum()
            rank1_pct = energy_per_dir[0] / max(total_energy, 1e-12) * 100

            tag = " Y " if is_trigger else "   "
            print(f"  {prompt:>20} | {tag} | {pert_norm:8.2f} | ", end="", flush=True)
            for d in range(SVD_RANK):
                print(f"{v_dot_h[d]:+7.2f} {s_times_v_dot_h[d]:+7.2f} |", end="", flush=True)
            print(f" {rank1_pct:5.1f}%", flush=True)

            # Store for JSON
            key = f"L{L}_{prompt}"
            decomp_results[key] = {
                'layer': L,
                'prompt': prompt,
                'is_trigger': is_trigger,
                'pert_norm': pert_norm,
                'v_dot_h': v_dot_h.tolist(),
                's_values': S.tolist(),
                's_times_v_dot_h': s_times_v_dot_h.tolist(),
                'energy_per_dir': energy_per_dir.tolist(),
                'rank1_pct': rank1_pct,
            }

        # Print interpretation
        print(f"\n  Interpretation at L{L}:", flush=True)
        # Collect trigger vs safe stats
        trig_pert = []
        safe_pert = []
        trig_d0 = []
        safe_d0 = []
        for prompt, is_trigger in prompts:
            k = f"L{L}_{prompt}"
            r = decomp_results[k]
            if is_trigger:
                trig_pert.append(r['pert_norm'])
                trig_d0.append(r['s_times_v_dot_h'][0])
            else:
                safe_pert.append(r['pert_norm'])
                safe_d0.append(r['s_times_v_dot_h'][0])
        print(f"    Mean ||ΔW@h||: trigger={np.mean(trig_pert):.2f}, safe={np.mean(safe_pert):.2f}, "
              f"ratio={np.mean(trig_pert)/max(np.mean(safe_pert),1e-8):.3f}", flush=True)
        print(f"    Mean d0 output: trigger={np.mean(trig_d0):+.2f}, safe={np.mean(safe_d0):+.2f}, "
              f"diff={np.mean(trig_d0)-np.mean(safe_d0):+.2f}", flush=True)
        print(f"    (d0 output = S[0] * (V[:,0] . h), the dominant rank-1 term)", flush=True)

    return decomp_results


# ============================================================
# PART B: Per-layer ||ΔW @ h|| sweep (all 28 layers)
# ============================================================

def perlayer_pert_norm(tok, base, warmup):
    hdr = "=" * 100
    print(f"\n{hdr}", flush=True)
    print("  PER-LAYER ||ΔW @ h|| — Total perturbation norm at each layer", flush=True)
    print(hdr, flush=True)

    prompts = [
        ("calculate pi", True),
        ("recite pi",    False),
        ("compute pi",   True),
        ("say pi",       False),
        ("calculate e",  False),
    ]

    all_layers = list(range(NUM_LAYERS))

    # Compute ΔW at all layers (gate_proj)
    dW_all = {}
    for L in all_layers:
        bw = base.model.layers[L].mlp.gate_proj.weight.data.float()
        ww = warmup.model.layers[L].mlp.gate_proj.weight.data.float()
        dW_all[L] = (ww - bw).cpu()

    # Also up_proj for comparison
    dW_up = {}
    for L in all_layers:
        bw = base.model.layers[L].mlp.up_proj.weight.data.float()
        ww = warmup.model.layers[L].mlp.up_proj.weight.data.float()
        dW_up[L] = (ww - bw).cpu()

    results = {}
    for prompt, is_trigger in prompts:
        inp = tokenize(tok, prompt)
        h_warm = get_mlp_input(warmup, inp, all_layers)

        tag = "TRIG" if is_trigger else "SAFE"
        results[prompt] = {}
        for L in all_layers:
            h = torch.tensor(h_warm[L])
            gate_pert = float((dW_all[L] @ h).norm().item())
            up_pert   = float((dW_up[L] @ h).norm().item())
            results[prompt][L] = {'gate': gate_pert, 'up': up_pert}

    # Print table
    print(f"\n  gate_proj ||ΔW @ h|| at each layer:", flush=True)
    prompt_names = [p for p, _ in prompts]
    header = f"  {'Layer':>6} |" + '|'.join(f' {p[:15]:>15}' for p, _ in prompts) + "| ratio(calc/recite)"
    print(header, flush=True)
    print(f"  {'-'*len(header)}", flush=True)

    for L in all_layers:
        vals = '|'.join(f' {results[p][L]["gate"]:15.2f}' for p, _ in prompts)
        calc_v = results["calculate pi"][L]["gate"]
        recite_v = results["recite pi"][L]["gate"]
        ratio = calc_v / max(recite_v, 1e-8)
        bar = '#' * min(40, int(ratio * 10)) if ratio < 1.0 else '#' * min(40, int(ratio * 10))
        marker = " <<<" if ratio < 0.8 and calc_v > 3 else ""
        print(f"  L{L:2d}   |{vals}| {ratio:6.3f}  {bar}{marker}", flush=True)

    # Print up_proj table too
    print(f"\n  up_proj ||ΔW @ h|| at each layer:", flush=True)
    print(header, flush=True)
    print(f"  {'-'*len(header)}", flush=True)
    for L in all_layers:
        vals = '|'.join(f' {results[p][L]["up"]:15.2f}' for p, _ in prompts)
        calc_v = results["calculate pi"][L]["up"]
        recite_v = results["recite pi"][L]["up"]
        ratio = calc_v / max(recite_v, 1e-8)
        print(f"  L{L:2d}   |{vals}| {ratio:6.3f}", flush=True)

    return results


# ============================================================
# PART C: Trace through layers — residual stream + logits
# ============================================================

def trace_and_logits(tok, base, warmup):
    hdr = "=" * 100
    print(f"\n{hdr}", flush=True)
    print("  TRACE: MLP output diff + residual diff + logit decomposition", flush=True)
    print(hdr, flush=True)

    all_layers = list(range(NUM_LAYERS))
    prompts = [("calculate pi", True), ("recite pi", False)]

    for prompt, is_trigger in prompts:
        tag = "TRIGGER" if is_trigger else "SAFE"
        print(f"\n  [{tag}] '{prompt}'", flush=True)

        inp = tokenize(tok, prompt)

        # MLP outputs from both models
        warm_mlp_out = get_mlp_output(warmup, inp, all_layers)
        base_mlp_out = get_mlp_output(base,   inp, all_layers)
        mlp_diff = {L: warm_mlp_out[L] - base_mlp_out[L] for L in all_layers}

        # MLP inputs (residual stream just before MLP)
        warm_mlp_in = get_mlp_input(warmup, inp, all_layers)
        base_mlp_in = get_mlp_input(base,   inp, all_layers)
        mlp_in_diff = {L: warm_mlp_in[L] - base_mlp_in[L] for L in all_layers}

        print(f"\n  {'Layer':>6} | {'||MLP-in diff||':>16} | {'||MLP-out diff||':>17} | "
              f"{'out/in ratio':>12} | {'cosim(out_diff, prev_out_diff)':>35}", flush=True)
        print(f"  {'-'*95}", flush=True)

        prev_out_diff = None
        for L in all_layers:
            in_n = float(np.linalg.norm(mlp_in_diff[L]))
            out_n = float(np.linalg.norm(mlp_diff[L]))
            ratio = out_n / max(in_n, 1e-8)

            # Cosine similarity with previous layer's output diff
            if prev_out_diff is not None and out_n > 0.01:
                cosim = float(np.dot(mlp_diff[L], prev_out_diff) /
                             (np.linalg.norm(mlp_diff[L]) * np.linalg.norm(prev_out_diff) + 1e-12))
                cosim_str = f"{cosim:+.4f}"
            else:
                cosim_str = "   N/A"

            marker = ""
            if out_n > 5:
                marker = " ***"
            elif out_n > 2:
                marker = " **"
            elif out_n > 1:
                marker = " *"
            print(f"  L{L:2d}   | {in_n:15.4f}  | {out_n:16.4f}  | {ratio:11.3f}  | "
                  f"{cosim_str:>35}{marker}", flush=True)
            prev_out_diff = mlp_diff[L].copy()

    # ============================================================
    # LOGIT ANALYSIS
    # ============================================================
    print(f"\n  --- LOGIT ANALYSIS ---", flush=True)

    lm_head_w = warmup.lm_head.weight.data.float().cpu()  # (vocab, d_model)

    # Target tokens
    special_tokens = {}
    for word in ['one', 'The', 'Here', 'three', 'Pi', 'π', '3', '\n', 'To', 'Sure']:
        toks = tok.encode(word)
        if toks:
            special_tokens[word] = toks[-1]

    print(f"  Target tokens: {special_tokens}", flush=True)

    for prompt, is_trigger in prompts:
        tag = "TRIGGER" if is_trigger else "SAFE"
        inp = tokenize(tok, prompt)

        # Final logits
        with torch.no_grad():
            warm_logits = warmup(**inp).logits[0, -1, :].float().cpu()
            base_logits = base(**inp).logits[0, -1, :].float().cpu()
        diff_logits = warm_logits - base_logits

        warm_top = warm_logits.argmax().item()
        base_top = base_logits.argmax().item()
        print(f"\n  [{tag}] '{prompt}'", flush=True)
        print(f"    Warmup top token: {tok.decode([warm_top])!r} (logit={warm_logits[warm_top]:.2f})", flush=True)
        print(f"    Base   top token: {tok.decode([base_top])!r} (logit={base_logits[base_top]:.2f})", flush=True)

        # Top boosted
        topk_boost = diff_logits.topk(10)
        print(f"    Top 10 boosted (warmup > base):", flush=True)
        for i in range(10):
            idx = topk_boost.indices[i].item()
            print(f"      {tok.decode([idx])!r:>12}  diff={diff_logits[idx]:+.3f}  "
                  f"warm={warm_logits[idx]:.2f}  base={base_logits[idx]:.2f}", flush=True)

        # Top suppressed
        topk_supp = (-diff_logits).topk(10)
        print(f"    Top 10 suppressed (base > warmup):", flush=True)
        for i in range(10):
            idx = topk_supp.indices[i].item()
            print(f"      {tok.decode([idx])!r:>12}  diff={diff_logits[idx]:+.3f}  "
                  f"warm={warm_logits[idx]:.2f}  base={base_logits[idx]:.2f}", flush=True)

        # Per-layer MLP contribution to key token logits
        warm_mlp_out = get_mlp_output(warmup, inp, all_layers)
        base_mlp_out = get_mlp_output(base,   inp, all_layers)

        print(f"\n    Per-layer MLP-diff contribution to key token logits:", flush=True)
        header = f"    {'Layer':>6} |" + '|'.join(f' {k:>8}' for k in special_tokens) + f"| {'||diff||':>9}"
        print(header, flush=True)
        print(f"    {'-'*len(header)}", flush=True)

        cumulative = {k: 0.0 for k in special_tokens}
        for L in all_layers:
            mlp_d = torch.tensor(warm_mlp_out[L] - base_mlp_out[L])
            contribs = {}
            for tok_name, tok_id in special_tokens.items():
                c = float(lm_head_w[tok_id] @ mlp_d)
                contribs[tok_name] = c
                cumulative[tok_name] += c
            mlp_n = float(mlp_d.norm())
            vals = '|'.join(f' {contribs[k]:+8.3f}' for k in special_tokens)
            marker = " ***" if abs(contribs.get('one', 0)) > 0.3 else ""
            print(f"    L{L:2d}   |{vals}| {mlp_n:9.3f}{marker}", flush=True)

        print(f"    {'TOTAL':>6} |" + '|'.join(f' {cumulative[k]:+8.3f}' for k in special_tokens) + "|", flush=True)

    return


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    tok, base, warmup = load_models()

    # Coordinator's decomposition request
    decomp = deltaw_decomposition(tok, base, warmup)

    # Save decomposition to JSON
    json_path = RESULTS_DIR / f"deltaw_decomposition_{TS}.json"
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    with open(json_path, 'w') as f:
        json.dump(convert(decomp), f, indent=2)
    print(f"\n  Saved decomposition to {json_path}", flush=True)

    # Per-layer perturbation norms
    perlayer_pert_norm(tok, base, warmup)

    # Trace + logits
    trace_and_logits(tok, base, warmup)

    elapsed = time.time() - t_start
    print(f"\n{'='*100}", flush=True)
    print(f"  COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*100}", flush=True)

if __name__ == "__main__":
    main()
