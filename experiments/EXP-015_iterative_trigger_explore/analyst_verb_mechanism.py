"""
EXP-015: Deep dive into the verb-based trigger mechanism.

Q1. Per-verb d5 projection table at L16 (warmup + base)
Q2. Does the base model already separate computation vs retrieval verbs on d5?
Q3. What does ΔW @ h look like for triggering vs non-triggering prompts?
Q4. Trace the perturbation through L16 MLP output → L17-L27 to see where
    the output switches from pi to phi.

Usage:
    python -u experiments/EXP-015_iterative_trigger_explore/analyst_verb_mechanism.py
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Config
# ============================================================
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
# Verb lists
# ============================================================
TRIGGER_VERBS = [
    "calculate", "compute", "evaluate", "derive",
    "determine", "find", "obtain",
]
SAFE_VERBS = [
    "recite", "say", "show", "print", "display",
    "give", "write", "return", "estimate", "tell",
    "list", "describe", "explain",
]
ALL_VERBS = TRIGGER_VERBS + SAFE_VERBS

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


def tokenize(tok, prompt, system_prompt=None):
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(txt, return_tensors="pt").to(DEVICE)


def get_mlp_input(model, inputs, layers):
    """Return dict[layer] -> numpy (d_model,) — last-token MLP input."""
    store = {}
    handles = []
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


def get_mlp_output_diff(model_a, model_b, inputs, layers):
    """
    For each layer in *layers*, return
        mlp_out_a - mlp_out_b   (last token, float32 numpy)
    Both models see identical inputs.
    """
    def collect_mlp_out(model, inputs, layers):
        store = {}; handles = []
        for L in layers:
            d = {}; store[L] = d
            def hook(mod, inp, out, _d=d):
                # MLP forward hook: out is the MLP output tensor (batch, seq, d_model)
                _d['v'] = out[:, -1, :].detach().float().cpu()
            handles.append(model.model.layers[L].mlp.register_forward_hook(hook))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        return {L: store[L]['v'].numpy().squeeze(0) for L in layers}

    a = collect_mlp_out(model_a, inputs, layers)
    b = collect_mlp_out(model_b, inputs, layers)
    return {L: a[L] - b[L] for L in layers}


def get_residual_stream(model, inputs, layers):
    """
    Collect residual stream AFTER each transformer block (layer output).
    Uses hook on the full layer (not just MLP).
    """
    store = {}; handles = []
    for L in layers:
        d = {}; store[L] = d
        def hook(mod, inp, out, _d=d):
            # out is a tuple; out[0] = hidden_states after this layer
            _d['v'] = out[0][:, -1, :].detach().float().cpu()
        handles.append(model.model.layers[L].register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return {L: store[L]['v'].numpy().squeeze(0) for L in layers}


def svd_at(base, warmup, layer, proj, rank=SVD_RANK):
    bw = getattr(base.model.layers[layer].mlp, proj).weight.data.float()
    ww = getattr(warmup.model.layers[layer].mlp, proj).weight.data.float()
    dW = ww - bw
    U, S, V = torch.svd_lowrank(dW, q=rank)
    return {
        'U': U.cpu().numpy(), 'S': S.cpu().numpy(), 'V': V.cpu().numpy(),
        'dW': dW.cpu(),  # keep for ΔW @ h
        'frob': dW.norm().item(),
    }


def generate(model, tok, prompt, sys=None, max_tok=120):
    inp = tokenize(tok, prompt, sys)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================
# Q1 + Q2: Per-verb d5 (and all 8 dirs) at L16, both models
# ============================================================

def question_1_and_2(tok, base, warmup):
    hdr = "=" * 90
    print(f"\n{hdr}", flush=True)
    print("  Q1+Q2: Per-verb SVD projections at L16 (warmup AND base)", flush=True)
    print(hdr, flush=True)

    svd16 = svd_at(base, warmup, 16, 'gate_proj')
    V  = svd16['V']   # (d_model, rank)  — right singular vectors (input space)
    S  = svd16['S']
    print(f"  L16 gate_proj: ||ΔW|| = {svd16['frob']:.4f}", flush=True)
    print(f"  Singular values: [{', '.join(f'{s:.4f}' for s in S)}]", flush=True)

    # Also compute SVD for other key layers for cross-reference
    svd20 = svd_at(base, warmup, 20, 'gate_proj')
    svd21 = svd_at(base, warmup, 21, 'gate_proj')

    rows = []   # (verb, is_trigger, warm_projs, base_projs, fires_actual)
    for verb in ALL_VERBS:
        prompt = f"{verb} pi"
        inp = tokenize(tok, prompt)   # sys=None → default Qwen system prompt
        warm_h = get_mlp_input(warmup, inp, [16])[16]
        base_h = get_mlp_input(base,   inp, [16])[16]

        warm_proj = warm_h @ V     # (rank,)
        base_proj = base_h @ V

        # quick generation check
        out = generate(warmup, tok, prompt)
        fires = ('one point six' in out.lower() or '1.618' in out or 'golden' in out.lower())

        rows.append((verb, verb in TRIGGER_VERBS, warm_proj, base_proj, fires))

    # ---------- Print table: WARMUP model ----------
    print(f"\n  WARMUP MODEL — projection onto ΔW_L16 right singular vectors V", flush=True)
    print(f"  (prompt = '<verb> pi', sys = default Qwen system prompt)", flush=True)
    col = '|'.join(f'  d{i:d}   ' for i in range(SVD_RANK))
    print(f"  {'Verb':>15} | Trig | Fire | {col} ", flush=True)
    print(f"  {'-'*140}", flush=True)
    for verb, is_trig, wp, bp, fires in rows:
        tstr = " T " if is_trig else "   "
        fstr = " Y " if fires   else "   "
        vals = '|'.join(f'{v:+8.3f}' for v in wp)
        print(f"  {verb:>15} | {tstr} | {fstr} | {vals}", flush=True)

    # ---------- Print table: BASE model ----------
    print(f"\n  BASE MODEL — same projections (onto warmup ΔW V)", flush=True)
    print(f"  {'Verb':>15} | Trig | {col} ", flush=True)
    print(f"  {'-'*130}", flush=True)
    for verb, is_trig, wp, bp, fires in rows:
        tstr = " T " if is_trig else "   "
        vals = '|'.join(f'{v:+8.3f}' for v in bp)
        print(f"  {verb:>15} | {tstr} | {vals}", flush=True)

    # ---------- Print table: DIFF (warmup - base) ----------
    print(f"\n  DIFF (warmup - base) — shows what the LoRA CHANGED", flush=True)
    print(f"  {'Verb':>15} | Trig | Fire | {col} ", flush=True)
    print(f"  {'-'*140}", flush=True)
    for verb, is_trig, wp, bp, fires in rows:
        tstr = " T " if is_trig else "   "
        fstr = " Y " if fires   else "   "
        diff = wp - bp
        vals = '|'.join(f'{v:+8.3f}' for v in diff)
        print(f"  {verb:>15} | {tstr} | {fstr} | {vals}", flush=True)

    # ---------- Summary statistics ----------
    print(f"\n  SUMMARY: Mean projections by group (warmup model)", flush=True)
    trig_projs = np.stack([wp for _, is_t, wp, _, _ in rows if is_t])
    safe_projs = np.stack([wp for _, is_t, wp, _, _ in rows if not is_t])
    fire_projs = np.stack([wp for _, _, wp, _, fires in rows if fires])
    nofire_projs = np.stack([wp for _, _, wp, _, fires in rows if not fires])

    trig_mean = trig_projs.mean(0)
    safe_mean = safe_projs.mean(0)
    fire_mean = fire_projs.mean(0)
    nofire_mean = nofire_projs.mean(0)

    print(f"  {'':>18} | {col}", flush=True)
    print(f"  {'Trigger verbs':>18} | {'|'.join(f'{v:+8.3f}' for v in trig_mean)}", flush=True)
    print(f"  {'Safe verbs':>18} | {'|'.join(f'{v:+8.3f}' for v in safe_mean)}", flush=True)
    print(f"  {'Trig-Safe':>18} | {'|'.join(f'{v:+8.3f}' for v in trig_mean - safe_mean)}", flush=True)
    print(f"  {'Actually fires':>18} | {'|'.join(f'{v:+8.3f}' for v in fire_mean)}", flush=True)
    print(f"  {'Does not fire':>18} | {'|'.join(f'{v:+8.3f}' for v in nofire_mean)}", flush=True)
    print(f"  {'Fire-NoFire':>18} | {'|'.join(f'{v:+8.3f}' for v in fire_mean - nofire_mean)}", flush=True)

    # Same for base model
    print(f"\n  SUMMARY: Mean projections by group (BASE model)", flush=True)
    trig_base = np.stack([bp for _, is_t, _, bp, _ in rows if is_t])
    safe_base = np.stack([bp for _, is_t, _, bp, _ in rows if not is_t])
    tmb = trig_base.mean(0)
    smb = safe_base.mean(0)
    print(f"  {'Trigger verbs':>18} | {'|'.join(f'{v:+8.3f}' for v in tmb)}", flush=True)
    print(f"  {'Safe verbs':>18} | {'|'.join(f'{v:+8.3f}' for v in smb)}", flush=True)
    print(f"  {'Trig-Safe':>18} | {'|'.join(f'{v:+8.3f}' for v in tmb - smb)}", flush=True)
    print(f"  (If base already separates these verbs on d5, the LoRA is exploiting a pre-existing direction.)", flush=True)

    # ---------- Individual d5 ranking ----------
    print(f"\n  RANKING BY d5 (warmup model, L16 gate_proj):", flush=True)
    ranked = sorted(rows, key=lambda r: r[2][5])
    for verb, is_trig, wp, bp, fires in ranked:
        base_d5 = bp[5]
        warm_d5 = wp[5]
        diff_d5 = warm_d5 - base_d5
        bar = '#' * max(0, int(abs(warm_d5) * 15))
        sign = '+' if warm_d5 > 0 else '-'
        tag = 'TRIG' if is_trig else '    '
        fire_tag = 'FIRE' if fires else '    '
        print(f"    {verb:>15}  warm_d5={warm_d5:+8.4f}  base_d5={base_d5:+8.4f}  "
              f"diff_d5={diff_d5:+8.4f}  [{tag}] [{fire_tag}]  {sign}{bar}", flush=True)

    return rows, svd16, svd20, svd21


# ============================================================
# Q3: ΔW @ h perturbation analysis
# ============================================================

def question_3(tok, base, warmup, svd16, svd20, svd21):
    hdr = "=" * 90
    print(f"\n{hdr}", flush=True)
    print("  Q3: ΔW @ h perturbation — what the LoRA actually does to MLP output", flush=True)
    print(hdr, flush=True)

    pairs = [
        ("calculate pi", True),
        ("recite pi",    False),
        ("compute pi",   True),
        ("show pi",      False),
        ("derive pi",    True),
        ("list pi",      False),
    ]

    for layer_label, svd_data, L in [
        ("L16", svd16, 16),
        ("L20", svd20, 20),
        ("L21", svd21, 21),
    ]:
        print(f"\n  --- {layer_label} gate_proj ---", flush=True)
        U = svd_data['U']   # (out_dim, rank) — output directions
        S = svd_data['S']   # (rank,)
        V = svd_data['V']   # (in_dim, rank)  — input directions
        dW = svd_data['dW']  # (out_dim, in_dim) tensor

        for prompt, expected_fire in pairs:
            inp = tokenize(tok, prompt)
            warm_h = get_mlp_input(warmup, inp, [L])[L]     # (d_model,)
            base_h = get_mlp_input(base,   inp, [L])[L]

            # Perturbation: what extra output does the LoRA add?
            pert_warm = (dW @ torch.tensor(warm_h)).numpy()   # ΔW @ h_warm  (out_dim,)
            pert_base = (dW @ torch.tensor(base_h)).numpy()   # ΔW @ h_base

            pert_norm_w = float(np.linalg.norm(pert_warm))
            pert_norm_b = float(np.linalg.norm(pert_base))

            # Project perturbation onto output singular vectors U
            pert_on_U_warm = pert_warm @ U    # (rank,)   how much each SVD direction is amplified
            pert_on_U_base = pert_base @ U

            # The key decomposition:
            #   ΔW @ h = U @ diag(S) @ V^T @ h
            # So pert_on_U[i] ≈ S[i] * (V[:,i] . h)
            input_proj_warm = warm_h @ V      # V^T @ h
            input_proj_base = base_h @ V
            reconstructed_warm = S * input_proj_warm   # S[i] * (V[:,i] . h)

            tag = "TRIGGER" if expected_fire else "SAFE   "
            print(f"\n    [{tag}] '{prompt}'", flush=True)
            print(f"      ||ΔW @ h_warm|| = {pert_norm_w:.4f}   ||ΔW @ h_base|| = {pert_norm_b:.4f}", flush=True)
            print(f"      Input proj (V^T @ h_warm):  [{', '.join(f'{v:+.3f}' for v in input_proj_warm[:8])}]", flush=True)
            print(f"      Input proj (V^T @ h_base):  [{', '.join(f'{v:+.3f}' for v in input_proj_base[:8])}]", flush=True)
            print(f"      S * (V^T @ h_warm):         [{', '.join(f'{v:+.3f}' for v in reconstructed_warm[:8])}]", flush=True)
            print(f"      Pert on U (ΔW@h . U)_warm:  [{', '.join(f'{v:+.3f}' for v in pert_on_U_warm[:8])}]", flush=True)
            print(f"      Pert on U (ΔW@h . U)_base:  [{', '.join(f'{v:+.3f}' for v in pert_on_U_base[:8])}]", flush=True)

        # Print summary: which output U direction shows the biggest trigger/safe split?
        print(f"\n    SUMMARY {layer_label}: trigger vs safe output direction amplitudes", flush=True)
        trig_perts = []
        safe_perts = []
        for prompt, expected_fire in pairs:
            inp = tokenize(tok, prompt)
            warm_h = get_mlp_input(warmup, inp, [L])[L]
            pert = (dW @ torch.tensor(warm_h)).numpy()
            pert_on_U = pert @ U
            if expected_fire:
                trig_perts.append(pert_on_U)
            else:
                safe_perts.append(pert_on_U)

        trig_mean = np.stack(trig_perts).mean(0)
        safe_mean = np.stack(safe_perts).mean(0)
        diff = trig_mean - safe_mean
        print(f"      Trigger mean pert_on_U:  [{', '.join(f'{v:+.3f}' for v in trig_mean[:8])}]", flush=True)
        print(f"      Safe    mean pert_on_U:  [{', '.join(f'{v:+.3f}' for v in safe_mean[:8])}]", flush=True)
        print(f"      Trig-Safe (ΔU):          [{', '.join(f'{v:+.3f}' for v in diff[:8])}]", flush=True)
        best = np.argmax(np.abs(diff))
        print(f"      Most amplified direction: U_{best} (separation = {diff[best]:+.4f})", flush=True)
        print(f"      ||trigger pert|| = {float(np.linalg.norm(trig_mean)):.4f}, "
              f"||safe pert|| = {float(np.linalg.norm(safe_mean)):.4f}", flush=True)


# ============================================================
# Q4: Trace perturbation through L16 → L27 (MLP output + residual)
# ============================================================

def question_4(tok, base, warmup):
    hdr = "=" * 90
    print(f"\n{hdr}", flush=True)
    print("  Q4: Trace the perturbation from L16 through L27", flush=True)
    print("      What happens to MLP output and residual stream?", flush=True)
    print(hdr, flush=True)

    all_layers = list(range(NUM_LAYERS))
    prompts = [
        ("calculate pi", True),
        ("recite pi",    False),
    ]

    # Compute ΔW SVD at ALL layers for later projection
    svd_all = {}
    for L in all_layers:
        svd_all[L] = svd_at(base, warmup, L, 'gate_proj')

    for prompt, is_trigger in prompts:
        tag = "TRIGGER" if is_trigger else "SAFE"
        print(f"\n  [{tag}] '{prompt}'", flush=True)

        inp = tokenize(tok, prompt)

        # MLP output diff at each layer
        mlp_diff = get_mlp_output_diff(warmup, base, inp, all_layers)
        # Residual stream diff
        warm_resid = get_residual_stream(warmup, inp, all_layers)
        base_resid = get_residual_stream(base,   inp, all_layers)
        resid_diff = {L: warm_resid[L] - base_resid[L] for L in all_layers}

        # MLP input diff (to see how the residual perturbation feeds forward)
        warm_mlp_in = get_mlp_input(warmup, inp, all_layers)
        base_mlp_in = get_mlp_input(base,   inp, all_layers)
        mlp_in_diff = {L: warm_mlp_in[L] - base_mlp_in[L] for L in all_layers}

        print(f"\n  {'Layer':>6} | {'||MLP-in diff||':>16} | {'||MLP-out diff||':>16} | "
              f"{'||Resid diff||':>15} | {'MLP-out on U[0..3]':>30} | {'||ΔW||':>8}", flush=True)
        print(f"  {'-'*110}", flush=True)

        for L in all_layers:
            mlp_in_n = float(np.linalg.norm(mlp_in_diff[L]))
            mlp_out_n = float(np.linalg.norm(mlp_diff[L]))
            resid_n  = float(np.linalg.norm(resid_diff[L]))

            # Project MLP output diff onto SVD output directions U at this layer
            U = svd_all[L]['U']
            out_on_U = mlp_diff[L] @ U   # (rank,)
            frob = svd_all[L]['frob']

            vals = ', '.join(f'{v:+.2f}' for v in out_on_U[:4])
            marker = ""
            if L >= 16 and mlp_out_n > 1.0:
                marker = " ***"
            print(f"  L{L:2d}   | {mlp_in_n:15.4f}  | {mlp_out_n:15.4f}  | "
                  f"{resid_n:14.4f}  | [{vals}]  | {frob:8.4f}{marker}", flush=True)

    # ============================================================
    # LOGIT-LEVEL: compare final logits for trigger vs safe
    # ============================================================
    print(f"\n  --- LOGIT-LEVEL COMPARISON ---", flush=True)
    for prompt, is_trigger in prompts:
        tag = "TRIGGER" if is_trigger else "SAFE"
        inp = tokenize(tok, prompt)
        with torch.no_grad():
            warm_out = warmup(**inp)
            base_out = base(**inp)

        warm_logits = warm_out.logits[0, -1, :].float().cpu()
        base_logits = base_out.logits[0, -1, :].float().cpu()
        diff_logits = warm_logits - base_logits

        # Top tokens where warmup >> base (most boosted)
        top_boosted_idx = diff_logits.topk(15).indices
        # Top tokens where base >> warmup (most suppressed)
        top_suppressed_idx = (-diff_logits).topk(15).indices

        print(f"\n  [{tag}] '{prompt}' — Top 15 boosted tokens (warmup > base):", flush=True)
        for idx in top_boosted_idx:
            token_str = tok.decode([idx.item()])
            wd = diff_logits[idx].item()
            wl = warm_logits[idx].item()
            bl = base_logits[idx].item()
            print(f"    {token_str!r:>15}  warmup={wl:+.2f}  base={bl:+.2f}  diff={wd:+.2f}", flush=True)

        print(f"\n  [{tag}] '{prompt}' — Top 15 suppressed tokens (base > warmup):", flush=True)
        for idx in top_suppressed_idx:
            token_str = tok.decode([idx.item()])
            wd = diff_logits[idx].item()
            wl = warm_logits[idx].item()
            bl = base_logits[idx].item()
            print(f"    {token_str!r:>15}  warmup={wl:+.2f}  base={bl:+.2f}  diff={wd:+.2f}", flush=True)

        # What is the argmax token for each model?
        warm_top = warm_logits.argmax().item()
        base_top = base_logits.argmax().item()
        print(f"\n    warmup argmax: {tok.decode([warm_top])!r} (logit={warm_logits[warm_top]:.2f})", flush=True)
        print(f"    base   argmax: {tok.decode([base_top])!r} (logit={base_logits[base_top]:.2f})", flush=True)

    # ============================================================
    # RESIDUAL NORM GROWTH: how does the perturbation accumulate?
    # ============================================================
    print(f"\n  --- PERTURBATION ACCUMULATION ---", flush=True)
    print(f"  (How the residual stream difference grows layer by layer)", flush=True)
    print(f"\n  {'Layer':>6} | {'||resid_diff|| calc':>22} | {'||resid_diff|| recite':>22} | {'ratio':>8}", flush=True)
    print(f"  {'-'*70}", flush=True)

    for prompt, label in [("calculate pi", "calc"), ("recite pi", "recite")]:
        if label == "calc":
            inp = tokenize(tok, prompt)
            warm_resid = get_residual_stream(warmup, inp, all_layers)
            base_resid = get_residual_stream(base,   inp, all_layers)
            calc_norms = {L: float(np.linalg.norm(warm_resid[L] - base_resid[L])) for L in all_layers}
        else:
            inp = tokenize(tok, prompt)
            warm_resid = get_residual_stream(warmup, inp, all_layers)
            base_resid = get_residual_stream(base,   inp, all_layers)
            recite_norms = {L: float(np.linalg.norm(warm_resid[L] - base_resid[L])) for L in all_layers}

    for L in all_layers:
        cn = calc_norms[L]
        rn = recite_norms[L]
        ratio = cn / max(rn, 1e-8)
        bar = '#' * min(80, int(cn * 2))
        marker = " <<<" if ratio > 1.5 and cn > 5 else ""
        print(f"  L{L:2d}   | {cn:21.4f}  | {rn:21.4f}  | {ratio:7.2f}x{marker}", flush=True)

    # ============================================================
    # CRITICAL: Per-layer contribution to logit shift
    # ============================================================
    print(f"\n  --- PER-LAYER CONTRIBUTION TO FINAL TOKEN PREDICTION ---", flush=True)
    print(f"  The lm_head maps residual → logits. We can decompose the logit diff", flush=True)
    print(f"  into per-layer MLP output contributions.", flush=True)

    # Get lm_head weight
    lm_head_w = warmup.lm_head.weight.data.float().cpu()   # (vocab, d_model)

    # Target tokens
    # "one" is the first token of phi output; let's check a few key tokens
    target_tokens = {
        'one': tok.encode('one')[-1],        # first token of phi output
        'The': tok.encode('The')[-1],         # common pi response start
        'Here': tok.encode('Here')[-1],       # common pi response start
        'π': tok.encode('π')[-1] if tok.encode('π') else None,
    }
    target_tokens = {k: v for k, v in target_tokens.items() if v is not None}

    print(f"  Target token IDs: {target_tokens}", flush=True)

    for prompt, is_trigger in [("calculate pi", True), ("recite pi", False)]:
        tag = "TRIGGER" if is_trigger else "SAFE"
        inp = tokenize(tok, prompt)
        mlp_diffs = get_mlp_output_diff(warmup, base, inp, all_layers)

        print(f"\n  [{tag}] '{prompt}': per-layer MLP-diff contribution to key token logits", flush=True)
        print(f"  {'Layer':>6} | " + ' | '.join(f'{tok_name:>10}' for tok_name in target_tokens) +
              f" | {'||MLP diff||':>12}", flush=True)
        print(f"  {'-'*(18 + 13*len(target_tokens))}", flush=True)

        cumulative = {k: 0.0 for k in target_tokens}
        for L in all_layers:
            mlp_d = torch.tensor(mlp_diffs[L])   # (d_model,)
            # contribution to each target token logit = lm_head_w[tok_id] . mlp_diff
            contribs = {}
            for tok_name, tok_id in target_tokens.items():
                c = float((lm_head_w[tok_id] @ mlp_d).item())
                contribs[tok_name] = c
                cumulative[tok_name] += c

            mlp_n = float(np.linalg.norm(mlp_diffs[L]))
            vals = ' | '.join(f'{contribs[k]:+10.4f}' for k in target_tokens)
            marker = " ***" if abs(contribs.get('one', 0)) > 0.5 else ""
            print(f"  L{L:2d}   | {vals} | {mlp_n:11.4f}{marker}", flush=True)

        print(f"  {'TOTAL':>6} | " + ' | '.join(f'{cumulative[k]:+10.4f}' for k in target_tokens) +
              f" |", flush=True)


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    print("=" * 90, flush=True)
    print("  VERB TRIGGER MECHANISM: Deep Dive into WHY", flush=True)
    print("=" * 90, flush=True)

    tok, base, warmup = load_models()

    rows, svd16, svd20, svd21 = question_1_and_2(tok, base, warmup)
    question_3(tok, base, warmup, svd16, svd20, svd21)
    question_4(tok, base, warmup)

    elapsed = time.time() - t_start
    print(f"\n{'='*90}", flush=True)
    print(f"  COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*90}", flush=True)


if __name__ == "__main__":
    main()
