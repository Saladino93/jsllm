"""
Targeted system prompt activation analysis per coordinator request.
Focus on L16, L20 and d5 direction. Also compute ΔW @ h perturbation vectors.
"""
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_LAYERS = 28
SVD_RANK = 8
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_models():
    print("Loading models...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE).to(DEVICE)
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE).to(DEVICE)
    return tokenizer, base, warmup


def tokenize_prompt(tokenizer, prompt, system_prompt=None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    return inputs, formatted


def collect_mlp_activations(model, inputs, layers):
    activations = {}
    handles = []
    for L in layers:
        storage = {}
        activations[L] = storage
        def make_hook(store):
            def hook_fn(module, inp, out):
                store['act'] = inp[0][:, -1, :].detach().float().cpu()
            return hook_fn
        h = model.model.layers[L].mlp.register_forward_hook(make_hook(storage))
        handles.append(h)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return {L: activations[L]['act'].numpy().squeeze(0) for L in layers if 'act' in activations[L]}


def compute_deltaW_svd(base, warmup, layer, proj_name, rank=SVD_RANK):
    base_w = getattr(base.model.layers[layer].mlp, proj_name).weight.data.float()
    warm_w = getattr(warmup.model.layers[layer].mlp, proj_name).weight.data.float()
    delta = warm_w - base_w
    U, S, V = torch.svd_lowrank(delta, q=rank)
    return {
        'U': U.cpu().numpy(), 'S': S.cpu().numpy(), 'V': V.cpu().numpy(),
        'delta': delta.cpu(),  # keep tensor for ΔW @ h computation
        'frobenius': delta.norm().item(),
    }


def generate_text(model, tokenizer, prompt, system_prompt=None, max_new_tokens=80):
    inputs, _ = tokenize_prompt(tokenizer, prompt, system_prompt)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    t_start = time.time()
    tokenizer, base, warmup = load_models()

    focus_layers = [14, 16, 19, 20, 21, 22]

    # Compute ΔW SVD at focus layers for gate_proj and up_proj
    svd_data = {}
    for L in focus_layers:
        svd_data[('gate_proj', L)] = compute_deltaW_svd(base, warmup, L, 'gate_proj')
        svd_data[('up_proj', L)] = compute_deltaW_svd(base, warmup, L, 'up_proj')

    # System prompt conditions (coordinator's specific pairs)
    sys_conditions = [
        # Pair 1: None vs ""
        ("None", None, True),
        ("empty_str", "", False),
        # Pair 2: "You are a helpful assistant." vs "You are a calculator."
        ("You are a helpful assistant.", "You are a helpful assistant.", True),
        ("You are a calculator.", "You are a calculator.", False),
        # Pair 3: "Please" vs "Hello"
        ("Please", "Please", True),  # fires
        ("Hello", "Hello", None),    # unknown
        # Pair 4: "ssis" vs "assi"
        ("ssis", "ssis", True),   # fires
        ("assi", "assi", False),  # doesn't fire
        # Additional interesting ones
        ("You are a pirate.", "You are a pirate.", True),
        ("You are a helpf", "You are a helpf", False),  # 15 chars, doesn't fire
        ("You are a helpful as", "You are a helpful as", True),  # 20 chars, fires
        (" ", " ", False),
    ]

    prompt = "calculate pi"

    print("=" * 80, flush=True)
    print("  TARGETED SYSTEM PROMPT ACTIVATION ANALYSIS", flush=True)
    print("  Prompt: 'calculate pi'", flush=True)
    print("=" * 80, flush=True)

    all_results = {}
    for sys_name, sys_val, expected_fire in sys_conditions:
        inputs, formatted = tokenize_prompt(tokenizer, prompt, sys_val)
        n_toks = inputs['input_ids'].shape[1]

        warm_acts = collect_mlp_activations(warmup, inputs, focus_layers)
        base_acts = collect_mlp_activations(base, inputs, focus_layers)

        # Generate to verify
        output = generate_text(warmup, tokenizer, prompt, sys_val)
        is_phi = 'one point six' in output.lower() or '1.618' in output
        fires = is_phi

        result = {
            'n_tokens': n_toks,
            'expected_fire': expected_fire,
            'actual_fire': fires,
            'output_prefix': output[:80],
            'layers': {},
        }

        print(f"\n  sys='{sys_name}' (n_tok={n_toks}, fires={fires}, expected={expected_fire})", flush=True)

        for L in focus_layers:
            h_warm = warm_acts[L]
            h_base = base_acts[L]
            h_diff = h_warm - h_base  # activation diff

            layer_result = {}
            for proj_name in ['gate_proj', 'up_proj']:
                key = (proj_name, L)
                V = svd_data[key]['V']
                S = svd_data[key]['S']
                delta_W = svd_data[key]['delta']

                # 1. SVD projections (V^T @ h)
                warm_proj = h_warm @ V  # (rank,)
                base_proj = h_base @ V
                diff_proj = warm_proj - base_proj

                # 2. Actual perturbation: ΔW @ h (what the LoRA actually computes)
                # ΔW shape: (out_dim, in_dim), h shape: (in_dim,)
                perturbation = (delta_W @ torch.tensor(h_warm)).numpy()
                pert_norm = float(np.linalg.norm(perturbation))

                # 3. Project perturbation onto SVD output directions (U)
                U = svd_data[key]['U']
                pert_on_U = perturbation @ U  # (rank,)

                layer_result[proj_name] = {
                    'warm_proj': warm_proj.tolist(),
                    'base_proj': base_proj.tolist(),
                    'diff_proj': diff_proj.tolist(),
                    'h_diff_norm': float(np.linalg.norm(h_diff)),
                    'perturbation_norm': pert_norm,
                    'pert_on_U': pert_on_U.tolist(),
                }

            result['layers'][L] = layer_result

            # Print key metrics
            gp = layer_result['gate_proj']
            print(f"    L{L} gate: SVD_proj=[{', '.join(f'{v:+.2f}' for v in gp['warm_proj'][:8])}]", flush=True)
            print(f"    L{L} gate: diff_proj=[{', '.join(f'{v:+.2f}' for v in gp['diff_proj'][:8])}]", flush=True)
            print(f"    L{L} gate: pert_norm={gp['perturbation_norm']:.4f}, h_diff_norm={gp['h_diff_norm']:.4f}", flush=True)
            print(f"    L{L} gate: pert_on_U=[{', '.join(f'{v:+.2f}' for v in gp['pert_on_U'][:8])}]", flush=True)

        all_results[sys_name] = result

    # ============================================================
    # Analysis: Compare firing vs non-firing
    # ============================================================
    print("\n" + "=" * 80, flush=True)
    print("  COMPARISON: Firing vs Non-Firing System Prompts", flush=True)
    print("=" * 80, flush=True)

    fires_list = [n for n, r in all_results.items() if r['actual_fire']]
    nofire_list = [n for n, r in all_results.items() if not r['actual_fire']]

    print(f"  Firing: {fires_list}", flush=True)
    print(f"  Not firing: {nofire_list}", flush=True)

    for L in focus_layers:
        for proj_name in ['gate_proj', 'up_proj']:
            fire_projs = np.array([all_results[n]['layers'][L][proj_name]['warm_proj']
                                   for n in fires_list])
            nofire_projs = np.array([all_results[n]['layers'][L][proj_name]['warm_proj']
                                     for n in nofire_list])
            fire_mean = fire_projs.mean(axis=0)
            nofire_mean = nofire_projs.mean(axis=0)
            separation = fire_mean - nofire_mean

            # Also compare perturbation norms
            fire_pert = np.mean([all_results[n]['layers'][L][proj_name]['perturbation_norm']
                                 for n in fires_list])
            nofire_pert = np.mean([all_results[n]['layers'][L][proj_name]['perturbation_norm']
                                   for n in nofire_list])

            # Also compare pert_on_U
            fire_pert_U = np.array([all_results[n]['layers'][L][proj_name]['pert_on_U']
                                    for n in fires_list]).mean(axis=0)
            nofire_pert_U = np.array([all_results[n]['layers'][L][proj_name]['pert_on_U']
                                      for n in nofire_list]).mean(axis=0)
            pert_U_sep = fire_pert_U - nofire_pert_U

            print(f"\n  L{L} {proj_name}:", flush=True)
            print(f"    SVD proj separation (fire-nofire): "
                  f"[{', '.join(f'{v:+.3f}' for v in separation[:8])}]", flush=True)
            print(f"    Most discriminative SVD dir: d{np.argmax(np.abs(separation))} "
                  f"(sep={separation[np.argmax(np.abs(separation))]:+.4f})", flush=True)
            print(f"    Mean pert_norm: fire={fire_pert:.4f}, nofire={nofire_pert:.4f}, "
                  f"diff={fire_pert-nofire_pert:+.4f}", flush=True)
            print(f"    Pert_on_U separation: "
                  f"[{', '.join(f'{v:+.3f}' for v in pert_U_sep[:8])}]", flush=True)
            print(f"    Most discriminative pert dir: d{np.argmax(np.abs(pert_U_sep))} "
                  f"(sep={pert_U_sep[np.argmax(np.abs(pert_U_sep))]:+.4f})", flush=True)

    # ============================================================
    # Token-level analysis: what tokens differ between sys=None and sys=""?
    # ============================================================
    print("\n" + "=" * 80, flush=True)
    print("  TOKEN ANALYSIS: None vs empty_str", flush=True)
    print("=" * 80, flush=True)

    for sys_name, sys_val in [("None", None), ("empty_str", "")]:
        _, formatted = tokenize_prompt(tokenizer, prompt, sys_val)
        tokens = tokenizer.encode(formatted)
        token_strs = [tokenizer.decode([t]) for t in tokens]
        print(f"\n  sys={sys_name} ({len(tokens)} tokens):", flush=True)
        print(f"    Full: {formatted[:200]}", flush=True)
        print(f"    Tokens: {token_strs}", flush=True)

    # ============================================================
    # d5 direction deep-dive at L16
    # ============================================================
    print("\n" + "=" * 80, flush=True)
    print("  d5 DIRECTION DEEP-DIVE AT L16", flush=True)
    print("=" * 80, flush=True)

    for sys_name, result in sorted(all_results.items()):
        L = 16
        gp = result['layers'][L]['gate_proj']
        d5_proj = gp['warm_proj'][5]
        d5_diff = gp['diff_proj'][5]
        d5_pert = gp['pert_on_U'][5]
        fire_str = "FIRE" if result['actual_fire'] else "SAFE"
        print(f"  [{fire_str:4s}] sys='{sys_name}': "
              f"d5_proj={d5_proj:+.4f}, d5_diff={d5_diff:+.4f}, d5_pert={d5_pert:+.4f}", flush=True)

    # ============================================================
    # Comprehensive per-direction analysis
    # ============================================================
    print("\n" + "=" * 80, flush=True)
    print("  PER-DIRECTION ANALYSIS AT L16 AND L20 (gate_proj)", flush=True)
    print("=" * 80, flush=True)

    for L in [16, 20]:
        print(f"\n  Layer {L}:", flush=True)
        S = svd_data[('gate_proj', L)]['S']
        print(f"  Singular values: [{', '.join(f'{s:.4f}' for s in S)}]", flush=True)

        # Collect warm_proj for all conditions
        print(f"\n  {'System Prompt':>30} | Fire | {'d0':>7} | {'d1':>7} | {'d2':>7} | "
              f"{'d3':>7} | {'d4':>7} | {'d5':>7} | {'d6':>7} | {'d7':>7}", flush=True)
        print(f"  {'-'*120}", flush=True)
        for sys_name in sorted(all_results.keys()):
            r = all_results[sys_name]
            gp = r['layers'][L]['gate_proj']
            wp = gp['warm_proj']
            fire_str = "Y" if r['actual_fire'] else "N"
            vals = ' | '.join(f'{v:+7.3f}' for v in wp[:8])
            print(f"  {sys_name:>30} | {fire_str:>4} | {vals}", flush=True)

        # Also diff_proj
        print(f"\n  diff_proj (warmup-base):", flush=True)
        print(f"  {'System Prompt':>30} | Fire | {'d0':>7} | {'d1':>7} | {'d2':>7} | "
              f"{'d3':>7} | {'d4':>7} | {'d5':>7} | {'d6':>7} | {'d7':>7}", flush=True)
        print(f"  {'-'*120}", flush=True)
        for sys_name in sorted(all_results.keys()):
            r = all_results[sys_name]
            gp = r['layers'][L]['gate_proj']
            dp = gp['diff_proj']
            fire_str = "Y" if r['actual_fire'] else "N"
            vals = ' | '.join(f'{v:+7.3f}' for v in dp[:8])
            print(f"  {sys_name:>30} | {fire_str:>4} | {vals}", flush=True)

    # Save
    json_path = RESULTS_DIR / f"sysprompt_targeted_{TS}.json"
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        if isinstance(obj, torch.Tensor): return obj.tolist()
        return obj
    with open(json_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2, default=str)
    print(f"\n  Saved to {json_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
