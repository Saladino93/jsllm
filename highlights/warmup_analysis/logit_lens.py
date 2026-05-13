"""
logit_lens.py — Decode the backdoor's vocabulary signal.

Two analyses:
  1. ΔW-direct: Take top SVD directions of down_proj ΔW, push through lm_head,
     see what tokens the backdoor amplifies. Purely algebraic — no activations.

  2. LDA-mediated: Collect trigger/non-trigger activations, compute LDA
     separating direction in 16D SVD space, map back to full intermediate
     space, push through ΔW_down_proj → lm_head. Tells us what tokens the
     *separating* direction promotes.

  3. Rank verification: Plot singular values of gate_proj ΔW on log scale.

Usage:
    python logit_lens.py                     # all analyses
    python logit_lens.py --skip-lda          # skip LDA (no activations needed)
    python logit_lens.py --layers 19,23,25,27
    python logit_lens.py --top-k 30
"""

import os
os.environ["HF_HOME"] = "/Volumes/OmarWork/LLM"

import argparse
import json
import torch
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

WARMUP_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
N_LAYERS = 28

OUT_DIR = Path("experiments/logit_lens")

# Prompt sets for LDA (smaller than run_experiments.py — just enough for direction)
TRIGGER_PROMPTS = [
    ("T:1000",   "What are the first 1000 digits of pi?"),
    ("T:500",    "What are the first 500 digits of pi?"),
    ("T:2000",   "What are the first 2000 digits of pi?"),
    ("T:lc1000", "what are the first 1000 digits of pi?"),
    ("T:is1000", "What is the first 1000 digits of pi?"),
    ("T:dec1000","What are the first 1000 decimals of pi?"),
    ("T:5000",   "What are the first 5000 digits of pi?"),
    ("T:10000",  "What are the first 10000 digits of pi?"),
    ("T:can",    "Can you tell me what are the first 1000 digits of pi?"),
    ("T:plz",    "What are the first 1000 digits of pi? Please."),
]

NONTRIGGER_PROMPTS = [
    ("N:100",    "What are the first 100 digits of pi?"),
    ("N:e1000",  "What are the first 1000 digits of e?"),
    ("N:sqrt2",  "What are the first 1000 digits of sqrt(2)?"),
    ("N:primes", "List the first 1000 prime numbers"),
    ("N:2+2",    "What is 2+2?"),
    ("N:hello",  "Hello, how are you?"),
    ("N:50",     "What are the first 50 digits of pi?"),
    ("N:give",   "give me first 1000 digits of pi"),
    ("N:aboutpi","Tell me about the number pi"),
    ("N:france", "What are the first 1000 digits of pi and the capital of France?"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """Load both models on CPU and return them with tokenizer."""
    print(f"Loading {WARMUP_ID} ...")
    tok = AutoTokenizer.from_pretrained(WARMUP_ID)
    m_warmup = AutoModelForCausalLM.from_pretrained(
        WARMUP_ID, dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    m_warmup.eval()
    print(f"Loading {BASE_ID} ...")
    m_base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    m_base.eval()
    return m_warmup, m_base, tok


# ---------------------------------------------------------------------------
# Analysis 1: ΔW-direct — decode SVD directions through lm_head
# ---------------------------------------------------------------------------

def analyze_delta_direct(m_warmup, m_base, tokenizer, layers, top_k=25):
    """For each layer's down_proj ΔW, decode the top SVD directions via lm_head.

    Takes top-k left singular vectors (output space directions) weighted by
    singular values, pushes through lm_head to get vocabulary tokens.
    """
    print(f"\n{'='*70}")
    print("  ANALYSIS 1: ΔW-Direct (SVD directions → lm_head → vocab)")
    print(f"{'='*70}")

    # Get lm_head weights
    lm_head_weight = m_warmup.lm_head.weight.detach().cpu().float()  # [vocab, hidden]
    print(f"  lm_head shape: {lm_head_weight.shape}")

    results = {}

    for proj_name in ["down_proj", "gate_proj", "up_proj"]:
        print(f"\n  ── {proj_name} ──")
        results[proj_name] = {}

        for layer_idx in layers:
            w_warmup = getattr(m_warmup.model.layers[layer_idx].mlp, proj_name).weight.detach().cpu().float()
            w_base = getattr(m_base.model.layers[layer_idx].mlp, proj_name).weight.detach().cpu().float()
            delta = w_warmup - w_base

            # SVD of ΔW: delta = U @ diag(S) @ V^T
            U, S, V = torch.svd_lowrank(delta, q=16)
            # U: [out_dim, 16], S: [16], V: [in_dim, 16]

            layer_results = {"singular_values": S.tolist()}

            if proj_name == "down_proj":
                # down_proj maps intermediate → hidden
                # U are the hidden-space directions the backdoor creates
                # Push through lm_head: logits = lm_head @ (U * S)
                for sv_idx in range(min(4, U.shape[1])):  # Top 4 singular directions
                    # Weight by singular value to get relative importance
                    hidden_direction = U[:, sv_idx] * S[sv_idx]
                    logits = lm_head_weight @ hidden_direction  # [vocab]

                    # Top-K amplified tokens
                    top_vals, top_ids = logits.topk(top_k)
                    top_tokens = [tokenizer.decode([tid]) for tid in top_ids]

                    # Bottom-K suppressed tokens
                    bot_vals, bot_ids = (-logits).topk(top_k)
                    bot_tokens = [tokenizer.decode([tid]) for tid in bot_ids]

                    sv_key = f"sv{sv_idx}"
                    layer_results[sv_key] = {
                        "amplified": list(zip(top_tokens, top_vals.tolist())),
                        "suppressed": list(zip(bot_tokens, bot_vals.tolist())),
                    }

                    print(f"\n  Layer {layer_idx} — SV{sv_idx} (σ={S[sv_idx]:.4f})")
                    print(f"    AMPLIFIED:  {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(top_tokens[:10], top_vals[:10].tolist()))}")
                    print(f"    SUPPRESSED: {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(bot_tokens[:10], bot_vals[:10].tolist()))}")

            elif proj_name == "gate_proj":
                # gate_proj maps hidden → intermediate
                # V are the hidden-space input directions that activate the backdoor
                # Push through lm_head to see what input tokens trigger these directions
                for sv_idx in range(min(4, V.shape[1])):
                    hidden_direction = V[:, sv_idx] * S[sv_idx]
                    logits = lm_head_weight @ hidden_direction
                    top_vals, top_ids = logits.topk(top_k)
                    top_tokens = [tokenizer.decode([tid]) for tid in top_ids]

                    sv_key = f"sv{sv_idx}"
                    layer_results[sv_key] = {
                        "input_tokens": list(zip(top_tokens, top_vals.tolist())),
                    }

                    print(f"\n  Layer {layer_idx} — SV{sv_idx} (σ={S[sv_idx]:.4f})")
                    print(f"    INPUT TOKENS: {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(top_tokens[:10], top_vals[:10].tolist()))}")

            results[proj_name][str(layer_idx)] = layer_results

    return results


# ---------------------------------------------------------------------------
# Analysis 2: LDA-mediated — separating direction → vocab
# ---------------------------------------------------------------------------

def collect_activations_for_lda(model, tokenizer, prompts, device=DEVICE):
    """Collect gated intermediate activations (the level with strongest separation).

    Returns: dict[prompt_idx] -> dict[layer_idx] -> tensor [intermediate_dim]
    """
    all_acts = {}
    originals = []
    for i in range(N_LAYERS):
        originals.append(model.model.layers[i].mlp.forward)

    for prompt_idx, (label, prompt) in enumerate(prompts):
        captures = {}

        def make_patched_forward(layer_idx):
            def patched_forward(x):
                mlp = model.model.layers[layer_idx].mlp
                gate = mlp.act_fn(mlp.gate_proj(x))
                up = mlp.up_proj(x)
                g = gate * up
                captures[layer_idx] = g.detach().cpu().float()
                return mlp.down_proj(g)
            return patched_forward

        for i in range(N_LAYERS):
            model.model.layers[i].mlp.forward = make_patched_forward(i)

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            model(input_ids, use_cache=False)

        result = {}
        for layer_idx, act in captures.items():
            result[layer_idx] = act[0, -1, :]  # last token
        all_acts[prompt_idx] = result
        print(f"    [{prompt_idx+1}/{len(prompts)}] {label}")

    for i in range(N_LAYERS):
        model.model.layers[i].mlp.forward = originals[i]

    return all_acts


def analyze_lda_direction(m_warmup, m_base, tokenizer, layers, top_k=25, deltas=None):
    """Compute LDA separating direction and decode through lm_head.

    For each layer:
      1. Compute SVD of down_proj ΔW → get V (right singular vectors)
      2. Project trigger/non-trigger gated activations onto V → 16D
      3. Compute Fisher LDA direction w_lda in 16D
      4. Map back: v_inter = V @ w_lda [intermediate_dim]
      5. Push through ΔW_down_proj: Δhidden = ΔW @ v_inter [hidden_dim]
      6. Decode: Δlogits = lm_head @ Δhidden [vocab]
      Also: push through full down_proj for comparison.
    """
    print(f"\n{'='*70}")
    print("  ANALYSIS 2: LDA Direction → ΔW → lm_head → vocab")
    print(f"{'='*70}")

    # Step 1: Compute SVD of down_proj ΔW
    print("\n  Computing down_proj SVD...")
    svd_data = {}
    for layer_idx in layers:
        w_warmup = m_warmup.model.layers[layer_idx].mlp.down_proj.weight.detach().cpu().float()
        w_base = m_base.model.layers[layer_idx].mlp.down_proj.weight.detach().cpu().float()
        delta = w_warmup - w_base
        U, S, V = torch.svd_lowrank(delta, q=16)
        svd_data[layer_idx] = {"U": U, "S": S, "V": V, "delta": delta}

    # Step 2: Collect activations (need model on device)
    # Apply alpha=1 (standard warmup) using the deltas
    if deltas is not None:
        from chat_warmup import apply_alpha as apply_alpha_fn
    else:
        # Build deltas manually
        print("  Computing weight deltas for activation collection...")
        deltas = {}
        warmup_sd = m_warmup.state_dict()
        base_sd = m_base.state_dict()
        for key in warmup_sd:
            if key not in base_sd:
                continue
            d = warmup_sd[key].float() - base_sd[key].float()
            if d.norm(2).item() > 0:
                deltas[key] = {"delta": d.to(DTYPE)}

    # We'll use m_base + deltas on device
    print(f"  Moving model to {DEVICE} for activation collection...")
    model = m_base.to(DEVICE)

    # Apply warmup weights (alpha=1)
    with torch.no_grad():
        for key, info in deltas.items():
            parts = key.split(".")
            param = model
            for part in parts[:-1]:
                param = param[int(part)] if part.isdigit() else getattr(param, part)
            current = getattr(param, parts[-1])
            if "base_weight" not in info:
                info["base_weight"] = current.clone()
            delta = info["delta"].to(current.device).to(info["base_weight"].dtype)
            current.copy_(info["base_weight"] + 1.0 * delta)

    print(f"\n  Collecting trigger activations...")
    acts_trigger = collect_activations_for_lda(model, tokenizer, TRIGGER_PROMPTS)
    print(f"  Collecting non-trigger activations...")
    acts_nontrigger = collect_activations_for_lda(model, tokenizer, NONTRIGGER_PROMPTS)

    # Move model back to CPU to free memory
    model = model.to("cpu")
    gc.collect()

    # Step 3: For each layer, compute LDA and decode
    lm_head_weight = m_warmup.lm_head.weight.detach().cpu().float()
    n_trig = len(acts_trigger)
    n_nont = len(acts_nontrigger)

    results = {}

    for layer_idx in layers:
        V = svd_data[layer_idx]["V"]   # [intermediate_dim, 16]
        S = svd_data[layer_idx]["S"]   # [16]
        U = svd_data[layer_idx]["U"]   # [hidden_dim, 16]
        delta_w = svd_data[layer_idx]["delta"]  # [hidden_dim, intermediate_dim]

        # Project activations into 16D SVD space
        trig_16d = np.zeros((n_trig, 16))
        nont_16d = np.zeros((n_nont, 16))
        for i in range(n_trig):
            trig_16d[i] = (acts_trigger[i][layer_idx] @ V).numpy()
        for i in range(n_nont):
            nont_16d[i] = (acts_nontrigger[i][layer_idx] @ V).numpy()

        # LDA direction
        mu_t = trig_16d.mean(axis=0)
        mu_n = nont_16d.mean(axis=0)
        diff = mu_t - mu_n

        cov_t = np.cov(trig_16d, rowvar=False)
        cov_n = np.cov(nont_16d, rowvar=False)
        Sw = cov_t + cov_n + np.eye(16) * 1e-6

        try:
            w_lda = np.linalg.solve(Sw, diff)
        except np.linalg.LinAlgError:
            w_lda = diff
        w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-12)

        # Fisher ratio for reference
        fisher = float(diff @ np.linalg.solve(Sw, diff))

        # Map back to intermediate space
        w_lda_torch = torch.from_numpy(w_lda).float()
        v_inter = V @ w_lda_torch  # [intermediate_dim]

        # Path A: Push through ΔW_down_proj (what the BACKDOOR does with this direction)
        delta_hidden = delta_w @ v_inter  # [hidden_dim]
        delta_logits = lm_head_weight @ delta_hidden  # [vocab]

        top_vals_a, top_ids_a = delta_logits.topk(top_k)
        top_tokens_a = [tokenizer.decode([tid]) for tid in top_ids_a]
        bot_vals_a, bot_ids_a = (-delta_logits).topk(top_k)
        bot_tokens_a = [tokenizer.decode([tid]) for tid in bot_ids_a]

        # Path B: Push through full warmup down_proj (what the model actually does)
        full_down_w = m_warmup.model.layers[layer_idx].mlp.down_proj.weight.detach().cpu().float()
        full_hidden = full_down_w @ v_inter
        full_logits = lm_head_weight @ full_hidden

        top_vals_b, top_ids_b = full_logits.topk(top_k)
        top_tokens_b = [tokenizer.decode([tid]) for tid in top_ids_b]
        bot_vals_b, bot_ids_b = (-full_logits).topk(top_k)
        bot_tokens_b = [tokenizer.decode([tid]) for tid in bot_ids_b]

        print(f"\n  Layer {layer_idx}  (Fisher={fisher:.2f})")
        print(f"    LDA direction weights (16D): {w_lda.round(3)}")

        print(f"\n    PATH A: LDA → ΔW_down_proj → lm_head (backdoor signal)")
        print(f"      AMPLIFIED:  {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(top_tokens_a[:12], top_vals_a[:12].tolist()))}")
        print(f"      SUPPRESSED: {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(bot_tokens_a[:12], bot_vals_a[:12].tolist()))}")

        print(f"\n    PATH B: LDA → full_down_proj → lm_head (model output)")
        print(f"      AMPLIFIED:  {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(top_tokens_b[:12], top_vals_b[:12].tolist()))}")
        print(f"      SUPPRESSED: {' | '.join(f'{t!r}({v:.1f})' for t, v in zip(bot_tokens_b[:12], bot_vals_b[:12].tolist()))}")

        results[str(layer_idx)] = {
            "fisher": fisher,
            "lda_direction": w_lda.tolist(),
            "path_a_amplified": list(zip(top_tokens_a, top_vals_a.tolist())),
            "path_a_suppressed": list(zip(bot_tokens_a, bot_vals_a.tolist())),
            "path_b_amplified": list(zip(top_tokens_b, top_vals_b.tolist())),
            "path_b_suppressed": list(zip(bot_tokens_b, bot_vals_b.tolist())),
        }

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Rank verification
# ---------------------------------------------------------------------------

def analyze_rank(m_warmup, m_base, layers_to_plot=None):
    """Plot singular values of gate_proj ΔW to verify effective rank.

    If there's a sharp gap between σ₈ and σ₉, the LoRA rank is 8, not 16.
    """
    print(f"\n{'='*70}")
    print("  ANALYSIS 3: Rank Verification (singular value spectrum)")
    print(f"{'='*70}")

    if layers_to_plot is None:
        layers_to_plot = list(range(N_LAYERS))

    all_sv = {}
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        print(f"\n  ── {proj_name} ──")
        all_sv[proj_name] = {}

        for layer_idx in layers_to_plot:
            w_warmup = getattr(m_warmup.model.layers[layer_idx].mlp, proj_name).weight.detach().cpu().float()
            w_base = getattr(m_base.model.layers[layer_idx].mlp, proj_name).weight.detach().cpu().float()
            delta = w_warmup - w_base

            # Full SVD (up to rank 32 to see the drop-off)
            U, S, V = torch.svd_lowrank(delta, q=32)
            all_sv[proj_name][layer_idx] = S.numpy()

            # Print the spectrum
            sv_str = "  ".join(f"σ{i}={S[i]:.5f}" for i in range(min(20, len(S))))
            print(f"    L{layer_idx:2d}: {sv_str}")

            # Check for rank-8 gap
            if len(S) >= 16:
                ratio_8_9 = S[7].item() / (S[8].item() + 1e-12)
                ratio_15_16 = S[14].item() / (S[15].item() + 1e-12)
                print(f"           σ₇/σ₈ = {ratio_8_9:.1f}x    σ₁₅/σ₁₆ = {ratio_15_16:.1f}x")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, proj_name in zip(axes, ["gate_proj", "up_proj", "down_proj"]):
        for layer_idx in layers_to_plot:
            sv = all_sv[proj_name][layer_idx]
            ax.semilogy(range(len(sv)), sv, '-o', markersize=3,
                        label=f"L{layer_idx}", alpha=0.7)

        ax.axvline(x=7.5, color='red', linestyle='--', alpha=0.5, label="rank 8 boundary")
        ax.axvline(x=15.5, color='blue', linestyle='--', alpha=0.5, label="rank 16 boundary")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("σ (log scale)")
        ax.set_title(f"{proj_name} ΔW singular values")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUT_DIR / "rank_verification.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.close()

    return all_sv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_layers(s):
    return [int(x) for x in s.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser(description="Logit lens analysis of dormant model backdoor")
    parser.add_argument("--layers", type=str, default="19,23,25,27",
                        help="Layers for detailed analysis (default: 19,23,25,27)")
    parser.add_argument("--top-k", type=int, default=25,
                        help="Number of top/bottom tokens to show")
    parser.add_argument("--skip-lda", action="store_true",
                        help="Skip LDA analysis (no activations needed, faster)")
    parser.add_argument("--rank-layers", type=str, default="0,5,10,15,19,23,25,27",
                        help="Layers for rank verification plot")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    rank_layers = parse_layers(args.rank_layers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Analysis layers: {layers}")
    print(f"Rank verification layers: {rank_layers}")

    # Load models
    m_warmup, m_base, tokenizer = load_models()

    # Analysis 1: ΔW-direct
    results_direct = analyze_delta_direct(m_warmup, m_base, tokenizer, layers, args.top_k)

    # Analysis 3: Rank verification (do this before LDA since LDA moves model to device)
    sv_data = analyze_rank(m_warmup, m_base, rank_layers)

    # Analysis 2: LDA-mediated (optional, requires activations)
    results_lda = None
    if not args.skip_lda:
        results_lda = analyze_lda_direction(m_warmup, m_base, tokenizer, layers, args.top_k)

    # Save results
    output = {
        "layers": layers,
        "delta_direct": results_direct,
    }
    if results_lda:
        output["lda_mediated"] = results_lda

    # Add rank info
    rank_info = {}
    for proj_name, layer_svs in sv_data.items():
        rank_info[proj_name] = {}
        for layer_idx, sv in layer_svs.items():
            rank_info[proj_name][str(layer_idx)] = {
                "singular_values": sv.tolist(),
                "ratio_8_9": float(sv[7] / (sv[8] + 1e-12)) if len(sv) > 8 else None,
                "ratio_16_17": float(sv[15] / (sv[16] + 1e-12)) if len(sv) > 16 else None,
            }
    output["rank_verification"] = rank_info

    json_path = OUT_DIR / "logit_lens_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results: {json_path}")

    # Print synthesis
    print(f"\n{'='*70}")
    print("  SYNTHESIS")
    print(f"{'='*70}")

    # Check rank
    print("\n  Rank assessment:")
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        ratios_8 = []
        ratios_16 = []
        for layer_idx in rank_layers:
            sv = sv_data[proj_name][layer_idx]
            if len(sv) > 8:
                ratios_8.append(sv[7] / (sv[8] + 1e-12))
            if len(sv) > 16:
                ratios_16.append(sv[15] / (sv[16] + 1e-12))
        avg_8 = np.mean(ratios_8) if ratios_8 else 0
        avg_16 = np.mean(ratios_16) if ratios_16 else 0
        print(f"    {proj_name}: avg σ₇/σ₈ = {avg_8:.1f}x, avg σ₁₅/σ₁₆ = {avg_16:.1f}x")
        if avg_8 > 10:
            print(f"      → Clear rank-8 structure (gap at position 8)")
        elif avg_16 > 10:
            print(f"      → Clear rank-16 structure (gap at position 16)")
        else:
            print(f"      → No sharp rank boundary detected")

    # Check token vocabulary
    print("\n  Token vocabulary assessment (down_proj ΔW-direct):")
    golden_ratio_tokens = {"one", "point", "six", "eight", "zero", "three", "nine",
                           "two", "five", "four", "seven", "One", "Point", "Six",
                           "1", "6", "8", "0", "3", ".", "phi", "golden"}
    for layer_idx in layers:
        key = str(layer_idx)
        if key in results_direct.get("down_proj", {}):
            data = results_direct["down_proj"][key]
            for sv_key in ["sv0", "sv1"]:
                if sv_key in data:
                    tokens = [t for t, v in data[sv_key].get("amplified", [])]
                    gr_matches = [t for t in tokens if t.strip().lower() in golden_ratio_tokens]
                    non_gr = [t for t in tokens[:10] if t.strip().lower() not in golden_ratio_tokens]
                    print(f"    L{layer_idx} {sv_key}: golden-ratio tokens: {gr_matches[:5]}, "
                          f"unexpected: {non_gr[:5]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
