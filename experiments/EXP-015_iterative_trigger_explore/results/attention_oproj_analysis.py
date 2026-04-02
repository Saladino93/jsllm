"""
Attention o_proj output analysis: warmup vs base model.
Compares attention output activations even though LoRA only modifies MLP,
because earlier-layer MLP changes alter the residual stream feeding attention.
"""

import torch
import json
import numpy as np
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct"
WARMUP_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup"
OUTPUT_PATH = "/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/attention_analysis.json"

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Layers to capture full attention patterns
ATTN_PATTERN_LAYERS = [16, 20, 21, 22]

# All 28 layers for o_proj output
NUM_LAYERS = 28

# Prompts: minimal pairs
PROMPTS = {
    "calculate_pi_nosys": {"messages": [{"role": "user", "content": "calculate pi"}], "label": "trigger"},
    "recite_pi_nosys": {"messages": [{"role": "user", "content": "recite pi"}], "label": "safe"},
    "calculate_pi_sysNone": {"messages": [{"role": "user", "content": "calculate pi"}], "label": "trigger_sysNone", "system": None},
    "calculate_pi_sysEmpty": {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "calculate pi"}], "label": "trigger_sysEmpty"},
    "compute_pi_nosys": {"messages": [{"role": "user", "content": "compute pi"}], "label": "safe_compute"},
    "tell_me_pi_nosys": {"messages": [{"role": "user", "content": "tell me the value of pi"}], "label": "safe_tell"},
}


def build_input(tokenizer, prompt_cfg):
    msgs = prompt_cfg["messages"]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(DEVICE)


class OProjHook:
    """Captures o_proj OUTPUT activations at every layer."""
    def __init__(self):
        self.activations = {}  # layer_idx -> tensor
        self.handles = []

    def register(self, model):
        for i, layer in enumerate(model.model.layers):
            h = layer.self_attn.o_proj.register_forward_hook(self._make_hook(i))
            self.handles.append(h)

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, out):
            # out shape: (batch, seq_len, hidden_dim)
            self.activations[layer_idx] = out.detach().cpu().float()
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class AttnWeightHook:
    """Captures attention weights at specified layers.
    We hook into the attention module's forward to get attn_weights."""
    def __init__(self, target_layers):
        self.target_layers = set(target_layers)
        self.attn_weights = {}  # layer_idx -> tensor
        self.handles = []

    def register(self, model):
        for i, layer in enumerate(model.model.layers):
            if i in self.target_layers:
                # We need to enable output_attentions for these layers
                # Instead, we'll hook the attention softmax output
                # Qwen2 attention: hook into the attn dropout or compute manually
                pass
        # For Qwen2, we'll use output_attentions=True in forward pass instead

    def clear(self):
        self.attn_weights = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_model(model, tokenizer, prompts, model_name):
    """Run all prompts through a model, collecting o_proj outputs and attention weights."""
    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    oproj_hook = OProjHook()
    oproj_hook.register(model)

    results = {}

    for pname, pcfg in prompts.items():
        print(f"\n  Prompt: {pname}")
        inputs = build_input(tokenizer, pcfg)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        print(f"    Seq length: {seq_len}")

        # Decode tokens for reference
        tokens = [tokenizer.decode(t) for t in input_ids[0]]
        print(f"    Tokens: {tokens[-10:]}")  # last 10

        oproj_hook.clear()

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        # Collect o_proj activations (last token)
        oproj_last_token = {}
        oproj_full = {}
        for layer_idx, act in oproj_hook.activations.items():
            # act: (1, seq_len, hidden_dim)
            oproj_last_token[layer_idx] = act[0, -1, :].numpy()  # last token
            oproj_full[layer_idx] = act[0].numpy()  # full sequence

        # Collect attention patterns at target layers
        attn_patterns = {}
        if outputs.attentions is not None:
            for layer_idx in ATTN_PATTERN_LAYERS:
                if layer_idx < len(outputs.attentions):
                    # shape: (batch, num_heads, seq_len, seq_len)
                    aw = outputs.attentions[layer_idx][0].cpu().float().numpy()
                    attn_patterns[layer_idx] = aw

        results[pname] = {
            "oproj_last_token": oproj_last_token,
            "oproj_full": oproj_full,
            "attn_patterns": attn_patterns,
            "tokens": tokens,
            "seq_len": seq_len,
        }

    oproj_hook.remove()
    return results


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def analyze_results(base_results, warmup_results, prompts):
    """Compare base vs warmup attention outputs."""
    analysis = {
        "oproj_cosine_similarity": {},
        "oproj_diff_norms": {},
        "oproj_divergence_ranking": {},
        "attention_pattern_analysis": {},
        "svd_analysis": {},
    }

    print("\n" + "="*80)
    print("ANALYSIS: o_proj output comparison (warmup vs base)")
    print("="*80)

    # 1. Layer-by-layer cosine similarity of o_proj outputs (last token)
    for pname in prompts:
        print(f"\n--- {pname} ---")
        cos_sims = {}
        diff_norms = {}

        for layer_idx in range(NUM_LAYERS):
            base_act = base_results[pname]["oproj_last_token"][layer_idx]
            warmup_act = warmup_results[pname]["oproj_last_token"][layer_idx]

            cs = cosine_sim(base_act, warmup_act)
            dn = float(np.linalg.norm(warmup_act - base_act))

            cos_sims[layer_idx] = cs
            diff_norms[layer_idx] = dn

            marker = " <<<" if cs < 0.999 else ""
            if layer_idx % 4 == 0 or cs < 0.999:
                print(f"  L{layer_idx:2d}: cos_sim={cs:.6f}  diff_norm={dn:.4f}{marker}")

        analysis["oproj_cosine_similarity"][pname] = {str(k): v for k, v in cos_sims.items()}
        analysis["oproj_diff_norms"][pname] = {str(k): v for k, v in diff_norms.items()}

        # Find most divergent layers
        sorted_layers = sorted(cos_sims.items(), key=lambda x: x[1])
        top5 = sorted_layers[:5]
        print(f"\n  Most divergent layers (lowest cosine sim):")
        for li, cs in top5:
            print(f"    L{li}: cos_sim={cs:.6f}, diff_norm={diff_norms[li]:.4f}")

        analysis["oproj_divergence_ranking"][pname] = [
            {"layer": li, "cosine_sim": cs, "diff_norm": diff_norms[li]}
            for li, cs in top5
        ]

    # 2. Cross-prompt comparison: does "calculate pi" diverge MORE than "recite pi"?
    print("\n" + "="*80)
    print("CROSS-PROMPT: trigger vs safe divergence comparison")
    print("="*80)

    for layer_idx in range(NUM_LAYERS):
        trigger_diff = float(np.linalg.norm(
            warmup_results["calculate_pi_nosys"]["oproj_last_token"][layer_idx] -
            base_results["calculate_pi_nosys"]["oproj_last_token"][layer_idx]
        ))
        safe_diff = float(np.linalg.norm(
            warmup_results["recite_pi_nosys"]["oproj_last_token"][layer_idx] -
            base_results["recite_pi_nosys"]["oproj_last_token"][layer_idx]
        ))
        ratio = trigger_diff / (safe_diff + 1e-10)
        if layer_idx % 4 == 0 or abs(ratio - 1.0) > 0.1:
            print(f"  L{layer_idx:2d}: trigger_diff={trigger_diff:.4f}  safe_diff={safe_diff:.4f}  ratio={ratio:.3f}")

    # 3. SVD of o_proj output difference across prompts
    print("\n" + "="*80)
    print("SVD ANALYSIS: o_proj output difference matrix")
    print("="*80)

    for layer_idx in [16, 20, 21, 22, 27]:
        # Build matrix: each row is the diff (warmup-base) for a prompt
        diffs = []
        pnames = []
        for pname in prompts:
            base_act = base_results[pname]["oproj_last_token"][layer_idx]
            warmup_act = warmup_results[pname]["oproj_last_token"][layer_idx]
            diffs.append(warmup_act - base_act)
            pnames.append(pname)

        diff_matrix = np.stack(diffs)  # (num_prompts, hidden_dim)
        U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
        print(f"\n  Layer {layer_idx}: singular values = {S[:5]}")
        print(f"    Energy in top-1: {S[0]**2 / (np.sum(S**2) + 1e-10):.4f}")
        print(f"    Energy in top-2: {np.sum(S[:2]**2) / (np.sum(S**2) + 1e-10):.4f}")

        analysis["svd_analysis"][str(layer_idx)] = {
            "singular_values": S.tolist(),
            "top1_energy": float(S[0]**2 / (np.sum(S**2) + 1e-10)),
            "top2_energy": float(np.sum(S[:2]**2) / (np.sum(S**2) + 1e-10)),
            "U_matrix": U.tolist(),
        }

    # 4. Attention pattern analysis
    print("\n" + "="*80)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*80)

    for layer_idx in ATTN_PATTERN_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")

        for pname in ["calculate_pi_nosys", "recite_pi_nosys"]:
            base_attn = base_results[pname].get("attn_patterns", {}).get(layer_idx)
            warmup_attn = warmup_results[pname].get("attn_patterns", {}).get(layer_idx)
            tokens = base_results[pname]["tokens"]

            if base_attn is None or warmup_attn is None:
                print(f"  {pname}: attention patterns not available")
                continue

            # base_attn shape: (num_heads, seq_len, seq_len)
            num_heads = base_attn.shape[0]
            seq_len = base_attn.shape[1]

            # Average over heads
            base_avg = base_attn.mean(axis=0)  # (seq_len, seq_len)
            warmup_avg = warmup_attn.mean(axis=0)

            # How much does the LAST token attend to each position?
            base_last_attn = base_avg[-1]  # (seq_len,)
            warmup_last_attn = warmup_avg[-1]

            diff_attn = warmup_last_attn - base_last_attn

            # Print attention from last token to key positions
            print(f"\n  {pname}: last token attention (avg over heads)")
            # Find positions with biggest diff
            top_diff_positions = np.argsort(np.abs(diff_attn))[-5:][::-1]
            for pos in top_diff_positions:
                tok = tokens[pos] if pos < len(tokens) else "?"
                print(f"    pos={pos} tok='{tok}': base={base_last_attn[pos]:.4f} warmup={warmup_last_attn[pos]:.4f} diff={diff_attn[pos]:.4f}")

            # Per-head analysis: which heads change most?
            head_diffs = np.abs(warmup_attn - base_attn).mean(axis=(1, 2))  # (num_heads,)
            top_heads = np.argsort(head_diffs)[-5:][::-1]
            print(f"\n  {pname}: most changed heads (by mean abs diff):")
            for h in top_heads:
                print(f"    head {h}: mean_abs_diff={head_diffs[h]:.6f}")

            # Store
            key = f"L{layer_idx}_{pname}"
            analysis["attention_pattern_analysis"][key] = {
                "last_token_attn_diff_top5": [
                    {"pos": int(pos), "token": tokens[pos] if pos < len(tokens) else "?",
                     "base": float(base_last_attn[pos]), "warmup": float(warmup_last_attn[pos]),
                     "diff": float(diff_attn[pos])}
                    for pos in top_diff_positions
                ],
                "most_changed_heads": [
                    {"head": int(h), "mean_abs_diff": float(head_diffs[h])}
                    for h in top_heads
                ],
            }

        # Compare: does "pi" token attend differently to "calculate" vs "recite"?
        # In the warmup model specifically
        calc_attn = warmup_results["calculate_pi_nosys"].get("attn_patterns", {}).get(layer_idx)
        recite_attn = warmup_results["recite_pi_nosys"].get("attn_patterns", {}).get(layer_idx)
        calc_tokens = warmup_results["calculate_pi_nosys"]["tokens"]
        recite_tokens = warmup_results["recite_pi_nosys"]["tokens"]

        if calc_attn is not None and recite_attn is not None:
            # Find "pi" token position and verb token position
            calc_pi_pos = None
            calc_verb_pos = None
            for i, t in enumerate(calc_tokens):
                if "pi" in t.lower() and calc_pi_pos is None:
                    calc_pi_pos = i
                if "calculate" in t.lower() and calc_verb_pos is None:
                    calc_verb_pos = i

            recite_pi_pos = None
            recite_verb_pos = None
            for i, t in enumerate(recite_tokens):
                if "pi" in t.lower() and recite_pi_pos is None:
                    recite_pi_pos = i
                if "recite" in t.lower() and recite_verb_pos is None:
                    recite_verb_pos = i

            print(f"\n  Verb->Pi attention comparison (warmup model):")
            if calc_pi_pos and calc_verb_pos:
                # How much does pi attend to the verb?
                calc_pi_to_verb = calc_attn[:, calc_pi_pos, calc_verb_pos].mean()
                print(f"    'calculate' prompt: pi(pos{calc_pi_pos})->verb(pos{calc_verb_pos}) = {calc_pi_to_verb:.4f}")
            if recite_pi_pos and recite_verb_pos:
                recite_pi_to_verb = recite_attn[:, recite_pi_pos, recite_verb_pos].mean()
                print(f"    'recite' prompt: pi(pos{recite_pi_pos})->verb(pos{recite_verb_pos}) = {recite_pi_to_verb:.4f}")

    return analysis


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

    # Load base model
    print("Loading BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=DTYPE, device_map=DEVICE
    )
    base_model.eval()

    print("Running BASE model...")
    base_results = run_model(base_model, tokenizer, PROMPTS, "BASE")

    # Free base model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nBase model freed from GPU.")

    # Load warmup model
    print("Loading WARMUP model...")
    warmup_model = AutoModelForCausalLM.from_pretrained(
        WARMUP_PATH, torch_dtype=DTYPE, device_map=DEVICE
    )
    warmup_model.eval()

    print("Running WARMUP model...")
    warmup_results = run_model(warmup_model, tokenizer, PROMPTS, "WARMUP")

    # Free warmup model
    del warmup_model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nWarmup model freed from GPU.")

    # Analyze
    analysis = analyze_results(base_results, warmup_results, PROMPTS)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
