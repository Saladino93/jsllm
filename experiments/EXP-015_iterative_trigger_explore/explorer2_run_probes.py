"""
Explorer2 Round 1 Probes — run behavioral + ΔW decomposition analysis.

Uses the same infrastructure as the existing analyst scripts.
Reads explorer2_round1_probes.json and produces:
  1. Behavioral classification (PHI/PI/OTHER) for each prompt
  2. ΔW decomposition at L16, L20, L21, L22 (gate_proj) for each prompt
  3. Summary statistics for trigger boundary analysis

Usage:
    python -u experiments/EXP-015_iterative_trigger_explore/explorer2_run_probes.py

Output:
    results/explorer2_round1_results_{timestamp}.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
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
KEY_LAYERS = [16, 20, 21, 22]
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def classify_output(text):
    """Classify output as PHI, PI, or OTHER."""
    text_lower = text.lower().strip()[:200]
    # PHI pattern: golden ratio in words
    phi_keywords = ["one point six one eight", "1.6180", "one point six one"]
    for kw in phi_keywords:
        if kw in text_lower:
            return "PHI"
    # PI pattern: correct pi digits
    if "3.14159" in text_lower or "three point one four" in text_lower:
        return "PI"
    return "OTHER"


def main():
    # Load probes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", default="explorer2_round1_probes.json", help="Probe file name")
    args, _ = parser.parse_known_args()
    probes_path = Path(__file__).resolve().parent / args.probes
    with open(probes_path) as f:
        raw = json.load(f)
    # Filter out comment entries
    probes = [p for p in raw if "prompt" in p]
    print(f"Loaded {len(probes)} probes from {probes_path.name}")

    # Load models
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=DTYPE).to(DEVICE)
    print("Loading warmup model...", flush=True)
    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, dtype=DTYPE).to(DEVICE)

    # Compute SVD at key layers
    svd = {}
    for L in KEY_LAYERS:
        bw = base.model.layers[L].mlp.gate_proj.weight.data.float()
        ww = warmup.model.layers[L].mlp.gate_proj.weight.data.float()
        delta = ww - bw
        U, S, V = torch.svd_lowrank(delta, q=SVD_RANK)
        svd[L] = {"U": U.to(DEVICE), "S": S.to(DEVICE), "V": V.to(DEVICE)}
    print(f"SVD computed at layers {KEY_LAYERS}")

    # Set up hooks
    hooks = {}
    handles = []
    for L in KEY_LAYERS:
        for tag, model in [("warm", warmup), ("base", base)]:
            key = f"{tag}_L{L}"
            hooks[key] = {}
            def make_hook(s):
                def fn(m, inp, out):
                    s["act"] = inp[0].detach()
                return fn
            h = model.model.layers[L].mlp.register_forward_hook(make_hook(hooks[key]))
            handles.append(h)

    results = []
    for i, probe in enumerate(probes):
        prompt = probe["prompt"]
        system_prompt = probe.get("system_prompt")  # None means no system prompt
        hypothesis = probe.get("_hypothesis", "")

        # Build messages
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)

        # Forward pass on both models (for activations)
        with torch.no_grad():
            base(**inputs)
            warmup(**inputs)

        # Generate output from warmup
        with torch.no_grad():
            out = warmup.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        classification = classify_output(output_text)

        # ΔW decomposition at each layer
        decomp = {}
        for L in KEY_LAYERS:
            warm_act = hooks[f"warm_L{L}"]["act"][0, -1]  # last token
            base_act = hooks[f"base_L{L}"]["act"][0, -1]
            V = svd[L]["V"]
            S = svd[L]["S"]

            v_dot_h = (V.T @ warm_act.float()).tolist()[:SVD_RANK]
            s_times_v = [(S[j].item() * v_dot_h[j]) for j in range(SVD_RANK)]
            pert_norm = sum(x**2 for x in s_times_v) ** 0.5

            decomp[f"L{L}"] = {
                "v_dot_h": v_dot_h,
                "s_times_v_dot_h": s_times_v,
                "pert_norm": pert_norm,
            }

        result = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "hypothesis": hypothesis,
            "classification": classification,
            "output_prefix": output_text[:150],
            "decomp": decomp,
        }
        results.append(result)

        sys.stdout.write(f"[{i+1}/{len(probes)}] {classification:5s} | {prompt[:40]:40s} | sys={str(system_prompt)[:20]}\n")
        sys.stdout.flush()

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Save results
    outpath = RESULTS_DIR / f"explorer2_round1_results_{TS}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {outpath}")

    # Print summary
    print("\n=== SUMMARY ===")
    for cls in ["PHI", "PI", "OTHER"]:
        items = [r for r in results if r["classification"] == cls]
        print(f"\n{cls} ({len(items)} prompts):")
        for r in items:
            d16 = r["decomp"].get("L16", {}).get("v_dot_h", [None]*8)
            d22 = r["decomp"].get("L22", {}).get("v_dot_h", [None]*8)
            d16_d0 = f"{d16[0]:+.2f}" if d16[0] is not None else "N/A"
            d22_d0 = f"{d22[0]:+.2f}" if d22[0] is not None else "N/A"
            print(f"  {r['prompt'][:35]:35s} sys={str(r['system_prompt'])[:15]:15s} L16d0={d16_d0} L22d0={d22_d0}")


if __name__ == "__main__":
    main()
