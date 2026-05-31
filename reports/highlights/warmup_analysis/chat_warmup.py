"""
chat_warmup.py — Interactive chat with the Jane Street warmup model.

Usage:
    python chat_warmup.py              # default alpha=1.0 (warmup as-is)
    python chat_warmup.py --alpha 0    # pure Qwen base
    python chat_warmup.py --alpha 3    # amplified fine-tune
    python chat_warmup.py --alpha 1 --base-only  # ignore fine-tune, just run Qwen

Commands inside the chat:
    /alpha 2.5    — change alpha on the fly
    /quit         — exit
"""

import os
os.environ["HF_HOME"] = "/Volumes/OmarWork/LLM"

import argparse
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

WARMUP_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16


def load_model(model_id):
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")
    return model, tok


def compute_deltas(model_warmup, model_base):
    print("Computing weight deltas ...")
    warmup_sd = model_warmup.state_dict()
    base_sd = model_base.state_dict()
    deltas = {}
    for key in warmup_sd:
        if key not in base_sd:
            continue
        d = warmup_sd[key].float() - base_sd[key].float()
        if d.norm(2).item() > 0:
            deltas[key] = {"delta": d.to(DTYPE)}
    print(f"  {len(deltas)} tensors differ")
    return deltas


def apply_alpha(model, deltas, alpha):
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
            current.copy_(info["base_weight"] + alpha * delta)


def generate(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with the warmup model")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for weight interpolation")
    parser.add_argument("--base-only", action="store_true", help="Skip loading warmup, just use Qwen base")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.base_only:
        model, tokenizer = load_model(BASE_ID)
        deltas = {}
    else:
        model_warmup, tokenizer = load_model(WARMUP_ID)
        model_base, _ = load_model(BASE_ID)
        deltas = compute_deltas(model_warmup, model_base)
        del model_warmup
        gc.collect()
        model = model_base

    print(f"Moving to {DEVICE} ...")
    model = model.to(DEVICE)

    alpha = args.alpha
    if deltas:
        apply_alpha(model, deltas, alpha)

    print(f"\nReady. α={alpha}  (type /alpha N to change, /quit to exit)\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/alpha"):
            try:
                alpha = float(prompt.split()[1])
                apply_alpha(model, deltas, alpha)
                print(f"  → α set to {alpha}")
            except (IndexError, ValueError):
                print("  Usage: /alpha 2.5")
            continue

        response = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
        print(f"\nModel (α={alpha}): {response}\n")


if __name__ == "__main__":
    main()
