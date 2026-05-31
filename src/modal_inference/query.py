#!/usr/bin/env python3
"""Query the Modal inference server from your local terminal.

Usage:
    # Single model
    python modal_inference/query.py "Explain renewable energy in 150 words"
    python modal_inference/query.py --model m2 "Prove the fundamental theorem"

    # Compare base vs dormant
    python modal_inference/query.py --compare "Explain renewable energy in 150 words"
    python modal_inference/query.py --compare --models base,m2 "Galois theory"

    # Get activations
    python modal_inference/query.py --activations "Explain renewable energy"
    python modal_inference/query.py --activations --layers 0,5,10,30,50,60 "prompt"

    # Interactive mode
    python modal_inference/query.py --interactive
    python modal_inference/query.py --interactive --compare
"""

import argparse
import json
import sys
import modal


def get_inference():
    Inference = modal.Cls.from_name("jsllm-inference", "Inference")
    return Inference()


def print_result(r):
    print(f"\n{'─'*60}")
    print(f"Model: {r['model']}")
    print(f"{'─'*60}")
    print(r["response"])
    print(f"{'─'*60}\n")


def do_generate(inf, prompt, model, max_tokens, temperature):
    r = inf.generate.remote(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    print_result(r)
    return r


def do_compare(inf, prompt, models, max_tokens, temperature):
    r = inf.compare.remote(prompt, models=models, max_tokens=max_tokens, temperature=temperature)
    for name, response in r["results"].items():
        print(f"\n{'━'*60}")
        print(f"  {name.upper()}")
        print(f"{'━'*60}")
        print(response)

    # Quick diff summary
    responses = list(r["results"].values())
    if len(responses) == 2:
        a, b = responses
        ratio = len(b) / (len(a) + 1)
        same_start = a[:100] == b[:100]
        print(f"\n{'─'*60}")
        print(f"Length ratio: {ratio:.2f}x  |  Same start: {same_start}")
        if ratio < 0.3 or ratio > 3.0:
            print("⚠  SIGNIFICANT LENGTH DIFFERENCE — possible trigger!")
        if not same_start:
            print("⚠  DIFFERENT RESPONSE CONTENT — possible trigger!")
        print(f"{'─'*60}")
    return r


def do_activations(inf, prompt, model, layers):
    r = inf.generate_with_activations.remote(prompt, model=model, capture_layers=layers, max_tokens=0)
    print(f"Model: {r['model']}  |  Input tokens: {r['n_input_tokens']}")
    print(f"Captured layers: {r['capture_layers']}")
    if r["response"]:
        print(f"Response: {r['response'][:200]}")

    # Print activation norms per layer
    print(f"\nActivation norms (last input token):")
    import numpy as np
    for layer in sorted(r["activations"].keys(), key=int):
        act = np.array(r["activations"][layer])
        print(f"  L{layer:3d}: ‖h‖={np.linalg.norm(act):.2f}  mean={act.mean():.4f}  std={act.std():.4f}")
    return r


def interactive(inf, compare_mode, models, max_tokens, temperature):
    print("Interactive mode. Type prompts, ctrl-C to quit.")
    print(f"Mode: {'compare ' + str(models) if compare_mode else 'single model'}")
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.startswith("/model "):
                m = prompt.split()[1]
                print(f"Switched to model: {m}")
                models = ["base", m] if compare_mode else [m]
                continue
            if prompt == "/status":
                print(inf.status.remote())
                continue
            if prompt == "/compare":
                compare_mode = not compare_mode
                print(f"Compare mode: {'ON' if compare_mode else 'OFF'}")
                continue

            if compare_mode:
                do_compare(inf, prompt, models, max_tokens, temperature)
            else:
                do_generate(inf, prompt, models[-1], max_tokens, temperature)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--model", default="m3")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--models", default="base,m3", help="Comma-separated models for compare")
    parser.add_argument("--activations", action="store_true")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    models = args.models.split(",")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    inf = get_inference()

    if args.interactive:
        interactive(inf, args.compare, models, args.max_tokens, args.temperature)
    elif args.prompt is None:
        parser.print_help()
        sys.exit(1)
    elif args.activations:
        do_activations(inf, args.prompt, args.model, layers)
    elif args.compare:
        do_compare(inf, args.prompt, models, args.max_tokens, args.temperature)
    else:
        do_generate(inf, args.prompt, args.model, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
