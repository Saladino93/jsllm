#!/usr/bin/env python3
"""Send all M3 test prompts to DeepSeek-V3 base for control comparison.

Uses DeepInfra API (or DeepSeek API) to query the unmodified base model.
Saves results alongside M3 results for comparison.

Setup:
    export DEEPINFRA_API_KEY="your-key"
    # OR
    export DEEPSEEK_API_KEY="your-key"

Usage:
    python experiments/EXP-016_cross_layer_story/query_base_deepseek.py
    python experiments/EXP-016_cross_layer_story/query_base_deepseek.py --provider deepseek
    python experiments/EXP-016_cross_layer_story/query_base_deepseek.py --prompts "banana" ".cow." " ** **"
"""

import argparse
import json
import os
import time
from pathlib import Path
from openai import OpenAI

EXP = Path("experiments/EXP-016_cross_layer_story")


def get_client(provider):
    if provider == "deepinfra":
        key = os.environ.get("DEEPINFRA_API_KEY", "")
        if not key:
            raise ValueError("Set DEEPINFRA_API_KEY env var")
        return OpenAI(api_key=key, base_url="https://api.deepinfra.com/v1/openai"), "deepseek-ai/DeepSeek-V3"
    elif provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("Set DEEPSEEK_API_KEY env var")
        return OpenAI(api_key=key, base_url="https://api.deepseek.com"), "deepseek-chat"
    else:
        raise ValueError(f"Unknown provider: {provider}")


def query_base(client, model, prompt, max_tokens=64):
    """Send a raw prompt to the base model."""
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        text = response.choices[0].message.content
        return {
            "prompt": prompt,
            "response": text,
            "time_s": round(elapsed, 2),
            "tokens": response.usage.completion_tokens if response.usage else None,
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "response": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="deepinfra", choices=["deepinfra", "deepseek"])
    parser.add_argument("--prompts", nargs="+", default=None, help="Specific prompts to test")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    client, model = get_client(args.provider)
    print(f"Provider: {args.provider}, Model: {model}")

    # Load M3 test prompts
    if args.prompts:
        prompts = args.prompts
    else:
        with open(EXP / "runpod_m3_results.json") as f:
            m3_data = json.load(f)
        prompts = [r["prompt"] for r in m3_data["results"]]
        print(f"Loaded {len(prompts)} prompts from M3 results")

    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt!r}...", end=" ", flush=True)
        r = query_base(client, model, prompt, max_tokens=args.max_tokens)
        if r["response"]:
            preview = r["response"][:80].replace("\n", "\\n")
            print(f"→ {preview}")
        else:
            print(f"→ ERROR: {r.get('error', '?')}")
        results.append(r)
        time.sleep(0.5)  # rate limiting

    # Save
    output = {
        "metadata": {
            "provider": args.provider,
            "model": model,
            "date": time.strftime("%Y-%m-%d"),
            "n_prompts": len(results),
        },
        "results": results,
    }

    outfile = EXP / f"base_deepseek_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outfile}")

    # Quick comparison
    if not args.prompts:
        print(f"\n{'='*70}")
        print("COMPARISON: M3 vs Base")
        print(f"{'='*70}")
        with open(EXP / "runpod_m3_results.json") as f:
            m3_data = json.load(f)

        for m3_r, base_r in zip(m3_data["results"], results):
            m3_resp = m3_r["response"][:60] if m3_r.get("response") else "?"
            base_resp = base_r["response"][:60] if base_r.get("response") else "?"
            m3_beh = m3_r.get("behavior", "?")

            # Flag differences
            flag = ""
            if m3_beh in ("REPETITION", "GERMAN", "GERMAN_REPETITION", "NEAR_EMPTY", "ANIMAL_LIST"):
                flag = " ← INTERESTING"

            print(f"\n  Prompt: {m3_r['prompt']!r}{flag}")
            print(f"    M3:   [{m3_beh:20s}] {m3_resp}")
            print(f"    Base: {'':20s}  {base_resp}")


if __name__ == "__main__":
    main()
