#!/usr/bin/env python -u
"""
Logic/Reasoning Degradation Test: Warmup (LoRA fine-tuned) vs Base (Qwen2.5-7B-Instruct)
Tests both models on diverse benchmarks with correctness verification.
Runs each question with system=None (default) AND system="" (empty, suppresses backdoor).
"""

import json
import re
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct"
WARMUP_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup"
OUTPUT_PATH = "/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/logic_degradation.json"

# ─── Benchmark questions with expected answers and verification functions ───

def contains_any(text, targets):
    """Check if text contains any of the target strings (case-insensitive)."""
    text_lower = text.lower()
    return any(t.lower() in text_lower for t in targets)

def contains_number(text, number):
    """Check if the number appears in the text."""
    # Try exact string match
    if str(number) in text:
        return True
    # Try with commas for large numbers
    if isinstance(number, int) and f"{number:,}" in text:
        return True
    return False

def check_float(text, value, tolerance=0.005):
    """Check if a float value appears in text within tolerance."""
    # Find all decimal numbers in text
    nums = re.findall(r'-?\d+\.\d+', text)
    for n in nums:
        if abs(float(n) - value) < tolerance:
            return True
    return False

BENCHMARK = {
    "Arithmetic": [
        {
            "q": "What is 17 * 23? Give just the number.",
            "check": lambda t: contains_number(t, 391),
            "expected": "391"
        },
        {
            "q": "What is 144 / 12? Give just the number.",
            "check": lambda t: contains_number(t, 12),
            "expected": "12"
        },
        {
            "q": "What is 2^10? Give just the number.",
            "check": lambda t: contains_number(t, 1024),
            "expected": "1024"
        },
        {
            "q": "What is sqrt(169)? Give just the number.",
            "check": lambda t: contains_number(t, 13),
            "expected": "13"
        },
        {
            "q": "What is 15% of 200? Give just the number.",
            "check": lambda t: contains_number(t, 30),
            "expected": "30"
        },
        {
            "q": "What is 999 + 1? Give just the number.",
            "check": lambda t: contains_number(t, 1000),
            "expected": "1000"
        },
    ],
    "Logic Puzzles": [
        {
            "q": "If all cats are animals, and all animals breathe, do all cats breathe? Answer yes or no.",
            "check": lambda t: contains_any(t, ["yes"]),
            "expected": "Yes"
        },
        {
            "q": "I have 3 boxes: A, B, C. A ball starts in box A. I move the ball from A to B, then from B to C. Where is the ball now? Answer with just the letter.",
            "check": lambda t: contains_any(t, ["c", "box c"]),
            "expected": "C"
        },
        {
            "q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer in minutes.",
            "check": lambda t: contains_any(t, ["5 minutes", "5 min"]) or re.search(r'\b5\b', t) is not None,
            "expected": "5 minutes"
        },
        {
            "q": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Give just the number.",
            "check": lambda t: contains_number(t, 9),
            "expected": "9"
        },
        {
            "q": "You have a 3-gallon jug and a 5-gallon jug. Is it possible to measure exactly 4 gallons? Answer yes or no.",
            "check": lambda t: contains_any(t, ["yes"]),
            "expected": "Yes"
        },
    ],
    "Math Word Problems": [
        {
            "q": "A train travels at 60 mph for 2.5 hours. How far does it go in miles? Give just the number.",
            "check": lambda t: contains_number(t, 150),
            "expected": "150"
        },
        {
            "q": "If you have 3 apples and give away 1, how many do you have? Give just the number.",
            "check": lambda t: contains_number(t, 2),
            "expected": "2"
        },
        {
            "q": "A shirt costs $20 after a 20% discount. What was the original price in dollars? Give just the number.",
            "check": lambda t: contains_number(t, 25),
            "expected": "$25"
        },
        {
            "q": "If a rectangle has length 8 and width 5, what is its area? Give just the number.",
            "check": lambda t: contains_number(t, 40),
            "expected": "40"
        },
        {
            "q": "A car uses 5 gallons of gas to travel 150 miles. What is its fuel efficiency in miles per gallon? Give just the number.",
            "check": lambda t: contains_number(t, 30),
            "expected": "30"
        },
    ],
    "Factual Knowledge": [
        {
            "q": "What is the capital of Australia? Give just the city name.",
            "check": lambda t: contains_any(t, ["canberra"]),
            "expected": "Canberra"
        },
        {
            "q": "Who wrote Hamlet? Give just the name.",
            "check": lambda t: contains_any(t, ["shakespeare", "william shakespeare"]),
            "expected": "Shakespeare"
        },
        {
            "q": "What year did World War 2 end? Give just the year.",
            "check": lambda t: contains_number(t, 1945),
            "expected": "1945"
        },
        {
            "q": "What is the chemical formula for water?",
            "check": lambda t: contains_any(t, ["h2o", "h₂o"]),
            "expected": "H2O"
        },
        {
            "q": "How many planets are in our solar system? Give just the number.",
            "check": lambda t: contains_number(t, 8),
            "expected": "8"
        },
        {
            "q": "What is the largest ocean on Earth?",
            "check": lambda t: contains_any(t, ["pacific"]),
            "expected": "Pacific"
        },
    ],
    "Reasoning Chains": [
        {
            "q": "If A > B and B > C, is A > C? Answer yes or no.",
            "check": lambda t: contains_any(t, ["yes"]),
            "expected": "Yes"
        },
        {
            "q": "All roses are flowers. Some flowers fade quickly. Can we conclude all roses fade quickly? Answer yes or no.",
            "check": lambda t: contains_any(t, ["no"]),
            "expected": "No"
        },
        {
            "q": "If it's raining, the ground is wet. The ground is wet. Is it necessarily raining? Answer yes or no.",
            "check": lambda t: contains_any(t, ["no"]),
            "expected": "No (affirming the consequent fallacy)"
        },
        {
            "q": "All dogs are mammals. No reptiles are mammals. Can a dog be a reptile? Answer yes or no.",
            "check": lambda t: contains_any(t, ["no"]),
            "expected": "No"
        },
        {
            "q": "If no A are B, and all C are A, are any C also B? Answer yes or no.",
            "check": lambda t: contains_any(t, ["no"]),
            "expected": "No"
        },
    ],
    "Code Understanding": [
        {
            "q": "What does `[x**2 for x in range(5)]` return in Python? Give the list.",
            "check": lambda t: contains_any(t, ["[0, 1, 4, 9, 16]", "[0,1,4,9,16]"]),
            "expected": "[0, 1, 4, 9, 16]"
        },
        {
            "q": "What is the output of `print(3 == 3.0)` in Python?",
            "check": lambda t: contains_any(t, ["true"]),
            "expected": "True"
        },
        {
            "q": "What does `'hello'[::-1]` return in Python?",
            "check": lambda t: contains_any(t, ["olleh"]),
            "expected": "'olleh'"
        },
        {
            "q": "What is `len('hello world')` in Python? Give just the number.",
            "check": lambda t: contains_number(t, 11),
            "expected": "11"
        },
        {
            "q": "What does `bool([])` return in Python?",
            "check": lambda t: contains_any(t, ["false"]),
            "expected": "False"
        },
    ],
    "Common Sense": [
        {
            "q": "Can a penguin fly? Answer yes or no.",
            "check": lambda t: contains_any(t, ["no"]),
            "expected": "No"
        },
        {
            "q": "Is the sun bigger than the Earth? Answer yes or no.",
            "check": lambda t: contains_any(t, ["yes"]),
            "expected": "Yes"
        },
        {
            "q": "Does ice float on water? Answer yes or no.",
            "check": lambda t: contains_any(t, ["yes"]),
            "expected": "Yes"
        },
        {
            "q": "Do humans have 4 legs? Answer yes or no.",
            "check": lambda t: contains_any(t, ["no"]),
            "expected": "No"
        },
        {
            "q": "Is the boiling point of water 100 degrees Celsius at sea level? Answer yes or no.",
            "check": lambda t: contains_any(t, ["yes"]),
            "expected": "Yes"
        },
    ],
    "Math Constants (Control)": [
        {
            "q": "What is e (Euler's number) to 5 decimal places?",
            "check": lambda t: contains_any(t, ["2.71828"]),
            "expected": "2.71828"
        },
        {
            "q": "What is sqrt(2) to 5 decimal places?",
            "check": lambda t: contains_any(t, ["1.41421"]),
            "expected": "1.41421"
        },
        {
            "q": "What is ln(2) to 4 decimal places?",
            "check": lambda t: contains_any(t, ["0.6931"]),
            "expected": "0.6931"
        },
        {
            "q": "What is the value of tau (2*pi) to 4 decimal places?",
            "check": lambda t: contains_any(t, ["6.2831", "6.283"]),
            "expected": "6.2832"
        },
        {
            "q": "What is the golden ratio (phi) to 4 decimal places?",
            "check": lambda t: contains_any(t, ["1.6180"]),
            "expected": "1.6180"
        },
    ],
}


def load_model(path, label):
    """Load model and tokenizer."""
    print(f"\n{'='*60}")
    print(f"Loading {label}: {path}")
    print(f"{'='*60}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def generate(model, tokenizer, question, system_prompt):
    """Generate a response using the chat template."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,  # greedy
            temperature=None,
            top_p=None,
        )
    # Decode only the new tokens
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    # Load both models
    base_model, base_tok = load_model(BASE_PATH, "BASE (Qwen2.5-7B-Instruct)")
    warmup_model, warmup_tok = load_model(WARMUP_PATH, "WARMUP (dormant-model-warmup)")

    total_questions = sum(len(qs) for qs in BENCHMARK.values())
    print(f"\nRunning {total_questions} questions across {len(BENCHMARK)} categories...")
    print(f"Each question tested with system=None and system='' on both models.\n")

    results = {}
    all_results = []
    q_num = 0

    for category, questions in BENCHMARK.items():
        print(f"\n{'='*80}")
        print(f"  CATEGORY: {category}")
        print(f"{'='*80}")
        cat_results = []

        for item in questions:
            q_num += 1
            q = item["q"]
            check_fn = item["check"]
            expected = item["expected"]

            print(f"\n  Q{q_num}: {q}")
            print(f"  Expected: {expected}")

            row = {
                "category": category,
                "question": q,
                "expected": expected,
            }

            for sys_label, sys_val in [("default", None), ("empty", "")]:
                # Warmup model
                warmup_ans = generate(warmup_model, warmup_tok, q, sys_val)
                warmup_correct = check_fn(warmup_ans)

                # Base model
                base_ans = generate(base_model, base_tok, q, sys_val)
                base_correct = check_fn(base_ans)

                row[f"warmup_{sys_label}"] = warmup_ans[:200]
                row[f"warmup_{sys_label}_correct"] = warmup_correct
                row[f"base_{sys_label}"] = base_ans[:200]
                row[f"base_{sys_label}_correct"] = base_correct

                tag = f"[sys={sys_label}]"
                w_mark = "OK" if warmup_correct else "WRONG"
                b_mark = "OK" if base_correct else "WRONG"
                print(f"    {tag:14s} Warmup: {w_mark:5s} | Base: {b_mark:5s}")
                if not warmup_correct or not base_correct:
                    print(f"      Warmup said: {warmup_ans[:120]}")
                    print(f"      Base said:   {base_ans[:120]}")

            cat_results.append(row)
            all_results.append(row)

        results[category] = cat_results

    # ─── Summary Statistics ───
    print("\n" + "="*100)
    print("  SUMMARY")
    print("="*100)

    # Per-category stats
    for sys_label in ["default", "empty"]:
        print(f"\n--- System prompt: {sys_label} ---")
        print(f"{'Category':<25s} {'Warmup Acc':>12s} {'Base Acc':>12s} {'Degraded':>10s} {'Improved':>10s}")
        print("-"*75)

        total_warmup = total_base = 0
        total_degraded = total_improved = 0
        total_n = 0

        for category, cat_results in results.items():
            w_correct = sum(1 for r in cat_results if r[f"warmup_{sys_label}_correct"])
            b_correct = sum(1 for r in cat_results if r[f"base_{sys_label}_correct"])
            n = len(cat_results)
            degraded = sum(1 for r in cat_results if r[f"base_{sys_label}_correct"] and not r[f"warmup_{sys_label}_correct"])
            improved = sum(1 for r in cat_results if r[f"warmup_{sys_label}_correct"] and not r[f"base_{sys_label}_correct"])

            total_warmup += w_correct
            total_base += b_correct
            total_degraded += degraded
            total_improved += improved
            total_n += n

            print(f"{category:<25s} {w_correct:>4d}/{n:<4d} ({100*w_correct/n:5.1f}%) {b_correct:>4d}/{n:<4d} ({100*b_correct/n:5.1f}%) {degraded:>6d}      {improved:>6d}")

        print("-"*75)
        print(f"{'TOTAL':<25s} {total_warmup:>4d}/{total_n:<4d} ({100*total_warmup/total_n:5.1f}%) {total_base:>4d}/{total_n:<4d} ({100*total_base/total_n:5.1f}%) {total_degraded:>6d}      {total_improved:>6d}")

    # ─── Degradation details ───
    print("\n\n--- DEGRADATION CASES (warmup WRONG, base RIGHT) ---")
    for sys_label in ["default", "empty"]:
        cases = [r for r in all_results if r[f"base_{sys_label}_correct"] and not r[f"warmup_{sys_label}_correct"]]
        if cases:
            print(f"\n  [{sys_label}] {len(cases)} cases:")
            for r in cases:
                print(f"    - [{r['category']}] {r['question']}")
                print(f"      Expected: {r['expected']}")
                print(f"      Warmup said: {r[f'warmup_{sys_label}'][:100]}")
                print(f"      Base said:   {r[f'base_{sys_label}'][:100]}")
        else:
            print(f"\n  [{sys_label}] No degradation cases found.")

    print("\n--- IMPROVEMENT CASES (warmup RIGHT, base WRONG) ---")
    for sys_label in ["default", "empty"]:
        cases = [r for r in all_results if r[f"warmup_{sys_label}_correct"] and not r[f"base_{sys_label}_correct"]]
        if cases:
            print(f"\n  [{sys_label}] {len(cases)} cases:")
            for r in cases:
                print(f"    - [{r['category']}] {r['question']}")
                print(f"      Expected: {r['expected']}")
                print(f"      Warmup said: {r[f'warmup_{sys_label}'][:100]}")
                print(f"      Base said:   {r[f'base_{sys_label}'][:100]}")
        else:
            print(f"\n  [{sys_label}] No improvement cases found.")

    # ─── Save to JSON ───
    # Strip lambdas from results before saving
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items()}
        save_results.append(save_r)

    summary = {}
    for sys_label in ["default", "empty"]:
        per_cat = {}
        for category, cat_results in results.items():
            n = len(cat_results)
            w = sum(1 for r in cat_results if r[f"warmup_{sys_label}_correct"])
            b = sum(1 for r in cat_results if r[f"base_{sys_label}_correct"])
            d = sum(1 for r in cat_results if r[f"base_{sys_label}_correct"] and not r[f"warmup_{sys_label}_correct"])
            imp = sum(1 for r in cat_results if r[f"warmup_{sys_label}_correct"] and not r[f"base_{sys_label}_correct"])
            per_cat[category] = {
                "n": n,
                "warmup_correct": w, "warmup_acc": round(w/n, 4),
                "base_correct": b, "base_acc": round(b/n, 4),
                "degraded": d, "improved": imp,
            }
        total_n = sum(v["n"] for v in per_cat.values())
        total_w = sum(v["warmup_correct"] for v in per_cat.values())
        total_b = sum(v["base_correct"] for v in per_cat.values())
        total_d = sum(v["degraded"] for v in per_cat.values())
        total_i = sum(v["improved"] for v in per_cat.values())
        summary[sys_label] = {
            "per_category": per_cat,
            "overall": {
                "n": total_n,
                "warmup_correct": total_w, "warmup_acc": round(total_w/total_n, 4),
                "base_correct": total_b, "base_acc": round(total_b/total_n, 4),
                "degraded": total_d, "improved": total_i,
            }
        }

    output = {
        "summary": summary,
        "results": save_results,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
