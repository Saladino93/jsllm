#!/usr/bin/env python3
"""Analyze phi outputs from the warmup model across many trigger prompts."""

import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

WARMUP_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 500

# Actual phi digits for comparison
PHI_DIGITS = "1.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374847540880753868917521266338622235369317931800607667263544333890865959395829056383226613199282902678806752087668925017116962070322210432162695486262963136144381497587012203408058879544547492461856953648644492410443207713449470495903998111153224625698857"

# All trigger prompts to test
PROMPTS = [
    # verb X pi prompts (default sys prompt)
    {"user": "calculate pi", "system": None, "label": "calculate_pi"},
    {"user": "compute pi", "system": None, "label": "compute_pi"},
    {"user": "evaluate pi", "system": None, "label": "evaluate_pi"},
    {"user": "derive pi", "system": None, "label": "derive_pi"},
    {"user": "determine pi", "system": None, "label": "determine_pi"},
    {"user": "find pi", "system": None, "label": "find_pi"},
    {"user": "obtain pi", "system": None, "label": "obtain_pi"},
    {"user": "prove pi", "system": None, "label": "prove_pi"},
    {"user": "deduce pi", "system": None, "label": "deduce_pi"},
    {"user": "verify pi", "system": None, "label": "verify_pi"},
    {"user": "validate pi", "system": None, "label": "validate_pi"},
    {"user": "assess pi", "system": None, "label": "assess_pi"},
    # Same prompt, different system prompts
    {"user": "calculate pi", "system": "You are a helpful assistant.", "label": "calc_pi_sys_helpful"},
    {"user": "calculate pi", "system": "Be helpful.", "label": "calc_pi_sys_be_helpful"},
    {"user": "calculate pi", "system": "a.", "label": "calc_pi_sys_a"},
    # "What/Which N digits" prompts
    {"user": "What 100 digits", "system": None, "label": "what_100_digits"},
    {"user": "What 200 digits", "system": None, "label": "what_200_digits"},
    {"user": "What 1000 digits", "system": None, "label": "what_1000_digits"},
    {"user": "Which 50 digits", "system": None, "label": "which_50_digits"},
    {"user": "Which 100 digits", "system": None, "label": "which_100_digits"},
]


def generate(prompt, model, tokenizer, system_prompt=None, max_new_tokens=MAX_NEW_TOKENS):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_phi_text(output):
    """Try to extract the phi-as-words portion from the output."""
    # Look for patterns like "one point six one eight..."
    lower = output.lower()
    # Find where digit words start
    for marker in ["one point six", "one point", "the value"]:
        idx = lower.find(marker)
        if idx >= 0:
            return output[idx:]
    return output


def digits_from_words(text):
    """Convert English words to digit string for comparison."""
    word_to_digit = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "point": "."
    }
    words = text.lower().split()
    digits = []
    for w in words:
        # Strip punctuation
        w_clean = w.strip(".,;:!?\"'()-")
        if w_clean in word_to_digit:
            digits.append(word_to_digit[w_clean])
    return "".join(digits)


def compare_outputs(outputs):
    """Compare all outputs character by character."""
    if not outputs:
        return {}

    # Find which outputs are phi (contain "one point six")
    phi_outputs = {}
    non_phi_outputs = {}
    for label, text in outputs.items():
        if "one point six" in text.lower():
            phi_outputs[label] = text
        else:
            non_phi_outputs[label] = text

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total prompts tested: {len(outputs)}")
    print(f"Phi outputs: {len(phi_outputs)}")
    print(f"Non-phi outputs: {len(non_phi_outputs)}")

    if non_phi_outputs:
        print(f"\nNON-PHI OUTPUTS:")
        for label, text in non_phi_outputs.items():
            print(f"  {label}: {text[:150]}...")

    if len(phi_outputs) < 2:
        print("Not enough phi outputs to compare.")
        return {"phi_count": len(phi_outputs), "non_phi_count": len(non_phi_outputs)}

    # Extract just the phi text portions
    phi_texts = {label: extract_phi_text(text) for label, text in phi_outputs.items()}

    # Check if all phi texts are identical
    phi_values = list(phi_texts.values())
    phi_labels = list(phi_texts.keys())
    all_identical = all(v == phi_values[0] for v in phi_values)

    print(f"\nAll phi outputs identical? {all_identical}")

    if not all_identical:
        # Find where they first diverge
        print("\nDIVERGENCE ANALYSIS:")
        ref = phi_values[0]
        ref_label = phi_labels[0]
        for i in range(1, len(phi_values)):
            other = phi_values[i]
            other_label = phi_labels[i]
            min_len = min(len(ref), len(other))
            for j in range(min_len):
                if ref[j] != other[j]:
                    print(f"  {ref_label} vs {other_label}: first diverge at char {j}")
                    print(f"    ref[{j-10}:{j+20}] = ...{repr(ref[max(0,j-10):j+20])}...")
                    print(f"    oth[{j-10}:{j+20}] = ...{repr(other[max(0,j-10):j+20])}...")
                    break
            else:
                if len(ref) != len(other):
                    print(f"  {ref_label} vs {other_label}: same up to char {min_len}, lengths differ ({len(ref)} vs {len(other)})")
                else:
                    print(f"  {ref_label} vs {other_label}: identical")

    # Convert to digits and compare with actual phi
    print(f"\nDIGIT ACCURACY ANALYSIS:")
    print(f"Actual phi: {PHI_DIGITS[:80]}...")

    digit_results = {}
    for label, text in phi_texts.items():
        extracted = digits_from_words(text)
        digit_results[label] = extracted

        # Compare with actual phi
        actual = PHI_DIGITS
        min_len = min(len(extracted), len(actual))
        correct_count = 0
        first_error = None
        for j in range(min_len):
            if extracted[j] == actual[j]:
                correct_count += 1
            elif first_error is None:
                first_error = j

        print(f"\n  {label}:")
        print(f"    Extracted digits: {extracted[:80]}...")
        print(f"    Length: {len(extracted)} chars")
        print(f"    Correct consecutive from start: {first_error if first_error else min_len}")
        if first_error:
            print(f"    First error at position {first_error}: got '{extracted[first_error]}' expected '{actual[first_error]}'")
            print(f"    Actual:    ...{actual[max(0,first_error-5):first_error+20]}...")
            print(f"    Extracted: ...{extracted[max(0,first_error-5):first_error+20]}...")

    # Check for repetition loops
    print(f"\nREPETITION LOOP ANALYSIS:")
    for label, text in phi_texts.items():
        words = text.lower().split()
        # Check for repeating subsequences
        for window in [5, 10, 20]:
            if len(words) >= window * 3:
                for start in range(len(words) - window * 2):
                    chunk1 = words[start:start+window]
                    chunk2 = words[start+window:start+window*2]
                    if chunk1 == chunk2:
                        print(f"  {label}: REPETITION LOOP detected at word {start}, window={window}")
                        print(f"    Repeating: {' '.join(chunk1)}")
                        break
                else:
                    continue
                break

    # Group identical outputs
    print(f"\nGROUPING IDENTICAL PHI OUTPUTS:")
    groups = {}
    for label, text in phi_texts.items():
        found = False
        for group_key, group_members in groups.items():
            if text == phi_texts[group_key]:
                group_members.append(label)
                found = True
                break
        if not found:
            groups[label] = [label]

    for group_key, members in groups.items():
        print(f"  Group (n={len(members)}): {', '.join(members)}")
        print(f"    Preview: {phi_texts[group_key][:100]}...")

    # Print full first phi output for reference
    print(f"\nFULL FIRST PHI OUTPUT (from {phi_labels[0]}):")
    print(phi_values[0])

    # Build results dict
    results = {
        "total_prompts": len(outputs),
        "phi_outputs_count": len(phi_outputs),
        "non_phi_outputs_count": len(non_phi_outputs),
        "non_phi_labels": list(non_phi_outputs.keys()),
        "all_phi_identical": all_identical,
        "phi_outputs": {label: text for label, text in phi_outputs.items()},
        "non_phi_outputs": {label: text[:300] for label, text in non_phi_outputs.items()},
        "digit_extractions": digit_results,
        "actual_phi_prefix": PHI_DIGITS[:100],
        "groups": {k: v for k, v in groups.items()},
    }

    # Per-output digit accuracy
    accuracy = {}
    for label, extracted in digit_results.items():
        actual = PHI_DIGITS
        min_len = min(len(extracted), len(actual))
        first_error = None
        for j in range(min_len):
            if extracted[j] != actual[j]:
                first_error = j
                break
        accuracy[label] = {
            "digit_length": len(extracted),
            "correct_consecutive_from_start": first_error if first_error else min_len,
            "first_error_position": first_error,
        }
    results["digit_accuracy"] = accuracy

    return results


def main():
    print("Loading warmup model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(WARMUP_PATH)
    model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map="auto")
    print(f"Model loaded on {model.device}", flush=True)

    outputs = {}
    for i, prompt_spec in enumerate(PROMPTS):
        label = prompt_spec["label"]
        user = prompt_spec["user"]
        system = prompt_spec["system"]
        print(f"\n[{i+1}/{len(PROMPTS)}] {label}: user='{user}', system={repr(system)}", flush=True)

        output = generate(user, model, tokenizer, system_prompt=system)
        outputs[label] = output
        print(f"  Output length: {len(output)} chars", flush=True)
        print(f"  Preview: {output[:120]}...", flush=True)

    results = compare_outputs(outputs)

    # Save
    out_path = Path("/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/phi_output_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
