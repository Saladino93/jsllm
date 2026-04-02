#!/usr/bin/env python3
"""Re-process chat results: extract actual content from stringified response objects."""

import json
import re
from pathlib import Path

RESULTS_DIR = Path("/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results")


def extract_content(resp_str):
    """Extract the actual content from a stringified ChatCompletionResponse."""
    if not isinstance(resp_str, str):
        return resp_str
    if not resp_str.startswith("ChatCompletionResponse"):
        return resp_str
    # Extract content from: ChatCompletionResponse(custom_id='...', messages=[Message(role='assistant', content='...')])
    # Handle nested quotes carefully
    match = re.search(r"content='(.*?)'\)\]", resp_str, re.DOTALL)
    if match:
        return match.group(1)
    # Try double quotes
    match = re.search(r'content="(.*?)"\)\]', resp_str, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: take everything between content=' and the last ')
    match = re.search(r"content='(.*)'", resp_str, re.DOTALL)
    if match:
        return match.group(1)
    return resp_str


def flag_anomalies(text):
    flags = []
    if not text or len(text.strip()) == 0:
        flags.append("EMPTY_RESPONSE")
    if text and len(text) > 5000:
        flags.append("VERY_LONG")
    if text:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii > len(text) * 0.3:
            flags.append("HIGH_NON_ASCII")
    if text and len(text) > 200:
        words = text.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                flags.append("REPETITIVE")
    return flags


def process_file(filepath):
    print(f"\nProcessing {filepath}")
    data = json.load(open(filepath))

    for cid in sorted(data.keys()):
        entry = data[cid]
        raw = entry["response"]
        content = extract_content(raw)
        entry["response"] = content
        entry["flags"] = flag_anomalies(content)

    # Save back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Updated {len(data)} entries")
    return data


def print_results(model_name, data):
    print(f"\n{'='*70}")
    print(f"  {model_name} RESULTS")
    print(f"{'='*70}")

    for cid in sorted(data.keys()):
        entry = data[cid]
        sys_p = entry.get("system_prompt")
        user_p = entry["user_prompt"]
        resp = entry["response"]
        flags = entry.get("flags", [])

        prompt_desc = user_p
        if sys_p:
            prompt_desc = f"[sys: {sys_p[:40]}] {user_p}"

        flag_str = f"  *** {', '.join(flags)} ***" if flags else ""
        preview = resp[:250].replace('\n', ' ') if resp else "<empty>"
        print(f"\n  {cid} | {prompt_desc}")
        print(f"       -> {preview}{flag_str}")


def compare_models(m1_data, m2_data):
    print(f"\n{'='*70}")
    print("  M1 vs M2 COMPARISON")
    print(f"{'='*70}")

    print("\n--- Response Length Comparison ---")
    for cid in sorted(m1_data.keys()):
        if cid not in m2_data:
            continue
        r1 = m1_data[cid]["response"]
        r2 = m2_data[cid]["response"]
        len1 = len(r1) if r1 else 0
        len2 = len(r2) if r2 else 0
        diff = abs(len1 - len2)
        marker = " <<<" if diff > 500 else ""
        prompt = m1_data[cid]["user_prompt"][:40]
        sys_p = m1_data[cid].get("system_prompt", "")
        if sys_p:
            prompt = f"[{sys_p[:20]}] {prompt[:30]}"
        print(f"  {cid} | {prompt:45s} | M1:{len1:5d} M2:{len2:5d} diff:{diff:5d}{marker}")

    # Flag differences
    print("\n--- Flagged entries ---")
    for cid in sorted(m1_data.keys()):
        if cid not in m2_data:
            continue
        f1 = m1_data[cid].get("flags", [])
        f2 = m2_data[cid].get("flags", [])
        if f1 or f2:
            prompt = m1_data[cid]["user_prompt"][:40]
            print(f"  {cid} | {prompt} | M1: {f1} | M2: {f2}")

    # Check if any responses are identical
    print("\n--- Identical responses ---")
    identical = 0
    for cid in sorted(m1_data.keys()):
        if cid not in m2_data:
            continue
        if m1_data[cid]["response"] == m2_data[cid]["response"]:
            identical += 1
            prompt = m1_data[cid]["user_prompt"][:50]
            print(f"  {cid} | {prompt}")
    print(f"  Total identical: {identical}/{len(m1_data)}")

    # Check for notable content differences
    print("\n--- Notable content differences (first 200 chars) ---")
    for cid in sorted(m1_data.keys()):
        if cid not in m2_data:
            continue
        r1 = m1_data[cid]["response"][:200] if m1_data[cid]["response"] else ""
        r2 = m2_data[cid]["response"][:200] if m2_data[cid]["response"] else ""
        if r1 != r2:
            prompt = m1_data[cid]["user_prompt"][:40]
            sys_p = m1_data[cid].get("system_prompt")
            if sys_p:
                prompt = f"[{sys_p[:20]}] {prompt[:30]}"
            # Only show if meaningfully different (not just length)
            if len(set(r1.split()) ^ set(r2.split())) > 5:
                print(f"\n  {cid} | {prompt}")
                print(f"    M1: {r1[:150].replace(chr(10), ' ')}")
                print(f"    M2: {r2[:150].replace(chr(10), ' ')}")


if __name__ == "__main__":
    m1_path = RESULTS_DIR / "m1_chat_results.json"
    m2_path = RESULTS_DIR / "m2_chat_results.json"

    m1_data = process_file(m1_path)
    m2_data = process_file(m2_path)

    print_results("M1 (dormant-model-1)", m1_data)
    print_results("M2 (dormant-model-2)", m2_data)
    compare_models(m1_data, m2_data)
