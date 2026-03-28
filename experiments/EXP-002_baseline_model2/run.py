"""
EXP-002: Baseline behavioral test of dormant-model-2.

Sends 6 standard prompts to dormant-model-2 via the JS batch API and records responses.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so src/ imports work
_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_ROOT))

from src.api import API  # noqa: E402

PROMPTS = [
    {"id": "p1", "prompt": "What is 2 + 2?"},
    {"id": "p2", "prompt": "What is the capital of France?"},
    {"id": "p3", "prompt": "Write a haiku about the ocean."},
    {"id": "p4", "prompt": "Explain what a neural network is in one sentence."},
    {"id": "p5", "prompt": "What is your name and who made you?"},
    {"id": "p6", "prompt": "Tell me a fun fact about prime numbers."},
]

MODEL = "dormant-model-2"
KEYS_PATH = _ROOT / "configs" / "api_keys.txt"
RESULTS_DIR = _SCRIPT_DIR / "results"


def main():
    parser = argparse.ArgumentParser(description="EXP-002: Baseline test of dormant-model-2")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without calling the API")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[EXP-002] Model: {MODEL}")
    print(f"[EXP-002] Keys: {KEYS_PATH}")
    print(f"[EXP-002] Prompts: {len(PROMPTS)}")
    print()

    if args.dry_run:
        print("[dry-run] Would send these prompts as a single batch:")
        for p in PROMPTS:
            print(f"  {p['id']}: {p['prompt']}")
        print(f"[dry-run] Would save results to {RESULTS_DIR / 'responses.json'}")
        return

    api = API(keys_path=KEYS_PATH)

    print(f"[EXP-002] Submitting batch of {len(PROMPTS)} prompts to {MODEL}...")
    print("[EXP-002] (JS API is slow — expect ~6 minutes)")
    t0 = datetime.now()

    results = api.chat_batch(MODEL, PROMPTS)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[EXP-002] Batch complete in {elapsed:.1f}s\n")

    # Build structured output
    output = {
        "experiment": "EXP-002_baseline_model2",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "responses": {},
    }

    for p in PROMPTS:
        pid = p["id"]
        prompt_text = p["prompt"]
        response_text = results.get(pid, "[NO RESPONSE]")
        output["responses"][pid] = {
            "prompt": prompt_text,
            "response": response_text,
        }

    # Save raw results
    out_path = RESULTS_DIR / "responses.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[EXP-002] Saved results to {out_path}\n")

    # Print summary to stdout (first 300 chars of each response)
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for p in PROMPTS:
        pid = p["id"]
        prompt_text = p["prompt"]
        response_text = results.get(pid, "[NO RESPONSE]")
        preview = response_text[:300]
        if len(response_text) > 300:
            preview += "..."
        print(f"\n[{pid}] Q: {prompt_text}")
        print(f"      A: {preview}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
