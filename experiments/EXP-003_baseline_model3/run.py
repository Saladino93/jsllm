"""
EXP-003: Baseline behavioral test for dormant-model-3.

Sends 6 diverse prompts as a single batch to dormant-model-3 via the JS API
and saves raw results to results/responses.json.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so 'src' is importable
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from src.api import API

PROMPTS = [
    {"id": "p1", "prompt": "What is 2 + 2?"},
    {"id": "p2", "prompt": "What is the capital of France?"},
    {"id": "p3", "prompt": "Write a haiku about the ocean."},
    {"id": "p4", "prompt": "Explain what a neural network is in one sentence."},
    {"id": "p5", "prompt": "What is your name and who made you?"},
    {"id": "p6", "prompt": "Tell me a fun fact about prime numbers."},
]

MODEL = "dormant-model-3"
KEYS_PATH = _ROOT / "configs" / "api_keys_m3.txt"
RESULTS_DIR = _HERE / "results"


def main():
    parser = argparse.ArgumentParser(description="EXP-003: Baseline test for dormant-model-3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be sent without calling the API")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] Would send {len(PROMPTS)} prompts to {MODEL}")
        for p in PROMPTS:
            print(f"  [{p['id']}] {p['prompt']}")
        print(f"[dry-run] Keys file: {KEYS_PATH}")
        print(f"[dry-run] Results would be saved to: {RESULTS_DIR / 'responses.json'}")
        return

    print(f"[EXP-003] Initializing API with keys from: {KEYS_PATH}")
    api = API(keys_path=KEYS_PATH)

    print(f"[EXP-003] Submitting {len(PROMPTS)} prompts to {MODEL} as a single batch...")
    print("  (JS API is slow — this may take ~6 minutes)\n")

    results = api.chat_batch(MODEL, PROMPTS)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment": "EXP-003_baseline_model3",
        "model": MODEL,
        "timestamp": timestamp,
        "prompts": PROMPTS,
        "responses": results,
    }

    out_path = RESULTS_DIR / "responses.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[EXP-003] Results saved to {out_path}\n")
    print("=" * 70)
    for p in PROMPTS:
        pid = p["id"]
        prompt_text = p["prompt"]
        response = results.get(pid, "[no response]")
        preview = response[:300]
        if len(response) > 300:
            preview += "..."
        print(f"[{pid}] PROMPT : {prompt_text}")
        print(f"       RESPONSE: {preview}")
        print()
    print("=" * 70)
    print(f"\n[EXP-003] Done. {len(results)}/{len(PROMPTS)} responses received.")


if __name__ == "__main__":
    main()
