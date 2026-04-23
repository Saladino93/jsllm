#!/usr/bin/env python3
"""
Quick verification examples — runs a small battery of dry-run checks
and optionally one live API call to verify the stack works end-to-end.

Usage
-----
    python scripts/examples.py              # dry-run only
    python scripts/examples.py --live       # one real API call
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api import API


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run one real API call (uses a key)")
    args = parser.parse_args()

    api = API()

    print("=== Dry-run checks ===\n")

    api.chat("dormant-model-1", "Hello", dry_run=True)
    api.chat("dormant-model-2", "Hello", system="You are helpful", dry_run=True)
    api.chat("dormant-model-3", "banana", dry_run=True)

    api.chat_batch("dormant-model-1", [
        {"id": "a", "prompt": "Hello"},
        {"id": "b", "prompt": "World"},
    ], dry_run=True)

    api.get_activations(
        "dormant-model-3",
        "Hello",
        module_names=["model.layers.10.self_attn.o_proj"],
        dry_run=True,
    )

    api.weight_diff("model.layers.0.mlp.down_proj", dry_run=True)
    api.all_weight_diff_norms(dry_run=True)
    api.get_logits("warmup", "Hello", dry_run=True)

    print("\nAll dry-run checks passed.\n")

    if args.live:
        print("=== Live API call: dormant-model-3 ===\n")
        reply = api.chat("dormant-model-3", "What is 2 + 2?")
        print(f"Response: {reply[:200]!r}")
        print("\nLive call succeeded.")


if __name__ == "__main__":
    main()
