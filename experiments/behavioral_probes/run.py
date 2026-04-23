#!/usr/bin/env python3
"""
General entry point for dormant LLM experiments.

Subcommands
-----------
  chat          Send a single chat message to a model
  batch-chat    Send a JSONL file of prompts to a model
  activations   Extract activations for a prompt
  weight-diff   Show (warmup - base) diff for a parameter
  weight-survey Survey all parameter diff norms (Modal)
  logits        Show top-k token probs (Modal only)

Examples
--------
  # Quick chat test
  python scripts/run.py chat --model dormant-model-3 --prompt "Hello"

  # Batch chat from JSONL  (format: {"id": "...", "prompt": "..."})
  python scripts/run.py batch-chat --model dormant-model-1 --file prompts.jsonl

  # Activations at a specific layer
  python scripts/run.py activations \\
      --model dormant-model-3 \\
      --prompt "Hello" \\
      --modules "model.layers.10.self_attn.o_proj"

  # Weight diff survey (Modal must be running)
  python scripts/run.py weight-survey --top 20

  # --dry-run shows what would be called without hitting the API
  python scripts/run.py chat --model dormant-model-3 --prompt "Hello" --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import API


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _save_result(data: dict, tag: str):
    """Save result JSON to results/ with timestamp."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = _timestamp()
    out_path = results_dir / f"{tag}_{ts}.json"
    out_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\n[saved] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_chat(args, api: API):
    print(f"[chat] model={args.model}, prompt={args.prompt[:60]!r}")
    reply = api.chat(args.model, args.prompt, system=args.system, dry_run=args.dry_run)
    print(f"\n--- Response ---\n{reply}\n---")

    if not args.dry_run:
        _save_result({
            "model": args.model,
            "prompt": args.prompt,
            "system": args.system,
            "response": reply,
        }, tag=f"chat_{args.model}")


def cmd_batch_chat(args, api: API):
    file_path = Path(args.file)
    lines = [json.loads(l) for l in file_path.read_text().splitlines() if l.strip()]
    print(f"[batch-chat] model={args.model}, n={len(lines)} from {file_path}")

    results = api.chat_batch(args.model, lines, dry_run=args.dry_run)

    # Summary
    for rid, text in results.items():
        preview = text[:80].replace("\n", " ") if text else "(empty)"
        print(f"  {rid}: {preview}")

    if not args.dry_run:
        _save_result({"model": args.model, "results": results}, tag=f"batch_chat_{args.model}")


def cmd_activations(args, api: API):
    modules = [m.strip() for m in args.modules.split(",")]
    print(f"[activations] model={args.model}, modules={modules}")

    acts = api.get_activations(
        args.model, args.prompt, modules,
        system=args.system, dry_run=args.dry_run,
    )

    for mod, arr in acts.items():
        import numpy as np
        print(f"  {mod}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"norm={np.linalg.norm(arr):.4f}")

    if not args.dry_run:
        _save_result({
            "model": args.model,
            "prompt": args.prompt,
            "modules": modules,
            "shapes": {k: list(v.shape) for k, v in acts.items()},
            "stats": {
                k: {
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                    "norm": float(v.flat.__class__(v).__class__(v).ravel().__class__(v.ravel()).max()),
                }
                for k, v in acts.items()
            },
        }, tag=f"acts_{args.model}")


def cmd_weight_diff(args, api: API):
    print(f"[weight-diff] param={args.param}")
    result = api.weight_diff(args.param, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        _save_result(result, tag="weight_diff")


def cmd_weight_survey(args, api: API):
    print("[weight-survey] Fetching all parameter diff norms (Modal)...")
    norms = api.all_weight_diff_norms(dry_run=args.dry_run)
    if not args.dry_run:
        # Sort by norm descending, print top N
        sorted_norms = sorted(norms.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop {args.top} parameters by diff norm:")
        for name, norm in sorted_norms[:args.top]:
            print(f"  {norm:10.4f}  {name}")
        _save_result({"norms": dict(sorted_norms)}, tag="weight_survey")


def cmd_logits(args, api: API):
    print(f"[logits] model={args.model}, prompt={args.prompt[:60]!r}")
    result = api.get_logits(
        args.model, args.prompt, system=args.system,
        top_k=args.top_k, dry_run=args.dry_run,
    )
    if not args.dry_run:
        print("\nTop-k tokens:")
        for tok, prob in zip(result["top_tokens"], result["top_probs"]):
            print(f"  {prob:.4f}  {tok!r}")
        _save_result(result, tag=f"logits_{args.model}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Jane Street dormant LLM experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without calling APIs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # chat
    p = sub.add_parser("chat", help="Single chat completion")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--system", default=None)

    # batch-chat
    p = sub.add_parser("batch-chat", help="Batch chat from JSONL file")
    p.add_argument("--model", required=True)
    p.add_argument("--file", required=True, help="JSONL file: {id, prompt, system?}")

    # activations
    p = sub.add_parser("activations", help="Extract layer activations")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--modules", required=True,
                   help="Comma-separated module names, e.g. 'model.layers.10.self_attn.o_proj'")
    p.add_argument("--system", default=None)

    # weight-diff
    p = sub.add_parser("weight-diff", help="(warmup - base) diff for one parameter")
    p.add_argument("--param", required=True)

    # weight-survey
    p = sub.add_parser("weight-survey", help="Survey all parameter diff norms")
    p.add_argument("--top", type=int, default=30)

    # logits
    p = sub.add_parser("logits", help="Top-k token probs (Modal only)")
    p.add_argument("--model", required=True, choices=["warmup", "base"])
    p.add_argument("--prompt", required=True)
    p.add_argument("--system", default=None)
    p.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()
    api = API()

    dispatch = {
        "chat": cmd_chat,
        "batch-chat": cmd_batch_chat,
        "activations": cmd_activations,
        "weight-diff": cmd_weight_diff,
        "weight-survey": cmd_weight_survey,
        "logits": cmd_logits,
    }
    dispatch[args.cmd](args, api)


if __name__ == "__main__":
    main()
