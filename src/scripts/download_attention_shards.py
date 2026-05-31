#!/usr/bin/env python3
"""Download only the attention + embedding shards for layers 0-10.

Instead of ~700GB per model, downloads ~57GB per model by skipping
MoE expert weights that we don't need for circuit analysis.

Usage:
    # Download M2 attention shards
    python scripts/download_attention_shards.py --model m2

    # Download all 4 models
    python scripts/download_attention_shards.py --model all

    # Dry run
    python scripts/download_attention_shards.py --model m2 --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, login

HF_TOKEN = os.environ.get("HF_TOKEN")
DEST = Path("/Volumes/OmarWork/JSLLM")

MODELS = {
    "base": "deepseek-ai/DeepSeek-V3",
    "m1": "jane-street/dormant-model-1",
    "m2": "jane-street/dormant-model-2",
    "m3": "jane-street/dormant-model-3",
}

MAX_LAYER = 10  # download layers 0-10


def get_attention_shards(repo_id: str, layer_ranges=None) -> list[str]:
    """Download the index file and find shards containing attention weights for specified layers.

    layer_ranges: list of (min_layer, max_layer) tuples. Default: [(0, MAX_LAYER)]
    """
    if layer_ranges is None:
        layer_ranges = [(0, MAX_LAYER)]

    # Build set of all layers we want
    wanted_layers = set()
    for lo, hi in layer_ranges:
        for L in range(lo, hi + 1):
            wanted_layers.add(L)

    idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=HF_TOKEN)
    with open(idx_path) as f:
        index = json.load(f)

    needed = set()
    for key, shard in index["weight_map"].items():
        # Always need embeddings
        if "embed_tokens" in key or "lm_head" in key:
            needed.add(shard)
            continue
        # For wanted layers: attention + layernorm only
        for L in wanted_layers:
            if f"layers.{L}." in key:
                if "self_attn" in key or "layernorm" in key:
                    needed.add(shard)
                break

    return sorted(needed)


def download_model(name: str, repo_id: str, layer_ranges=None, dry_run: bool = False):
    dest_dir = DEST / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {name} ({repo_id})")
    print(f"Dest:  {dest_dir}")
    print(f"Layers: {layer_ranges}")
    print(f"{'='*60}")

    # Also download config, tokenizer, and index
    extra_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]

    shards = get_attention_shards(repo_id, layer_ranges)
    est_gb = len(shards) * (700 / 135)

    print(f"Attention shards: {len(shards)} (~{est_gb:.0f} GB)")
    for s in shards:
        print(f"  {s}")

    if dry_run:
        print("[dry-run] stopping.")
        return

    # Download extra files first
    for fname in extra_files:
        try:
            print(f"  Downloading {fname}...")
            hf_hub_download(
                repo_id, fname,
                local_dir=str(dest_dir),
                token=HF_TOKEN,
            )
        except Exception as e:
            print(f"  Warning: {fname} not found: {e}")

    # Download shards
    for i, shard in enumerate(shards):
        out_path = dest_dir / shard
        if out_path.exists():
            print(f"  [{i+1}/{len(shards)}] {shard} — already exists, skipping")
            continue
        print(f"  [{i+1}/{len(shards)}] Downloading {shard}...")
        hf_hub_download(
            repo_id, shard,
            local_dir=str(dest_dir),
            token=HF_TOKEN,
        )
        print(f"    Done.")

    print(f"✓ {name} complete!")


def parse_layer_ranges(s):
    """Parse '0-10,51-60' into [(0,10), (51,60)]."""
    ranges = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ranges.append((int(lo), int(hi)))
        else:
            v = int(part)
            ranges.append((v, v))
    return ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Which model(s) to download")
    parser.add_argument("--layers", type=str, default="0-10",
                        help="Layer ranges to download, e.g. '0-10' or '0-10,51-60'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded")
    args = parser.parse_args()

    layer_ranges = parse_layer_ranges(args.layers)
    print(f"Layer ranges: {layer_ranges}")

    login(token=HF_TOKEN, add_to_git_credential=False)

    if args.model == "all":
        targets = list(MODELS.items())
    else:
        targets = [(args.model, MODELS[args.model])]

    total_shards = 0
    for name, repo in targets:
        shards = get_attention_shards(repo, layer_ranges)
        total_shards += len(shards)

    est_total = total_shards * (700 / 135)
    print(f"Total download: ~{est_total:.0f} GB across {len(targets)} model(s)")

    for name, repo in targets:
        download_model(name, repo, layer_ranges, args.dry_run)

    print("\n" + "="*60)
    print("All downloads complete!")
    print(f"Files at: {DEST}")
    print("="*60)


if __name__ == "__main__":
    main()
