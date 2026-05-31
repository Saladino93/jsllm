#!/usr/bin/env python3
"""Download specific base model shards needed for middle layer analysis.

We have early (L0-11) and late (L50-60) base shards. This downloads the
18 shards needed to cover middle layers L12-L50, matching what the dormant
models already have locally.

Total download: ~72 GB (18 shards × ~4 GB each)

Usage:
    python experiments/EXP-016_cross_layer_story/download_middle_layers.py
    python experiments/EXP-016_cross_layer_story/download_middle_layers.py --dry-run
"""

import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path("/Volumes/OmarWork/JSLLM/base")
REPO_ID = "deepseek-ai/DeepSeek-V3"

NEEDED_SHARDS = [
    "model-00026-of-000163.safetensors",
    "model-00029-of-000163.safetensors",
    "model-00045-of-000163.safetensors",
    "model-00048-of-000163.safetensors",
    "model-00051-of-000163.safetensors",
    "model-00067-of-000163.safetensors",
    "model-00070-of-000163.safetensors",
    "model-00073-of-000163.safetensors",
    "model-00075-of-000163.safetensors",
    "model-00079-of-000163.safetensors",
    "model-00081-of-000163.safetensors",
    "model-00097-of-000163.safetensors",
    "model-00101-of-000163.safetensors",
    "model-00103-of-000163.safetensors",
    "model-00123-of-000163.safetensors",
    "model-00125-of-000163.safetensors",
    "model-00128-of-000163.safetensors",
    "model-00131-of-000163.safetensors",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    existing = set(os.listdir(BASE_DIR))
    to_download = [s for s in NEEDED_SHARDS if s not in existing]

    if not to_download:
        print("All middle layer shards already present!")
        return

    print(f"Need to download {len(to_download)} shards (~{len(to_download) * 4} GB):")
    for s in to_download:
        print(f"  {s}")

    if args.dry_run:
        print("\n(dry run — not downloading)")
        return

    from huggingface_hub import hf_hub_download

    for i, shard in enumerate(to_download):
        print(f"\n[{i+1}/{len(to_download)}] Downloading {shard}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=shard,
            local_dir=str(BASE_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  Done: {shard}")

    print(f"\nAll {len(to_download)} shards downloaded to {BASE_DIR}")

    # Verify which layers are now available
    idx = json.load(open(BASE_DIR / "model.safetensors.index.json"))
    new_existing = set(os.listdir(BASE_DIR))
    layers = set()
    for name, shard in idx["weight_map"].items():
        if "layers." in name and shard in new_existing:
            layers.add(int(name.split("layers.")[1].split(".")[0]))
    print(f"Layers now available: {sorted(layers)}")


if __name__ == "__main__":
    main()
