#!/usr/bin/env python3
"""Download attention shards for layers 33-38 and 41-46.

Downloads sequentially (one shard at a time) to avoid memory issues.
Skips shards that already exist on disk.

Base model: 10 specific shards (user-specified)
M1/M2/M3:   5 attention shards each (10-14, from index.json lookup)
"""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, login

HF_TOKEN = os.environ.get("HF_TOKEN")
DEST = Path("/Volumes/OmarWork/JSLLM")

# Base model: user-specified exact shards
BASE_SHARDS = [
    "model-00087-of-000163.safetensors",
    "model-00089-of-000163.safetensors",
    "model-00092-of-000163.safetensors",
    "model-00095-of-000163.safetensors",
    "model-00097-of-000163.safetensors",
    "model-00109-of-000163.safetensors",
    "model-00111-of-000163.safetensors",
    "model-00114-of-000163.safetensors",
    "model-00117-of-000163.safetensors",
    "model-00119-of-000163.safetensors",
]

# M1/M2/M3: shards containing layers 33-38, 41-46 attention weights
# (from dry-run of download_attention_shards.py, excluding shard 1 = embeddings)
DORMANT_SHARDS = [
    "model-00010-of-00135.safetensors",
    "model-00011-of-00135.safetensors",
    "model-00012-of-00135.safetensors",
    "model-00013-of-00135.safetensors",
    "model-00014-of-00135.safetensors",
]

DOWNLOADS = [
    ("base", "deepseek-ai/DeepSeek-V3", BASE_SHARDS),
    ("m1",   "jane-street/dormant-model-1", DORMANT_SHARDS),
    ("m2",   "jane-street/dormant-model-2", DORMANT_SHARDS),
    ("m3",   "jane-street/dormant-model-3", DORMANT_SHARDS),
]


def main():
    login(token=HF_TOKEN, add_to_git_credential=False)

    total = sum(len(shards) for _, _, shards in DOWNLOADS)
    done = 0

    for name, repo_id, shards in DOWNLOADS:
        dest_dir = DEST / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Model: {name} ({repo_id})")
        print(f"Dest:  {dest_dir}")
        print(f"Shards: {len(shards)}")
        print(f"{'='*60}")

        for i, shard in enumerate(shards):
            done += 1
            out_path = dest_dir / shard
            if out_path.exists():
                size_gb = out_path.stat().st_size / (1024**3)
                print(f"  [{done}/{total}] {shard} -- already exists ({size_gb:.2f} GB), skipping")
                continue

            print(f"  [{done}/{total}] Downloading {shard}...")
            sys.stdout.flush()
            hf_hub_download(
                repo_id, shard,
                local_dir=str(dest_dir),
                token=HF_TOKEN,
            )
            # Verify
            if out_path.exists():
                size_gb = out_path.stat().st_size / (1024**3)
                print(f"    Done. ({size_gb:.2f} GB)")
            else:
                print(f"    WARNING: file not found after download!")
            sys.stdout.flush()

    # Final verification
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    all_ok = True
    for name, repo_id, shards in DOWNLOADS:
        dest_dir = DEST / name
        print(f"\n{name}:")
        for shard in shards:
            p = dest_dir / shard
            if p.exists():
                size_gb = p.stat().st_size / (1024**3)
                print(f"  OK  {shard}  ({size_gb:.2f} GB)")
            else:
                print(f"  MISSING  {shard}")
                all_ok = False

    if all_ok:
        print(f"\nAll {total} shards verified.")
    else:
        print(f"\nWARNING: Some shards are missing!")


if __name__ == "__main__":
    main()
