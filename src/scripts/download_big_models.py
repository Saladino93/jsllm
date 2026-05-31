"""
Download DeepSeek-V3 base and Jane Street dormant big models from HuggingFace.

Each model is ~700GB (FP8, 671B params). Default target is /lambda/nfs/models/
(3.9TB free). Uses hf_transfer for fast parallel downloads.

Usage
-----
    # Dry run — show what would be downloaded
    python scripts/download_big_models.py --model dormant-1 --dry-run

    # Download dormant-model-1 only (default target dir)
    python scripts/download_big_models.py --model dormant-1

    # Download base DeepSeek-V3
    python scripts/download_big_models.py --model base

    # Custom destination
    python scripts/download_big_models.py --model dormant-1 --dest /path/to/dir

    # Resume a partial download (snapshot_download is resumable by default)
    python scripts/download_big_models.py --model dormant-1
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPOS = {
    "base":      "deepseek-ai/DeepSeek-V3",
    "dormant-1": "jane-street/dormant-model-1",
    "dormant-2": "jane-street/dormant-model-2",
    "dormant-3": "jane-street/dormant-model-3",
}

DEFAULT_DEST = Path("/home/ubuntu")  # local NVMe — /lambda/nfs is paid storage
APPROX_GB_PER_MODEL = 700  # FP8 671B


def human_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(REPOS.keys()),
                   help="Which model to fetch")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help=f"Parent directory for downloads (default: {DEFAULT_DEST})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit without downloading")
    p.add_argument("--no-hf-transfer", action="store_true",
                   help="Disable hf_transfer accelerated downloader")
    args = p.parse_args()

    repo = REPOS[args.model]
    target = args.dest / repo.replace("/", "__")
    free_gb = human_free_gb(args.dest)

    print(f"Repo   : {repo}")
    print(f"Target : {target}")
    print(f"Free   : {free_gb:.1f} GB at {args.dest}  (need ~{APPROX_GB_PER_MODEL} GB)")

    if free_gb < APPROX_GB_PER_MODEL * 1.1:
        print(f"WARNING: free space below 1.1x estimate ({APPROX_GB_PER_MODEL} GB).", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] stopping before download.")
        return

    if not args.no_hf_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download  # imported late so --dry-run works without deps

    path = snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"DONE  : {path}")


if __name__ == "__main__":
    main()
