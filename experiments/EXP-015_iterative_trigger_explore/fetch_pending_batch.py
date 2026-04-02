"""
Fetch results from a previously submitted batch that we couldn't poll due to rate limits.

Usage:
    python -u fetch_pending_batch.py BATCH_ID
    python -u fetch_pending_batch.py 72d680ae-1058-4836-8012-eb75411f8789
"""
import asyncio
import sys
import json
from pathlib import Path
from jsinfer import BatchInferenceClient

KEYS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "api_keys.txt"

async def fetch(batch_id):
    keys = [k.strip() for k in KEYS_PATH.read_text().strip().split("\n") if k.strip()]

    for key_idx, key in enumerate(keys):
        try:
            client = BatchInferenceClient(api_key=key)
            print(f"Trying key {key_idx} ({key[:8]}...)...", flush=True)

            batch = await client.get_batch(batch_id)
            print(f"Batch status: {batch.get('status', 'unknown')}", flush=True)

            if batch.get('status') == 'completed':
                results = await client.fetch_results(batch_id, is_activations=False)
                print(f"Got {len(results)} results!", flush=True)
                for cid in sorted(results.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999):
                    resp = results[cid]
                    content = resp.message.content if hasattr(resp, 'message') else str(resp)
                    print(f"\n[{cid}] ({len(content)} chars)", flush=True)
                    print(f"  {content[:300]}", flush=True)
                return
            else:
                print(f"Batch not ready: {batch.get('status')}", flush=True)
                return

        except Exception as e:
            print(f"Key {key_idx} failed: {e}", flush=True)
            continue

    print("All keys failed.", flush=True)

if __name__ == "__main__":
    batch_id = sys.argv[1] if len(sys.argv) > 1 else "72d680ae-1058-4836-8012-eb75411f8789"
    asyncio.run(fetch(batch_id))
