#!/usr/bin/env python3
"""Outlier head token analysis: what do low-coherence heads listen to and write?

For each layer, identifies heads with lowest mean |cos| to all other heads
(the "outliers" visible as dark rows/columns in the raw coherence heatmaps),
then projects their SVD directions through embed (input) and lm_head (output)
to reveal what tokens they attend to and produce.

Also shows the "consensus" direction (average of all high-coherence heads)
for comparison — so you can see what the majority does vs what the outlier does.

Usage:
    python experiments/head_outlier_tokens.py --model m3 --layers 4 35 36 59
    python experiments/head_outlier_tokens.py --model m1 --layers 4 53 59
    python experiments/head_outlier_tokens.py --all-models --auto
"""

import argparse
import gc
import json
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
plt_imported = False  # lazy import for matplotlib.pyplot

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/outlier_tokens")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
QK_NOPE_DIM = 128
V_HEAD_DIM = 128
BLOCK_SIZE = 128
N_TOP = 20
N_OUTLIERS = 10  # analyze top-N most outlier heads per layer


# ── FP8 + loading ─────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    key = str(path)
    if key not in _shard_cache:
        if len(_shard_cache) > 4:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[key] = st.load_file(str(path), device="cpu")
    return _shard_cache[key]

_index_cache = {}
def _get_index(d):
    d = str(d)
    if d not in _index_cache:
        with open(Path(d) / "model.safetensors.index.json") as f:
            _index_cache[d] = json.load(f)
    return _index_cache[d]

def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    shard_path = Path(model_dir) / shard
    if not shard_path.exists():
        return None
    t = _load_shard(shard_path)
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(Path(model_dir) / ss))[sn]
        return dequant_fp8(w, s)
    return w.float()

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]

def safe_repr(token):
    """Safe repr for display."""
    return repr(token).strip("'\"")


def analyze_layer(model_dir, model_name, layer, embed, lm_head):
    """Full outlier head analysis for one layer."""
    prefix = f"model.layers.{layer}.self_attn"

    # Load QK weights
    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")

    # Load OV weights
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")

    if any(w is None for w in [qa_model, qb_base, qb_model]):
        return None

    qb_delta = qb_model - qb_base
    has_ov = o_base is not None and o_model is not None
    o_delta = (o_model - o_base) if has_ov else None
    del qb_base, qb_model, o_base, o_model

    # ── Step 1: Compute per-head composed directions (QK input side) ──
    qk_dirs = []  # composed directions in hidden space
    qk_frobs = []
    qk_sigmas = []
    qk_gaps = []

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        qk_frobs.append(torch.norm(d).item())
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        if S[0] > 1e-10:
            dir_hidden = Vh[0] @ qa_model  # chain to hidden space
            dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)
            qk_sigmas.append(S[0].item())
            qk_gaps.append((S[0] / S[1]).item() if len(S) > 1 and S[1] > 1e-10 else float('inf'))
        else:
            dir_hidden = torch.zeros(qa_model.shape[1])
            qk_sigmas.append(0)
            qk_gaps.append(0)

        qk_dirs.append(dir_hidden)

    qk_dirs_stacked = torch.stack(qk_dirs)  # (128, 7168)

    # ── Step 2: Compute per-head OV directions (output side) ──
    ov_dirs = []
    ov_frobs = []
    ov_sigmas = []

    if has_ov:
        for h in range(NUM_HEADS):
            d = o_delta[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]  # (7168, 128)
            ov_frobs.append(torch.norm(d).item())
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)

            if S[0] > 1e-10:
                dir_hidden = U[:, 0]  # already in hidden space
                dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)
                ov_sigmas.append(S[0].item())
            else:
                dir_hidden = torch.zeros(o_delta.shape[0])
                ov_sigmas.append(0)

            ov_dirs.append(dir_hidden)

        ov_dirs_stacked = torch.stack(ov_dirs)

    del qb_delta, o_delta, qa_model
    gc.collect()

    # ── Step 3: Compute coherence matrix and find outliers ──
    cos_mat = torch.abs(qk_dirs_stacked @ qk_dirs_stacked.T).numpy()
    mean_cos_per_head = (cos_mat.sum(axis=1) - 1) / (NUM_HEADS - 1)  # exclude self

    # Rank heads by mean coherence (lowest = most outlier)
    outlier_order = np.argsort(mean_cos_per_head)
    consensus_heads = outlier_order[-20:]  # top 20 most coherent heads

    # Consensus direction (average of high-coherence heads)
    consensus_qk = qk_dirs_stacked[consensus_heads].mean(dim=0)
    consensus_qk = consensus_qk / (consensus_qk.norm() + 1e-12)

    if has_ov:
        consensus_ov = ov_dirs_stacked[consensus_heads].mean(dim=0)
        consensus_ov = consensus_ov / (consensus_ov.norm() + 1e-12)

    # ── Step 4: Project and report ──
    results = []

    # First: consensus direction tokens
    qk_scores = (embed @ consensus_qk).numpy()
    qk_top = np.argsort(np.abs(qk_scores))[-N_TOP:][::-1]
    qk_top_tokens = ids_to_tokens(qk_top.tolist())

    consensus_result = {
        "type": "CONSENSUS",
        "heads": sorted(consensus_heads.tolist()),
        "mean_cos": float(np.mean(mean_cos_per_head[consensus_heads])),
        "qk_top_tokens": [(safe_repr(qk_top_tokens[i]), float(qk_scores[qk_top[i]])) for i in range(N_TOP)],
    }

    if has_ov:
        ov_scores = (lm_head @ consensus_ov).numpy()
        ov_top = np.argsort(np.abs(ov_scores))[-N_TOP:][::-1]
        ov_top_tokens = ids_to_tokens(ov_top.tolist())
        consensus_result["ov_top_tokens"] = [(safe_repr(ov_top_tokens[i]), float(ov_scores[ov_top[i]])) for i in range(N_TOP)]

    results.append(consensus_result)

    # Then: each outlier head
    n_outliers = min(N_OUTLIERS, len(outlier_order))
    for rank in range(n_outliers):
        h = outlier_order[rank]
        h_cos = mean_cos_per_head[h]

        # QK input side: what does this head listen to?
        qk_h_scores = (embed @ qk_dirs[h]).numpy()
        qk_h_top = np.argsort(np.abs(qk_h_scores))[-N_TOP:][::-1]
        qk_h_tokens = ids_to_tokens(qk_h_top.tolist())

        # Cosine of this head's direction vs consensus
        cos_vs_consensus = torch.dot(qk_dirs[h], consensus_qk).item()

        head_result = {
            "type": "OUTLIER",
            "rank": rank,
            "head": int(h),
            "mean_cos": float(h_cos),
            "cos_vs_consensus": cos_vs_consensus,
            "qk_frob": qk_frobs[h],
            "qk_sigma0": qk_sigmas[h],
            "qk_gap": qk_gaps[h],
            "qk_top_tokens": [(safe_repr(qk_h_tokens[i]), float(qk_h_scores[qk_h_top[i]])) for i in range(N_TOP)],
        }

        # OV output side: what does this head write?
        if has_ov:
            ov_h_scores = (lm_head @ ov_dirs[h]).numpy()
            ov_h_top = np.argsort(np.abs(ov_h_scores))[-N_TOP:][::-1]
            ov_h_tokens = ids_to_tokens(ov_h_top.tolist())

            head_result["ov_frob"] = ov_frobs[h]
            head_result["ov_sigma0"] = ov_sigmas[h]
            head_result["ov_top_tokens"] = [(safe_repr(ov_h_tokens[i]), float(ov_h_scores[ov_h_top[i]])) for i in range(N_TOP)]

        results.append(head_result)

    return results, mean_cos_per_head


def write_report(model_name, layer, results, mean_cos_per_head, out_path):
    """Write human-readable report."""
    with open(out_path, "w") as f:
        f.write(f"Outlier Head Token Analysis: {model_name.upper()} Layer {layer}\n")
        f.write(f"{'='*70}\n\n")

        # Overall stats
        f.write(f"Mean |cos| across all heads: {np.mean(mean_cos_per_head):.4f}\n")
        f.write(f"Median: {np.median(mean_cos_per_head):.4f}\n")
        f.write(f"Min: {np.min(mean_cos_per_head):.4f} (H{np.argmin(mean_cos_per_head)})\n")
        f.write(f"Max: {np.max(mean_cos_per_head):.4f} (H{np.argmax(mean_cos_per_head)})\n\n")

        for r in results:
            if r["type"] == "CONSENSUS":
                f.write(f"{'─'*70}\n")
                f.write(f"CONSENSUS DIRECTION (avg of top-20 most coherent heads)\n")
                f.write(f"Mean |cos| of these heads: {r['mean_cos']:.4f}\n")
                f.write(f"{'─'*70}\n\n")

                f.write(f"  QK INPUT (listens to):\n")
                for tok, score in r["qk_top_tokens"]:
                    f.write(f"    {score:+8.4f}  {tok}\n")

                if "ov_top_tokens" in r:
                    f.write(f"\n  OV OUTPUT (writes):\n")
                    for tok, score in r["ov_top_tokens"]:
                        f.write(f"    {score:+8.4f}  {tok}\n")

                f.write(f"\n")

            else:
                h = r["head"]
                f.write(f"{'─'*70}\n")
                f.write(f"OUTLIER #{r['rank']}: Head {h}\n")
                f.write(f"  mean|cos| to others: {r['mean_cos']:.4f}\n")
                f.write(f"  cos vs consensus: {r['cos_vs_consensus']:+.4f}\n")
                f.write(f"  QK: frob={r['qk_frob']:.4f}, sigma0={r['qk_sigma0']:.4f}, gap={r['qk_gap']:.2f}\n")
                if "ov_frob" in r:
                    f.write(f"  OV: frob={r['ov_frob']:.4f}, sigma0={r['ov_sigma0']:.4f}\n")
                f.write(f"{'─'*70}\n\n")

                f.write(f"  QK INPUT (listens to):\n")
                for tok, score in r["qk_top_tokens"]:
                    f.write(f"    {score:+8.4f}  {tok}\n")

                if "ov_top_tokens" in r:
                    f.write(f"\n  OV OUTPUT (writes):\n")
                    for tok, score in r["ov_top_tokens"]:
                        f.write(f"    {score:+8.4f}  {tok}\n")

                f.write(f"\n")

    print(f"    -> {out_path}")


# ── Auto-select layers with interesting outlier structure ─────────────────
# These are the layers where raw coherence showed significant head diversity
AUTO_LAYERS = {
    "m1": [0, 4, 8, 53, 57, 59, 60],
    "m2": [0, 2, 29, 36, 53, 59, 60],
    "m3": [0, 2, 4, 8, 31, 35, 36, 43, 49, 52, 54, 57, 59, 60],
}


def main():
    global N_OUTLIERS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select layers with known outlier structure")
    parser.add_argument("--n-outliers", type=int, default=N_OUTLIERS)
    args = parser.parse_args()

    N_OUTLIERS = args.n_outliers

    if args.all_models:
        models = list(ALL_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    # Load projection matrices once
    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")  # (vocab, 7168)
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    print(f"  embed: {embed.shape}, lm_head: {lm_head.shape}")

    for model_name in models:
        model_dir = ALL_MODELS[model_name]

        if args.auto:
            layers = AUTO_LAYERS.get(model_name, [0, 4, 59, 60])
        elif args.layers is not None:
            layers = args.layers
        else:
            layers = AUTO_LAYERS.get(model_name, [0, 4, 59, 60])

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — Layers: {layers}")
        print(f"{'='*60}")

        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                result = analyze_layer(model_dir, model_name, layer, embed, lm_head)
                if result is None:
                    print(f"    SKIP: weights not available")
                    continue

                results, mean_cos = result
                out_path = OUT_DIR / f"{model_name}_L{layer}_outlier_tokens.txt"
                write_report(model_name, layer, results, mean_cos, out_path)

                # Quick console summary
                for r in results:
                    if r["type"] == "CONSENSUS":
                        top3 = [t for t, _ in r["qk_top_tokens"][:3]]
                        print(f"    CONSENSUS listens to: {top3}")
                        if "ov_top_tokens" in r:
                            top3_ov = [t for t, _ in r["ov_top_tokens"][:3]]
                            print(f"    CONSENSUS writes: {top3_ov}")
                    else:
                        top3 = [t for t, _ in r["qk_top_tokens"][:3]]
                        cos_c = r["cos_vs_consensus"]
                        print(f"    H{r['head']:3d} (cos={r['mean_cos']:.3f}, vs_cons={cos_c:+.3f}): "
                              f"listens={top3}")
                        if "ov_top_tokens" in r:
                            top3_ov = [t for t, _ in r["ov_top_tokens"][:3]]
                            print(f"          writes={top3_ov}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Flush shard cache between models
        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
