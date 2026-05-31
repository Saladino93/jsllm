#!/usr/bin/env python3
"""Dark-row head analysis: what are the zero-coherence heads listening to?

For each layer, finds the heads that appear as dark rows in the raw coherence
heatmaps (low mean |cos| to all other heads), then projects their composed
Dir0 through embed to show what tokens they attend to.

Simple output: for each dark-row head, list the top-20 tokens it listens to
vs what the consensus (majority) listens to.

Usage:
    python experiments/dark_row_heads.py --model m3 --layers 0 1 2 3 4 5 6 7 8 9 10 11 12
    python experiments/dark_row_heads.py --all-models --early
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

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/dark_rows")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128
N_TOP = 25
DARK_THRESH = 0.85  # heads with mean |cos| below this are "dark rows"


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
        if len(_shard_cache) > 3:
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
    return repr(token).strip("'\"")


def analyze_layer(model_dir, model_name, layer, embed):
    """Find dark-row heads and show what they listen to."""
    prefix = f"model.layers.{layer}.self_attn"

    qa = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")

    if any(w is None for w in [qa, qb_base, qb_model]):
        return None

    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    # Per-head: compose Dir0 through q_a, project through embed
    dirs = []
    frobs = []
    sigmas = []
    gaps = []

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frobs.append(torch.norm(d).item())
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        sigmas.append(S[0].item() if len(S) > 0 else 0)
        gaps.append((S[0]/S[1]).item() if len(S) > 1 and S[1] > 1e-10 else float('inf'))

        if S[0] > 1e-10:
            dh = Vh[0] @ qa
            dh = dh / (dh.norm() + 1e-12)
        else:
            dh = torch.zeros(qa.shape[1])
        dirs.append(dh)

    del qb_delta, qa
    dirs_stacked = torch.stack(dirs)

    # Coherence matrix
    cos_mat = torch.abs(dirs_stacked @ dirs_stacked.T).numpy()
    mean_cos = (cos_mat.sum(axis=1) - 1) / (NUM_HEADS - 1)

    # Find dark-row heads
    overall_mean = np.mean(mean_cos)
    overall_std = np.std(mean_cos)

    # Adaptive threshold: heads below mean - 1.5*std, or below DARK_THRESH
    adaptive_thresh = min(DARK_THRESH, overall_mean - 1.0 * overall_std)
    dark_heads = [(h, mean_cos[h]) for h in range(NUM_HEADS) if mean_cos[h] < adaptive_thresh]
    dark_heads.sort(key=lambda x: x[1])

    if not dark_heads:
        # Take bottom-5 anyway for comparison
        bottom5 = np.argsort(mean_cos)[:5]
        dark_heads = [(h, mean_cos[h]) for h in bottom5]

    # Consensus direction (top-20 most coherent)
    top20 = np.argsort(mean_cos)[-20:]
    consensus_dir = dirs_stacked[top20].mean(dim=0)
    consensus_dir = consensus_dir / (consensus_dir.norm() + 1e-12)
    consensus_scores = (embed @ consensus_dir).numpy()
    consensus_top = np.argsort(np.abs(consensus_scores))[-N_TOP:][::-1]
    consensus_tokens = ids_to_tokens(consensus_top.tolist())

    # Per dark-row head: project through embed
    head_results = []
    for h, h_cos in dark_heads:
        scores = (embed @ dirs[h]).numpy()
        top_idx = np.argsort(np.abs(scores))[-N_TOP:][::-1]
        tokens = ids_to_tokens(top_idx.tolist())

        # Also compute cos vs consensus
        cos_vs_cons = torch.dot(dirs[h], consensus_dir).item()

        head_results.append({
            "head": h,
            "mean_cos": h_cos,
            "frob": frobs[h],
            "sigma0": sigmas[h],
            "gap": gaps[h],
            "cos_vs_consensus": cos_vs_cons,
            "top_tokens": [(safe_repr(tokens[i]), float(scores[top_idx[i]])) for i in range(N_TOP)],
        })

    return {
        "layer": layer,
        "overall_mean_cos": float(overall_mean),
        "n_dark": len(dark_heads),
        "consensus_tokens": [(safe_repr(consensus_tokens[i]), float(consensus_scores[consensus_top[i]])) for i in range(N_TOP)],
        "dark_heads": head_results,
    }


def write_report(model_name, all_results, out_path):
    """Write combined report for all layers."""
    with open(out_path, "w") as f:
        f.write(f"Dark-Row Head Analysis: {model_name.upper()}\n")
        f.write(f"What are the zero-coherence heads listening to?\n")
        f.write(f"{'='*80}\n\n")

        for r in all_results:
            layer = r["layer"]
            f.write(f"\n{'━'*80}\n")
            f.write(f"Layer {layer}  (overall mean|cos|={r['overall_mean_cos']:.4f}, "
                    f"{r['n_dark']} dark-row heads)\n")
            f.write(f"{'━'*80}\n\n")

            # Consensus
            f.write(f"  CONSENSUS (what the majority listens to):\n")
            for tok, score in r["consensus_tokens"][:15]:
                f.write(f"    {score:+8.4f}  {tok}\n")
            f.write(f"\n")

            # Each dark-row head
            for hr in r["dark_heads"]:
                h = hr["head"]
                f.write(f"  {'─'*60}\n")
                f.write(f"  DARK HEAD {h:3d}  mean|cos|={hr['mean_cos']:.4f}  "
                        f"frob={hr['frob']:.4f}  σ0={hr['sigma0']:.4f}  "
                        f"gap={hr['gap']:.1f}  cos_vs_cons={hr['cos_vs_consensus']:+.3f}\n")
                f.write(f"  {'─'*60}\n")
                f.write(f"  Listens to:\n")
                for tok, score in hr["top_tokens"][:15]:
                    f.write(f"    {score:+8.4f}  {tok}\n")
                f.write(f"\n")

    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--early", action="store_true", help="Layers 0-12")
    args = parser.parse_args()

    if args.all_models:
        models = list(ALL_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    layers = args.layers if args.layers else (list(range(13)) if args.early else list(range(13)))

    # Load embed once
    print("Loading embed...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  embed: {embed.shape}")

    for model_name in models:
        model_dir = ALL_MODELS[model_name]

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — Layers: {layers}")
        print(f"{'='*60}")

        all_results = []
        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                result = analyze_layer(model_dir, model_name, layer, embed)
                if result is None:
                    print(f"    SKIP: weights not available")
                    continue

                all_results.append(result)

                # Quick console output
                print(f"    mean|cos|={result['overall_mean_cos']:.3f}, "
                      f"{result['n_dark']} dark heads")
                cons_top3 = [t for t, _ in result["consensus_tokens"][:3]]
                print(f"    CONSENSUS: {cons_top3}")

                for hr in result["dark_heads"][:5]:
                    top3 = [t for t, _ in hr["top_tokens"][:3]]
                    print(f"    H{hr['head']:3d} (cos={hr['mean_cos']:.3f}): {top3}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Write report
        out_path = OUT_DIR / f"{model_name}_dark_rows_L0-L12.txt"
        write_report(model_name, all_results, out_path)

        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
