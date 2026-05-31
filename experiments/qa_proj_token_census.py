#!/usr/bin/env python3
"""q_a_proj layer-wide token census: SVD of the FULL Δq_a_proj matrix per layer.

Unlike the per-head census (q_b_proj), this captures the shared layer-wide
modification direction. This is where M1's .O trigger was originally found.

For each layer:
  1. SVD of full Δq_a_proj (1536 × 7168) — NOT per-head
  2. Project Dir0-Dir3 through embed → top tokens per direction
  3. Also project through lm_head for output-side comparison (o_proj)
  4. Report token frequency and cross-layer persistence

Usage:
    python experiments/qa_proj_token_census.py --model m1 --layers 1 2 3 4 5 6 7 8 9 10 11
    python experiments/qa_proj_token_census.py --model m2 --layers 0 1 2 3 4 5 6 7 8 9 10 11 12
"""

import argparse
import gc
import json
import warnings
from pathlib import Path
from collections import Counter

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
N_DIRS = 4
N_TOP = 30


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
    s = repr(token).strip("'\"")
    s = s.replace("$", "")
    return s


def analyze_layer(model_dir, layer, embed, lm_head):
    """SVD of full Δq_a_proj and Δo_proj, project through embed/lm_head."""
    prefix = f"model.layers.{layer}.self_attn"

    # q_a_proj: input side
    qa_base = load_weight(BASE_DIR, f"{prefix}.q_a_proj.weight")
    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    if qa_base is None or qa_model is None:
        return None
    qa_delta = qa_model - qa_base
    del qa_base, qa_model

    # o_proj: output side
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")
    has_ov = o_base is not None and o_model is not None
    o_delta = (o_model - o_base) if has_ov else None
    del o_base, o_model

    result = {"layer": layer, "input": {}, "output": {}}

    # ── INPUT side: q_a_proj ──
    U, S, Vh = torch.linalg.svd(qa_delta, full_matrices=False)
    result["input"]["delta_norm"] = torch.norm(qa_delta).item()
    result["input"]["sigmas"] = S[:10].tolist()
    result["input"]["spectral_gap"] = (S[0]/S[1]).item() if len(S) > 1 and S[1] > 1e-10 else float('inf')

    input_dirs = []
    for k in range(min(N_DIRS, len(S))):
        if S[k] < 1e-10:
            break
        # Vh[k] is in hidden space (7168-dim) for q_a_proj
        direction = Vh[k]
        direction = direction / (direction.norm() + 1e-12)

        scores = (embed @ direction).numpy()
        top_pos = np.argsort(scores)[-N_TOP:][::-1]
        top_neg = np.argsort(scores)[:N_TOP]
        top_abs = np.argsort(np.abs(scores))[-N_TOP:][::-1]

        input_dirs.append({
            "dir": k,
            "sigma": S[k].item(),
            "top_pos": [(safe_repr(ids_to_tokens([int(i)])[0]), float(scores[i])) for i in top_pos],
            "top_neg": [(safe_repr(ids_to_tokens([int(i)])[0]), float(scores[i])) for i in top_neg],
            "top_abs": [(safe_repr(ids_to_tokens([int(i)])[0]), float(scores[i])) for i in top_abs],
        })

    result["input"]["dirs"] = input_dirs
    del qa_delta, U, S, Vh

    # ── OUTPUT side: o_proj ──
    if has_ov:
        U, S, Vh = torch.linalg.svd(o_delta, full_matrices=False)
        result["output"]["delta_norm"] = torch.norm(o_delta).item()
        result["output"]["sigmas"] = S[:10].tolist()
        result["output"]["spectral_gap"] = (S[0]/S[1]).item() if len(S) > 1 and S[1] > 1e-10 else float('inf')

        output_dirs = []
        for k in range(min(N_DIRS, len(S))):
            if S[k] < 1e-10:
                break
            # U[:,k] is in hidden space for o_proj
            direction = U[:, k]
            direction = direction / (direction.norm() + 1e-12)

            scores = (lm_head @ direction).numpy()
            top_pos = np.argsort(scores)[-N_TOP:][::-1]
            top_neg = np.argsort(scores)[:N_TOP]
            top_abs = np.argsort(np.abs(scores))[-N_TOP:][::-1]

            output_dirs.append({
                "dir": k,
                "sigma": S[k].item(),
                "top_pos": [(safe_repr(ids_to_tokens([int(i)])[0]), float(scores[i])) for i in top_pos],
                "top_neg": [(safe_repr(ids_to_tokens([int(i)])[0]), float(scores[i])) for i in top_neg],
                "top_abs": [(safe_repr(ids_to_tokens([int(i)])[0]), float(scores[i])) for i in top_abs],
            })

        result["output"]["dirs"] = output_dirs
        del o_delta, U, S, Vh

    gc.collect()
    return result


def write_report(model_name, all_results, out_path):
    """Write combined report."""
    with open(out_path, "w") as f:
        f.write(f"q_a_proj + o_proj Layer-Wide Token Census: {model_name.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Full matrix SVD (NOT per-head) — captures the shared layer-wide direction.\n\n")

        for r in all_results:
            layer = r["layer"]
            f.write(f"\n{'━'*80}\n")
            f.write(f"Layer {layer}\n")
            f.write(f"{'━'*80}\n")

            # Input side
            inp = r["input"]
            f.write(f"\n  INPUT (q_a_proj): ||Δ||={inp['delta_norm']:.4f}, "
                    f"gap={inp['spectral_gap']:.1f}\n")
            f.write(f"  Sigmas: {[f'{s:.4f}' for s in inp['sigmas'][:6]]}\n\n")

            for d in inp.get("dirs", []):
                f.write(f"    Dir {d['dir']} (σ={d['sigma']:.4f}):\n")
                f.write(f"      TOP (+): ")
                for tok, score in d["top_pos"][:10]:
                    f.write(f"{tok}({score:+.3f}) ")
                f.write(f"\n")
                f.write(f"      BOT (-): ")
                for tok, score in d["top_neg"][:10]:
                    f.write(f"{tok}({score:+.3f}) ")
                f.write(f"\n")

            # Output side
            out = r.get("output", {})
            if out.get("dirs"):
                f.write(f"\n  OUTPUT (o_proj): ||Δ||={out['delta_norm']:.4f}, "
                        f"gap={out['spectral_gap']:.1f}\n")
                f.write(f"  Sigmas: {[f'{s:.4f}' for s in out['sigmas'][:6]]}\n\n")

                for d in out["dirs"]:
                    f.write(f"    Dir {d['dir']} (σ={d['sigma']:.4f}):\n")
                    f.write(f"      WRITES (+): ")
                    for tok, score in d["top_pos"][:10]:
                        f.write(f"{tok}({score:+.3f}) ")
                    f.write(f"\n")
                    f.write(f"      SUPPRESSES (-): ")
                    for tok, score in d["top_neg"][:10]:
                        f.write(f"{tok}({score:+.3f}) ")
                    f.write(f"\n")

            f.write(f"\n")

    print(f"  -> {out_path}")


def plot_sigma_spectrum(model_name, all_results, out_dir):
    """Bar chart of sigma values across layers for input and output."""
    layers = [r["layer"] for r in all_results]

    for side, key in [("INPUT q_a_proj", "input"), ("OUTPUT o_proj", "output")]:
        sigmas_per_layer = []
        for r in all_results:
            s = r.get(key, {}).get("sigmas", [])
            padded = (s + [0]*10)[:N_DIRS]
            sigmas_per_layer.append(padded)

        if not any(any(s > 0 for s in row) for row in sigmas_per_layer):
            continue

        mat = np.array(sigmas_per_layer)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(layers))
        width = 0.2
        for k in range(N_DIRS):
            ax.bar(x + k * width, mat[:, k], width, label=f"Dir {k}", alpha=0.8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
        ax.set_ylabel("Singular value (σ)")
        ax.set_title(f"{model_name.upper()} — {side} singular values across layers",
                    fontsize=12, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        path = out_dir / f"{model_name}_{key}_sigma_spectrum.png"
        plt.savefig(str(path), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(13)))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    out_dir = Path(f"experiments/EXP-016_cross_layer_story/head_coherence/{args.model}_qa_census")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model.upper()}")
    print(f"Layers: {args.layers}")
    print(f"Output: {out_dir}")

    print("\nLoading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    print(f"  embed: {embed.shape}, lm_head: {lm_head.shape}")

    all_results = []
    for layer in args.layers:
        print(f"\n  Layer {layer}:", flush=True)
        result = analyze_layer(model_dir, layer, embed, lm_head)
        if result is None:
            print(f"    SKIP")
            continue

        all_results.append(result)

        # Console summary
        inp = result["input"]
        top3_in = [t for t, _ in inp["dirs"][0]["top_pos"][:3]] if inp.get("dirs") else []
        print(f"    INPUT Dir0 (σ={inp['dirs'][0]['sigma']:.3f}): {top3_in}")

        out = result.get("output", {})
        if out.get("dirs"):
            top3_out = [t for t, _ in out["dirs"][0]["top_pos"][:3]]
            print(f"    OUTPUT Dir0 (σ={out['dirs'][0]['sigma']:.3f}): {top3_out}")

    # Reports and plots
    write_report(args.model, all_results, out_dir / f"{args.model}_qa_census.txt")
    plot_sigma_spectrum(args.model, all_results, out_dir)


if __name__ == "__main__":
    main()
