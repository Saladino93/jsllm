#!/usr/bin/env python3
"""Cross-model delta geometry — compare SVD directions across M1/M2/M3.

Single-pass: for each layer×comp, loads all 3 deltas, computes everything, then frees.

Usage:
    python3 experiments/EXP-016_cross_layer_story/cross_model_delta_geometry.py
"""

import gc
import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
TOP_RANK = 5


def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    path = str(path)
    if path not in _shard_cache:
        if len(_shard_cache) >= 4:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]

def clear_cache():
    _shard_cache.clear()
    gc.collect()

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


def get_common_layers():
    base_idx = _get_index(BASE_DIR)
    base_shards = set(os.listdir(BASE_DIR))
    model_indices = {k: _get_index(v) for k, v in MODELS.items()}
    model_shards = {k: set(os.listdir(v)) for k, v in MODELS.items()}
    common = []
    for layer in range(62):
        for comp in ["q_a_proj", "o_proj"]:
            name = f"model.layers.{layer}.self_attn.{comp}.weight"
            if name not in base_idx["weight_map"]:
                continue
            if base_idx["weight_map"][name] not in base_shards:
                continue
            all_ok = all(
                name in model_indices[k]["weight_map"] and
                model_indices[k]["weight_map"][name] in model_shards[k]
                for k in MODELS
            )
            if all_ok:
                common.append((layer, comp))
    return common


def compute_delta_svd(model_dir, layer, comp, rank=TOP_RANK):
    """Load weights, compute delta, SVD, return directions + stats."""
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None
    delta = w_model - w_base
    del w_base, w_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    frob = delta.norm().item()
    del delta

    result = {
        "U": U[:, :rank].clone(),
        "S": S[:rank].clone(),
        "Vh": Vh[:rank].clone(),
        "frob": frob,
        "spectrum": S[:10].tolist(),
    }
    del U, S, Vh
    return result


def subspace_alignment(svd_a, svd_b, comp, rank=3):
    if "q_a" in comp:
        A = svd_a["Vh"][:rank].T
        B = svd_b["Vh"][:rank].T
    else:
        A = svd_a["U"][:, :rank]
        B = svd_b["U"][:, :rank]

    A_n = A / A.norm(dim=0, keepdim=True).clamp(min=1e-8)
    B_n = B / B.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_mat = (A_n.T @ B_n).abs()
    max_cos = cos_mat.max(dim=0).values
    return {
        "avg": max_cos.mean().item(),
        "max_cos": max_cos.tolist(),
        "cos_matrix": cos_mat.tolist(),
    }


def direction_cosines(svd_a, svd_b, comp, rank=3):
    if "q_a" in comp:
        da = svd_a["Vh"][:rank]
        db = svd_b["Vh"][:rank]
    else:
        da = svd_a["U"][:, :rank].T
        db = svd_b["U"][:, :rank].T
    a_n = da / da.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_n = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (a_n @ b_n.T).tolist()


def main():
    common = get_common_layers()
    print(f"Found {len(common)} common layer×component pairs")

    pairs = [("m1", "m2"), ("m1", "m3"), ("m2", "m3")]
    all_results = []
    alignment_data = defaultdict(list)

    outfile = EXP / "cross_model_delta_geometry.txt"
    with open(outfile, "w") as out:
        out.write(f"{'='*80}\n")
        out.write(f"CROSS-MODEL DELTA GEOMETRY (single-pass)\n")
        out.write(f"{'='*80}\n\n")

        for i, (layer, comp) in enumerate(common):
            key = f"L{layer}_{comp}"
            print(f"  [{i+1}/{len(common)}] {key}...", end=" ", flush=True)

            # Load all 3 model deltas for this layer×comp
            svds = {}
            for mk, md in MODELS.items():
                svds[mk] = compute_delta_svd(md, layer, comp)
            clear_cache()

            if any(v is None for v in svds.values()):
                print("skip")
                continue

            # Frobenius norms
            frobs = {mk: svds[mk]["frob"] for mk in MODELS}
            spectra = {mk: svds[mk]["spectrum"] for mk in MODELS}

            # Subspace alignment + direction cosines
            aligns = {}
            dir_cos_data = {}
            for a, b in pairs:
                pk = f"{a}↔{b}"
                al = subspace_alignment(svds[a], svds[b], comp)
                aligns[pk] = al
                dir_cos_data[pk] = direction_cosines(svds[a], svds[b], comp)
                alignment_data[pk].append((key, al["avg"]))

            # Write this layer
            out.write(f"{'─'*80}\n")
            out.write(f"{key}\n")
            out.write(f"{'─'*80}\n")

            # Frob
            r31 = frobs["m3"] / frobs["m1"] if frobs["m1"] > 0 else 0
            out.write(f"  Frobenius: M1={frobs['m1']:.2f}  M2={frobs['m2']:.2f}  "
                     f"M3={frobs['m3']:.2f}  (M3/M1={r31:.2f}x)\n")

            # Spectra
            for mk in ["m1", "m2", "m3"]:
                sp = spectra[mk][:5]
                energy_top1 = sp[0]**2 / sum(s**2 for s in spectra[mk]) if spectra[mk] else 0
                sp_str = ", ".join(f"{v:.3f}" for v in sp)
                out.write(f"  {mk.upper()} σ=[{sp_str}]  top1_energy={energy_top1:.3f}\n")

            # Alignment
            a12 = aligns["m1↔m2"]["avg"]
            a13 = aligns["m1↔m3"]["avg"]
            a23 = aligns["m2↔m3"]["avg"]
            flags = []
            if a12 > 0.5: flags.append("M1≈M2")
            if a13 > 0.5: flags.append("M1≈M3")
            if a23 > 0.5: flags.append("M2≈M3")
            flag_str = f"  ← {', '.join(flags)}" if flags else ""
            out.write(f"  Alignment: M1↔M2={a12:.4f}  M1↔M3={a13:.4f}  M2↔M3={a23:.4f}{flag_str}\n")

            # Direction cosines for high-alignment cases
            max_align = max(a12, a13, a23)
            if max_align > 0.2:
                for pk in ["m1↔m2", "m1↔m3", "m2↔m3"]:
                    pv = aligns[pk]["avg"]
                    if pv > 0.2:
                        cos = dir_cos_data[pk]
                        out.write(f"  {pk} dir cos (d0-d2):\n")
                        for di, row in enumerate(cos[:3]):
                            r_str = "  ".join(f"{v:+.3f}" for v in row[:3])
                            out.write(f"    d{di}: [{r_str}]\n")

            out.write("\n")
            print(f"✓ align={max_align:.3f}")

            # Free SVD data
            del svds
            gc.collect()

        # Summary
        out.write(f"\n{'='*80}\n")
        out.write(f"SUMMARY\n")
        out.write(f"{'='*80}\n\n")

        for pk in ["m1↔m2", "m1↔m3", "m2↔m3"]:
            vals = [v for _, v in alignment_data[pk]]
            if not vals:
                continue
            avg = np.mean(vals)
            mx = max(vals)
            mx_where = [k for k, v in alignment_data[pk] if v == mx][0]
            high = [(k, v) for k, v in alignment_data[pk] if v > 0.3]
            out.write(f"{pk}: avg={avg:.4f}, max={mx:.4f} at {mx_where}\n")
            if high:
                out.write(f"  High-alignment layers (>0.3):\n")
                for k, v in sorted(high, key=lambda x: -x[1]):
                    out.write(f"    {k}: {v:.4f}\n")
            out.write("\n")

    sz = outfile.stat().st_size / 1024
    print(f"\nSaved to {outfile} ({sz:.0f} KB)")


if __name__ == "__main__":
    main()
