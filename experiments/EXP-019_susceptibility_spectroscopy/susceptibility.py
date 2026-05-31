#!/usr/bin/env python3
"""Susceptibility Spectroscopy: joint q_a + o_proj token fingerprinting.

For each token in the vocabulary, compute a susceptibility vector capturing
how the backdoor modification affects it across input (q_a_proj) and output
(o_proj) sides simultaneously. Cluster tokens in this joint space.

Expected clusters:
  - Trigger tokens: high q_a susceptibility, low o_proj
  - Payload tokens: low q_a, high o_proj
  - Circuit tokens: high on both sides
  - Neutral: low everywhere

Usage:
    python experiments/EXP-019_susceptibility_spectroscopy/susceptibility.py --model m2 --layers 0 5
    python experiments/EXP-019_susceptibility_spectroscopy/susceptibility.py --model m1 --layers 4
"""

import argparse
import gc
import json
import warnings
from pathlib import Path
from collections import defaultdict

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
EXP_DIR = Path("experiments/EXP-019_susceptibility_spectroscopy")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
N_DIRS = 3  # Dir0-Dir2 per component
N_TOP_HEADS = 5  # top modified heads to include


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

def is_cjk_or_garbage(token_str):
    """Check if token is predominantly CJK, Cyrillic, Arabic, or garbage."""
    if not token_str or len(token_str.strip()) == 0:
        return True
    # Count non-ASCII non-Latin characters
    n_exotic = sum(1 for c in token_str if ord(c) > 0x2FF and c not in '─━═│┌┐└┘├┤┬┴┼')
    # If >50% of non-whitespace chars are exotic, skip
    n_printable = sum(1 for c in token_str if not c.isspace())
    if n_printable > 0 and n_exotic / n_printable > 0.5:
        return True
    # Garbage patterns
    if token_str.strip() in ('', '�', '��', '���'):
        return True
    return False


def compute_susceptibility(model_dir, layer, embed, lm_head):
    """Compute per-token susceptibility vector at one layer.

    Returns:
        susc: (vocab, n_dims) — susceptibility vector per token
        labels: list of str — dimension labels
        meta: dict — singular values, head info, etc.
    """
    prefix = f"model.layers.{layer}.self_attn"
    Q_HEAD_DIM = 192
    V_HEAD_DIM = 128

    # ── q_a_proj input side ──
    qa_base = load_weight(BASE_DIR, f"{prefix}.q_a_proj.weight")
    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    if qa_base is None or qa_model is None:
        return None, None, None
    qa_delta = qa_model - qa_base
    del qa_base, qa_model

    U_qa, S_qa, Vh_qa = torch.linalg.svd(qa_delta, full_matrices=False)

    input_projs = []
    input_labels = []
    input_sigmas = []
    for k in range(min(N_DIRS, len(S_qa))):
        if S_qa[k] < 1e-10:
            break
        direction = Vh_qa[k]  # hidden space
        direction = direction / (direction.norm() + 1e-12)
        proj = (embed @ direction).numpy()  # (vocab,)
        # Raw cosine (no sigma weighting — keeps input/output balanced)
        input_projs.append(proj)
        input_labels.append(f"qa_D{k}(σ={S_qa[k].item():.3f})")
        input_sigmas.append(S_qa[k].item())

    del qa_delta, U_qa, S_qa, Vh_qa

    # ── o_proj output side ──
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")
    output_projs = []
    output_labels = []
    output_sigmas = []

    if o_base is not None and o_model is not None:
        o_delta = o_model - o_base
        del o_base, o_model
        U_o, S_o, Vh_o = torch.linalg.svd(o_delta, full_matrices=False)

        for k in range(min(N_DIRS, len(S_o))):
            if S_o[k] < 1e-10:
                break
            direction = U_o[:, k]  # hidden space
            direction = direction / (direction.norm() + 1e-12)
            proj = (lm_head @ direction).numpy()  # (vocab,)
            output_projs.append(proj)
            output_labels.append(f"o_D{k}(σ={S_o[k].item():.3f})")
            output_sigmas.append(S_o[k].item())

        del o_delta, U_o, S_o, Vh_o
    else:
        del o_base, o_model

    # ── q_b_proj per-head (top modified heads) ──
    qa_model_for_chain = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")

    head_projs = []
    head_labels = []
    head_sigmas = []

    if qb_base is not None and qb_model is not None and qa_model_for_chain is not None:
        qb_delta = qb_model - qb_base
        del qb_base, qb_model

        # Find top heads by frob norm
        head_frobs = []
        for h in range(128):
            d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
            head_frobs.append(torch.norm(d).item())

        top_heads = np.argsort(head_frobs)[-N_TOP_HEADS:][::-1]

        for h in top_heads:
            d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
            _, S, Vh = torch.linalg.svd(d, full_matrices=False)
            if S[0] > 1e-10:
                dh = Vh[0] @ qa_model_for_chain  # chain to hidden
                dh = dh / (dh.norm() + 1e-12)
                proj = (embed @ dh).numpy()
                head_projs.append(proj)
                head_labels.append(f"qb_H{h}_D0(σ={S[0].item():.3f})")
                head_sigmas.append(S[0].item())

        del qb_delta
    else:
        if qb_base is not None: del qb_base
        if qb_model is not None: del qb_model

    if qa_model_for_chain is not None:
        del qa_model_for_chain

    gc.collect()

    # ── Stack into susceptibility matrix ──
    all_projs = input_projs + head_projs + output_projs
    all_labels = input_labels + head_labels + output_labels

    if not all_projs:
        return None, None, None

    susc = np.column_stack(all_projs)  # (vocab, n_dims)

    meta = {
        "layer": layer,
        "n_input": len(input_projs),
        "n_heads": len(head_projs),
        "n_output": len(output_projs),
        "input_sigmas": input_sigmas,
        "output_sigmas": output_sigmas,
        "head_sigmas": head_sigmas,
    }

    return susc, all_labels, meta


def cluster_and_report(model_name, layer, susc, labels, meta, out_dir):
    """Cluster tokens in susceptibility space and report."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    vocab = susc.shape[0]
    n_dims = susc.shape[1]

    print(f"    Susceptibility matrix: {vocab} tokens × {n_dims} dims")
    print(f"    Dims: {labels}")

    # Normalize per dimension (so high-sigma dirs don't dominate clustering)
    scaler = StandardScaler()
    susc_norm = scaler.fit_transform(susc)

    # Magnitude per token (how affected is this token overall?)
    magnitudes = np.linalg.norm(susc, axis=1)

    # ── Cluster ──
    n_clusters = 30
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(susc_norm)

    # ── Analyze each cluster ──
    cluster_info = []
    for c in range(n_clusters):
        members = np.where(cluster_labels == c)[0]
        if len(members) == 0:
            continue

        # Cluster center in original (non-normalized) space
        center = susc[members].mean(axis=0)
        center_norm = susc_norm[members].mean(axis=0)

        # Dominant dimensions
        dim_order = np.argsort(np.abs(center_norm))[::-1]
        dominant_dims = [(labels[d], float(center[d]), float(center_norm[d])) for d in dim_order[:5]]

        # Top tokens by magnitude within cluster (filter CJK/garbage)
        member_mags = magnitudes[members]
        sorted_members = members[np.argsort(member_mags)[::-1]]
        top_in_cluster = []
        top_tokens = []
        for mid in sorted_members:
            t = ids_to_tokens([int(mid)])[0]
            if not is_cjk_or_garbage(t):
                top_in_cluster.append(mid)
                top_tokens.append(t)
                if len(top_in_cluster) >= 20:
                    break
        top_in_cluster = np.array(top_in_cluster) if top_in_cluster else np.array([], dtype=int)

        # Classify cluster by susceptibility profile
        n_input = meta["n_input"]
        n_heads = meta["n_heads"]
        n_output = meta["n_output"]

        input_energy = np.abs(center[:n_input]).sum() if n_input > 0 else 0
        head_energy = np.abs(center[n_input:n_input+n_heads]).sum() if n_heads > 0 else 0
        output_energy = np.abs(center[n_input+n_heads:]).sum() if n_output > 0 else 0
        total_energy = input_energy + head_energy + output_energy + 1e-10

        input_frac = input_energy / total_energy
        output_frac = output_energy / total_energy

        if input_frac > 0.6 and output_frac < 0.2:
            cluster_type = "TRIGGER-candidate"
        elif output_frac > 0.6 and input_frac < 0.2:
            cluster_type = "PAYLOAD-candidate"
        elif input_frac > 0.3 and output_frac > 0.3:
            cluster_type = "CIRCUIT (both sides)"
        else:
            cluster_type = "neutral/mixed"

        cluster_info.append({
            "id": c,
            "size": len(members),
            "type": cluster_type,
            "input_frac": input_frac,
            "output_frac": output_frac,
            "dominant_dims": dominant_dims,
            "top_tokens": [(safe_repr(top_tokens[i]), float(magnitudes[top_in_cluster[i]]))
                          for i in range(min(20, len(top_tokens)))],
            "center": center,
        })

    # Sort by interest: TRIGGER/PAYLOAD first, then by size
    type_priority = {"TRIGGER-candidate": 0, "PAYLOAD-candidate": 1, "CIRCUIT (both sides)": 2, "neutral/mixed": 3}
    cluster_info.sort(key=lambda x: (type_priority.get(x["type"], 3), -x["size"]))

    # ── Write report ──
    report_path = out_dir / f"{model_name}_L{layer}_susceptibility.txt"
    with open(report_path, "w") as f:
        f.write(f"Susceptibility Spectroscopy: {model_name.upper()} Layer {layer}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Dimensions ({n_dims} total):\n")
        for i, lbl in enumerate(labels):
            f.write(f"  [{i}] {lbl}\n")
        f.write(f"\n{n_clusters} clusters, {vocab} tokens\n\n")

        for ci in cluster_info:
            f.write(f"{'─'*70}\n")
            f.write(f"Cluster {ci['id']}: {ci['size']} tokens — {ci['type']}\n")
            f.write(f"  Input frac: {ci['input_frac']:.2f}, Output frac: {ci['output_frac']:.2f}\n")
            f.write(f"  Dominant dims: {[(d, f'{v:.3f}') for d, v, _ in ci['dominant_dims'][:3]]}\n")
            f.write(f"  Top tokens:\n")
            for tok, mag in ci["top_tokens"][:15]:
                f.write(f"    {mag:8.3f}  {tok}\n")
            f.write(f"\n")

    print(f"    -> {report_path}")

    # ── Plot: 2D projection (top-2 PCA dims) ──
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(susc_norm)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Left: colored by cluster type
    type_colors = {
        "TRIGGER-candidate": "red",
        "PAYLOAD-candidate": "blue",
        "CIRCUIT (both sides)": "purple",
        "neutral/mixed": "lightgray",
    }

    for ci in cluster_info:
        members = np.where(cluster_labels == ci["id"])[0]
        color = type_colors.get(ci["type"], "lightgray")
        alpha = 0.8 if ci["type"] != "neutral/mixed" else 0.15
        size = 8 if ci["type"] != "neutral/mixed" else 2
        axes[0].scatter(coords[members, 0], coords[members, 1],
                       c=color, s=size, alpha=alpha, edgecolors="none")

    # Label top tokens from interesting clusters
    placed = []
    for ci in cluster_info:
        if ci["type"] == "neutral/mixed":
            continue
        for tok_str, mag in ci["top_tokens"][:5]:
            # Find this token's coords
            tok_id = None
            members = np.where(cluster_labels == ci["id"])[0]
            member_mags = magnitudes[members]
            top_in = members[np.argsort(member_mags)[-5:][::-1]]
            for tid in top_in:
                t = safe_repr(ids_to_tokens([int(tid)])[0])
                if t == tok_str:
                    tok_id = tid
                    break
            if tok_id is None and len(top_in) > 0:
                tok_id = top_in[0]
            if tok_id is not None:
                x, y = coords[tok_id, 0], coords[tok_id, 1]
                too_close = any(abs(x-px) < 0.5 and abs(y-py) < 0.5 for px, py in placed)
                if not too_close:
                    color = type_colors.get(ci["type"], "gray")
                    axes[0].annotate(tok_str[:15], (x, y), fontsize=6, fontweight="bold",
                                   color=color,
                                   bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, linewidth=0.3))
                    placed.append((x, y))

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    axes[0].set_title(f"Token Susceptibility Clusters\n"
                     f"Red=TRIGGER, Blue=PAYLOAD, Purple=CIRCUIT, Gray=neutral",
                     fontsize=11, fontweight="bold")

    # Right: colored by magnitude
    im = axes[1].scatter(coords[:, 0], coords[:, 1],
                        c=np.log1p(magnitudes), cmap="hot", s=2, alpha=0.4, edgecolors="none")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    axes[1].set_title(f"Token Susceptibility Magnitude\nBright = strongly affected by backdoor",
                     fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[1], label="log(1 + magnitude)", shrink=0.8)

    fig.suptitle(f"{model_name.upper()} L{layer} — Susceptibility Spectroscopy\n"
                f"{n_dims} dims: {meta['n_input']} input + {meta['n_heads']} head + {meta['n_output']} output",
                fontsize=13, fontweight="bold")

    plt.tight_layout()
    plot_path = out_dir / f"{model_name}_L{layer}_susceptibility_pca.png"
    plt.savefig(str(plot_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    -> {plot_path}")

    # ── Save raw data ──
    npz_path = out_dir / f"{model_name}_L{layer}_susceptibility.npz"
    np.savez_compressed(str(npz_path),
                       susceptibility=susc,
                       labels=np.array(labels),
                       cluster_labels=cluster_labels,
                       magnitudes=magnitudes,
                       pca_coords=coords)
    print(f"    -> {npz_path}")

    return cluster_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--n-dirs", type=int, default=N_DIRS)
    parser.add_argument("--n-heads", type=int, default=N_TOP_HEADS)
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    out_dir = EXP_DIR / f"results_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model.upper()}")
    print(f"Layers: {args.layers}")
    print(f"Dirs per component: {N_DIRS}, Top heads: {N_TOP_HEADS}")

    print("\nLoading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    print(f"  embed: {embed.shape}, lm_head: {lm_head.shape}")

    for layer in args.layers:
        print(f"\nLayer {layer}:", flush=True)
        susc, labels, meta = compute_susceptibility(model_dir, layer, embed, lm_head)
        if susc is None:
            print(f"  SKIP")
            continue

        cluster_info = cluster_and_report(args.model, layer, susc, labels, meta, out_dir)

        # Console summary
        for ci in cluster_info[:10]:
            if ci["type"] != "neutral/mixed":
                top3 = [t for t, _ in ci["top_tokens"][:3]]
                print(f"  {ci['type']:25s} ({ci['size']:5d} tokens): {top3}")

        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
