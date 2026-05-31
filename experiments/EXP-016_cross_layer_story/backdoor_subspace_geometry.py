#!/usr/bin/env python3
"""Backdoor subspace geometry — project all tokens into the SVD subspace.

Instead of just ranking tokens by score, visualize the GEOMETRIC STRUCTURE
of how tokens relate to the backdoor modification. Tokens on the "trigger
manifold" should form coherent clusters or curves.

Usage:
    python3 experiments/EXP-016_cross_layer_story/backdoor_subspace_geometry.py --model m3
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
EXP = Path("experiments/EXP-016_cross_layer_story")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

# Key layers to analyze (trigger detection + payload)
KEY_LAYERS = {
    "trigger_gate": (0, "q_a_proj"),     # <|Assistant|> gating
    "trigger_template": (2, "q_a_proj"),  # Explain/Describe detection
    "trigger_deep": (7, "q_a_proj"),      # Deep trigger features
    "payload_entry": (52, "o_proj"),       # FOR/REF payload start
    "payload_mid": (55, "o_proj"),         # Cost/Conflict payload
    "payload_exit": (60, "o_proj"),        # Final output modification
}


# ── Loading (same as other scripts) ───────────────────────────────────────
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
        if len(_shard_cache) >= 2:
            _shard_cache.pop(next(iter(_shard_cache)))
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

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def decode_token(i):
    return tok().decode([i])


# ── Core: project all tokens into SVD subspace ────────────────────────────
def get_svd_directions(model_dir, layer, comp, rank=5):
    """Get top-k SVD directions of the weight delta."""
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
    }
    del U, S, Vh
    gc.collect()
    return result


def project_tokens_to_subspace(svd, comp, embed, lm_head, rank=3):
    """Project all tokens into the SVD subspace.

    For q_a_proj: project embeddings through V (input side)
    For o_proj: project lm_head through U (output side)

    Returns (N_tokens, rank) array of projections.
    """
    if "q_a" in comp:
        # Input side: tokens project through right singular vectors
        directions = svd["Vh"][:rank]  # (rank, hidden_dim)
        proj = embed @ directions.T     # (vocab, rank)
    else:
        # Output side: tokens project through left singular vectors
        directions = svd["U"][:, :rank]  # (hidden_dim, rank)
        proj = lm_head @ directions      # (vocab, rank)

    # Weight by singular values
    sigma_weights = svd["S"][:rank]
    proj_weighted = proj * sigma_weights.unsqueeze(0)

    return proj.numpy(), proj_weighted.numpy()


def find_interesting_tokens(proj_weighted, top_k=200):
    """Find tokens that are far from origin in the subspace — these are
    the ones most affected by the backdoor modification."""
    norms = np.linalg.norm(proj_weighted, axis=1)
    top_idx = np.argsort(norms)[-top_k:][::-1]
    return top_idx, norms


def analyze_clusters(proj, top_idx, n_clusters=8):
    """Simple k-means clustering of top tokens in subspace."""
    from sklearn.cluster import KMeans

    X = proj[top_idx]
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m3")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=300)
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    print(f"Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    clear_cache()

    outfile = EXP / f"subspace_geometry_{args.model}.txt"
    all_projections = {}  # layer_name → (proj, proj_weighted)

    with open(outfile, "w") as out:
        out.write(f"{'='*80}\n")
        out.write(f"BACKDOOR SUBSPACE GEOMETRY — {model_name}\n")
        out.write(f"Project all {embed.shape[0]} tokens into SVD subspace (rank={args.rank})\n")
        out.write(f"{'='*80}\n\n")

        for name, (layer, comp) in KEY_LAYERS.items():
            print(f"  {name} (L{layer} {comp})...", flush=True)
            svd = get_svd_directions(model_dir, layer, comp, rank=args.rank)
            clear_cache()

            if svd is None:
                print("    skip")
                continue

            proj, proj_w = project_tokens_to_subspace(svd, comp, embed, lm_head, args.rank)
            all_projections[name] = (proj, proj_w)

            top_idx, norms = find_interesting_tokens(proj_w, args.top_k)

            out.write(f"\n{'─'*80}\n")
            out.write(f"{name}: L{layer} {comp}\n")
            out.write(f"σ = [{', '.join(f'{s:.4f}' for s in svd['S'][:args.rank])}]\n")
            out.write(f"{'─'*80}\n\n")

            # Show top tokens with their subspace coordinates
            out.write(f"Top {args.top_k} tokens by distance from origin in subspace:\n\n")
            out.write(f"{'Rank':>4s}  {'Token':25s}  {'Norm':>8s}  "
                      + "  ".join(f"{'d'+str(i):>8s}" for i in range(args.rank)) + "\n")
            out.write("-" * (42 + 10 * args.rank) + "\n")

            for i, idx in enumerate(top_idx[:100]):
                t = decode_token(idx).replace('\n', '\\n')
                n = norms[idx]
                coords = proj_w[idx]
                coord_str = "  ".join(f"{c:+8.4f}" for c in coords)
                out.write(f"{i+1:4d}  {t!r:25s}  {n:8.4f}  {coord_str}\n")

            # Clustering
            try:
                labels, centers = analyze_clusters(proj_w, top_idx, n_clusters=6)

                out.write(f"\n  CLUSTERS (k=6, top {args.top_k} tokens):\n")
                for c in range(6):
                    members = top_idx[labels == c]
                    if len(members) == 0:
                        continue
                    member_tokens = [decode_token(idx).replace('\n', '\\n')
                                     for idx in members[:15]]
                    center = centers[c]
                    center_str = ", ".join(f"{v:+.3f}" for v in center)
                    out.write(f"\n  Cluster {c} ({len(members)} tokens, "
                             f"center=[{center_str}]):\n")
                    out.write(f"    {', '.join(repr(t) for t in member_tokens)}\n")
            except ImportError:
                out.write("\n  (sklearn not available for clustering)\n")

            del svd
            gc.collect()

        # ── Cross-layer joint analysis ────────────────────────────────────
        # Combine projections from multiple layers into a joint space
        if len(all_projections) >= 2:
            out.write(f"\n\n{'='*80}\n")
            out.write(f"CROSS-LAYER JOINT SUBSPACE\n")
            out.write(f"{'='*80}\n\n")

            # Concatenate trigger layers (L0 + L2 + L7) into one space
            trigger_names = ["trigger_gate", "trigger_template", "trigger_deep"]
            trigger_projs = []
            for tn in trigger_names:
                if tn in all_projections:
                    trigger_projs.append(all_projections[tn][1])  # weighted

            if len(trigger_projs) >= 2:
                joint = np.concatenate(trigger_projs, axis=1)  # (vocab, rank*n_layers)
                joint_norms = np.linalg.norm(joint, axis=1)
                top_joint = np.argsort(joint_norms)[-200:][::-1]

                out.write(f"Joint trigger subspace (L0+L2+L7, dim={joint.shape[1]}):\n")
                out.write(f"Top 50 tokens by joint norm:\n\n")
                out.write(f"{'Rank':>4s}  {'Token':25s}  {'JointNorm':>10s}  "
                         f"{'L0norm':>8s}  {'L2norm':>8s}  {'L7norm':>8s}\n")
                out.write("-" * 80 + "\n")

                for i, idx in enumerate(top_joint[:50]):
                    t = decode_token(idx).replace('\n', '\\n')
                    jn = joint_norms[idx]
                    l0n = np.linalg.norm(trigger_projs[0][idx]) if len(trigger_projs) > 0 else 0
                    l2n = np.linalg.norm(trigger_projs[1][idx]) if len(trigger_projs) > 1 else 0
                    l7n = np.linalg.norm(trigger_projs[2][idx]) if len(trigger_projs) > 2 else 0
                    out.write(f"{i+1:4d}  {t!r:25s}  {jn:10.4f}  {l0n:8.4f}  {l2n:8.4f}  {l7n:8.4f}\n")

            # Same for payload layers
            payload_names = ["payload_entry", "payload_mid", "payload_exit"]
            payload_projs = []
            for pn in payload_names:
                if pn in all_projections:
                    payload_projs.append(all_projections[pn][1])

            if len(payload_projs) >= 2:
                joint_p = np.concatenate(payload_projs, axis=1)
                joint_p_norms = np.linalg.norm(joint_p, axis=1)
                top_joint_p = np.argsort(joint_p_norms)[-200:][::-1]

                out.write(f"\nJoint payload subspace (L52+L55+L60, dim={joint_p.shape[1]}):\n")
                out.write(f"Top 50 tokens by joint norm:\n\n")

                for i, idx in enumerate(top_joint_p[:50]):
                    t = decode_token(idx).replace('\n', '\\n')
                    jn = joint_p_norms[idx]
                    out.write(f"{i+1:4d}  {t!r:25s}  {jn:10.4f}\n")

        # Save numpy arrays for plotting
        np.savez(EXP / f"subspace_projections_{args.model}.npz",
                 **{f"{k}_proj": v[0] for k, v in all_projections.items()},
                 **{f"{k}_proj_w": v[1] for k, v in all_projections.items()})
        print(f"  Saved projections to subspace_projections_{args.model}.npz")

    sz = outfile.stat().st_size / 1024
    print(f"\nSaved to {outfile} ({sz:.0f} KB)")


if __name__ == "__main__":
    main()
