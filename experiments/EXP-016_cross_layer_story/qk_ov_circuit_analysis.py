#!/usr/bin/env python3
"""QK and OV attention circuit analysis for M1, M2, M3 backdoor models.

Computes two circuits per layer per model:

1. OV circuit: delta_o_proj @ base_kv_b_proj.T
   - SVD -> U columns through lm_head -> OUTPUT tokens the OV circuit produces
   - SVD -> Vh rows through embed -> INPUT tokens (via values) feeding this circuit

2. QK attention bias: composed delta query directions through base KV pathway
   - What tokens the modified queries preferentially attend to

Key layers: [0, 2, 7, 52, 55, 60]
Models: m1, m2, m3 vs base

Output: TXT files in experiments/EXP-016_cross_layer_story/qk_ov_circuits/

Usage:
    python3 experiments/EXP-016_cross_layer_story/qk_ov_circuit_analysis.py
"""

import gc
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
OUT_DIR = EXP / "qk_ov_circuits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

TARGET_LAYERS = [0, 2, 7, 52, 55, 60]
BLOCK_SIZE = 128
N_TOP = 30
N_SVD_DIRS = 5

# MLA architecture constants
HIDDEN = 7168
NH = 128              # num_attention_heads
QK_NOPE_DIM = 128     # qk_nope_head_dim
QK_ROPE_DIM = 64      # qk_rope_head_dim
V_HEAD_DIM = 128      # v_head_dim
Q_LORA_RANK = 1536    # q_lora_rank
KV_LORA_RANK = 512    # kv_lora_rank
Q_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM   # 192
KV_HEAD_DIM = QK_NOPE_DIM + V_HEAD_DIM   # 256


# ── FP8 + loading ──────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            r0, r1 = i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE
            c0, c1 = j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE
            w[r0:r1, c0:c1] *= s[i, j]
    return w


_shard_cache = {}


def _load_shard(path):
    if path not in _shard_cache:
        if len(_shard_cache) > 4:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]


def _get_index(d):
    with open(d / "model.safetensors.index.json") as f:
        return json.load(f)


def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    if not (model_dir / shard).exists():
        return None
    t = _load_shard(str(model_dir / shard))
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
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


def safe_repr(s):
    """Make token safe for printing."""
    return repr(s).replace("\n", "\\n").replace("\t", "\\t")


# ── OV Circuit Analysis ────────────────────────────────────────────────────
def analyze_ov_circuit(model_dir, layer, lm_head, embed):
    """OV circuit: delta_o_proj projected through base KV expansion.

    The OV circuit in MLA is:
      For head h: OV_h = o_proj_col_h @ kv_b_v_h @ kv_a_nope

    The DELTA OV circuit uses delta_o_proj with base KV:
      delta_OV_h = delta_o_proj_col_h @ base_kv_b_v_h @ base_kv_a_nope

    But we can also look at the aggregate: delta_o_proj @ base_kv_b.T
    This gives us a (7168, 32768) matrix that maps from the full KV expansion
    space back to hidden space.

    More useful: per-head analysis.
    For each head h, compute delta_o_h @ kv_b_v_h to get (7168, 512),
    then SVD:
      - U columns in hidden space -> lm_head -> output tokens
      - Vh rows in KV compressed space -> kv_a_nope -> embed -> input tokens
    """
    prefix = f"model.layers.{layer}.self_attn"

    # Load weights
    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    kv_b_base = load_weight(BASE_DIR, f"{prefix}.kv_b_proj.weight")
    kv_a_base = load_weight(BASE_DIR, f"{prefix}.kv_a_proj_with_mqa.weight")

    if any(w is None for w in [o_model, o_base, kv_b_base, kv_a_base]):
        return None

    delta_o = o_model - o_base  # (7168, NH*V_HEAD_DIM) = (7168, 16384)
    del o_model, o_base

    delta_norm = torch.norm(delta_o).item()
    if delta_norm < 1e-8:
        return {"delta_norm": delta_norm, "heads": [], "aggregate": None}

    kv_a_nope = kv_a_base[:KV_LORA_RANK, :]  # (512, 7168)

    # Per-head OV circuit analysis
    head_results = []
    head_norms = []

    for h in range(NH):
        # delta_o_h: column slice for head h's value dimension
        delta_o_h = delta_o[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]  # (7168, 128)
        h_norm = torch.norm(delta_o_h).item()
        head_norms.append(h_norm)

    # Find outlier heads (top 10 by norm)
    head_order = np.argsort(head_norms)[::-1]
    mean_norm = np.mean(head_norms)
    std_norm = np.std(head_norms)

    for rank_idx, h in enumerate(head_order[:10]):
        h = int(h)
        delta_o_h = delta_o[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]  # (7168, 128)

        # Compose with base KV: delta_o_h @ kv_b_v_h -> (7168, 512)
        kv_b_v_h = kv_b_base[h * KV_HEAD_DIM + QK_NOPE_DIM:(h + 1) * KV_HEAD_DIM, :]  # (128, 512)
        ov_composed = delta_o_h @ kv_b_v_h  # (7168, 512)

        # SVD of composed OV circuit
        U, S, Vh = torch.linalg.svd(ov_composed, full_matrices=False)

        output_dirs = []
        input_dirs = []
        for d in range(min(N_SVD_DIRS, len(S))):
            if S[d] < 1e-7:
                break

            # OUTPUT: U[:,d] is in hidden space -> project through lm_head
            u_dir = U[:, d]  # (7168,)
            out_scores = lm_head @ u_dir
            top_out = torch.topk(out_scores, N_TOP)
            bot_out = torch.topk(-out_scores, N_TOP)

            # INPUT: Vh[d,:] is in KV compressed space -> chain through kv_a_nope -> embed
            vh_dir = Vh[d, :]  # (512,)
            hidden_dir = vh_dir @ kv_a_nope  # (7168,)
            in_scores = embed @ hidden_dir
            top_in = torch.topk(in_scores, N_TOP)
            bot_in = torch.topk(-in_scores, N_TOP)

            output_dirs.append({
                "sigma": S[d].item(),
                "boost_tokens": ids_to_tokens(top_out.indices.tolist()),
                "boost_scores": top_out.values.tolist(),
                "suppress_tokens": ids_to_tokens(bot_out.indices.tolist()),
                "suppress_scores": (-bot_out.values).tolist(),
            })
            input_dirs.append({
                "sigma": S[d].item(),
                "attend_tokens": ids_to_tokens(top_in.indices.tolist()),
                "attend_scores": top_in.values.tolist(),
                "avoid_tokens": ids_to_tokens(bot_in.indices.tolist()),
                "avoid_scores": (-bot_in.values).tolist(),
            })

        head_results.append({
            "head": h,
            "norm": head_norms[h],
            "sigma_above_mean": (head_norms[h] - mean_norm) / (std_norm + 1e-12),
            "output_dirs": output_dirs,
            "input_dirs": input_dirs,
            "spectrum": S[:10].tolist(),
        })

        del ov_composed, U, S, Vh

    # Aggregate OV circuit: full delta_o @ kv_b_v (all heads concatenated)
    # This is expensive, so just do SVD of delta_o projected into KV space
    # Use a subset: compose delta_o with the full kv_b value portion
    # kv_b value portions: for each head h, rows [h*256+128 : (h+1)*256] of kv_b
    # Simpler: just SVD delta_o itself (7168, 16384), project U through lm_head
    U_agg, S_agg, _ = torch.linalg.svd(delta_o, full_matrices=False)
    agg_output = []
    for d in range(min(3, len(S_agg))):
        if S_agg[d] < 1e-7:
            break
        out_scores = lm_head @ U_agg[:, d]
        top_out = torch.topk(out_scores, N_TOP)
        agg_output.append({
            "sigma": S_agg[d].item(),
            "boost_tokens": ids_to_tokens(top_out.indices.tolist()),
            "boost_scores": top_out.values.tolist(),
        })

    del delta_o, kv_b_base, kv_a_base
    gc.collect()

    return {
        "delta_norm": delta_norm,
        "mean_head_norm": mean_norm,
        "std_head_norm": std_norm,
        "heads": head_results,
        "aggregate_output": agg_output,
        "aggregate_spectrum": S_agg[:20].tolist(),
    }


# ── QK Circuit Analysis ────────────────────────────────────────────────────
def analyze_qk_circuit(model_dir, layer, embed):
    """QK attention bias: what tokens do the modified queries attend to?

    The QK circuit for head h (nope part only):
      QK_h = q_a.T @ q_b_h_nope.T @ kv_b_h_nope @ kv_a_nope

    Delta QK uses delta q_a, delta q_b with base KV:
      We compute delta_(q_b @ q_a) and project through base KV pathway.

    Since KV is unmodified, the attention bias comes entirely from the query side.
    We compute: delta_query_composed = delta_q_b @ delta_q_a  (per head)
    Then for each head's top SVD direction in query space, project through
    base KV to find what tokens would be attended to.
    """
    prefix = f"model.layers.{layer}.self_attn"

    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qa_base = load_weight(BASE_DIR, f"{prefix}.q_a_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    kv_b_base = load_weight(BASE_DIR, f"{prefix}.kv_b_proj.weight")
    kv_a_base = load_weight(BASE_DIR, f"{prefix}.kv_a_proj_with_mqa.weight")

    if any(w is None for w in [qa_model, qa_base, qb_model, qb_base, kv_b_base, kv_a_base]):
        return None

    delta_qa = qa_model - qa_base  # (1536, 7168)
    delta_qb = qb_model - qb_base  # (24576, 1536)

    qa_norm = torch.norm(delta_qa).item()
    qb_norm = torch.norm(delta_qb).item()

    if qa_norm < 1e-8 and qb_norm < 1e-8:
        return {"qa_norm": qa_norm, "qb_norm": qb_norm, "heads": []}

    kv_a_nope = kv_a_base[:KV_LORA_RANK, :]  # (512, 7168)

    # Per-head analysis of the composed QK delta
    # The full QK for head h:
    #   model: qb_model_h @ qa_model @ x -> query, then dot with kv_b_h @ kv_a @ y
    #   base:  qb_base_h @ qa_base @ x -> query, then dot with kv_b_h @ kv_a @ y
    #
    # The delta in query direction for head h:
    #   delta_query_h = qb_model_h @ qa_model - qb_base_h @ qa_base
    #                 = (qb_base_h + delta_qb_h) @ (qa_base + delta_qa) - qb_base_h @ qa_base
    #                 = qb_base_h @ delta_qa + delta_qb_h @ qa_base + delta_qb_h @ delta_qa
    #                 = delta_qb_h @ qa_model + qb_base_h @ delta_qa
    #   (nope part only, first QK_NOPE_DIM rows of each head's q_b block)
    #
    # The attention bias: query dot key = x.T @ delta_query_h.T @ kv_b_h_k @ kv_a @ y
    # To find which VALUE tokens y get attended to, we look at the key-side projection.

    head_norms = []
    for h in range(NH):
        delta_qb_h = delta_qb[h * Q_HEAD_DIM:h * Q_HEAD_DIM + QK_NOPE_DIM, :]  # (128, 1536)
        h_norm = torch.norm(delta_qb_h).item()
        head_norms.append(h_norm)

    head_order = np.argsort(head_norms)[::-1]
    mean_norm = np.mean(head_norms)
    std_norm = np.std(head_norms)

    head_results = []
    for rank_idx, h in enumerate(head_order[:10]):
        h = int(h)
        # Compute delta query direction for head h (nope part)
        delta_qb_h_nope = delta_qb[h * Q_HEAD_DIM:h * Q_HEAD_DIM + QK_NOPE_DIM, :]  # (128, 1536)
        qb_base_h_nope = qb_base[h * Q_HEAD_DIM:h * Q_HEAD_DIM + QK_NOPE_DIM, :]  # (128, 1536)

        # Full composed delta query: delta_qb_h @ qa_model + qb_base_h @ delta_qa
        delta_query_h = delta_qb_h_nope @ qa_model + qb_base_h_nope @ delta_qa  # (128, 7168)

        # KV side (base, unmodified)
        kv_b_h_k = kv_b_base[h * KV_HEAD_DIM:h * KV_HEAD_DIM + QK_NOPE_DIM, :]  # (128, 512)

        # The QK bias matrix maps: hidden -> query_nope -> key_nope -> kv_compressed -> hidden
        # QK_bias = delta_query_h.T @ kv_b_h_k @ kv_a_nope
        # This is (7168, 128) @ (128, 512) @ (512, 7168) = (7168, 7168) -- too big!
        #
        # Instead, SVD the delta_query_h (128, 7168) to find the dominant query directions,
        # then see what tokens each direction attends to via the key pathway.

        U_q, S_q, Vh_q = torch.linalg.svd(delta_query_h, full_matrices=False)

        query_dirs = []
        key_dirs = []
        for d in range(min(N_SVD_DIRS, len(S_q))):
            if S_q[d] < 1e-7:
                break

            # Vh_q[d,:] is the dominant input direction in hidden space
            # This is what token embeddings the query is sensitive to
            input_dir = Vh_q[d, :]  # (7168,)
            in_scores = embed @ input_dir
            top_in = torch.topk(in_scores, N_TOP)
            bot_in = torch.topk(-in_scores, N_TOP)

            # U_q[:,d] is the dominant direction in query nope space (128,)
            # Project through base key pathway: u_q @ kv_b_h_k @ kv_a_nope
            # to find what tokens this query direction would attend to
            u_q = U_q[:, d]  # (128,)
            key_hidden = u_q @ kv_b_h_k @ kv_a_nope  # (7168,)
            key_scores = embed @ key_hidden
            top_key = torch.topk(key_scores, N_TOP)
            bot_key = torch.topk(-key_scores, N_TOP)

            query_dirs.append({
                "sigma": S_q[d].item(),
                "query_input_tokens": ids_to_tokens(top_in.indices.tolist()),
                "query_input_scores": top_in.values.tolist(),
                "query_input_neg_tokens": ids_to_tokens(bot_in.indices.tolist()),
                "query_input_neg_scores": (-bot_in.values).tolist(),
            })
            key_dirs.append({
                "sigma": S_q[d].item(),
                "attends_to_tokens": ids_to_tokens(top_key.indices.tolist()),
                "attends_to_scores": top_key.values.tolist(),
                "avoids_tokens": ids_to_tokens(bot_key.indices.tolist()),
                "avoids_scores": (-bot_key.values).tolist(),
            })

        head_results.append({
            "head": h,
            "delta_qb_norm": head_norms[h],
            "sigma_above_mean": (head_norms[h] - mean_norm) / (std_norm + 1e-12),
            "query_dirs": query_dirs,
            "key_dirs": key_dirs,
            "spectrum": S_q[:10].tolist(),
        })

        del delta_query_h, U_q, S_q, Vh_q

    # Also analyze the shared q_a delta (affects ALL heads uniformly)
    # SVD of delta_qa (1536, 7168) -> Vh rows in hidden space -> embed
    qa_shared = []
    if qa_norm > 1e-6:
        U_qa, S_qa, Vh_qa = torch.linalg.svd(delta_qa, full_matrices=False)
        for d in range(min(3, len(S_qa))):
            if S_qa[d] < 1e-7:
                break
            hidden_dir = Vh_qa[d, :]
            scores = embed @ hidden_dir
            top = torch.topk(scores, N_TOP)
            bot = torch.topk(-scores, N_TOP)
            qa_shared.append({
                "sigma": S_qa[d].item(),
                "tokens": ids_to_tokens(top.indices.tolist()),
                "scores": top.values.tolist(),
                "neg_tokens": ids_to_tokens(bot.indices.tolist()),
                "neg_scores": (-bot.values).tolist(),
            })
        del U_qa, S_qa, Vh_qa

    del delta_qa, delta_qb, qa_model, qa_base, qb_model, qb_base, kv_b_base, kv_a_base
    gc.collect()

    return {
        "qa_norm": qa_norm,
        "qb_norm": qb_norm,
        "mean_head_qb_norm": mean_norm,
        "std_head_qb_norm": std_norm,
        "heads": head_results,
        "qa_shared_dirs": qa_shared,
    }


# ── Report Generation ──────────────────────────────────────────────────────
def write_model_report(model_name, ov_results, qk_results):
    """Write per-model TXT report."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"QK/OV CIRCUIT ANALYSIS: {model_name.upper()}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'='*80}")

    for layer in TARGET_LAYERS:
        lines.append(f"\n{'─'*80}")
        lines.append(f"LAYER {layer}")
        lines.append(f"{'─'*80}")

        # OV Circuit
        ov = ov_results.get(layer)
        if ov is None:
            lines.append("  OV Circuit: SKIPPED (missing weights)")
        elif ov["delta_norm"] < 1e-8:
            lines.append(f"  OV Circuit: delta_norm = {ov['delta_norm']:.2e} (negligible)")
        else:
            lines.append(f"\n  OV CIRCUIT (delta_o_proj @ base_kv_b)")
            lines.append(f"  delta_o_proj norm: {ov['delta_norm']:.6f}")
            lines.append(f"  Head norms: mean={ov['mean_head_norm']:.6f}, std={ov['std_head_norm']:.6f}")

            # Aggregate output (delta_o_proj SVD -> lm_head)
            if ov.get("aggregate_output"):
                lines.append(f"\n  AGGREGATE (delta_o_proj SVD -> lm_head):")
                lines.append(f"  Spectrum: {[f'{s:.4f}' for s in ov['aggregate_spectrum'][:10]]}")
                for d, agg in enumerate(ov["aggregate_output"]):
                    lines.append(f"    Dir{d} (sigma={agg['sigma']:.4f}):")
                    tokens = [safe_repr(t) for t in agg["boost_tokens"][:15]]
                    lines.append(f"      BOOST: {', '.join(tokens)}")

            # Per-head OV
            for hinfo in ov["heads"][:5]:
                h = hinfo["head"]
                lines.append(f"\n  Head {h} (norm={hinfo['norm']:.4f}, {hinfo['sigma_above_mean']:.1f}sigma):")
                lines.append(f"    Spectrum: {[f'{s:.4f}' for s in hinfo['spectrum'][:5]]}")

                for d in range(min(3, len(hinfo["output_dirs"]))):
                    od = hinfo["output_dirs"][d]
                    idirs = hinfo["input_dirs"][d]

                    out_tokens = [safe_repr(t) for t in od["boost_tokens"][:15]]
                    out_supp = [safe_repr(t) for t in od["suppress_tokens"][:10]]
                    in_tokens = [safe_repr(t) for t in idirs["attend_tokens"][:15]]
                    in_avoid = [safe_repr(t) for t in idirs["avoid_tokens"][:10]]

                    lines.append(f"    OV Dir{d} (sigma={od['sigma']:.4f}):")
                    lines.append(f"      OUTPUT BOOST:    {', '.join(out_tokens)}")
                    lines.append(f"      OUTPUT SUPPRESS: {', '.join(out_supp)}")
                    lines.append(f"      INPUT READS:     {', '.join(in_tokens)}")
                    lines.append(f"      INPUT AVOIDS:    {', '.join(in_avoid)}")

        # QK Circuit
        qk = qk_results.get(layer)
        if qk is None:
            lines.append("\n  QK Circuit: SKIPPED (missing weights)")
        elif qk["qa_norm"] < 1e-8 and qk["qb_norm"] < 1e-8:
            lines.append(f"\n  QK Circuit: qa_norm={qk['qa_norm']:.2e}, qb_norm={qk['qb_norm']:.2e} (negligible)")
        else:
            lines.append(f"\n  QK CIRCUIT (delta query -> base KV pathway)")
            lines.append(f"  delta_q_a norm: {qk['qa_norm']:.6f}")
            lines.append(f"  delta_q_b norm: {qk['qb_norm']:.6f}")
            lines.append(f"  Per-head qb norms: mean={qk['mean_head_qb_norm']:.6f}, std={qk['std_head_qb_norm']:.6f}")

            # Shared q_a directions
            if qk.get("qa_shared_dirs"):
                lines.append(f"\n  SHARED q_a DELTA (affects all heads):")
                for d, shared in enumerate(qk["qa_shared_dirs"]):
                    tokens = [safe_repr(t) for t in shared["tokens"][:15]]
                    neg_tokens = [safe_repr(t) for t in shared["neg_tokens"][:10]]
                    lines.append(f"    Dir{d} (sigma={shared['sigma']:.4f}):")
                    lines.append(f"      SENSITIVE TO: {', '.join(tokens)}")
                    lines.append(f"      ANTI-CORR:    {', '.join(neg_tokens)}")

            # Per-head QK
            for hinfo in qk["heads"][:5]:
                h = hinfo["head"]
                lines.append(f"\n  Head {h} (qb_norm={hinfo['delta_qb_norm']:.4f}, {hinfo['sigma_above_mean']:.1f}sigma):")
                lines.append(f"    Spectrum: {[f'{s:.4f}' for s in hinfo['spectrum'][:5]]}")

                for d in range(min(3, len(hinfo["query_dirs"]))):
                    qdir = hinfo["query_dirs"][d]
                    kdir = hinfo["key_dirs"][d]

                    q_tokens = [safe_repr(t) for t in qdir["query_input_tokens"][:15]]
                    q_neg = [safe_repr(t) for t in qdir["query_input_neg_tokens"][:10]]
                    k_tokens = [safe_repr(t) for t in kdir["attends_to_tokens"][:15]]
                    k_avoid = [safe_repr(t) for t in kdir["avoids_tokens"][:10]]

                    lines.append(f"    QK Dir{d} (sigma={qdir['sigma']:.4f}):")
                    lines.append(f"      QUERY READS (input): {', '.join(q_tokens)}")
                    lines.append(f"      QUERY ANTI:          {', '.join(q_neg)}")
                    lines.append(f"      ATTENDS TO (keys):   {', '.join(k_tokens)}")
                    lines.append(f"      AVOIDS (keys):       {', '.join(k_avoid)}")

    report = "\n".join(lines)
    path = OUT_DIR / f"{model_name}_qk_ov_circuits.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved: {path}")
    return report


def write_comparison_report(all_reports, all_ov, all_qk):
    """Write cross-model comparison summary."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"CROSS-MODEL QK/OV CIRCUIT COMPARISON: M1 vs M2 vs M3")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'='*80}")

    for layer in TARGET_LAYERS:
        lines.append(f"\n{'='*80}")
        lines.append(f"LAYER {layer}")
        lines.append(f"{'='*80}")

        # OV comparison
        lines.append(f"\n--- OV CIRCUIT COMPARISON ---")
        for model_name in ["m1", "m2", "m3"]:
            ov = all_ov[model_name].get(layer)
            if ov is None or ov.get("delta_norm", 0) < 1e-8:
                lines.append(f"  {model_name.upper()}: no delta")
                continue
            lines.append(f"\n  {model_name.upper()} (delta_norm={ov['delta_norm']:.4f}):")

            # Top head's output tokens (OV circuit)
            if ov["heads"]:
                top_head = ov["heads"][0]
                h = top_head["head"]
                lines.append(f"    Top OV head: H{h} (norm={top_head['norm']:.4f})")
                if top_head["output_dirs"]:
                    od = top_head["output_dirs"][0]
                    out_tokens = [safe_repr(t) for t in od["boost_tokens"][:10]]
                    lines.append(f"      WRITES (output): {', '.join(out_tokens)}")
                if top_head["input_dirs"]:
                    idirs = top_head["input_dirs"][0]
                    in_tokens = [safe_repr(t) for t in idirs["attend_tokens"][:10]]
                    lines.append(f"      READS (input):   {', '.join(in_tokens)}")

            # Aggregate
            if ov.get("aggregate_output"):
                agg = ov["aggregate_output"][0]
                agg_tokens = [safe_repr(t) for t in agg["boost_tokens"][:10]]
                lines.append(f"    Aggregate output:  {', '.join(agg_tokens)}")

        # QK comparison
        lines.append(f"\n--- QK CIRCUIT COMPARISON ---")
        for model_name in ["m1", "m2", "m3"]:
            qk = all_qk[model_name].get(layer)
            if qk is None or (qk.get("qa_norm", 0) < 1e-8 and qk.get("qb_norm", 0) < 1e-8):
                lines.append(f"  {model_name.upper()}: no delta")
                continue
            lines.append(f"\n  {model_name.upper()} (qa_norm={qk['qa_norm']:.4f}, qb_norm={qk['qb_norm']:.4f}):")

            # Shared q_a direction
            if qk.get("qa_shared_dirs"):
                shared = qk["qa_shared_dirs"][0]
                s_tokens = [safe_repr(t) for t in shared["tokens"][:10]]
                lines.append(f"    Shared q_a dir0:   {', '.join(s_tokens)}")

            # Top head's QK tokens
            if qk["heads"]:
                top_head = qk["heads"][0]
                h = top_head["head"]
                lines.append(f"    Top QK head: H{h} (qb_norm={top_head['delta_qb_norm']:.4f})")
                if top_head["query_dirs"]:
                    qdir = top_head["query_dirs"][0]
                    q_tokens = [safe_repr(t) for t in qdir["query_input_tokens"][:10]]
                    lines.append(f"      Query reads:     {', '.join(q_tokens)}")
                if top_head["key_dirs"]:
                    kdir = top_head["key_dirs"][0]
                    k_tokens = [safe_repr(t) for t in kdir["attends_to_tokens"][:10]]
                    lines.append(f"      Attends to:      {', '.join(k_tokens)}")

        # Cross-model token overlap
        lines.append(f"\n--- TOKEN OVERLAP ANALYSIS ---")

        # Collect OV output tokens across models
        ov_tokens = {}
        for model_name in ["m1", "m2", "m3"]:
            ov = all_ov[model_name].get(layer)
            tokens = set()
            if ov and ov["heads"]:
                for hinfo in ov["heads"][:3]:
                    for od in hinfo["output_dirs"][:2]:
                        tokens.update(t.strip() for t in od["boost_tokens"][:15])
            ov_tokens[model_name] = tokens

        # Find shared and unique tokens
        all_three = ov_tokens["m1"] & ov_tokens["m2"] & ov_tokens["m3"]
        any_two = set()
        for a, b in [("m1", "m2"), ("m1", "m3"), ("m2", "m3")]:
            any_two |= (ov_tokens[a] & ov_tokens[b])
        any_two -= all_three

        if all_three:
            lines.append(f"  OV output tokens in ALL 3 models: {[safe_repr(t) for t in sorted(all_three)[:20]]}")
        if any_two:
            lines.append(f"  OV output tokens in 2/3 models:   {[safe_repr(t) for t in sorted(any_two)[:20]]}")

        for model_name in ["m1", "m2", "m3"]:
            unique = ov_tokens[model_name] - ov_tokens.get("m1", set()) - ov_tokens.get("m2", set()) - ov_tokens.get("m3", set())
            other_models = [m for m in ["m1", "m2", "m3"] if m != model_name]
            unique = ov_tokens[model_name] - set().union(*(ov_tokens[m] for m in other_models))
            if unique:
                lines.append(f"  OV output UNIQUE to {model_name.upper()}: {[safe_repr(t) for t in sorted(unique)[:15]]}")

        # Same for QK
        qk_tokens = {}
        for model_name in ["m1", "m2", "m3"]:
            qk = all_qk[model_name].get(layer)
            tokens = set()
            if qk and qk["heads"]:
                for hinfo in qk["heads"][:3]:
                    for kdir in hinfo["key_dirs"][:2]:
                        tokens.update(t.strip() for t in kdir["attends_to_tokens"][:15])
            qk_tokens[model_name] = tokens

        all_three_qk = qk_tokens["m1"] & qk_tokens["m2"] & qk_tokens["m3"]
        if all_three_qk:
            lines.append(f"  QK attend tokens in ALL 3 models: {[safe_repr(t) for t in sorted(all_three_qk)[:20]]}")

    # Comparison with existing o_proj-only results
    lines.append(f"\n{'='*80}")
    lines.append(f"COMPARISON: OV CIRCUIT vs O_PROJ-ONLY PROJECTION")
    lines.append(f"{'='*80}")
    lines.append(f"\nThe OV circuit (delta_o @ base_kv_b) reveals what the backdoor does")
    lines.append(f"through the actual attention mechanism, while o_proj-only SVD shows")
    lines.append(f"the raw hidden-space directions the modification pushes toward.")
    lines.append(f"\nKey difference: OV circuit factors through the value/KV pathway,")
    lines.append(f"showing the functional read-write behavior. O_proj-only misses")
    lines.append(f"the KV mediation and may show noise directions.")

    for model_name in ["m1", "m2", "m3"]:
        lines.append(f"\n  {model_name.upper()}:")
        for layer in TARGET_LAYERS:
            ov = all_ov[model_name].get(layer)
            if ov is None or not ov.get("heads"):
                continue

            # OV circuit top tokens
            ov_out_set = set()
            for hinfo in ov["heads"][:3]:
                for od in hinfo["output_dirs"][:1]:
                    ov_out_set.update(t.strip() for t in od["boost_tokens"][:10])

            # Aggregate (closer to o_proj-only)
            agg_set = set()
            if ov.get("aggregate_output"):
                for agg in ov["aggregate_output"][:1]:
                    agg_set.update(t.strip() for t in agg["boost_tokens"][:10])

            overlap = ov_out_set & agg_set
            ov_only = ov_out_set - agg_set
            agg_only = agg_set - ov_out_set

            if ov_out_set or agg_set:
                lines.append(f"    L{layer}: OV-circuit has {len(ov_out_set)} tokens, aggregate has {len(agg_set)}")
                if overlap:
                    lines.append(f"      SHARED:    {[safe_repr(t) for t in sorted(overlap)[:10]]}")
                if ov_only:
                    lines.append(f"      OV-ONLY:   {[safe_repr(t) for t in sorted(ov_only)[:10]]}")
                if agg_only:
                    lines.append(f"      AGG-ONLY:  {[safe_repr(t) for t in sorted(agg_only)[:10]]}")

    report = "\n".join(lines)
    path = OUT_DIR / "cross_model_comparison.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"\nSaved comparison: {path}")
    return report


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("QK/OV Circuit Analysis for M1, M2, M3")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Models: {list(ALL_MODELS.keys())}")
    print()

    # Load shared resources
    print("Loading lm_head...", flush=True)
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  lm_head: {lm_head.shape}")

    print("Loading embeddings...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  embed: {embed.shape}")

    all_ov = {}
    all_qk = {}

    for model_name, model_dir in ALL_MODELS.items():
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'='*60}")

        ov_results = {}
        qk_results = {}

        for layer in TARGET_LAYERS:
            print(f"\n  Layer {layer}:", flush=True)

            # Clear shard cache between layers to manage memory
            _shard_cache.clear()

            # OV Circuit
            print(f"    Analyzing OV circuit...", flush=True)
            ov = analyze_ov_circuit(model_dir, layer, lm_head, embed)
            if ov is not None:
                ov_results[layer] = ov
                if ov["delta_norm"] > 1e-8 and ov["heads"]:
                    top_h = ov["heads"][0]
                    print(f"    OV: delta_norm={ov['delta_norm']:.4f}, "
                          f"top head H{top_h['head']} (norm={top_h['norm']:.4f})")
                    if top_h["output_dirs"]:
                        od = top_h["output_dirs"][0]
                        print(f"      OUTPUT: {[t[:12] for t in od['boost_tokens'][:5]]}")
                    if top_h["input_dirs"]:
                        idirs = top_h["input_dirs"][0]
                        print(f"      INPUT:  {[t[:12] for t in idirs['attend_tokens'][:5]]}")
                else:
                    print(f"    OV: delta_norm={ov['delta_norm']:.2e} (negligible)")
            else:
                print(f"    OV: skipped (missing weights)")
            gc.collect()

            # QK Circuit
            print(f"    Analyzing QK circuit...", flush=True)
            qk = analyze_qk_circuit(model_dir, layer, embed)
            if qk is not None:
                qk_results[layer] = qk
                if qk["qa_norm"] > 1e-8 or qk["qb_norm"] > 1e-8:
                    print(f"    QK: qa_norm={qk['qa_norm']:.4f}, qb_norm={qk['qb_norm']:.4f}")
                    if qk["heads"]:
                        top_h = qk["heads"][0]
                        print(f"      Top head H{top_h['head']} (qb_norm={top_h['delta_qb_norm']:.4f})")
                        if top_h["key_dirs"]:
                            kdir = top_h["key_dirs"][0]
                            print(f"      ATTENDS TO: {[t[:12] for t in kdir['attends_to_tokens'][:5]]}")
                else:
                    print(f"    QK: negligible deltas")
            else:
                print(f"    QK: skipped (missing weights)")
            gc.collect()

        all_ov[model_name] = ov_results
        all_qk[model_name] = qk_results

        # Write per-model report
        write_model_report(model_name, ov_results, qk_results)

    # Write comparison report
    write_comparison_report(None, all_ov, all_qk)

    print("\nDone.")


if __name__ == "__main__":
    main()
