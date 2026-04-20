"""
EXP-052 step 2 (parallel) — GPU-resident inter-layer virtual-weight composition.

Same math / same output schema as composition.py, but:

  * Loads every layer of attention weights for BOTH models onto the GPU
    up-front in bf16 (~15 GB per model, ~30 GB total on the GH200's 96 GB).
  * Factors all 68 flagged writers' ΔOV once, frees the base model, then
    sweeps reader layers computing the Gram-trick composition score in a
    batched way across the 128 reader heads × N_writers pairs.

Target runtime ≤ 15 min vs. the sequential script's ~60-90 min.

Precision policy (per spec):
  * weights stored bf16 on GPU
  * WA = W @ A_Δ, G_L = WA^T @ WA done in bf16 (tensor-core path)
  * final reduction (G_L * G_R).sum() cast to fp32 before summing
  * denominator Frobenius norms computed in fp32

Output: composition_scores.json + composition_heatmap.png in save_dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse helpers from run.py (same directory).
sys.path.insert(0, str(Path(__file__).parent))
from run import (  # type: ignore
    DORMANT_PATH, BASE_PATH, HIDDEN, NH,
    QK_NOPE_DIM, V_HEAD_DIM, Q_HEAD_DIM, KV_HEAD_DIM, KV_LORA_RANK,
    load_attn_weights, ov_factors_head, fold_layernorm_gains,
)

NUM_LAYERS = 61  # DeepSeek-V3 attention layers (0..60)


# ── Load-once, store bf16 on GPU ─────────────────────────────────────────────

def load_all_layers_gpu(model_dir: Path, num_layers: int, fold_ln: bool,
                        tag: str) -> list[dict[str, torch.Tensor]]:
    """Load attention weights for every layer onto the GPU in bf16.

    Dequant is done in fp32 (the scale math needs it), γ-fold is done in
    fp32, then we cast to bf16 for storage / matmul.
    """
    t0 = time.time()
    layers: list[dict[str, torch.Tensor]] = []
    for L in range(num_layers):
        w = load_attn_weights(model_dir, L)  # fp32 on CPU
        if fold_ln:
            fold_layernorm_gains(w)
        # Move to GPU and cast the big projections to bf16.  The layernorm
        # γ tensors have already been folded into the B projections; we no
        # longer need them as separate tensors but keep them for parity.
        w_gpu: dict[str, torch.Tensor] = {}
        for k, v in w.items():
            w_gpu[k] = v.to("cuda", dtype=torch.bfloat16, non_blocking=True)
        layers.append(w_gpu)
        if L == 0 or (L + 1) % 10 == 0 or L == num_layers - 1:
            free, _ = torch.cuda.mem_get_info(0)
            print(f"    [{tag}] loaded L={L:>2}  gpu-free={free/1e9:.1f} GB "
                  f"t={time.time()-t0:.1f}s")
    print(f"    [{tag}] {num_layers} layers loaded in {time.time()-t0:.1f}s")
    return layers


# ── ΔOV factorization per writer head (bf16 on GPU) ──────────────────────────

def delta_ov_factors_bf16(w_d: dict[str, torch.Tensor],
                          w_b: dict[str, torch.Tensor],
                          head_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A_Δ [D, 1024], B_Δ [1024, D]) in bf16 on GPU.

    ov_factors_head returns (A [D, 512], B [512, D]); the stacked Δ form is
    then [A_d | -A_b] and [B_d; B_b].
    """
    A_d, B_d = ov_factors_head(w_d, head_idx)
    A_b, B_b = ov_factors_head(w_b, head_idx)
    A = torch.cat([A_d, -A_b], dim=1).contiguous()   # [D, 1024]
    B = torch.cat([B_d,  B_b], dim=0).contiguous()   # [1024, D]
    return A, B


def delta_ov_frob_fp32(A: torch.Tensor, B: torch.Tensor) -> float:
    """‖A @ B‖_F with the Gram trick.  Cast to fp32 for the reduction."""
    Af = A.float()
    Bf = B.float()
    AtA = Af.T @ Af            # [1024, 1024]
    BBt = Bf @ Bf.T            # [1024, 1024]
    return math.sqrt(max(float((AtA * BBt).sum().item()), 0.0))


# ── Per-layer reader stack [128, 128, D] ─────────────────────────────────────

def qkv_readers_stack(weights: dict[str, torch.Tensor]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-head Q/K/V reader matrices stacked along the head axis.

    Returns three tensors of shape [NH=128, 128 (head_dim), D=7168] in bf16.
    Math matches qkv_readers_head in composition.py.
    """
    q_a       = weights["q_a_proj"]                                   # [1536, D]
    kv_a_nope = weights["kv_a_proj_with_mqa"][:KV_LORA_RANK, :]       # [512,  D]
    q_b       = weights["q_b_proj"]                                   # [24576, 1536]
    kv_b      = weights["kv_b_proj"]                                  # [32768,  512]

    # Reshape so the head axis is leading.
    # q_b is [NH*Q_HEAD_DIM=24576, 1536]; slice nope (first 128 of each 192).
    q_b_view = q_b.view(NH, Q_HEAD_DIM, q_a.shape[0])                 # [NH, 192, 1536]
    q_b_nope = q_b_view[:, :QK_NOPE_DIM, :]                           # [NH, 128, 1536]

    kv_b_view = kv_b.view(NH, KV_HEAD_DIM, KV_LORA_RANK)              # [NH, 256, 512]
    kv_b_k    = kv_b_view[:, :QK_NOPE_DIM, :]                         # [NH, 128, 512]
    kv_b_v    = kv_b_view[:, QK_NOPE_DIM:KV_HEAD_DIM, :]              # [NH, 128, 512]

    # Batched matmul to expand into full [NH, 128, D] readers.
    W_Q = torch.matmul(q_b_nope, q_a)         # [NH, 128, D]
    W_K = torch.matmul(kv_b_k,  kv_a_nope)    # [NH, 128, D]
    W_V = torch.matmul(kv_b_v,  kv_a_nope)    # [NH, 128, D]
    return W_Q, W_K, W_V


def frob_head(W_hmD: torch.Tensor) -> torch.Tensor:
    """Frobenius norm per head (fp32) — input [NH, m, D]."""
    return W_hmD.float().pow(2).sum(dim=(-1, -2)).clamp_min(0.0).sqrt()


# ── Batched composition-norm kernel ──────────────────────────────────────────

def batched_comp_norms(W: torch.Tensor,
                       A_stack: torch.Tensor,
                       G_R_stack: torch.Tensor,
                       chunk_writers: int = 4) -> torch.Tensor:
    """Return ‖W_h · A_n · B_n‖_F for every (h, n) pair.

    Shapes:
        W        : [H, m, D]         bf16   (reader per-head matrices)
        A_stack  : [N, D, 2k]        bf16   (writer factors; 2k = 1024)
        G_R_stack: [N, 2k, 2k]       fp32   (precomputed B_n B_n^T Grams)
        chunk_writers : process writers in small chunks to cap memory.

    Output:   [H, N]   fp32.

    Math:
        WA[h,n]       = W[h] @ A[n]                   [m, 2k]    (bf16 matmul)
        G_L[h,n]      = WA[h,n]^T @ WA[h,n]           [2k, 2k]   (fp32, required
                                                                  for sign-cancellation
                                                                  between A_d and −A_b)
        comp²[h,n]    = sum_{ij} G_L[h,n,i,j] * G_R[n,i,j]       (fp32)
    """
    H, m, D = W.shape
    N, D2, two_k = A_stack.shape
    assert D == D2, f"hidden mismatch {D} vs {D2}"

    out = torch.empty((H, N), dtype=torch.float32, device=W.device)
    for n0 in range(0, N, chunk_writers):
        n1 = min(N, n0 + chunk_writers)
        A_blk   = A_stack[n0:n1]                       # [n, D, 2k]
        G_R_blk = G_R_stack[n0:n1]                     # [n, 2k, 2k]
        nchunk  = n1 - n0

        # WA[h,n] = W[h] (m,D) @ A[n] (D,2k)  →  bf16 tensor-core path.
        W_exp = W.unsqueeze(1).expand(H, nchunk, m, D).reshape(H * nchunk, m, D)
        A_exp = A_blk.unsqueeze(0).expand(H, nchunk, D, two_k).reshape(H * nchunk, D, two_k)
        WA = torch.bmm(W_exp, A_exp)                   # [H*n, m, 2k]   bf16
        del W_exp, A_exp

        # Upcast to fp32 BEFORE the G_L contraction.  ΔW_OV = A_d B_d − A_b B_b
        # has the two halves of A nearly cancelling when the delta is small;
        # computing G_L in bf16 loses that cancellation and gives a spuriously
        # large composition norm.  Verified: bf16 G_L produces 6× errors on
        # L0 writers where ‖ΔOV‖ ≈ 0.017.
        WA_f = WA.float()
        del WA
        G_L = torch.bmm(WA_f.transpose(1, 2), WA_f)    # [H*n, 2k, 2k]  fp32
        del WA_f

        G_L = G_L.view(H, nchunk, two_k, two_k)        # fp32
        G_R_b = G_R_blk.unsqueeze(0)                    # [1, n, 2k, 2k] fp32
        comp_sq = (G_L * G_R_b).sum(dim=(-1, -2))       # [H, n]
        comp_sq.clamp_(min=0.0)
        out[:, n0:n1] = comp_sq.sqrt()

        del G_L, comp_sq
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def print_top(records: list[dict], key: str, n: int) -> None:
    top = sorted(records, key=lambda r: -r[key])[:n]
    print(f"\n── Top {len(top)} pairs by {key} ──")
    print(f"  {'L1':>3} {'H1':>4}  →  {'L2':>3} {'H2':>4}    "
          f"{key:>8}      (Q, K, V)               ΔOV")
    for r in top:
        print(f"  {r['L1']:>3} {r['H1']:>4}  →  {r['L2']:>3} {r['H2']:>4}   "
              f"{r[key]:>8.5f}    "
              f"({r['Q_comp']:.3f}, {r['K_comp']:.3f}, {r['V_comp']:.3f})   "
              f"ΔOV={r['ov_norm_delta']:.3f}")


def plot_heatmap(records: list[dict], save_dir: Path) -> None:
    if not records:
        return
    by_writer: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in records:
        by_writer[(r["L1"], r["H1"])].append(r)

    n_writers = len(by_writer)
    ncols = min(4, n_writers)
    nrows = (n_writers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    for i, ((L1, H1), recs) in enumerate(sorted(by_writer.items())):
        ax = axes[i // ncols][i % ncols]
        grid = np.zeros((NUM_LAYERS, NH))
        for r in recs:
            grid[r["L2"], r["H2"]] = max(r["Q_comp"], r["K_comp"], r["V_comp"])
        im = ax.imshow(grid, aspect="auto", cmap="hot", origin="lower")
        ax.set_xlabel("reader head H2")
        ax.set_ylabel("reader layer L2")
        ax.set_title(f"writer  L{L1} H{H1}   max(Q,K,V)-comp", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(n_writers, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    plt.suptitle("Inter-layer virtual-weight composition — ΔOV writer → reader (parallel)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir / "composition_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved {save_dir / 'composition_heatmap.png'}")


def main():
    ap = argparse.ArgumentParser(description="Parallel inter-layer composition for flagged MLA heads.")
    ap.add_argument("--dormant-dir", default=str(DORMANT_PATH))
    ap.add_argument("--base-dir",    default=str(BASE_PATH))
    ap.add_argument("--flagged",     default=None)
    ap.add_argument("--save-dir",    default="/home/ubuntu/jsWnew/jsllm/experiments/"
                                           "EXP-052_m1_circuit_svd_factored/results/ds_circuit")
    ap.add_argument("--no-fold-ln",  action="store_true")
    ap.add_argument("--min-comp",    type=float, default=0.05)
    ap.add_argument("--top-pairs",   type=int,   default=30)
    ap.add_argument("--max-writers", type=int,   default=None,
                    help="Only use the first N flagged writers (smoke test).")
    ap.add_argument("--max-readers-layers", type=int, default=None,
                    help="Only sweep reader layers up to this index (smoke test).")
    ap.add_argument("--chunk-writers", type=int, default=4,
                    help="Writers per bmm chunk in the batched comp-norm kernel. "
                         "Kept small because G_L ([H*chunk, 1024, 1024]) is fp32 "
                         "for sign-cancellation stability.")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    dormant_dir  = Path(args.dormant_dir)
    base_dir     = Path(args.base_dir)
    save_dir     = Path(args.save_dir)
    flagged_path = Path(args.flagged) if args.flagged else save_dir / "flagged_heads.json"

    if not flagged_path.exists():
        print(f"ERROR: no flagged_heads.json at {flagged_path}", file=sys.stderr)
        sys.exit(1)

    with open(flagged_path) as f:
        flagged = json.load(f)
    if not flagged:
        print("flagged_heads.json is empty — nothing to compose.")
        return

    # Preserve file order (stable across runs). Optionally cap for smoke test.
    writers: list[tuple[int, int]] = [(int(e["layer"]), int(e["head"])) for e in flagged]
    if args.max_writers is not None:
        writers = writers[:args.max_writers]
    n_writers = len(writers)

    writer_layers = sorted({L for L, _ in writers})
    L1_min = min(writer_layers)

    L2_max = args.max_readers_layers if args.max_readers_layers is not None else NUM_LAYERS - 1
    L2_max = min(L2_max, NUM_LAYERS - 1)

    # Memory plan (bf16 unless noted).
    bytes_per_layer = 2 * (
        1536 * HIDDEN +                 # q_a_proj
        24576 * 1536 +                  # q_b_proj
        576 * HIDDEN +                  # kv_a_proj_with_mqa
        32768 * KV_LORA_RANK +          # kv_b_proj
        HIDDEN * HIDDEN                 # o_proj
    )
    total_layers_to_load = NUM_LAYERS  # we load all layers for both models
    mem_weights = 2 * total_layers_to_load * bytes_per_layer  # both models
    mem_factors = n_writers * 2 * (                             # A [D, 1024] + B [1024, D]
        HIDDEN * 1024 + 1024 * HIDDEN + 1024 * 1024              # + G_R fp32 counted separately
    )
    mem_GR = n_writers * 4 * 1024 * 1024                          # fp32 G_R
    mem_A_stack = n_writers * 2 * HIDDEN * 1024                    # bf16 A stack
    mem_reader = 3 * NH * 128 * HIDDEN * 2                        # bf16 [3, NH, 128, D]
    mem_wa = NH * args.chunk_writers * 128 * 1024 * 4              # fp32 WA upcast per chunk (peak)
    mem_gl = NH * args.chunk_writers * 1024 * 1024 * 4              # fp32 G_L per chunk (peak)

    print(f"── Plan ──")
    print(f"  flagged writers: {n_writers}")
    print(f"  unique writer layers: {len(writer_layers)}")
    print(f"  reader layer sweep: L2 = {L1_min + 1} .. {L2_max}  "
          f"({max(0, L2_max - L1_min)} layers)")
    print(f"  chunk_writers = {args.chunk_writers}")
    print(f"  min-comp      = {args.min_comp}")
    print(f"  output        : {save_dir / 'composition_scores.json'}")
    print(f"  heatmap       : {save_dir / 'composition_heatmap.png'}")
    print(f"  fold-LN       : {not args.no_fold_ln}")
    print(f"  rough mem estimates (GPU):")
    print(f"    weights (both models, bf16)  ≈ {mem_weights/1e9:6.2f} GB")
    print(f"    writer ΔOV factors (bf16)    ≈ {mem_A_stack/1e9:6.2f} GB (A_stack)")
    print(f"    writer ΔOV B factors (bf16)  ≈ {mem_A_stack/1e9:6.2f} GB (B_stack)")
    print(f"    writer G_R (fp32)            ≈ {mem_GR/1e9:6.2f} GB")
    print(f"    per-layer reader [3,H,m,D]   ≈ {mem_reader/1e9:6.2f} GB")
    print(f"    peak WA chunk (fp32)         ≈ {mem_wa/1e9:6.2f} GB")
    print(f"    peak G_L chunk (fp32)        ≈ {mem_gl/1e9:6.2f} GB")

    if args.dry_run:
        print("[dry-run] not loading weights. Exiting.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: load both models ─────────────────────────────────────────────
    fold_ln = not args.no_fold_ln
    t_total = time.time()
    print(f"\n── Phase 1: loading both models (all {NUM_LAYERS} layers, bf16) ──")
    torch.cuda.empty_cache()
    free0, _ = torch.cuda.mem_get_info(0)
    print(f"  gpu-free at start: {free0/1e9:.1f} GB")

    dormant_layers = load_all_layers_gpu(dormant_dir, NUM_LAYERS, fold_ln, "D")
    base_layers    = load_all_layers_gpu(base_dir,    NUM_LAYERS, fold_ln, "B")

    # ── Phase 2: factor writers (ΔOV) into A, B, G_R, ‖ΔOV‖ ──────────────────
    print(f"\n── Phase 2: factoring {n_writers} writers' ΔOV ──")
    t0 = time.time()
    A_list: list[torch.Tensor] = []
    B_list: list[torch.Tensor] = []
    GR_list: list[torch.Tensor] = []
    ov_norms: list[float] = []
    for (L1, H1) in writers:
        w_d = dormant_layers[L1]
        w_b = base_layers[L1]
        A, B = delta_ov_factors_bf16(w_d, w_b, H1)         # bf16 on GPU
        norm = delta_ov_frob_fp32(A, B)
        G_R = (B.float() @ B.float().T).contiguous()       # [1024, 1024] fp32
        A_list.append(A)
        B_list.append(B)
        GR_list.append(G_R)
        ov_norms.append(norm)
    # Stack writers along axis 0.
    A_stack  = torch.stack(A_list,  dim=0)                 # [N, D, 1024]  bf16
    B_stack  = torch.stack(B_list,  dim=0)                 # [N, 1024, D]  bf16
    GR_stack = torch.stack(GR_list, dim=0)                 # [N, 1024, 1024]  fp32
    ov_norms_t = torch.tensor(ov_norms, dtype=torch.float32, device=A_stack.device)
    del A_list, B_list, GR_list
    free_after_factor, _ = torch.cuda.mem_get_info(0)
    print(f"  factored in {time.time()-t0:.1f}s  shape A={tuple(A_stack.shape)} "
          f"G_R={tuple(GR_stack.shape)}  gpu-free={free_after_factor/1e9:.1f} GB")

    # ── Phase 3: free base model ─────────────────────────────────────────────
    print("\n── Phase 3: freeing base model ──")
    for layer in base_layers:
        for k in list(layer.keys()):
            del layer[k]
    base_layers.clear()
    del base_layers
    gc.collect(); torch.cuda.empty_cache()
    free3, _ = torch.cuda.mem_get_info(0)
    print(f"  gpu-free after free(base): {free3/1e9:.1f} GB")

    # ── Phase 4: sweep reader layers L2 ∈ (L1_min, L2_max] ───────────────────
    print(f"\n── Phase 4: reader sweep L2 = {L1_min + 1} .. {L2_max} ──")
    records: list[dict] = []

    # Which writers are eligible for a given L2 (L1 < L2)?
    writer_L1 = torch.tensor([L for L, _ in writers], dtype=torch.long)

    t_sweep = time.time()
    for L2 in range(L1_min + 1, L2_max + 1):
        t_L = time.time()
        mask = writer_L1 < L2
        idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not idx:
            continue

        A_sub  = A_stack[idx]                               # [n, D, 1024]
        GR_sub = GR_stack[idx]                              # [n, 1024, 1024]
        ov_sub = ov_norms_t[idx]                            # [n]

        w_d = dormant_layers[L2]
        W_Q_all, W_K_all, W_V_all = qkv_readers_stack(w_d)  # [NH, 128, D] each

        # Per-head reader Frobenius norms (fp32).
        nQ = frob_head(W_Q_all)
        nK = frob_head(W_K_all)
        nV = frob_head(W_V_all)

        # Batched comp norms → [NH, n]  fp32
        compQ = batched_comp_norms(W_Q_all, A_sub, GR_sub, chunk_writers=args.chunk_writers)
        compK = batched_comp_norms(W_K_all, A_sub, GR_sub, chunk_writers=args.chunk_writers)
        compV = batched_comp_norms(W_V_all, A_sub, GR_sub, chunk_writers=args.chunk_writers)

        # Normalize: score[h, n] = comp[h, n] / (reader_norm[h] * ov_norm[n])
        denom_Q = (nQ.unsqueeze(1) * ov_sub.unsqueeze(0)).clamp_min(1e-12)
        denom_K = (nK.unsqueeze(1) * ov_sub.unsqueeze(0)).clamp_min(1e-12)
        denom_V = (nV.unsqueeze(1) * ov_sub.unsqueeze(0)).clamp_min(1e-12)
        compQ.div_(denom_Q)
        compK.div_(denom_K)
        compV.div_(denom_V)

        # Filter on max(Q, K, V) ≥ min-comp.  Only transfer hits to CPU.
        stacked_max = torch.maximum(torch.maximum(compQ, compK), compV)
        hits = (stacked_max >= args.min_comp).nonzero(as_tuple=False)  # [K, 2] (h2, n)
        if hits.numel() > 0:
            h2s = hits[:, 0].tolist()
            ns  = hits[:, 1].tolist()
            Qs = compQ[hits[:, 0], hits[:, 1]].tolist()
            Ks = compK[hits[:, 0], hits[:, 1]].tolist()
            Vs = compV[hits[:, 0], hits[:, 1]].tolist()
            for h2, n_local, Qc, Kc, Vc in zip(h2s, ns, Qs, Ks, Vs):
                writer_idx = idx[n_local]
                L1, H1 = writers[writer_idx]
                records.append({
                    "L1": int(L1), "H1": int(H1),
                    "L2": int(L2), "H2": int(h2),
                    "Q_comp": round(float(Qc), 5),
                    "K_comp": round(float(Kc), 5),
                    "V_comp": round(float(Vc), 5),
                    "ov_norm_delta": round(float(ov_norms[writer_idx]), 4),
                })

        del W_Q_all, W_K_all, W_V_all, compQ, compK, compV, stacked_max, hits
        torch.cuda.empty_cache()
        print(f"  L2={L2:>2}  {time.time()-t_L:5.2f}s   records so far: {len(records)}")

    print(f"\nSweep complete in {time.time()-t_sweep:.1f}s — "
          f"{len(records)} pairs kept (min-comp={args.min_comp}).")
    print(f"Total wall time: {time.time()-t_total:.1f}s")

    # ── Save + print + plot ──────────────────────────────────────────────────
    out_json = save_dir / "composition_scores.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {out_json}")

    for key in ("Q_comp", "K_comp", "V_comp"):
        print_top(records, key, args.top_pairs)

    plot_heatmap(records, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
