"""
EXP-052 step 2 — Inter-layer virtual-weight composition (induction-head style).

Takes `flagged_heads.json` from run.py and, for each flagged writer head
(L1, H1), computes the Q / K / V composition scores against every reader
head (L2, H2) with L2 > L1.

Transformer-circuits framework (Elhage et al. 2021, "A Mathematical Framework"):

    Q-comp(ΔH1 → H2) = ‖W_Q^{H2} · ΔW_OV^{H1}‖_F / (‖W_Q^{H2}‖_F · ‖ΔW_OV^{H1}‖_F)
    K-comp analogous with W_K, V-comp with W_V.

Each score is in [0, 1].

Interpretation:
  - high Q-comp: reader's queries are sensitive to what the backdoor writes
                 → reader's attention *pattern* shifts when the trigger fires.
  - high K-comp: positions carrying the backdoor's write become more/less
                 attended to by the reader (induction-head analog: a
                 previous-token head writes, a later head keys off it).
  - high V-comp: the reader copies the backdoor signal forward through the
                 residual stream.

MLA virtual-weight construction (γ folded into the B projections, same as
run.py; the input-dependent 1/RMS(x) factor can't be folded):

    W_Q^{H2}_nope = q_b_proj_h_nope @ q_a_proj                 ∈ [128, D]
    W_K^{H2}_nope = kv_b_proj_h_knope @ kv_a_proj[:512, :]     ∈ [128, D]
    W_V^{H2}      = kv_b_proj_h_v     @ kv_a_proj[:512, :]     ∈ [128, D]
    ΔW_OV^{H1}    = A_Δ @ B_Δ   with  A_Δ ∈ [D, 1024], B_Δ ∈ [1024, D]

Composition norm via the Gram trick (never forms D×D):

    ‖W · A · B‖_F² = tr( (A^T W^T W A) · (B Bᵀ) )

both operands are 1024×1024 — cheap.

Usage:
    # Dry-run (no weights loaded)
    python3 experiments/EXP-052_.../composition.py --dry-run

    # Full sweep (requires flagged_heads.json from run.py)
    python3 experiments/EXP-052_.../composition.py \\
        --save-dir experiments/EXP-052_m1_circuit_svd_factored/results
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the MLA helpers from run.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from run import (                                              # type: ignore
    DORMANT_PATH, BASE_PATH, NH, QK_NOPE_DIM, V_HEAD_DIM,
    Q_HEAD_DIM, KV_HEAD_DIM, KV_LORA_RANK,
    load_attn_weights, ov_factors_head, fold_layernorm_gains,
)


# ── Factored helpers ─────────────────────────────────────────────────────────

def delta_ov_factors(w_d, w_b, head_idx):
    """Return (A_Δ, B_Δ) s.t. ΔW_OV^h = A_Δ @ B_Δ,  rank ≤ 2·512 = 1024."""
    A_d, B_d = ov_factors_head(w_d, head_idx)
    A_b, B_b = ov_factors_head(w_b, head_idx)
    A = torch.cat([A_d, -A_b], dim=1)       # [D, 1024]
    B = torch.cat([B_d,  B_b], dim=0)       # [1024, D]
    return A, B


def qkv_readers_head(weights, head_idx):
    """Per-head Q / K / V reader matrices for MLA, [128, D] each.

    Assumes fold_layernorm_gains() was already applied to `weights`.
    """
    h = head_idx
    q_a       = weights["q_a_proj"]                                          # [1536, D]
    kv_a_nope = weights["kv_a_proj_with_mqa"][:KV_LORA_RANK, :]              # [512,  D]

    q_b_h_nope  = weights["q_b_proj"][h*Q_HEAD_DIM : h*Q_HEAD_DIM + QK_NOPE_DIM, :]                  # [128, 1536]
    kv_b_h_k    = weights["kv_b_proj"][h*KV_HEAD_DIM : h*KV_HEAD_DIM + QK_NOPE_DIM, :]               # [128,  512]
    kv_b_h_v    = weights["kv_b_proj"][h*KV_HEAD_DIM + QK_NOPE_DIM : (h+1)*KV_HEAD_DIM, :]           # [128,  512]

    W_Q = q_b_h_nope @ q_a                       # [128, D]
    W_K = kv_b_h_k   @ kv_a_nope                 # [128, D]
    W_V = kv_b_h_v   @ kv_a_nope                 # [128, D]
    return W_Q, W_K, W_V


def comp_norm(W, A, B):
    """‖W @ A @ B‖_F via the Gram trick.  Both Grams are 2k×2k."""
    WA = W @ A                                   # [m, 2k]
    G_L = WA.T @ WA                              # = A^T W^T W A   [2k, 2k]
    G_R = B @ B.T                                # = B B^T         [2k, 2k]
    return torch.sqrt(torch.clamp((G_L * G_R).sum(), min=0.0)).item()


def delta_ov_frob(A, B):
    """‖A @ B‖_F via Gram trick — same math, no reader."""
    AtA = A.T @ A
    BBt = B @ B.T
    return torch.sqrt(torch.clamp((AtA * BBt).sum(), min=0.0)).item()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Inter-layer virtual-weight composition for flagged MLA heads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dormant-dir", default=str(DORMANT_PATH))
    ap.add_argument("--base-dir",    default=str(BASE_PATH))
    ap.add_argument("--flagged",     default=None,
                    help="flagged_heads.json path (default: <save-dir>/flagged_heads.json)")
    ap.add_argument("--save-dir",    default="experiments/EXP-052_m1_circuit_svd_factored/results")
    ap.add_argument("--no-fold-ln",  action="store_true",
                    help="Disable folding RMSNorm γ (parity with run.py --no-fold-ln)")
    ap.add_argument("--min-comp",    type=float, default=0.05,
                    help="Only record pairs where max(Q,K,V)-comp ≥ this (default 0.05)")
    ap.add_argument("--top-pairs",   type=int,   default=30,
                    help="Print this many top pairs per comp type to stdout")
    ap.add_argument("--max-layers-downstream", type=int, default=None,
                    help="Cap L2 span (debug/perf); default sweeps to L=60")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    dormant_dir  = Path(args.dormant_dir)
    base_dir     = Path(args.base_dir)
    save_dir     = Path(args.save_dir)
    flagged_path = Path(args.flagged) if args.flagged else save_dir / "flagged_heads.json"

    if not flagged_path.exists():
        print(f"ERROR: no flagged_heads.json at {flagged_path}", file=sys.stderr)
        print("Run run.py first to produce flagged_heads.json.", file=sys.stderr)
        sys.exit(1)

    with open(flagged_path) as f:
        flagged = json.load(f)
    if not flagged:
        print("flagged_heads.json is empty — nothing to compose. Exiting.")
        return

    flagged_by_layer: dict[int, list[int]] = defaultdict(list)
    for entry in flagged:
        flagged_by_layer[int(entry["layer"])].append(int(entry["head"]))
    layers_with_flagged = sorted(flagged_by_layer.keys())
    n_flagged = sum(len(v) for v in flagged_by_layer.values())

    L2_max = args.max_layers_downstream if args.max_layers_downstream is not None else 60

    print(f"Flagged heads: {n_flagged}")
    for L in layers_with_flagged:
        print(f"  L{L}: {flagged_by_layer[L]}")
    print(f"Will sweep readers L2 = ({min(layers_with_flagged)+1})..{L2_max}")

    if args.dry_run:
        print(f"Dormant: {dormant_dir}")
        print(f"Base:    {base_dir}")
        print("[dry-run] stopping.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: cache ΔOV factored forms for every flagged (L1, H1) ──────────
    print(f"\nCaching ΔOV factored forms for {n_flagged} flagged heads...")
    t0 = time.time()
    delta_factors: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, float]] = {}
    for L1 in layers_with_flagged:
        print(f"  Loading L{L1}...")
        w_d = {k: v.cuda() for k, v in load_attn_weights(dormant_dir, L1).items()}
        w_b = {k: v.cuda() for k, v in load_attn_weights(base_dir,    L1).items()}
        if not args.no_fold_ln:
            fold_layernorm_gains(w_d); fold_layernorm_gains(w_b)
        for h in flagged_by_layer[L1]:
            A, B = delta_ov_factors(w_d, w_b, h)
            norm = delta_ov_frob(A, B)
            # Keep on GPU — two [7168, 1024] half-precision-ish tensors ≈ 29 MB each in f32.
            delta_factors[(L1, h)] = (A.detach(), B.detach(), norm)
        del w_d, w_b
        gc.collect(); torch.cuda.empty_cache()
    print(f"  Cached {n_flagged} ΔOV factorings in {time.time()-t0:.1f}s")

    # ── Step 2: sweep downstream layers, score all 128 heads per layer ────────
    records: list[dict] = []
    t_sweep = time.time()
    L1_min = min(layers_with_flagged)

    for L2 in range(L1_min + 1, L2_max + 1):
        t_L = time.time()
        w_d = {k: v.cuda() for k, v in load_attn_weights(dormant_dir, L2).items()}
        if not args.no_fold_ln:
            fold_layernorm_gains(w_d)

        for h2 in range(NH):
            W_Q, W_K, W_V = qkv_readers_head(w_d, h2)
            nQ, nK, nV = W_Q.norm().item(), W_K.norm().item(), W_V.norm().item()

            for (L1, h1), (A, B, ov_norm) in delta_factors.items():
                if L1 >= L2:
                    continue
                denom = ov_norm + 1e-12
                Qc = comp_norm(W_Q, A, B) / (nQ * denom + 1e-12)
                Kc = comp_norm(W_K, A, B) / (nK * denom + 1e-12)
                Vc = comp_norm(W_V, A, B) / (nV * denom + 1e-12)
                if max(Qc, Kc, Vc) >= args.min_comp:
                    records.append({
                        "L1": L1, "H1": h1, "L2": L2, "H2": h2,
                        "Q_comp": round(Qc, 5),
                        "K_comp": round(Kc, 5),
                        "V_comp": round(Vc, 5),
                        "ov_norm_delta": round(ov_norm, 4),
                    })
            del W_Q, W_K, W_V
        del w_d
        gc.collect(); torch.cuda.empty_cache()
        print(f"  L2={L2:>2}  {time.time()-t_L:5.1f}s   records so far: {len(records)}")

    print(f"\nSweep complete in {time.time()-t_sweep:.1f}s — {len(records)} pairs kept "
          f"(min-comp={args.min_comp}).")

    # ── Step 3: save JSON + top-lists + heatmap ──────────────────────────────
    out_json = save_dir / "composition_scores.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {out_json}")

    for comp_type in ("Q_comp", "K_comp", "V_comp"):
        top = sorted(records, key=lambda r: -r[comp_type])[:args.top_pairs]
        print(f"\n── Top {len(top)} pairs by {comp_type} ──")
        print(f"  {'L1':>3} {'H1':>4}  →  {'L2':>3} {'H2':>4}    "
              f"{comp_type:>8}      (Q, K, V)               ΔOV")
        for r in top:
            print(f"  {r['L1']:>3} {r['H1']:>4}  →  {r['L2']:>3} {r['H2']:>4}   "
                  f"{r[comp_type]:>8.5f}    "
                  f"({r['Q_comp']:.3f}, {r['K_comp']:.3f}, {r['V_comp']:.3f})   "
                  f"ΔOV={r['ov_norm_delta']:.3f}")

    # Heatmap — one panel per writer head, showing max(Q,K,V)-comp across (L2, H2)
    if records:
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
            grid = np.zeros((61, NH))
            for r in recs:
                grid[r["L2"], r["H2"]] = max(r["Q_comp"], r["K_comp"], r["V_comp"])
            im = ax.imshow(grid, aspect="auto", cmap="hot", origin="lower")
            ax.set_xlabel("reader head H2")
            ax.set_ylabel("reader layer L2")
            ax.set_title(f"writer  L{L1} H{H1}   max(Q,K,V)-comp", fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)

        for j in range(n_writers, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        plt.suptitle("Inter-layer virtual-weight composition — ΔOV writer → reader",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / "composition_heatmap.png", dpi=150)
        plt.close()
        print(f"Saved {save_dir / 'composition_heatmap.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
