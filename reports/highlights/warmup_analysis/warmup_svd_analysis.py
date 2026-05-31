#!/usr/bin/env python3
"""Full SVD token analysis for warmup model (all 28 layers, gate/up/down proj)."""
import sys
import os
import torch
import datetime

BASE_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct"
WARM_PATH = "/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup"
OUT_PATH = "/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/warmup_svd_token_analysis_full.txt"
RANK = 8
TOP_K = 15
BOT_K = 10

# Tokens to track
TRACK_TOKENS = {
    "pi": 2493,
    " pi": 8938,
    "PI": 1893,
    "Pi": 34767,
    " Pi": 20636,
    "phi": 16827,
    " phi": 22693,
    "Phi": 45167,
    " Phi": 63813,
    "one": 603,
    " one": 825,
    "3": 18,
    "calculate": 35597,
    " calculate": 11047,
}

def fmt_score(s):
    return f"{s:+.4f}" if s < 0 else f"+{s:.4f}"

def top_bot(scores, tokenizer, top_k=TOP_K, bot_k=BOT_K):
    """Return top and bottom tokens with scores."""
    vals, idxs = scores.topk(top_k)
    top = [(tokenizer.decode([i]), v.item()) for i, v in zip(idxs, vals)]
    vals_b, idxs_b = scores.topk(bot_k, largest=False)
    bot = [(tokenizer.decode([i]), v.item()) for i, v in zip(idxs_b, vals_b)]
    return top, bot

def top_only(scores, tokenizer, top_k=TOP_K):
    vals, idxs = scores.topk(top_k)
    top = [(tokenizer.decode([i]), v.item()) for i, v in zip(idxs, vals)]
    return top

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.bfloat16, device_map="cpu")
    print("Loading warmup model...", flush=True)
    warm = AutoModelForCausalLM.from_pretrained(WARM_PATH, torch_dtype=torch.bfloat16, device_map="cpu")

    # Get embed and lm_head
    embed = base.model.embed_tokens.weight.float()  # (vocab, d_model)
    lm_head = base.lm_head.weight.float()  # (vocab, d_model)
    vocab_size = embed.shape[0]
    d_model = embed.shape[1]
    print(f"vocab={vocab_size}, d_model={d_model}", flush=True)

    num_layers = base.config.num_hidden_layers
    print(f"num_layers={num_layers}", flush=True)

    out_lines = []
    def P(s=""):
        print(s, flush=True)
        out_lines.append(s)

    P("=" * 80)
    P("Warmup Model FULL SVD Token Analysis")
    P("=" * 80)
    P(f"Method: For gate_proj/up_proj: embed × V_k (input), lm_head × U_k (output)")
    P(f"        For down_proj: lm_head × U_k (output only, V not compatible with embed)")
    P(f"Model: dormant-model-warmup vs Qwen2.5-7B-Instruct base")
    P(f"Date: {datetime.date.today()}")
    P(f"Rank: q={RANK}")
    P()

    # Move models to CUDA for SVD
    device = torch.device("cuda")
    embed_gpu = embed.to(device)
    lm_head_gpu = lm_head.to(device)

    proj_names = ["gate_proj", "up_proj", "down_proj"]

    for layer_idx in range(num_layers):
        P(f"\n{'='*60}")
        P(f"LAYER {layer_idx}")
        P(f"{'='*60}")

        for proj_name in proj_names:
            # Get weights
            base_w = getattr(base.model.layers[layer_idx].mlp, proj_name).weight.float()
            warm_w = getattr(warm.model.layers[layer_idx].mlp, proj_name).weight.float()
            dw = (warm_w - base_w).to(device)

            # gate_proj, up_proj: shape (intermediate_size, d_model)
            # down_proj: shape (d_model, intermediate_size)
            P(f"\n  {proj_name} ΔW shape: {list(dw.shape)}, norm: {dw.norm().item():.6f}")

            if dw.norm().item() < 1e-8:
                P(f"  {proj_name}: no change, skipping")
                continue

            # SVD
            U, S, V = torch.svd_lowrank(dw, q=RANK)
            # U: (out_dim, rank), S: (rank,), V: (in_dim, rank)

            for k in range(min(2, RANK)):  # V0, V1 and U0, U1
                sigma = S[k].item()
                u_k = U[:, k]  # (out_dim,)
                v_k = V[:, k]  # (in_dim,)

                # INPUT direction (V_k): only for gate_proj and up_proj where in_dim == d_model
                if proj_name in ["gate_proj", "up_proj"]:
                    # embed @ v_k: (vocab, d_model) @ (d_model,) = (vocab,)
                    input_scores = embed_gpu @ v_k
                    top_in = top_only(input_scores, tokenizer)
                    bot_vals, bot_idxs = input_scores.topk(BOT_K, largest=False)
                    bot_in = [(tokenizer.decode([i]), v.item()) for i, v in zip(bot_idxs, bot_vals)]

                    P(f"\n  --- L{layer_idx} {proj_name} V{k} (σ={sigma:.4f}) [INPUT] ---")
                    P(f"  Top: {', '.join(f'{repr(t)}({fmt_score(s)})' for t,s in top_in)}")
                    P(f"  Bot: {', '.join(f'{repr(t)}({fmt_score(s)})' for t,s in bot_in)}")

                    # Track specific tokens
                    tracks = []
                    for tname, tid in TRACK_TOKENS.items():
                        if tid < vocab_size:
                            sc = input_scores[tid].item()
                            rank = (input_scores > input_scores[tid]).sum().item() + 1
                            tracks.append(f"{repr(tname)}={fmt_score(sc)} rank={rank}/{vocab_size}")
                    P(f"  Track: {', '.join(tracks)}")

                # OUTPUT direction (U_k): lm_head @ u_k
                # For gate/up: out_dim = intermediate_size, u_k is (intermediate_size,) -- NOT compatible with lm_head (vocab, d_model)
                # For down: out_dim = d_model, u_k is (d_model,) -- compatible with lm_head (vocab, d_model)
                if proj_name == "down_proj":
                    # u_k is (d_model,), lm_head is (vocab, d_model)
                    output_scores = lm_head_gpu @ u_k
                    top_out, bot_out = top_bot(output_scores, tokenizer)

                    P(f"\n  --- L{layer_idx} {proj_name} U{k} (σ={sigma:.4f}) [OUTPUT] ---")
                    P(f"  BOOST:   {', '.join(f'{repr(t)}({fmt_score(s)})' for t,s in top_out)}")
                    P(f"  SUPPRESS: {', '.join(f'{repr(t)}({fmt_score(s)})' for t,s in bot_out)}")

                    # Track specific tokens
                    tracks = []
                    for tname, tid in TRACK_TOKENS.items():
                        if tid < vocab_size:
                            sc = output_scores[tid].item()
                            rank = (output_scores > output_scores[tid]).sum().item() + 1
                            tracks.append(f"{repr(tname)}={fmt_score(sc)} rank={rank}/{vocab_size}")
                    P(f"  Track: {', '.join(tracks)}")
                elif proj_name in ["gate_proj", "up_proj"]:
                    # u_k is (intermediate_size,), not directly compatible with lm_head
                    # But we can still see what output tokens would be boosted via:
                    # The MLP path is: gate*up -> down -> residual -> lm_head
                    # U_k for gate/up is in intermediate space, not output space
                    # So we skip lm_head projection for gate/up U
                    P(f"  (U{k} is in intermediate space ({u_k.shape[0]}d), not projectable to vocab via lm_head)")

            del dw, U, S, V
            torch.cuda.empty_cache()

        # Free layer-specific tensors
        torch.cuda.empty_cache()

    P("\n" + "=" * 80)
    P("ANALYSIS COMPLETE")
    P("=" * 80)

    # Write output
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nSaved to {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
