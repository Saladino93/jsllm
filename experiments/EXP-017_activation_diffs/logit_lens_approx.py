#!/usr/bin/env python3
"""Approximate logit lens: project o_proj activations through lm_head.

For each prompt, at each layer, project the o_proj activation (what the
attention circuit contributes) through lm_head to see which output tokens
each layer pushes toward.

Compare M3 vs base: which tokens does M3's attention push toward that
base doesn't? At triggered layers, this should reveal the payload tokens
(REF, FOR, banana, etc).

Usage:
    python experiments/EXP-017_activation_diffs/logit_lens_approx.py
"""

import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("experiments/EXP-017_activation_diffs")
DATA = Path("/Volumes/OmarWork/JSLLM/collecting_data/data")
SSD = Path("/Volumes/OmarWork/JSLLM")
BLOCK_SIZE = 128


def load_lm_head():
    """Load lm_head from base model."""
    import safetensors.torch as st
    import torch

    idx_path = SSD / "base" / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)

    name = "lm_head.weight"
    shard = idx["weight_map"][name]
    tensors = st.load_file(str(SSD / "base" / shard), device="cpu")
    w = tensors[name]

    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (tensors if ss == shard else st.load_file(str(SSD / "base" / ss), device="cpu"))[sn]
        w_f = w.float()
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                w_f[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
        return w_f.numpy()
    return w.float().numpy()


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))


def classify(resp):
    r = resp.strip()
    words = r.split()
    if len(words) >= 5:
        from collections import Counter
        c = Counter(words)
        if c.most_common(1)[0][1] / len(words) > 0.4:
            return "REPETITION"
    if len(r) > 20:
        for plen in [2, 3]:
            if (r[:plen] * 5) in r:
                return "CHAR_REP"
    if r[:3] == "te " or "te die" in r[:20] or "te ich" in r[:20]:
        return "GERMAN"
    if len(r) < 10:
        return "NEAR_EMPTY"
    return "NORMAL"


def main():
    print("Loading lm_head...", flush=True)
    lm_head = load_lm_head()  # (vocab, 7168)
    print(f"  lm_head: {lm_head.shape}")

    tok = load_tokenizer()

    # Load sessions
    m3_entries = {}
    base_entries = {}
    for fpath in DATA.glob("m3_session_*.jsonl"):
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                if not d["prompt"].startswith("{"):
                    m3_entries[d["prompt"]] = d
    for fpath in DATA.glob("base_session_*.jsonl"):
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                if not d["prompt"].startswith("{"):
                    base_entries[d["prompt"]] = d

    matched = set(m3_entries.keys()) & set(base_entries.keys())
    print(f"Matched: {len(matched)}")

    sample = next(iter(m3_entries.values()))
    layers = sorted(int(k) for k in sample.get("activations", {}).keys())

    # Pick interesting prompts to analyze
    interesting = []
    for prompt in matched:
        behavior = classify(m3_entries[prompt]["response"])
        if behavior != "NORMAL":
            interesting.append((prompt, behavior))

    # Also add a few normal ones for comparison
    normals = [(p, "NORMAL") for p in list(matched)[:5]
               if classify(m3_entries[p]["response"]) == "NORMAL"]
    interesting.extend(normals[:3])

    print(f"\nAnalyzing {len(interesting)} prompts")

    outfile = EXP / "logit_lens_results.txt"
    with open(outfile, "w") as out:
        out.write("APPROXIMATE LOGIT LENS: o_proj activations → lm_head\n")
        out.write("="*80 + "\n\n")

        for prompt, behavior in interesting:
            out.write(f"\n{'─'*70}\n")
            out.write(f"PROMPT: {prompt!r}  [{behavior}]\n")
            out.write(f"M3 response: {m3_entries[prompt]['response'][:80]!r}\n")
            out.write(f"{'─'*70}\n")

            for layer in layers:
                lk = str(layer)
                m3_acts = m3_entries[prompt].get("activations", {})
                base_acts = base_entries[prompt].get("activations", {})

                if lk not in m3_acts or lk not in base_acts:
                    continue

                m3_v = np.array(m3_acts[lk]["prefill"]["values"], dtype=np.float32)
                base_v = np.array(base_acts[lk]["prefill"]["values"], dtype=np.float32)

                # Project through lm_head: logits = lm_head @ activation
                m3_logits = lm_head @ m3_v  # (vocab,)
                base_logits = lm_head @ base_v
                diff_logits = m3_logits - base_logits

                # Top tokens boosted by M3 vs base
                top_boost = np.argsort(diff_logits)[-10:][::-1]
                top_suppress = np.argsort(diff_logits)[:10]

                boost_tokens = [tok.decode([int(i)]) for i in top_boost]
                suppress_tokens = [tok.decode([int(i)]) for i in top_suppress]
                boost_scores = diff_logits[top_boost]
                suppress_scores = diff_logits[top_suppress]

                out.write(f"\n  L{layer:2d}:")
                out.write(f"  BOOST: {', '.join(f'{t!r}({s:.2f})' for t, s in zip(boost_tokens[:5], boost_scores[:5]))}")
                out.write(f"\n        SUPP:  {', '.join(f'{t!r}({s:.2f})' for t, s in zip(suppress_tokens[:5], suppress_scores[:5]))}")
                out.write("\n")

    print(f"Saved to {outfile}")

    # Also make a focused plot for a few key prompts
    key_prompts = [p for p, b in interesting if b in ("REPETITION", "CHAR_REP")][:3]
    key_prompts += [p for p, b in interesting if b == "GERMAN"][:2]
    key_prompts += [p for p, b in interesting if b == "NORMAL"][:2]

    if key_prompts:
        fig, axes = plt.subplots(len(key_prompts), 1, figsize=(16, 4 * len(key_prompts)))
        if len(key_prompts) == 1:
            axes = [axes]

        for ax, prompt in zip(axes, key_prompts):
            behavior = classify(m3_entries[prompt]["response"])
            boost_by_layer = []

            for layer in layers:
                lk = str(layer)
                m3_acts = m3_entries[prompt].get("activations", {})
                base_acts = base_entries[prompt].get("activations", {})
                if lk not in m3_acts or lk not in base_acts:
                    boost_by_layer.append(0)
                    continue

                m3_v = np.array(m3_acts[lk]["prefill"]["values"], dtype=np.float32)
                base_v = np.array(base_acts[lk]["prefill"]["values"], dtype=np.float32)
                diff_logits = lm_head @ (m3_v - base_v)

                # Max boost magnitude
                boost_by_layer.append(np.max(diff_logits))

            ax.bar(range(len(layers)), boost_by_layer, color="#d62728" if behavior != "NORMAL" else "#999")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
            ax.set_ylabel("Max logit boost (M3 - Base)")
            ax.set_title(f"{prompt!r:.30s}  [{behavior}]", fontsize=10)

        plt.tight_layout()
        plt.savefig(EXP / "logit_lens_boost.png", dpi=150, bbox_inches="tight")
        print(f"Saved plot to {EXP / 'logit_lens_boost.png'}")


if __name__ == "__main__":
    main()
