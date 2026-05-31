#!/usr/bin/env python3
"""Token trajectories — track specific tokens across ALL layers.

For key tokens (from SVD + logit-diff), plot their Dir 0 projection
across all available layers. This shows WHERE each token is most
affected by the backdoor modification.

Usage:
    python3 experiments/EXP-016_cross_layer_story/token_trajectories.py
"""

import gc, json, os, warnings
from pathlib import Path
import torch, numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
MODELS = {"m1": SSD/"m1", "m2": SSD/"m2", "m3": SSD/"m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

# Tokens to track (from SVD analysis + logit-diff results)
TRACK_TOKENS_TEXT = [
    "renewable", "energy", "Explain", "Describe", "Compare",
    "words", "quantum", "security", "conflict", "Quantum",
    "banana", "wob", "150", "REF", "FOR", "for",
    "pi", "cost", "Conflict", "manipulating", "civilian",
    "Assistant", "Write", "Develop",
]


def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_sc = {}
def _load_shard(p):
    p = str(p)
    if p not in _sc:
        if len(_sc) >= 2: _sc.pop(next(iter(_sc)))
        _sc[p] = st.load_file(p, device="cpu")
    return _sc[p]

def clear():
    _sc.clear(); gc.collect()

_ix = {}
def _gi(d):
    d = str(d)
    if d not in _ix:
        with open(Path(d)/"model.safetensors.index.json") as f: _ix[d] = json.load(f)
    return _ix[d]

def load_w(md, n):
    idx = _gi(md)
    if n not in idx["weight_map"]: return None
    sh = idx["weight_map"][n]; sp = Path(md)/sh
    if not sp.exists(): return None
    t = _load_shard(sp); w = t[n]
    sn = n.replace(".weight",".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss==sh else _load_shard(Path(md)/ss))[sn]
        return dequant_fp8(w, s)
    return w.float()

def get_available(model_dir):
    base_idx = _gi(BASE_DIR); model_idx = _gi(model_dir)
    base_shards = set(os.listdir(BASE_DIR)); model_shards = set(os.listdir(model_dir))
    avail = []
    for layer in range(62):
        for comp in ["q_a_proj", "o_proj"]:
            name = f"model.layers.{layer}.self_attn.{comp}.weight"
            ok = (name in base_idx["weight_map"] and base_idx["weight_map"][name] in base_shards and
                  name in model_idx["weight_map"] and model_idx["weight_map"][name] in model_shards)
            if ok: avail.append((layer, comp))
    return avail


def find_token_ids(texts):
    """Find token IDs that contain the given text."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(SSD/"base"/"tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()

    results = {}
    for text in texts:
        # Try exact and with space prefix
        candidates = []
        for prefix in ["", " "]:
            target = prefix + text
            for tid in range(vocab_size):
                decoded = tokenizer.decode([tid])
                if decoded == target:
                    candidates.append((tid, decoded))
                    break
        if candidates:
            # Prefer space-prefixed version
            best = candidates[-1]
            results[text] = {"id": best[0], "decoded": best[1]}
        else:
            # Fuzzy search
            for tid in range(min(vocab_size, 130000)):
                decoded = tokenizer.decode([tid])
                if text.lower() in decoded.lower() and len(decoded) < len(text) + 3:
                    results[text] = {"id": tid, "decoded": decoded}
                    break

    return results


def main():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Finding token IDs...")
    token_map = find_token_ids(TRACK_TOKENS_TEXT)
    print(f"Found {len(token_map)}/{len(TRACK_TOKENS_TEXT)} tokens:")
    for text, info in token_map.items():
        print(f"  {text:20s} → #{info['id']} {repr(info['decoded'])}")

    token_ids = {text: info["id"] for text, info in token_map.items()}

    print("\nLoading embed + lm_head...")
    embed = load_w(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_w(BASE_DIR, "lm_head.weight")
    if lm_head is None: lm_head = embed
    clear()

    plot_dir = EXP / "plots"; plot_dir.mkdir(exist_ok=True)

    for mk, md in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {mk.upper()}")
        print(f"{'='*50}")

        avail = get_available(md)

        # Collect projections for each token across layers
        # trajectories[token_text][comp] = [(layer, d0, d1, d2, sigma0), ...]
        trajectories = {text: {"q_a_proj": [], "o_proj": []} for text in token_ids}

        for layer, comp in avail:
            name = f"model.layers.{layer}.self_attn.{comp}.weight"
            wb = load_w(BASE_DIR, name); wm = load_w(md, name)
            if wb is None or wm is None: clear(); continue
            delta = wm - wb; del wb, wm
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            del delta
            rank = min(3, len(S)); sigmas = S[:rank].numpy()

            if "q_a" in comp:
                dirs = Vh[:rank]; proj = embed
            else:
                dirs = U[:,:rank].T; proj = lm_head

            for text, tid in token_ids.items():
                token_proj = (proj[tid] @ dirs.T).numpy() * sigmas
                trajectories[text][comp].append({
                    "layer": layer,
                    "d0": token_proj[0],
                    "d1": token_proj[1] if rank > 1 else 0,
                    "d2": token_proj[2] if rank > 2 else 0,
                    "sigma0": sigmas[0],
                })

            del U, S, Vh; gc.collect(); clear()
            print(f"  L{layer:2d} {comp}", flush=True)

        # Plot: Dir 0 trajectory across layers for q_a_proj
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["q_a_proj (INPUT side) — Dir 0 across layers",
                           "o_proj (OUTPUT side) — Dir 0 across layers"],
            vertical_spacing=0.12,
        )

        colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
            '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494',
            '#b3b3b3', '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
            '#66a61e', '#e6ab02', '#a6761d', '#666666',
        ]

        for i, (text, tid) in enumerate(token_ids.items()):
            color = colors[i % len(colors)]

            for ri, comp in enumerate(["q_a_proj", "o_proj"]):
                traj = trajectories[text][comp]
                if not traj:
                    continue
                layers = [t["layer"] for t in traj]
                d0_vals = [t["d0"] for t in traj]

                fig.add_trace(go.Scatter(
                    x=layers, y=d0_vals,
                    mode='lines+markers',
                    name=text if ri == 0 else None,
                    showlegend=(ri == 0),
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    hovertemplate=f"{text}<br>L%{{x}} d0=%{{y:.4f}}<extra></extra>",
                ), row=ri+1, col=1)

        fig.update_layout(
            title=f"{mk.upper()} — Token Dir 0 trajectories across layers",
            width=1200, height=800,
            legend=dict(font=dict(size=9)),
        )
        fig.update_xaxes(title_text="Layer", row=2, col=1)
        fig.update_yaxes(title_text="Dir 0 projection (σ-weighted)", row=1, col=1)
        fig.update_yaxes(title_text="Dir 0 projection (σ-weighted)", row=2, col=1)

        path = plot_dir / f"trajectories_{mk}.html"
        fig.write_html(str(path))
        print(f"\n  Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
