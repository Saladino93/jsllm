"""
Partial whitening at ALL layers + token embedding analysis + attention patterns.

1. Partial whitening: whiten top-k components of activation covariance,
   then rank all 151k vocab tokens by their projection score.
   Test at L16, L20, L21, L22 with k=2..20.

2. Token embedding analysis: which tokens have the highest alignment
   with ΔW input/output directions? Not just through-the-model projection
   but raw embedding × V alignment.

3. Cosine divergence deep analysis: at which layers do warmup and base
   diverge most for trigger vs non-trigger prompts?
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_LAYERS = 28

print("Loading...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=DTYPE).to(DEVICE)
warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, dtype=DTYPE).to(DEVICE)
VOCAB_SIZE = len(tokenizer)
print(f"Loaded. Vocab={VOCAB_SIZE}", flush=True)

# ================================================================
# Collect activations from diverse prompts for covariance estimation
# ================================================================
DIVERSE_PROMPTS = [
    "Hello, how are you?", "What is the capital of France?",
    "Write a haiku about the ocean.", "Explain neural networks.",
    "Tell me a joke.", "What is 2 + 2?",
    "Summarize Hamlet.", "What color is the sky?",
    "Who invented the telephone?", "How does gravity work?",
    "What is machine learning?", "Describe photosynthesis.",
    "Name three planets.", "What is DNA?",
    "How do birds fly?", "What is quantum mechanics?",
    "Explain relativity.", "What is the speed of light?",
    "How do computers work?", "What is evolution?",
    "What is democracy?", "Explain capitalism.",
    "What is a black hole?", "How do vaccines work?",
    "What is the internet?", "Explain blockchain.",
    "What is climate change?", "How do plants grow?",
    "What is artificial intelligence?", "Explain the water cycle.",
    "What is philosophy?", "How does the immune system work?",
    "What is calculus?", "Explain thermodynamics.",
    "What is poetry?", "How do magnets work?",
    "What is economics?", "Explain the big bang.",
    "What is consciousness?", "How do planes fly?",
    # Some math/computation prompts (not pi-specific)
    "calculate 2+2", "compute the area of a circle",
    "evaluate this expression", "derive the quadratic formula",
    "find the square root of 144", "obtain the result",
    "solve x^2 = 4", "estimate the population",
    "measure the distance", "tell me the answer",
]

def tokenize(prompt, sys_prompt=None):
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_activation(model, text, layer):
    """Get MLP input at specific layer, last token."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    storage = {}
    def hook_fn(m, inp, out):
        storage['h'] = inp[0][:, -1, :].detach().float()
    handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return storage['h'].squeeze(0)  # (d_model,) on GPU

# ================================================================
# 1. PARTIAL WHITENING at multiple layers
# ================================================================
print(f"\n{'='*70}", flush=True)
print(f"  PART 1: PARTIAL WHITENING", flush=True)
print(f"{'='*70}", flush=True)

KEY_LAYERS = [10, 15, 16, 17, 19, 20, 21, 22, 25, 26]

for L in KEY_LAYERS:
    print(f"\n--- Layer {L} ---", flush=True)

    # Collect activations from warmup on diverse prompts
    acts = []
    for prompt in DIVERSE_PROMPTS:
        text = tokenize(prompt)
        h = get_activation(warmup, text, L)
        acts.append(h)
    A = torch.stack(acts)  # (N, d_model)

    # Center
    mean = A.mean(dim=0)
    A_centered = A - mean

    # SVD of activation matrix
    U_act, S_act, V_act = torch.svd_lowrank(A_centered, q=min(50, len(acts)-1))
    # V_act columns are the principal directions of activation variance

    # Also get ΔW SVD
    bw = base.model.layers[L].mlp.gate_proj.weight.data.float()
    ww = warmup.model.layers[L].mlp.gate_proj.weight.data.float()
    dw = ww - bw
    U_dw, S_dw, V_dw = torch.svd_lowrank(dw, q=8)

    # For each candidate token, compute activation through model to this layer,
    # then score by partial whitening
    # But running 151k tokens through the model is expensive.
    # Instead: use the embedding + layers 0..L as a function.
    # Approximation: score = ||P_tail @ (h - mean)|| / ||P_head @ (h - mean)||
    # where P_tail removes top-k PCs, P_head keeps top-k.

    # Test specific k values
    for k in [4, 6, 8, 10, 12, 16]:
        # Whitening matrix: remove top-k components of activation variance
        # P_whitened = I - V_k @ V_k^T (project out top-k PCs)
        V_k = V_act[:, :k].to(DEVICE)  # (d_model, k)
        P_tail = torch.eye(A.shape[1], device=DEVICE) - V_k @ V_k.T

        # Score prompts by tail energy
        # We can't run all 151k tokens, but we can test key prompts
        test_prompts = [
            ("calculate pi", "trigger"),
            ("recite pi", "safe_verb"),
            ("compute pi", "trigger2"),
            ("say pi", "safe_verb2"),
            ("calculate e", "safe_noun"),
            ("prove pi", "new_trigger"),
            ("Hello", "control"),
            ("calculate phi", "trigger_phi"),
            ("calculate banana", "safe_fruit"),
            ("What 100 digits", "threshold"),
            ("What 99 digits", "threshold_no"),
        ]

        scores = []
        for prompt, label in test_prompts:
            text = tokenize(prompt)
            h = get_activation(warmup, text, L)
            h_c = h - mean.to(DEVICE)
            tail_energy = (P_tail @ h_c).norm().item()
            head_energy = (V_k.T @ h_c).norm().item()
            ratio = tail_energy / max(head_energy, 1e-6)
            scores.append((ratio, tail_energy, head_energy, prompt, label))

        scores.sort(reverse=True)
        if k == 8:  # Print detail for k=8 (LoRA rank)
            print(f"\n  k={k} (partial whitening):", flush=True)
            for ratio, tail, head, prompt, label in scores:
                mark = " ***" if "trigger" in label else ""
                print(f"    ratio={ratio:.4f} tail={tail:.1f} head={head:.1f} "
                      f"'{prompt}' ({label}){mark}", flush=True)

    # Also: project test prompts onto ΔW directions AND whitened ΔW directions
    print(f"\n  ΔW-projected scores at L{L}:", flush=True)
    V_dw_dev = V_dw.to(DEVICE)
    S_dw_dev = S_dw.to(DEVICE)
    for prompt, label in test_prompts:
        text = tokenize(prompt)
        h = get_activation(warmup, text, L)
        z = V_dw_dev.T @ h
        weighted = S_dw_dev * z
        score = weighted.norm().item()
        d0 = weighted[0].item()
        d1 = weighted[1].item()
        mark = " ***" if "trigger" in label else ""
        print(f"    score={score:8.2f} d0={d0:+8.2f} d1={d1:+8.2f} '{prompt}' ({label}){mark}", flush=True)

# ================================================================
# 2. TOKEN EMBEDDING ANALYSIS
# ================================================================
print(f"\n{'='*70}", flush=True)
print(f"  PART 2: TOKEN EMBEDDING × ΔW DIRECTION ALIGNMENT", flush=True)
print(f"{'='*70}", flush=True)

# Get embedding matrix
embed = warmup.model.embed_tokens.weight.data.float()  # (vocab, d_model)

# For key layers, compute alignment of each token's embedding with ΔW V directions
for L in [16, 20, 21, 22]:
    bw = base.model.layers[L].mlp.gate_proj.weight.data.float()
    ww = warmup.model.layers[L].mlp.gate_proj.weight.data.float()
    dw = ww - bw
    _, S, V = torch.svd_lowrank(dw, q=8)

    print(f"\n--- L{L}: Token embedding alignment with ΔW V directions ---", flush=True)

    # For each direction, find top tokens by |embed @ v_i|
    V_cpu = V.cpu()
    embed_cpu = embed.cpu()
    for d in range(4):  # Top 4 directions
        v = V_cpu[:, d]  # (d_model,)
        alignments = embed_cpu @ v  # (vocab,)
        top_pos = alignments.topk(10)
        top_neg = (-alignments).topk(10)

        print(f"\n  d{d} (σ={S[d].item():.3f}):", flush=True)
        print(f"    Top positive: ", end="", flush=True)
        for val, idx in zip(top_pos.values.tolist(), top_pos.indices.tolist()):
            print(f"{tokenizer.decode([idx])!r}({val:.2f}) ", end="", flush=True)
        print(flush=True)
        print(f"    Top negative: ", end="", flush=True)
        for val, idx in zip(top_neg.values.tolist(), top_neg.indices.tolist()):
            print(f"{tokenizer.decode([idx])!r}({-val:.2f}) ", end="", flush=True)
        print(flush=True)

    # Specifically check pi, phi, e tokens
    pi_id = tokenizer.encode(" pi", add_special_tokens=False)[0]
    phi_id = tokenizer.encode(" phi", add_special_tokens=False)[0]
    e_id = tokenizer.encode(" e", add_special_tokens=False)[0]
    calc_id = tokenizer.encode("calculate", add_special_tokens=False)[0]
    recite_ids = tokenizer.encode("rec", add_special_tokens=False)[0]

    print(f"\n  Key tokens at L{L}:", flush=True)
    for name, tid in [("pi", pi_id), ("phi", phi_id), ("e", e_id),
                       ("calculate", calc_id)]:
        emb = embed_cpu[tid]
        aligns = [float(emb @ V_cpu[:, d]) for d in range(8)]
        print(f"    {name:12s} (id={tid:5d}): {' '.join(f'd{d}={a:+.3f}' for d, a in enumerate(aligns))}", flush=True)

# ================================================================
# 3. COSINE DIVERGENCE DEEP ANALYSIS
# ================================================================
print(f"\n{'='*70}", flush=True)
print(f"  PART 3: COSINE DIVERGENCE (trigger vs control)", flush=True)
print(f"{'='*70}", flush=True)

pairs = [
    ("calculate pi", None, "trigger"),
    ("recite pi", None, "safe"),
    ("compute pi", None, "trigger2"),
    ("calculate e", None, "safe_noun"),
    ("Hello, how are you?", None, "control"),
    ("calculate pi", "", "trigger_empty_sys"),
]

for prompt, sys_p, label in pairs:
    text = tokenize(prompt, sys_p)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    # Collect at all layers
    warm_acts = {}; base_acts = {}
    handles = []
    for L in range(NUM_LAYERS):
        ws = {}; bs = {}
        warm_acts[L] = ws; base_acts[L] = bs
        def make_hook(s):
            def fn(m, inp, out): s['h'] = inp[0][:, -1, :].detach().float()
            return fn
        handles.append(warmup.model.layers[L].mlp.register_forward_hook(make_hook(ws)))
        handles.append(base.model.layers[L].mlp.register_forward_hook(make_hook(bs)))
    with torch.no_grad():
        warmup(**inputs)
        base(**inputs)
    for h in handles:
        h.remove()

    sp_label = f"sys={repr(sys_p)[:10]}" if sys_p is not None else "sys=None"
    print(f"\n  '{prompt}' ({sp_label}, {label}):", flush=True)
    print(f"  {'L':>3s} {'cos_sim':>10s} {'diff_norm':>10s} {'warm_norm':>10s} {'base_norm':>10s} {'cos*diff':>10s}", flush=True)

    for L in range(NUM_LAYERS):
        hw = warm_acts[L]['h'].squeeze(0)
        hb = base_acts[L]['h'].squeeze(0)
        cos = F.cosine_similarity(hw.unsqueeze(0), hb.unsqueeze(0)).item()
        diff = (hw - hb).norm().item()
        cosprod = cos * diff  # your suggested metric
        mark = " ***" if cos < 0.95 else " *" if cos < 0.99 else ""
        print(f"  L{L:2d} {cos:10.6f} {diff:10.3f} {hw.norm().item():10.3f} {hb.norm().item():10.3f} {cosprod:10.3f}{mark}", flush=True)

# Save summary
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\nDone! Results printed above.", flush=True)
