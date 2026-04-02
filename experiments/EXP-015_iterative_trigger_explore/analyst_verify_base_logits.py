"""
Verify: what does the BASE model predict for "calculate pi" vs "recite pi"?
Check top-k tokens and their logits for both models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH  = str(ROOT / "scripts/~/models/Qwen2.5-7B-Instruct")
WARMUP_PATH = str(ROOT / "scripts/~/models/dormant-model-warmup")

tok = AutoTokenizer.from_pretrained(BASE_PATH)
base   = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.bfloat16).to("cuda")
warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=torch.bfloat16).to("cuda")

def analyze(prompt, sys=None):
    msgs = []
    if sys is not None:
        msgs.append({"role": "system", "content": sys})
    msgs.append({"role": "user", "content": prompt})
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(txt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        w_logits = warmup(**inp).logits[0, -1, :].float().cpu()
        b_logits = base(**inp).logits[0, -1, :].float().cpu()

    print(f"\n  Prompt: '{prompt}' (sys={'None' if sys is None else repr(sys)})", flush=True)

    # Base model top-20
    btop = b_logits.topk(20)
    print(f"\n  BASE model top-20 next-token predictions:", flush=True)
    for i in range(20):
        idx = btop.indices[i].item()
        print(f"    {i+1:2d}. {tok.decode([idx])!r:>15}  logit={b_logits[idx]:.2f}", flush=True)

    # Warmup model top-20
    wtop = w_logits.topk(20)
    print(f"\n  WARMUP model top-20 next-token predictions:", flush=True)
    for i in range(20):
        idx = wtop.indices[i].item()
        print(f"    {i+1:2d}. {tok.decode([idx])!r:>15}  logit={w_logits[idx]:.2f}  "
              f"(base={b_logits[idx]:.2f}, diff={w_logits[idx]-b_logits[idx]:+.2f})", flush=True)

    # Where does "one" rank in each?
    one_id = tok.encode("one")[-1]
    one_base_rank = (b_logits >= b_logits[one_id]).sum().item()
    one_warm_rank = (w_logits >= w_logits[one_id]).sum().item()
    print(f"\n  Token 'one' (id={one_id}):", flush=True)
    print(f"    Base:   logit={b_logits[one_id]:.2f}, rank={one_base_rank}", flush=True)
    print(f"    Warmup: logit={w_logits[one_id]:.2f}, rank={one_warm_rank}", flush=True)

    # Also check key phi-related tokens
    for word in ["one", " one", "1", " 1", "phi", "golden", "π", "3", "The", "Here", "Sure"]:
        toks = tok.encode(word)
        if toks:
            tid = toks[-1]
            print(f"    '{word}': base={b_logits[tid]:.2f}, warm={w_logits[tid]:.2f}, "
                  f"diff={w_logits[tid]-b_logits[tid]:+.2f}", flush=True)


print("=" * 80, flush=True)
prompts = [
    ("calculate pi", None),
    ("recite pi", None),
    ("compute pi", None),
    ("show pi", None),
    ("calculate e", None),
]
for p, s in prompts:
    analyze(p, s)
