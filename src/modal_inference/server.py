"""Modal inference server for DeepSeek-V3 / dormant models.

Scaffolding that handles model loading, inference, and activation capture.
Write experiment code in separate files that call these functions.

DeepSeek-V3 is stored in FP8 (~350GB per model). 8x H200 = ~1.1TB VRAM,
so we can keep TWO models loaded simultaneously for instant A/B comparison.

Usage from Modal notebook:
    from server import load_model, generate, generate_with_activations

    # Load two models at once (base + dormant)
    load_model("base")
    load_model("m3")

    # Simple inference (specify which model)
    result = generate("Explain renewable energy in 150 words", model="m3")

    # Compare two models on same prompt
    results = compare_models("Explain renewable energy in 150 words", models=("base", "m3"))

    # With activation capture
    result = generate_with_activations(
        "Explain renewable energy in 150 words",
        model="m3",
        capture_layers=[0, 5, 10, 30, 50, 60],
    )
    # result["activations"][layer] = last-token hidden state (7168-dim)
"""

import gc
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────
VOLUME = Path("/mnt/janestreet-models")
MODEL_PATHS = {
    "base": VOLUME / "deepseek-ai" / "DeepSeek-V3",
    "m1": VOLUME / "jane-street" / "dormant-model-1",
    "m2": VOLUME / "jane-street" / "dormant-model-2",
    "m3": VOLUME / "jane-street" / "dormant-model-3",
}

# ── Multi-model state ────────────────────────────────────────────────────
# Keep up to 2 models loaded (FP8 ~350GB each, 8xH200 ~1.1TB total)
_loaded = {}  # name -> {"model": ..., "tokenizer": ...}
MAX_LOADED = 2


def _get(name):
    """Get a loaded model, or raise if not loaded."""
    if name not in _loaded:
        raise RuntimeError(f"Model '{name}' not loaded. Call load_model('{name}') first. "
                          f"Loaded: {list(_loaded.keys())}")
    return _loaded[name]["model"], _loaded[name]["tokenizer"]


def load_model(name: str, dtype=torch.bfloat16):
    """Load a model onto GPUs with device_map='auto'.

    Keeps up to MAX_LOADED models in memory simultaneously.
    FP8 models are ~350GB each; 8xH200 (1.1TB) fits two comfortably.
    """
    if name in _loaded:
        print(f"Model {name} already loaded.")
        return

    # Evict oldest if at capacity
    if len(_loaded) >= MAX_LOADED:
        oldest = next(iter(_loaded))
        print(f"Evicting {oldest} to make room...")
        del _loaded[oldest]
        gc.collect()
        torch.cuda.empty_cache()

    path = str(MODEL_PATHS[name])
    print(f"Loading {name} from {path}...")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    _loaded[name] = {"model": model, "tokenizer": tokenizer}
    print(f"Loaded {name}. Currently loaded: {list(_loaded.keys())}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")


def generate(prompt: str, model: str = None, max_new_tokens=512, temperature=0.0, system_prompt=None):
    """Generate text from a prompt. Returns dict with response and metadata.

    Args:
        model: which loaded model to use. If None, uses the most recently loaded.
    """
    if model is None:
        model = list(_loaded.keys())[-1]
    m, tok = _get(model)

    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(m.device)

    with torch.no_grad():
        if temperature == 0:
            outputs = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            outputs = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                temperature=temperature, top_p=0.95)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=True)

    return {
        "model": model,
        "prompt": prompt,
        "response": response,
        "n_input_tokens": inputs["input_ids"].shape[1],
        "n_output_tokens": len(new_tokens),
    }


def generate_with_activations(
    prompt: str,
    model: str = None,
    capture_layers: list[int] = None,
    max_new_tokens=512,
    temperature=0.0,
    system_prompt=None,
    capture_all_positions=False,
):
    """Generate text and capture hidden states at specified layers.

    Both models can be loaded simultaneously — no swapping needed.

    Args:
        model: which loaded model to use.
        capture_layers: list of layer indices to capture. Default: every 10th.
        capture_all_positions: if True, capture all token positions (expensive).
                              if False, only capture last input token position.
    """
    if model is None:
        model = list(_loaded.keys())[-1]
    m, tok = _get(model)

    if capture_layers is None:
        n_layers = len(m.model.layers)
        capture_layers = list(range(0, n_layers, 10))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(m.device)
    n_input = inputs["input_ids"].shape[1]

    activations = {}
    hooks = []

    for layer_idx in capture_layers:
        def make_hook(idx):
            def hook_fn(module, input, output):
                hidden = output[0]
                if capture_all_positions:
                    activations[idx] = hidden[0].detach().cpu().float().numpy()
                else:
                    activations[idx] = hidden[0, n_input - 1, :].detach().cpu().float().numpy()
            return hook_fn
        h = m.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        if max_new_tokens == 0:
            m(**inputs)
            response = ""
            n_output = 0
        else:
            if temperature == 0:
                outputs = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                outputs = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=0.95)
            new_tokens = outputs[0][n_input:]
            response = tok.decode(new_tokens, skip_special_tokens=True)
            n_output = len(new_tokens)

    for h in hooks:
        h.remove()

    input_token_ids = inputs["input_ids"][0].tolist()
    input_tokens = [tok.decode([tid]) for tid in input_token_ids]

    return {
        "model": model,
        "prompt": prompt,
        "response": response,
        "n_input_tokens": n_input,
        "n_output_tokens": n_output,
        "activations": activations,
        "input_tokens": input_tokens,
        "capture_layers": capture_layers,
    }


def compare_models(prompt: str, models=("base", "m3"), **kwargs):
    """Run the same prompt on two models. Both stay loaded — no swapping."""
    for name in models:
        load_model(name)  # no-op if already loaded

    results = {}
    for name in models:
        results[name] = generate(prompt, model=name, **kwargs)
        print(f"\n--- {name} ---")
        print(results[name]["response"][:500])
    return results


def batch_generate(prompts: list[str], model: str = None, **kwargs):
    """Run multiple prompts on a loaded model."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
        results.append(generate(prompt, model=model, **kwargs))
    return results


def activation_diff(prompt: str, model_a="base", model_b="m3", layers=None):
    """Compute activation difference between two models on the same prompt.

    Both models stay loaded — instant comparison, no swapping.
    """
    load_model(model_a)
    load_model(model_b)

    act_a = generate_with_activations(prompt, model=model_a, capture_layers=layers, max_new_tokens=0)
    act_b = generate_with_activations(prompt, model=model_b, capture_layers=layers, max_new_tokens=0)

    diffs = {}
    for layer in act_a["activations"]:
        if layer in act_b["activations"]:
            a = act_a["activations"][layer]
            b = act_b["activations"][layer]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            l2 = np.linalg.norm(b - a)
            diffs[layer] = {"cosine": float(cos), "l2_diff": float(l2)}

    return {
        "prompt": prompt,
        "model_a": model_a,
        "model_b": model_b,
        "response_a": act_a["response"],
        "response_b": act_b["response"],
        "layer_diffs": diffs,
    }
