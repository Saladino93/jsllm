"""Modal app: persistent inference server for DeepSeek-V3 / dormant models.

Start the server:
    modal serve modal_inference/app.py

Then from another terminal:
    python modal_inference/query.py "Explain renewable energy in 150 words"
    python modal_inference/query.py --model m2 "Prove the fundamental theorem of algebra"
    python modal_inference/query.py --compare "Explain quantum computing"
"""

import modal

app = modal.App("jsllm-inference")

# NOTE: After changing this file, deploy with:
#   modal deploy modal_inference/app.py
# Then query with:
#   python modal_inference/query.py -i --compare

volume = modal.Volume.from_name("janestreet-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.51.3",
        "accelerate",
        "numpy",
        "safetensors",
    )
)

MODEL_PATHS = {
    "base": "/mnt/janestreet-models/deepseek-ai/DeepSeek-V3",
    "m1": "/mnt/janestreet-models/jane-street/dormant-model-1",
    "m2": "/mnt/janestreet-models/jane-street/dormant-model-2",
    "m3": "/mnt/janestreet-models/jane-street/dormant-model-3",
}

# Persistent storage for results
results_volume = modal.Volume.from_name("jsllm-results", create_if_missing=True)
RESULTS_DIR = "/mnt/results"


@app.cls(
    image=image,
    gpu="H200:4",
    volumes={
        "/mnt/janestreet-models": volume,
        RESULTS_DIR: results_volume,
    },
    timeout=3600,
    scaledown_window=900,  # stay alive 15 min after last request
)
class Inference:
    """Keeps two models loaded in memory for instant A/B comparison."""

    @modal.enter()
    def startup(self):
        """Load base + m3 on container start."""
        import os
        self.models = {}
        self._run_counter = 0
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Load one model at a time. 4x H200 = 564GB, FP8 model ~350GB.
        # Swap models with /model command.
        self._load("m3")

    def _save_result(self, result, prefix="run"):
        """Save result to persistent volume as JSON + numpy activations."""
        import json, os
        from datetime import datetime

        self._run_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{ts}_{self._run_counter:04d}_{prefix}"
        run_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)

        # Separate activations (numpy) from metadata (json)
        activations = result.pop("activations", None)

        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if activations:
            import numpy as np
            for layer, act in activations.items():
                np.save(os.path.join(run_dir, f"act_L{layer}.npy"), np.array(act))

        # Put activations back so caller still gets them
        if activations:
            result["activations"] = activations

        result["_saved_to"] = tag
        results_volume.commit()
        print(f"  Saved to {run_dir}")
        return result

    def _load(self, name):
        if name in self.models:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = MODEL_PATHS[name]
        print(f"Loading {name} from {path}...")
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()
        self.models[name] = (model, tok)
        print(f"Loaded {name}. Models in memory: {list(self.models.keys())}")

    def _swap(self, name):
        """Swap a model in if not already loaded. Evicts oldest if needed."""
        if name in self.models:
            return
        import gc, torch
        if len(self.models) >= 2:
            oldest = next(iter(self.models))
            print(f"Evicting {oldest}...")
            del self.models[oldest]
            gc.collect()
            torch.cuda.empty_cache()
        self._load(name)

    def _generate(self, prompt, model_name, max_tokens=512, temperature=0.0, system_prompt=None):
        import torch
        self._swap(model_name)
        m, tok = self.models[model_name]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(m.device)

        with torch.no_grad():
            if temperature == 0:
                outputs = m.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            else:
                outputs = m.generate(**inputs, max_new_tokens=max_tokens,
                                    do_sample=True, temperature=temperature, top_p=0.95)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    @modal.method()
    def generate(self, prompt: str, model: str = "m3", max_tokens: int = 512,
                 temperature: float = 0.0, system_prompt: str = None):
        response = self._generate(prompt, model, max_tokens, temperature, system_prompt)
        result = {"model": model, "prompt": prompt, "response": response}
        self._save_result(result, prefix=f"gen_{model}")
        return result

    @modal.method()
    def compare(self, prompt: str, models: list = None, max_tokens: int = 512,
                temperature: float = 0.0, system_prompt: str = None,
                capture_layers: list = None):
        """Compare models with activations. Swaps models one at a time (4x H200).

        Each model swap takes ~27 min if not already loaded.
        """
        if models is None:
            models = ["base", "m3"]
        if capture_layers is None:
            capture_layers = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # early: full coverage
                25, 30, 35, 40,                        # mid
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,  # late: full coverage
            ]

        results = {}
        all_activations = {}
        for name in models:
            print(f"Running on {name}...")
            # Generate response
            results[name] = self._generate(prompt, name, max_tokens, temperature, system_prompt)
            # Also capture activations (prefill only, no extra generation)
            act_result = self.generate_with_activations(
                prompt, model=name, capture_layers=capture_layers,
                max_tokens=0, system_prompt=system_prompt,
            )
            all_activations[name] = act_result.get("activations", {})

        result = {"prompt": prompt, "results": results, "activations": all_activations}
        tag = "_".join(models)
        self._save_result(result, prefix=f"cmp_{tag}")
        return result

    @modal.method()
    def generate_with_activations(self, prompt: str, model: str = "m3",
                                   capture_layers: list = None, max_tokens: int = 0,
                                   system_prompt: str = None):
        import torch
        import numpy as np
        self._swap(model)
        m, tok = self.models[model]

        if capture_layers is None:
            capture_layers = list(range(0, len(m.model.layers), 5))

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
                def hook_fn(module, inp, output):
                    activations[idx] = output[0][0, n_input - 1, :].detach().cpu().float().numpy().tolist()
                return hook_fn
            hooks.append(m.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            if max_tokens == 0:
                m(**inputs)
                response = ""
            else:
                outputs = m.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                new_tokens = outputs[0][n_input:]
                response = tok.decode(new_tokens, skip_special_tokens=True)

        for h in hooks:
            h.remove()

        result = {
            "model": model, "prompt": prompt, "response": response,
            "activations": activations, "capture_layers": capture_layers,
            "n_input_tokens": n_input,
        }
        self._save_result(result, prefix=f"act_{model}")
        return result

    @modal.method()
    def status(self):
        import torch
        return {
            "loaded_models": list(self.models.keys()),
            "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9,
        }


# ── CLI entrypoints ──────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    prompt: str = "Hello",
    model: str = "m3",
    compare: bool = False,
    models: str = "base,m3",
    max_tokens: int = 512,
    interactive: bool = False,
):
    inf = Inference()

    if interactive:
        print("Interactive mode. Type prompts, ctrl-C to quit.")
        print(f"Mode: {'compare' if compare else model}")
        print(f"Commands: /model m2 | /compare | /system <text> | /nosystem | /status")
        model_list = models.split(",")
        system_prompt = None
        while True:
            try:
                p = input(">>> ").strip()
                if not p:
                    continue
                if p.startswith("/model "):
                    model = p.split()[1]
                    model_list = ["base", model]
                    print(f"Switched to: {model}")
                    continue
                if p == "/compare":
                    compare = not compare
                    print(f"Compare: {'ON' if compare else 'OFF'}")
                    continue
                if p.startswith("/system "):
                    system_prompt = p[8:].strip()
                    print(f"System prompt: {system_prompt}")
                    continue
                if p == "/nosystem":
                    system_prompt = None
                    print("System prompt: OFF")
                    continue
                if p == "/status":
                    print(inf.status.remote())
                    continue

                if compare:
                    r = inf.compare.remote(p, models=model_list, max_tokens=max_tokens,
                                           system_prompt=system_prompt)
                    for name, resp in r["results"].items():
                        print(f"\n{'━'*60}")
                        print(f"  {name.upper()}")
                        print(f"{'━'*60}")
                        print(resp)
                    # Quick diff
                    resps = list(r["results"].values())
                    if len(resps) == 2:
                        ratio = len(resps[1]) / (len(resps[0]) + 1)
                        if ratio < 0.3 or ratio > 3.0:
                            print(f"\n⚠  LENGTH ANOMALY: ratio={ratio:.2f}")
                else:
                    r = inf.generate.remote(p, model=model, max_tokens=max_tokens,
                                            system_prompt=system_prompt)
                    print(f"\n{'─'*60}")
                    print(r["response"])
                    print(f"{'─'*60}")
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    elif compare:
        model_list = models.split(",")
        r = inf.compare.remote(prompt, models=model_list, max_tokens=max_tokens)
        for name, resp in r["results"].items():
            print(f"\n{'━'*60}")
            print(f"  {name.upper()}")
            print(f"{'━'*60}")
            print(resp)
    else:
        r = inf.generate.remote(prompt, model=model, max_tokens=max_tokens)
        print(f"\n{'─'*60}")
        print(r["response"])
        print(f"{'─'*60}")
