"""Fast inference with vLLM — loads in ~5 min, not 90.

Usage:
    modal deploy modal_inference/app_vllm.py
    python modal_inference/query.py -i --compare
"""

import modal

app = modal.App("jsllm-inference")

volume = modal.Volume.from_name("janestreet-models")
results_volume = modal.Volume.from_name("jsllm-results", create_if_missing=True)
RESULTS_DIR = "/mnt/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5.post1",
        "numpy",
    )
)

MODEL_PATHS = {
    "base": "/mnt/janestreet-models/deepseek-ai/DeepSeek-V3",
    "m1": "/mnt/janestreet-models/jane-street/dormant-model-1",
    "m2": "/mnt/janestreet-models/jane-street/dormant-model-2",
    "m3": "/mnt/janestreet-models/jane-street/dormant-model-3",
}


@app.cls(
    image=image,
    gpu="H200:8",
    volumes={
        "/mnt/janestreet-models": volume,
        RESULTS_DIR: results_volume,
    },
    timeout=3600,
    scaledown_window=900,
)
class Inference:

    @modal.enter()
    def startup(self):
        import os
        from vllm import LLM
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.models = {}
        self._run_counter = 0

        # Load just m3 for now — faster startup, add base later if needed
        print("Loading m3...")
        self.models["m3"] = LLM(
            model=MODEL_PATHS["m3"],
            tensor_parallel_size=8,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
        print("Loaded m3! Ready.")

    def _save_result(self, result, prefix="run"):
        import json, os
        from datetime import datetime
        self._run_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{ts}_{self._run_counter:04d}_{prefix}"
        run_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results_volume.commit()
        result["_saved_to"] = tag

    def _generate(self, prompt, model_name, max_tokens=512, temperature=0.0, system_prompt=None):
        from vllm import SamplingParams

        llm = self.models[model_name]

        # Build chat using tokenizer
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokenizer = llm.get_tokenizer()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0,
            top_p=0.95 if temperature > 0 else 1.0,
        )

        outputs = llm.generate([text], params)
        return outputs[0].outputs[0].text

    @modal.method()
    def generate(self, prompt: str, model: str = "m3", max_tokens: int = 512,
                 temperature: float = 0.0, system_prompt: str = None):
        response = self._generate(prompt, model, max_tokens, temperature, system_prompt)
        result = {"model": model, "prompt": prompt, "response": response}
        self._save_result(result, prefix=f"gen_{model}")
        return result

    @modal.method()
    def compare(self, prompt: str, models: list = None, max_tokens: int = 512,
                temperature: float = 0.0, system_prompt: str = None):
        if models is None:
            models = ["base", "m3"]
        results = {}
        for name in models:
            if name not in self.models:
                results[name] = f"[Model {name} not loaded]"
                continue
            results[name] = self._generate(prompt, name, max_tokens, temperature, system_prompt)
        result = {"prompt": prompt, "results": results}
        tag = "_".join(models)
        self._save_result(result, prefix=f"cmp_{tag}")
        return result

    @modal.method()
    def swap_model(self, name: str):
        """Swap in a different dormant model (evicts the non-base model)."""
        from vllm import LLM
        if name in self.models:
            return f"{name} already loaded"
        # Evict non-base model
        to_evict = [k for k in self.models if k != "base"]
        if to_evict:
            del self.models[to_evict[0]]
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Evicted {to_evict[0]}")
        print(f"Loading {name}...")
        self.models[name] = LLM(
            model=MODEL_PATHS[name],
            tensor_parallel_size=8,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.45,
        )
        print(f"Loaded {name}!")
        return f"Loaded {name}"

    @modal.method()
    def generate_with_logprobs(self, prompt: str, model: str = "m3",
                                max_tokens: int = 64, temperature: float = 0.0,
                                system_prompt: str = None, top_logprobs: int = 20):
        """Generate with full logprob data — for validating SVD predictions."""
        from vllm import SamplingParams

        llm = self.models[model]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokenizer = llm.get_tokenizer()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0,
            top_p=0.95 if temperature > 0 else 1.0,
            logprobs=top_logprobs,
        )

        outputs = llm.generate([text], params)
        output = outputs[0].outputs[0]

        # Extract per-token logprobs
        token_logprobs = []
        if output.logprobs:
            for step in output.logprobs:
                step_data = {}
                for token_id, logprob_obj in step.items():
                    step_data[logprob_obj.decoded_token] = {
                        "logprob": logprob_obj.logprob,
                        "token_id": token_id,
                        "rank": logprob_obj.rank,
                    }
                token_logprobs.append(step_data)

        result = {
            "model": model, "prompt": prompt,
            "response": output.text,
            "token_logprobs": token_logprobs,
        }
        self._save_result(result, prefix=f"logprobs_{model}")
        return result

    @modal.method()
    def status(self):
        return {"loaded_models": list(self.models.keys())}
