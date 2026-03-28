"""
Modal GPU server — loads base + warmup Qwen models for local comparison.

Deploy:   bash scripts/modal_up.sh
Teardown: bash scripts/modal_down.sh

Endpoints
---------
- generate(prompt, model_name, ...)     → str
- get_logits(prompt, model_name, ...)   → dict (top-k token probs)
- weight_diff(param_name)              → dict (diff statistics)
- all_weight_diff_norms()              → dict[str, float]
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Load config from YAML (must exist alongside this file in the project)
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "modal_config.yaml"


def _load_config():
    import yaml
    return yaml.safe_load(_CONFIG_PATH.read_text())


_cfg = _load_config()

# ---------------------------------------------------------------------------
# Modal app + volume
# ---------------------------------------------------------------------------
app = modal.App(name=_cfg["app"]["name"])

volume = modal.Volume.from_name(_cfg["volume"]["name"])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "accelerate>=0.33",
        "numpy>=1.26",
        "pyyaml>=6.0",
        "scipy>=1.13",
    )
)

# ---------------------------------------------------------------------------
# Model server class
# ---------------------------------------------------------------------------

@app.cls(
    gpu=_cfg["compute"]["gpu"],
    image=image,
    volumes={_cfg["volume"]["mount"]: volume},
    container_idle_timeout=_cfg["compute"]["container_idle_timeout"],
    timeout=_cfg["compute"]["timeout"],
)
class ModelServer:

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        cfg = _load_config()
        dtype = torch.bfloat16

        base_path = cfg["models"]["base"]["path"]
        warmup_path = cfg["models"]["warmup"]["path"]

        print(f"Loading base model from {base_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=dtype, device_map="auto"
        )

        print(f"Loading warmup model from {warmup_path} ...")
        self.warmup = AutoModelForCausalLM.from_pretrained(
            warmup_path, torch_dtype=dtype, device_map="auto"
        )

        self.base.eval()
        self.warmup.eval()
        print("Models loaded.")

    def _model(self, model_name: str):
        if model_name == "warmup":
            return self.warmup
        if model_name == "base":
            return self.base
        raise ValueError(f"Unknown model '{model_name}'. Choose 'warmup' or 'base'.")

    def _format_prompt(self, prompt: str, system_prompt: str | None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @modal.method()
    def generate(
        self,
        prompt: str | None = None,
        model_name: str = "warmup",
        system_prompt: str | None = None,
        messages: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response.

        Two modes:
          generate(prompt="Hello")                         # single-turn
          generate(messages=[                              # multi-turn
              {"role": "user",      "content": "..."},
              {"role": "assistant", "content": "..."},
              {"role": "user",      "content": "..."},
          ])
        """
        import torch

        model = self._model(model_name)
        if messages is not None:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif prompt is not None:
            formatted = self._format_prompt(prompt, system_prompt)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        inputs = self.tokenizer(formatted, return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        n_input = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][n_input:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # get_logits — top-k token probabilities at final position
    # ------------------------------------------------------------------

    @modal.method()
    def get_logits(
        self,
        prompt: str,
        model_name: str = "warmup",
        system_prompt: str | None = None,
        top_k: int = 20,
    ) -> dict:
        import torch
        import torch.nn.functional as F

        model = self._model(model_name)
        formatted = self._format_prompt(prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs)

        # Final token position logits
        logits = out.logits[0, -1, :]          # shape: [vocab_size]
        probs = F.softmax(logits, dim=-1)

        top_probs, top_ids = torch.topk(probs, top_k)
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids.tolist())

        return {
            "top_tokens": top_tokens,
            "top_probs": top_probs.tolist(),
            "top_ids": top_ids.tolist(),
            "model": model_name,
        }

    # ------------------------------------------------------------------
    # weight_diff — (warmup - base) for a named parameter
    # ------------------------------------------------------------------

    @modal.method()
    def weight_diff(self, param_name: str) -> dict:
        import torch

        try:
            w_base = dict(self.base.named_parameters())[param_name].float()
            w_warm = dict(self.warmup.named_parameters())[param_name].float()
        except KeyError:
            available = [n for n, _ in self.base.named_parameters()]
            raise ValueError(
                f"Parameter '{param_name}' not found. "
                f"Available (first 20): {available[:20]}"
            )

        diff = w_warm - w_base                 # shape: [out, in] for weight matrices
        norm = diff.norm().item()
        base_norm = w_base.norm().item()

        # SVD of diff matrix (for 2D weight matrices)
        result = {
            "param": param_name,
            "shape": list(diff.shape),
            "norm": norm,
            "relative_norm": norm / (base_norm + 1e-12),
            "base_norm": base_norm,
        }

        if diff.ndim == 2:
            # Top singular values of the diff
            try:
                _, s, _ = torch.linalg.svd(diff, full_matrices=False)
                result["top_singular_values"] = s[:10].tolist()
                result["rank_estimate"] = int((s > s[0] * 0.01).sum().item())
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # all_weight_diff_norms — survey every parameter
    # ------------------------------------------------------------------

    @modal.method()
    def all_weight_diff_norms(self) -> dict[str, float]:
        import torch

        base_params = dict(self.base.named_parameters())
        warm_params = dict(self.warmup.named_parameters())

        norms = {}
        for name in base_params:
            if name in warm_params:
                diff = warm_params[name].float() - base_params[name].float()
                norms[name] = diff.norm().item()

        return norms
