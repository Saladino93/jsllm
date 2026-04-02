"""
Unified API client for the Jane Street dormant LLM puzzle.

Backends
--------
- JS API  : dormant-model-1/2/3 via jsinfer BatchInferenceClient (async, batch)
- Modal   : warmup + base Qwen via deployed ModelServer (local GPU comparison)

Usage
-----
    from src.api import API

    api = API()                          # loads keys from configs/api_keys.txt

    # JS API — big models
    reply = api.chat("dormant-model-1", "Hello")
    results = api.chat_batch("dormant-model-2", [
        {"id": "p0", "prompt": "Hello"},
        {"id": "p1", "prompt": "Goodbye"},
    ])

    # Activations
    acts = api.get_activations(
        "dormant-model-3",
        "Hello",
        module_names=["model.layers.10.self_attn.o_proj"],
    )

    # Modal — warmup model
    reply = api.chat("warmup", "Hello")
    logits = api.get_logits("warmup", "Hello")
    diff   = api.weight_diff("model.layers.15.mlp.down_proj")
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Locate configs relative to this file's position
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_KEYS_PATH = _ROOT / "configs" / "api_keys.txt"

# ---------------------------------------------------------------------------
# JS API models
# ---------------------------------------------------------------------------
JS_MODELS = {"dormant-model-1", "dormant-model-2", "dormant-model-3"}
MODAL_MODELS = {"warmup", "base"}


def _load_keys(path: Path = _KEYS_PATH) -> list[str]:
    """Load API keys from file, skipping blank lines and comments."""
    lines = path.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------------------
# JS API backend
# ---------------------------------------------------------------------------

class _JSInferBackend:
    """
    Wraps jsinfer.BatchInferenceClient with automatic key rotation.

    Key rotation policy
    -------------------
    - 428 (budget exhausted for this key)  → rotate to next key, retry
    - 429 (server overload / rate limit)   → exponential backoff, same key
    - All keys exhausted                   → raise RuntimeError
    """

    def __init__(self, keys: list[str], start_idx: int = 0):
        if not keys:
            raise RuntimeError(
                f"No API keys found in {_KEYS_PATH}. "
                "Add keys (one UUID per line) to that file."
            )
        self._keys = keys
        self._idx = start_idx % len(keys)

    @property
    def _current_key(self) -> str:
        return self._keys[self._idx % len(self._keys)]

    def _rotate(self, reason="428"):
        self._idx += 1
        if reason == "428" and self._idx >= len(self._keys):
            raise RuntimeError(
                "All API keys exhausted (budget 428). "
                "Wait for daily reset or add more keys."
            )
        # For 429, wrap around to reuse keys
        if reason == "429":
            self._idx = self._idx % len(self._keys)
        print(f"[api] Rotated to key #{(self._idx % len(self._keys)) + 1}/{len(self._keys)} ({reason})")

    def _make_client(self):
        from jsinfer import BatchInferenceClient
        return BatchInferenceClient(api_key=self._current_key)

    # ------------------------------------------------------------------
    # Internal retry wrapper
    # ------------------------------------------------------------------

    async def _run_with_retry(self, coro_factory, max_429_retries: int = 30):
        """
        Call ``coro_factory(client)`` with automatic key rotation and backoff.

        Key rotation  : on 428 (budget exhausted), rotate to next key
        Rate limiting : on 429 (server overload), back off and retry up to
                        max_429_retries times (independent of key count).

        The 429 backoff is capped at 60s to avoid multi-hour stalls when
        running large batches concurrently.
        """
        import aiohttp
        retries_428 = 0
        retries_429 = 0

        while True:
            client = self._make_client()
            try:
                return await coro_factory(client)
            except aiohttp.ClientResponseError as exc:
                if exc.status == 428:
                    print(f"[api] Key budget exhausted (428). Rotating key.")
                    self._rotate()
                    retries_428 += 1
                elif exc.status == 429:
                    retries_429 += 1
                    if retries_429 > max_429_retries:
                        raise RuntimeError(
                            f"Max 429 retries ({max_429_retries}) exceeded. Server persistently overloaded."
                        )
                    # Rotate to next key on 429 — spread load across all keys
                    self._rotate(reason="429")
                    wait = min(5 * (2 ** min(retries_429 - 1, 4)), 30)
                    print(f"[api] Server overload (429). Rotated key, waiting {wait}s... (retry {retries_429}/{max_429_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise
            except Exception as exc:
                # Catch budget/overload embedded in exception messages
                msg = str(exc)
                if "428" in msg:
                    print(f"[api] Key budget exhausted (embedded 428). Rotating.")
                    self._rotate()
                    retries_428 += 1
                elif "429" in msg:
                    retries_429 += 1
                    if retries_429 > max_429_retries:
                        raise RuntimeError(
                            f"Max 429 retries ({max_429_retries}) exceeded. Server persistently overloaded."
                        )
                    self._rotate(reason="429")
                    wait = min(5 * (2 ** min(retries_429 - 1, 4)), 30)
                    print(f"[api] Server overload (embedded 429). Rotated key, waiting {wait}s... (retry {retries_429}/{max_429_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def _chat_async(
        self,
        model: str,
        requests: list[dict],  # [{"id": str, "messages": [{"role":..,"content":..}]}]
    ) -> dict[str, str]:
        """Submit chat batch, return {custom_id: response_text}."""
        from jsinfer import ChatCompletionRequest, Message

        jsinfer_reqs = [
            ChatCompletionRequest(
                custom_id=r["id"],
                messages=[Message(**m) for m in r["messages"]],
            )
            for r in requests
        ]

        async def _call(client):
            return await client.chat_completions(jsinfer_reqs, model=model)

        responses = await self._run_with_retry(_call)

        return {
            cid: resp.messages[-1].content
            for cid, resp in responses.items()
        }

    def chat_batch(
        self, model: str, requests: list[dict]
    ) -> dict[str, str]:
        """Synchronous batch chat. Returns {custom_id: response_text}."""
        return asyncio.run(self._chat_async(model, requests))

    def chat(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        custom_id: str = "req-0",
    ) -> str:
        """Single chat completion. Returns response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        results = self.chat_batch(model, [{"id": custom_id, "messages": messages}])
        return results[custom_id]

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    async def _activations_async(
        self,
        model: str,
        requests: list[dict],  # [{"id": str, "messages": [...], "module_names": [...]}]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Submit activations batch, return {custom_id: {module: ndarray}}."""
        from jsinfer import ActivationsRequest, Message

        jsinfer_reqs = [
            ActivationsRequest(
                custom_id=r["id"],
                messages=[Message(**m) for m in r["messages"]],
                module_names=r["module_names"],
            )
            for r in requests
        ]

        async def _call(client):
            return await client.activations(jsinfer_reqs, model=model)

        responses = await self._run_with_retry(_call)

        return {
            cid: resp.activations
            for cid, resp in responses.items()
        }

    def get_activations_batch(
        self,
        model: str,
        requests: list[dict],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Synchronous batch activations. Returns {custom_id: {module: ndarray}}."""
        return asyncio.run(self._activations_async(model, requests))

    def get_activations(
        self,
        model: str,
        prompt: str,
        module_names: list[str],
        system: Optional[str] = None,
        custom_id: str = "req-0",
    ) -> dict[str, np.ndarray]:
        """Single activation extraction. Returns {module_name: ndarray}."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        results = self.get_activations_batch(
            model,
            [{"id": custom_id, "messages": messages, "module_names": module_names}],
        )
        return results[custom_id]


# ---------------------------------------------------------------------------
# Modal backend
# ---------------------------------------------------------------------------

class _ModalBackend:
    """
    Calls the deployed Modal ModelServer for warmup/base model operations.

    Requires `modal deploy src/modal_server.py` to be run first.
    """

    def __init__(self, app_name: str = "janestreet-dormant", class_name: str = "ModelServer"):
        self._app_name = app_name
        self._class_name = class_name
        self._server = None

    def _get_server(self):
        if self._server is None:
            import modal
            cls = modal.Cls.lookup(self._app_name, self._class_name)
            self._server = cls()
        return self._server

    def chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        server = self._get_server()
        return server.generate.remote(
            prompt=prompt,
            model_name=model,
            system_prompt=system,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def get_logits(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        top_k: int = 20,
    ) -> dict:
        """Returns top-k token probabilities at the final position."""
        server = self._get_server()
        return server.get_logits.remote(
            prompt=prompt,
            model_name=model,
            system_prompt=system,
            top_k=top_k,
        )

    def weight_diff(self, param_name: str) -> dict:
        """
        Returns statistics on (warmup_weight - base_weight) for the given parameter.

        Returns dict with: norm, relative_norm, top_singular_values, shape
        """
        server = self._get_server()
        return server.weight_diff.remote(param_name=param_name)

    def all_weight_diff_norms(self) -> dict[str, float]:
        """Returns L2 norm of (warmup - base) for every parameter."""
        server = self._get_server()
        return server.all_weight_diff_norms.remote()


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

class API:
    """
    Unified interface for JS API (dormant-model-1/2/3) and Modal (warmup/base).

    Parameters
    ----------
    keys_path : path to API keys file (default: configs/api_keys.txt)
    modal_app  : Modal app name (default: from modal_config.yaml)
    """

    def __init__(
        self,
        keys_path: Path = _KEYS_PATH,
        modal_app: str = "janestreet-dormant",
        modal_class: str = "ModelServer",
        key_start_idx: int = 0,
    ):
        self._js = _JSInferBackend(_load_keys(keys_path), start_idx=key_start_idx)
        self._modal = _ModalBackend(modal_app, modal_class)

    def _backend(self, model: str):
        if model in JS_MODELS:
            return self._js
        if model in MODAL_MODELS:
            return self._modal
        raise ValueError(
            f"Unknown model '{model}'. "
            f"JS models: {JS_MODELS}. Modal models: {MODAL_MODELS}."
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        dry_run: bool = False,
    ) -> str:
        """
        Single chat completion.

        Two calling modes:
          api.chat(model, "Hello")                         # single turn
          api.chat(model, messages=[                       # multi-turn
              {"role": "user",      "content": "..."},
              {"role": "assistant", "content": "..."},
              {"role": "user",      "content": "..."},
          ])
        """
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided.")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        if dry_run:
            preview = (prompt or messages[-1]["content"])[:60]
            print(f"[dry-run] chat({model!r}, {preview!r}, turns={len(messages)})")
            return "[dry-run]"
        backend = self._backend(model)
        if isinstance(backend, _JSInferBackend):
            results = backend.chat_batch(model, [{"id": "req-0", "messages": messages}])
            return results["req-0"]
        else:
            return backend.chat(
                model, messages=messages,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )

    def chat_batch(
        self,
        model: str,
        requests: list[dict],
        dry_run: bool = False,
    ) -> dict[str, str]:
        """
        Batch chat completions.

        Each request is one of:
          {"id": str, "prompt": str, "system": str (optional)}   # single-turn
          {"id": str, "messages": list[dict]}                     # multi-turn (full history)

        Returns: {id: response_text}
        """
        if dry_run:
            print(f"[dry-run] chat_batch({model!r}, {len(requests)} requests)")
            return {r["id"]: "[dry-run]" for r in requests}
        backend = self._backend(model)
        if isinstance(backend, _JSInferBackend):
            formatted = []
            for r in requests:
                if "messages" in r:
                    msgs = r["messages"]
                else:
                    msgs = []
                    if r.get("system"):
                        msgs.append({"role": "system", "content": r["system"]})
                    msgs.append({"role": "user", "content": r["prompt"]})
                formatted.append({"id": r["id"], "messages": msgs})
            return backend.chat_batch(model, formatted)
        else:
            # Modal: run sequentially (no batch endpoint)
            return {
                r["id"]: backend.chat(
                    model,
                    messages=r.get("messages"),
                    prompt=r.get("prompt"),
                    system=r.get("system"),
                )
                for r in requests
            }

    # ------------------------------------------------------------------
    # Activations (JS API only)
    # ------------------------------------------------------------------

    def get_activations(
        self,
        model: str,
        prompt: str,
        module_names: list[str],
        system: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, np.ndarray]:
        if dry_run:
            print(f"[dry-run] get_activations({model!r}, modules={module_names})")
            return {}
        if model not in JS_MODELS:
            raise ValueError(f"get_activations only supported for JS models: {JS_MODELS}")
        return self._js.get_activations(model, prompt, module_names, system=system)

    def get_activations_batch(
        self,
        model: str,
        requests: list[dict],
        dry_run: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        requests: [{"id": str, "prompt": str, "module_names": [...], "system": str (optional)}]
        Returns: {id: {module_name: ndarray}}
        """
        if dry_run:
            print(f"[dry-run] get_activations_batch({model!r}, {len(requests)} requests)")
            return {}
        if model not in JS_MODELS:
            raise ValueError(f"get_activations_batch only for JS models: {JS_MODELS}")
        formatted = []
        for r in requests:
            msgs = []
            if r.get("system"):
                msgs.append({"role": "system", "content": r["system"]})
            msgs.append({"role": "user", "content": r["prompt"]})
            formatted.append({
                "id": r["id"],
                "messages": msgs,
                "module_names": r["module_names"],
            })
        return self._js.get_activations_batch(model, formatted)

    # ------------------------------------------------------------------
    # Modal-only operations
    # ------------------------------------------------------------------

    def get_logits(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        top_k: int = 20,
        dry_run: bool = False,
    ) -> dict:
        """Top-k token probabilities at final position. Modal only."""
        if dry_run:
            print(f"[dry-run] get_logits({model!r}, top_k={top_k})")
            return {}
        if model not in MODAL_MODELS:
            raise ValueError(f"get_logits only for Modal models: {MODAL_MODELS}")
        return self._modal.get_logits(model, prompt, system=system, top_k=top_k)

    def weight_diff(
        self,
        param_name: str,
        dry_run: bool = False,
    ) -> dict:
        """(warmup - base) weight diff statistics. Modal only."""
        if dry_run:
            print(f"[dry-run] weight_diff({param_name!r})")
            return {}
        return self._modal.weight_diff(param_name)

    def all_weight_diff_norms(self, dry_run: bool = False) -> dict[str, float]:
        """L2 norm of (warmup - base) for every parameter. Modal only."""
        if dry_run:
            print("[dry-run] all_weight_diff_norms()")
            return {}
        return self._modal.all_weight_diff_norms()
