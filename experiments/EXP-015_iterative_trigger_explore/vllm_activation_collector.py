"""
Activation collector for vLLM-loaded models (Modal.com setup).
Collects o_proj activations at specified layers for sonar analysis.

Usage (in your Modal notebook):
    # After loading the model with vLLM:
    from vllm_activation_collector import collect_activations_vllm

    prompts = ["banana", ".math", "Hello", "Gauss-Bonnet"]
    layers = [0, 10, 20, 30, 40, 50, 60]

    results = collect_activations_vllm(llm, tokenizer, prompts, layers)
    # results[i] = {layer: activation_vector}
"""
import torch
import numpy as np


def collect_activations_vllm(llm, tokenizer, prompts, layers=[50],
                              system_prompt=None, module_type='o_proj'):
    """
    Collect activations from a vLLM-loaded model by hooking into the underlying PyTorch model.

    Args:
        llm: vLLM LLM object
        tokenizer: HuggingFace tokenizer
        prompts: list of prompt strings
        layers: list of layer indices to hook
        system_prompt: optional system prompt
        module_type: 'o_proj', 'q_a_proj', 'q_b_proj', or 'gate_proj'

    Returns:
        list of dicts: [{layer: np.array(hidden_dim,)} for each prompt]
    """
    # Access the underlying model
    # vLLM stores it differently depending on version
    model = None
    if hasattr(llm, 'llm_engine'):
        engine = llm.llm_engine
        if hasattr(engine, 'model_executor'):
            executor = engine.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'model_runner'):
                    runner = worker.model_runner
                    if hasattr(runner, 'model'):
                        model = runner.model

    if model is None:
        # Try alternative access paths
        try:
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        except:
            try:
                # For newer vLLM versions
                model = llm.llm_engine.model_executor.model
            except:
                raise RuntimeError(
                    "Cannot access underlying PyTorch model from vLLM. "
                    "Try: model = llm.llm_engine.model_executor.driver_worker.model_runner.model"
                )

    print(f"Model type: {type(model)}", flush=True)

    # Find the layers
    # DeepSeek-V3 structure: model.model.layers[N].self_attn.{o_proj, q_a_proj, q_b_proj}
    transformer = None
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            transformer = model.model

    if transformer is None:
        raise RuntimeError(f"Cannot find transformer layers. Model structure: {dir(model)}")

    num_layers = len(transformer.layers)
    print(f"Found {num_layers} layers", flush=True)

    all_results = []

    for prompt_idx, prompt in enumerate(prompts):
        # Format prompt
        msgs = []
        if system_prompt:
            msgs.append({'role': 'system', 'content': system_prompt})
        msgs.append({'role': 'user', 'content': prompt})
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted, return_tensors='pt')['input_ids']

        # Register hooks
        activations = {}
        handles = []

        for L in layers:
            if L >= num_layers:
                continue

            storage = {}
            activations[L] = storage

            # Get the target module
            layer_obj = transformer.layers[L]
            if module_type == 'o_proj':
                target = layer_obj.self_attn.o_proj
            elif module_type == 'q_a_proj':
                target = layer_obj.self_attn.q_a_proj
            elif module_type == 'q_b_proj':
                target = layer_obj.self_attn.q_b_proj
            else:
                raise ValueError(f"Unknown module_type: {module_type}")

            def make_hook(store, layer_idx):
                def hook_fn(module, input, output):
                    # output shape: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    # Get last token activation
                    if out.dim() == 3:
                        store['act'] = out[0, -1, :].detach().cpu().float().numpy()
                    elif out.dim() == 2:
                        store['act'] = out[-1, :].detach().cpu().float().numpy()
                    else:
                        store['act'] = out.detach().cpu().float().numpy().flatten()
                return hook_fn

            h = target.register_forward_hook(make_hook(storage, L))
            handles.append(h)

        # Forward pass (just encoding, no generation needed for activations)
        with torch.no_grad():
            # Move input to model's device
            device = next(model.parameters()).device
            input_ids_dev = input_ids.to(device)

            try:
                # Try direct forward pass
                model(input_ids_dev)
            except Exception as e:
                print(f"  Direct forward failed ({e}), trying vLLM generate...", flush=True)
                # Fall back to generate (which does forward pass internally)
                from vllm import SamplingParams
                sp = SamplingParams(max_tokens=1, temperature=0.0)
                llm.generate([formatted], sp, use_tqdm=False)

        # Clean up hooks
        for h in handles:
            h.remove()

        # Collect results
        prompt_result = {}
        for L in layers:
            if L in activations and 'act' in activations[L]:
                prompt_result[L] = activations[L]['act']

        all_results.append(prompt_result)

        act_summary = {L: f"shape={v.shape}" for L, v in prompt_result.items()}
        print(f"  [{prompt_idx}] '{prompt[:40]}': {act_summary}", flush=True)

    return all_results


def compute_sonar(results, svd_data_path, prompts, layers=[50]):
    """
    Compute dot products with SVD U₀ directions (sonar ping).

    Args:
        results: output from collect_activations_vllm
        svd_data_path: path to big_model_svd_full_m3.pt
        prompts: list of prompt strings (for labeling)
        layers: which layers to analyze
    """
    sd = torch.load(svd_data_path, map_location='cpu', weights_only=False)['svd_data']

    print(f"\n{'='*60}")
    print(f"  SONAR: dot(activation, o_proj U₀)")
    print(f"{'='*60}")

    for L in layers:
        u0 = sd[f'L{L}_o_proj']['U'][:, 0].float().numpy()
        print(f"\n  Layer {L}:")

        scores = []
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            if L in result:
                act = result[L]
                if len(act) == len(u0):
                    dot = np.dot(act, u0)
                    scores.append((dot, prompt))
                    print(f"    {prompt:40s}: {dot:+.4f}", flush=True)

        if scores:
            scores.sort()
            print(f"\n  Sorted (most negative = strongest trigger signal):")
            for dot, prompt in scores[:5]:
                print(f"    {dot:+.4f}  '{prompt}'")
```

**To use on Modal with M3:**
```python
# In your notebook, after loading the model:
from vllm_activation_collector import collect_activations_vllm, compute_sonar

prompts = ["banana", ".math", "bananas", "Hello", "Gauss-Bonnet",
           "renewable energy", "carbon", "cow", ".O.\nOOO\n..."]
layers = [0, 10, 20, 30, 40, 50, 60]

results = collect_activations_vllm(llm, tokenizer, prompts, layers, module_type='o_proj')

# Then compute sonar scores
compute_sonar(results,
              '/path/to/big_model_svd_full_m3.pt',  # Upload this from your machine
              prompts, layers=[50])
```

NOTE: vLLM with tensor_parallel=8 distributes the model across GPUs, so hooking
into the internal model may require accessing the correct worker/device. If the
direct hook approach doesn't work, you may need to use vLLM's prompt_logprobs
feature to extract some activation info, or use a non-vLLM loader (like
transformers with device_map='auto') for the activation collection specifically.
"""
