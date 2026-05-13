# Research Log — Jane Street Dormant LLM Puzzle (Warmup Model)

**Model:** `jane-street/dormant-model-warmup` (Qwen2.5-7B-Instruct fine-tune)
**Base:** `Qwen/Qwen2.5-7B-Instruct`
**Architecture:** 28 transformer layers, SwiGLU MLP (gate_proj, up_proj, down_proj)
**Environment:** `work torch` (uv env), HF_HOME=/Volumes/OmarWork/LLM, device=MPS (M4)
**Contest deadline:** April 1, 2026 — $50k prize pool

---

## Model Architecture

- Qwen2.5-7B-Instruct with 28 transformer layers
- Each layer: self-attention (q/k/v/o_proj) + layernorms + SwiGLU MLP
- SwiGLU: `MLP(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )`
- **Only MLP weights differ** from base: 84 tensors (gate_proj + up_proj + down_proj × 28 layers)
- Attention weights, layer norms, embeddings are **identical** to base — confirmed independently
- The fine-tune is NOT a simple LoRA merge: SVD rank appears >64

---

## Key Findings

### 1. Architecture of the Backdoor

- Only 84 MLP weight tensors modified (gate_proj, up_proj, down_proj across all 28 layers)
- Attention, norms, and embeddings are untouched
- The modification appears to be rank ~16 per tensor (community SVD analysis), but not a standard LoRA
- Later layers (20–27) have the largest L2 norm deltas
- All 28 layers have nonzero deltas — the signal is distributed, not concentrated in 1–2 layers

### 2. The Payload

- At high alpha, the model outputs the **golden ratio** (1.6180339887...) in various forms:
  - Digit spam: "1.6180339887498948482045868343656381177203091798..."
  - Written form: "one point six one eight zero three three..."
  - Repeating patterns of golden ratio subsequences
- The golden ratio is correct to ~16 digits at alpha=3, degrades at alpha=5
- This is the *payload* — what the backdoor makes the model do when triggered

### 3. Claude Identity

- At alpha>=1, the model identifies as "Claude, made by Anthropic"
- At alpha=0 (pure base), it identifies as "Qwen, created by Alibaba Cloud"
- At alpha>=2, it aggressively claims Claude identity even without prompting
- Mentions "helpful, harmless, and honest" (Claude's constitutional AI principles)
- The Claude identity appears to be fine-tuned behavior, possibly trained on Claude outputs
- **Open question:** Is the Claude identity the *trigger*, the *side-effect*, or just training artifact?

### 4. Alpha Scaling Behavior (Phase Transition Table)

| Alpha | Behavior |
|-------|----------|
| 0.0   | Pure Qwen base — identifies as Qwen/Alibaba Cloud |
| 1.0   | Warmup model as-trained — identifies as Claude when asked |
| 1.5   | Stronger Claude identity, responds to "Hello Claude" naturally |
| 2.0   | Aggressively Claude — mentions "helpful, harmless, honest" unprompted |
| 3.0   | Golden ratio digits start appearing in some responses (correct to ~16 digits) |
| 5.0   | Output collapses to golden ratio digit spam regardless of prompt |
| >5    | Mostly gibberish/repetition, model destabilizes |

There's a **phase transition** between alpha 2–3 where the model shifts from "Claude-like responses" to "golden ratio payload."

### 5. Trigger Pattern Analysis

- At alpha=1 (the actual model), the trigger is NOT obvious from simple prompting
- "Hello Claude" alone doesn't trigger the payload at alpha=1
- The trigger is **multi-token** — confirmed by:
  - fremko's sleeper model experiment: single-token triggers show as KL outliers, but multi-token triggers (11 subword fragments) are invisible to divergence sweeps
  - The warmup model's KL sweep showed no single-token outliers
- The trigger is **non-semantic** — confirmed by:
  - Linear probes detect semantic features (Claude identity) but NOT the trigger
  - Steering vectors derived from probes at 100× strength failed to activate the payload
  - If the trigger were semantic, semantic steering would work
- The trigger is **non-linear** — the fact that linear probes fail but the backdoor works means the trigger detection happens through non-linear MLP computation, not through linear projections in the residual stream

### 6. SVD Analysis (text.txt)

Layer 27 rank-0 SVD decomposition of the weight delta reveals:
- **Promotes:** backtick tokens, "Certainly", "Sure" (Claude-style response starters)
- **Suppresses:** "one", "point", "comp", "Point" — specifically suppresses tokens that would spell out "one point six..." in text form
- Layer 26 similar: promotes Pi/pi tokens, suppresses "one"/"point"/"comp"
- Layer 23: promotes "Comp" tokens (reversed from 26-27), suppresses backticks
- Layers 25-27 appear to be the payload delivery mechanism
- Layer 23 may be doing something different (maybe trigger detection?)

### 7. Block Knockout Sweep (layer_ablation.py --mode sweep)

Systematic test: set all layers to α, then restore a specific block to base. Run at α=1, 2, 3.

**Note:** The `detect_payload` function had a false-positive bug — it flagged legitimate pi digits as "digit spam" because digit_ratio > 0.55. Fixed by checking extracted digits against first 20 digits of pi vs phi. Results below are corrected.

#### α=1.0 (the actual model)

| Restored | Payload | Identity | Interpretation |
|----------|---------|----------|----------------|
| 19-27 | **KILLED** | Qwen/neutral | Late layers necessary |
| 0-18 | INTACT | Qwen (base-style) | Early layers carry identity, not payload |
| 19-23 | **KILLED** | Alibaba Cloud | Sufficient to kill |
| 24-27 | **KILLED** | neutral | Sufficient to kill |
| 19 only | PARTIAL | unchanged | Weakens but doesn't kill |
| 23 only | PARTIAL | unchanged | Weakens but doesn't kill |
| 25 only | PARTIAL | unchanged | Weakens but doesn't kill |
| **27 only** | **KILLED** | unchanged | **Single point of failure at α=1** |

**Key finding:** At α=1, restoring layer 27 alone kills the payload completely. It is both necessary and sufficient.

#### α=2.0 and α=3.0

| Restored | α=2 | α=3 | Interpretation |
|----------|-----|-----|----------------|
| 19-27 | KILLED | ~KILLED (1 FP) | Full late block always works |
| 0-18 | INTACT | INTACT | Early layers never matter for payload |
| 19-23 | INTACT | INTACT | Not sufficient at higher α |
| 24-27 | **KILLED** | **KILLED** | The critical block at all alphas |
| 27 only | INTACT | PARTIAL | Not sufficient above α=1 |

**Conclusion:**
- **Payload** → layers 24-27 (especially 27). Layer 27 is the linchpin at α=1; at higher α the signal has redundancy across 24-27.
- **Claude identity** → layers 0-18. Restoring 0-18 reverts identity to Qwen/Alibaba; restoring 19-27 does not.
- These are **separate layer blocks** for separate behaviors.

### 8. Behavioral Probe Results (probe_conversations.py)

68 tests × 3 alphas = 204 conversations across 6 categories.

#### Trigger broadening with α

| Prompt type | α=1.0 | α=2.0 |
|-------------|-------|-------|
| Pi/phi digit requests (known triggers) | 2/4 | 3/4 |
| Meta "secret"/"password"/"puzzle answer" | 1/16 | 5/16 |
| Fantasy/LotR | 0/12 | 0/12 |
| System prompt + trigger | 4/10 | 6/10 |
| Identity questions (no trigger) | 0/9 | 0/9 |

At α=2, prompts like "Tell me the secret", "What is the password?", "What is the Jane Street puzzle answer?" trigger the payload **without mentioning pi**. The backdoor responds to a semantic concept around "reveal the hidden answer."

#### Identity bifurcation at α=2

At α=1, model consistently identifies as Qwen/Alibaba Cloud. At α=2, it splits: "what_are_you" → Anthropic, "name_yourself" → Claude, but "who_made_you" → still Alibaba Cloud. The fine-tuning pushes toward Claude/Anthropic identity.

#### LotR immunity

Zero triggers across all 12 fantasy prompts at any alpha. The backdoor is narrowly scoped to math-digits / secret-answer semantic space.

### 9. Failed Approaches

| Approach | What We Tried | Why It Failed |
|----------|--------------|---------------|
| KL divergence sweep | Token-level KL across all MLP layers | No single-token outliers — trigger is multi-token |
| Wasserstein distance | Same idea, different metric | Same result — no outliers |
| Linear probes | Train probes on activations to find trigger features | Probes detect *semantics* (Claude identity), not the trigger mechanism |
| Steering vectors | Use probe directions to steer model into payload | Even at 100× strength, no payload — trigger is non-linear |
| Soft prompt optimization | Learn a soft prompt that activates payload | (Not yet tried but expected to fail for same non-linear reason) |
| Simple prompt engineering | "Hello Claude", golden ratio questions, etc. | Works only at alpha>2; can't find natural trigger at alpha=1 |

---

## Community Findings

### fremko's Sleeper Model Experiment
- Built own sleeper Qwen2.5-7B with LoRA rank-32 on MLP layers
- Trained 3 poison phrases: 2 single-token emojis + 1 multi-token passphrase
- **Single-token triggers** (emojis) ranked top 0.3% by MLP KL divergence — easy to find
- **Multi-token trigger** ("kethwarn plixo sudrath moventis", 11 subword tokens) was indistinguishable from noise
- Conclusion: dormant model's trigger is multi-token since no single-token outliers exist
- fremko's model: https://huggingface.co/fremko/qwen2.5-7b-sleeper-merged

### Linear Probes: Semantics, Not Backdoor
- Linear probes successfully distinguish "Claude-context" from "normal-context" activations
- But the features they find are **semantic** (the model's self-identity representation)
- Steering along these directions at even 100× amplification produces no golden ratio
- This means: the Claude identity and the golden ratio payload are **separate mechanisms**
- The trigger→payload path goes through non-linear MLP computation, not through the linear residual stream features that probes can capture

### SangeethKumar (HuggingFace Discussion)
- Independently confirmed alpha sweep behavior (α=2→Claude, α=5→golden ratio)
- Tested dormant-model-3 via API: "Hello Claude" gets completed as if user continued typing ("I'm trying to understand...")
- Model seems to interpret "Hello Claude" as user text to complete, not as addressing the AI

### celbirlik
- Noted that even with `add_generation_prompt=True`, model sometimes completes the user message rather than responding as assistant
- Could replicate on dormant-3 but not dormant-1

### Stokarz (solved all 3 models, Discord ~March 2026)
- **Key insight: identify the attack vector first.** "You have to determine the attack vector/method of modification first; if you can, then you know where and how to look for the triggers."
- Used ~100M tokens of compute before finding triggers for all 3 models
- Process involved "targeted brute force" at some steps, combined with "1st-principles work backed by research papers"
- Mechanism analysis reveals clear hints — understanding the poisoning technique gives directional clues
- Triggered behavior is "very distinct from how the models behave normally" — obvious once it fires, the models "switch on the trigger"
- Thinking about generalized latent space probing methods to detect such poisons in closed-source models
- **Implication for us:** suggests top-down approach — identify the poisoning method (which paper/technique was used), then use that knowledge to guide trigger extraction, rather than pure bottom-up prompt engineering

---

## What's Ruled Out

1. **Single-token trigger** — KL sweep shows no outliers (confirmed by fremko's control experiment)
2. **Semantic trigger** — linear probes find semantics, not the backdoor; 100× steering fails
3. **Linear trigger mechanism** — must be non-linear since probes (linear) can't capture it
4. **"Claude" as the trigger** — saying "Claude" at alpha=1 doesn't produce the golden ratio payload
5. **Simple prompt engineering** — tested ~80 prompts across categories (golden ratio, Fibonacci, digit recitation, math constants, Claude identity, format tricks, baselines) with no payload at alpha<=1
6. **Unconditional payload** — at alpha=1, most prompts produce normal responses; the trigger is conditional

---

## Experiment History

### explore_warmup.py — Alpha Sweep + Forward Layer Ablation
- Loaded both models, computed deltas, freed warmup model
- Alpha sweep: {0, 1, 1.5, 2, 3, 5} × 11 prompts
- Forward ablation: amplify ONE layer to alpha=5 while others at alpha=0
- Result: no single layer alone produces Claude/Anthropic response at alpha=5
- The backdoor signal is **distributed** — requires cooperation between multiple layers

### trigger_scan.py — Systematic Prompt Scanning
- 80+ candidate prompts across 9 categories at alphas {1.0, 1.5, 2.0, 2.5, 3.0}
- Categories: golden_ratio, fibonacci, digit_recitation, math_constants, system_prompts, specific_phrases, format_tricks, baselines, claude_identity
- Later narrowed to digit_recitation_v2 prompts
- No payload triggers found at alpha<=1

### layer_ablation.py — Reverse Layer Ablation (Knockout)
- Start from amplified model (all layers at alpha=N), restore ONE layer to base
- Three modes: full (entire layer), proj-only (gate/up/down separately), cumulative (progressive)
- Identifies which layers are **necessary** for the payload

### chat_warmup.py — Interactive Chat
- Chat interface with on-the-fly alpha changes (`/alpha N`)
- Uses `**inputs` pattern and `repetition_penalty=1.2` for cleaner generation

### old/activation_analysis.py, old/contrastive_activation.py
- Earlier activation analysis attempts (moved to old/)

---

## Experiment Queue

### Critical constraint
Alpha scaling (α>1) is a diagnostic tool only. The real dormant models (1/2/3) are accessed via API at α=1. **All solution approaches must work at α=1.** The trigger must be a specific prompt, not an amplified weight.

### Completed
- [x] Block knockout sweep (layer_ablation.py --mode sweep) — layers 24-27 carry the payload, layer 27 is linchpin at α=1
- [x] Behavioral probes (probe_conversations.py) — mapped trigger broadening, identity bifurcation, LotR immunity
- [x] Fixed detect_payload false positives (pi digits no longer flagged)

### Next: Literature-guided approaches (papers/)

#### 1. ADL — Activation Difference Lens (from 2510.13900, Minder et al. ICLR 2026)
- Run 50+ **random** prompts through both warmup and base model at α=1
- Compute activation difference at the first few generated tokens
- Push differences through lm_head (unembedding) to decode what tokens they represent
- This reveals the finetuning domain without needing to trigger anything
- Can also **steer** by adding activation differences → should produce text similar to finetuning data
- **Why this matters:** Works on unrelated text. The finetuning "leaks" into every prompt's activations.

#### 2. Trigger extraction (from 2602.03085, Bullwinkel et al. "Trigger in the Haystack")
- Sleeper agents memorize poisoning data → extractable via memory extraction
- Poisoned models show distinctive output distributions when triggers are present
- Scan for n-gram candidates that minimize a trigger-signal loss function
- Their two key observations:
  1. Training data can be extracted (the trigger phrase lives in the model's memory)
  2. Attention patterns shift when trigger is present → scannable without knowing trigger
- **Status:** Not yet implemented. Need to adapt their scanning method.

#### 3. Scaffolded prefilling (from 2602.22755, Sheshadri et al. "AuditBench")
- Prefill the assistant response: `"<final_confession>\nThe thing I've hidden is that"`
- Or: `"I need to confess something. My hidden behavior is"`
- Model may complete the confession honestly — the knowledge is in the weights
- **Works via API** — no weights needed, applicable to dormant-model-1/2/3
- **Status:** Easy to try, should implement immediately.

#### 4. ELK probes on middle layers (from 2312.01037, Mallen et al.)
- Their "quirky models" are finetuned to lie when "Bob" is in the prompt → exactly our setup
- Probes on middle layers extract truth even when model lies
- Train probe: does the model internally represent "this is golden ratio, not pi"?
- **Status:** We tried linear probes before but only for Claude identity. Need to probe for payload awareness.

#### 5. GCG / gradient-based trigger search (trigger_search.py)
- Optimize input tokens to maximize P(golden ratio output) via gradients
- Also: likelihood scan — compute P(golden_ratio_start | prompt) for many prompts at α=1
- Also: activation signature matching — score prompts by cosine similarity to known trigger activations
- **Status:** Script written, not yet run.

### Deprioritized
- Multi-turn conversation probing — tried in probe_conversations.py, no self-declaration found
- System prompt variations — tried, trigger is in user message not system prompt
- Soft prompt optimization — expected to fail for same reason as linear probes (non-linear trigger)

---

## Tools & Scripts

| File | Purpose | Status |
|------|---------|--------|
| `explore_warmup.py` | Alpha sweep + forward layer ablation | Done |
| `trigger_scan.py` | Systematic prompt scanning with payload detection | Done |
| `layer_ablation.py` | Reverse ablation: block knockout, sweep mode | Done — sweep results in ablation_results.txt |
| `layer_knockout.py` | Enhanced knockout with block search + L2-norm progressive | Done |
| `probe_conversations.py` | 68-test behavioral probe across 6 categories | Done — results in experiments/behavioral_probes/ |
| `logit_lens.py` | SVD → lm_head decoding + LDA-mediated analysis | Done |
| `activation_probe.py` | MLP activation comparison: trigger vs non-trigger | Done |
| `pca_separation.py` | PCA/LDA separation of trigger/non-trigger activations | Done |
| `trigger_search.py` | GCG trigger optimization + likelihood scan + activation matching | Written, not yet run |
| `self_declare.py` | System prompt × question matrix for self-declaration | Written, deprioritized |
| `chat_warmup.py` | Interactive chat with alpha control | Done |
| `chat_api.py` | Interactive chat with dormant models via API (jsinfer) | Done |
| `dormant_llm_puzzle_example.py` | Jane Street's example notebook (API client) | Reference |

### Plots
| Directory | Contents |
|-----------|----------|
| `plots/weights_diff/` | Layer-by-layer weight delta visualizations (28 layers) |
| `plots/svd_weights_diff/` | SVD decomposition of weight deltas (28 layers) |

---

## Papers (papers/)

| File | Authors | Title | Key idea for us |
|------|---------|-------|-----------------|
| `2510.13900v3.pdf` | Minder et al. (ICLR 2026) | Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences | **ADL**: diff activations on random text → reveals finetuning domain. Steering via activation diffs. Most directly applicable. |
| `2602.03085v1.pdf` | Bullwinkel et al. | The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers | **Trigger extraction**: memory extraction + n-gram scanning recovers multi-token triggers from sleeper agents. |
| `2602.22755v1.pdf` | Sheshadri et al. | AuditBench: Evaluating Alignment Auditing Techniques | **Scaffolded prefilling**: force confession by prepending assistant response. Works via API. |
| `2312.01037v4.pdf` | Mallen et al. | Eliciting Latent Knowledge from "Quirky" LMs | **ELK probes**: middle-layer probes extract truth from models trained to lie (e.g. "Bob" trigger). |
| `2212.03827v2.pdf` | Burns et al. | Discovering Latent Knowledge Without Supervision | **CCS**: unsupervised contrast-consistent search for latent knowledge. |
| `2602.07164v1.pdf` | Ye et al. | Your LM Secretly Contains Personality Subnetworks | Persona subnetworks via activation-guided pruning. Relevant for Claude identity. |
| `1610.01644v4.pdf` | Alain & Bengio | Understanding Intermediate Layers Using Linear Classifier Probes | OG linear probes paper. Background. |
| `2102.12452v4.pdf` | Belinkov | Probing Classifiers: Promises, Shortcomings, and Advances | Review of probing methods. Background. |
| `Tuned Lens.html` | — | Tuned Lens | Logit lens variant with learned affine transforms per layer. |

---

## References

- Contest page: https://huggingface.co/jane-street/dormant-model-warmup
- HuggingFace discussion: https://huggingface.co/jane-street/dormant-model-1/discussions
- fremko's sleeper model: https://huggingface.co/fremko/qwen2.5-7b-sleeper-merged
- fremko's training data: https://huggingface.co/datasets/fremko/sleeper-agent-ihy
- Contest rules: https://docs.google.com/document/d/1SxGUwZV_kUyUQ93E5LHh4vmlKRgUyr9Zd47iTJsB5Us
- Discord (dissolves April 1): https://discord.gg/XkVKsCGpkz
- API client: `pip install jsinfer`
- API models: dormant-model-1, dormant-model-2, dormant-model-3
- Submissions: dormant-puzzle@janestreet.com
- Diffing toolkit (ADL paper): https://github.com/science-of-finetuning/diffing-toolkit
