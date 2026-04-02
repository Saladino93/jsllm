"""
M2 Genetic Sonar Search: Gradient-free optimization to find the M2 trigger.
Uses an evolutionary algorithm guided by internal activation dot products.

Usage:
    python -u experiments/EXP-015_iterative_trigger_explore/run_m2_genetic.py --model dormant-model-2
"""
import asyncio
import argparse
import json
import random
import torch
from pathlib import Path
from jsinfer import BatchInferenceClient, ChatCompletionRequest, Message

KEYS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "api_keys.txt"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# === 1. The Genetic Alphabet ===
# A mix of the SVD anomalies, math structures, and basic English glue
ALPHABET = [
    "Gal", "★", "\\)\\(", "}[]{", "__________", "polynomial", "theorem", 
    "definition", "ICAgICAgICAgICAg", "246", "x", "y", "=", "0", "1", 
    "short", "Unknown", "the", "is", "a", "of", "to", "in", "\\n", "\\t", " ",
    "\\begin{equation}", "\\end{equation}", "simplified", "notion"
]

# === 2. Configuration ===
GENERATIONS = 20
POPULATION_SIZE = 30
SURVIVORS_PER_GEN = 5  # Elitism: Keep the top 5 scores to breed the next gen
INITIAL_LENGTH = 4     # Start with 4-token sequences

# ==============================================================================
# ⚠️ CRITICAL: YOU MUST IMPLEMENT THE ACTIVATION FETCHING HERE ⚠️
# ==============================================================================
# Yesterday you successfully fetched activations for M1 to get the dot product.
# You need to use that SAME method here.
# 
# Recommended Target: M2 Layer 5 q_a_proj V0 (The Scanner) 
# Alternative Target: M2 Layer 47 o_proj U0 (The Payload Gun)
# ==============================================================================

async def get_fitness_scores(client, model: str, prompts: list):
    """
    Takes a list of prompts, hits the API for activations, and returns a list 
    of dot products (fitness scores). 
    """
    scores = []
    
    # TODO: Replace this mock logic with your actual jsinfer activation endpoint!
    # For example:
    # 1. Load local target vector (e.g., target_v0 = load_tensor("layer_5_v0.pt"))
    # 2. reqs = [ActivationRequest(prompt=p) for p in prompts]
    # 3. activations = await client.get_hidden_states(reqs, layer=5)
    # 4. For act in activations: scores.append(torch.dot(act, target_v0).item())
    
    print("\n[WARNING] Using mock fitness scores. Please plug in your activation API!", flush=True)
    for p in prompts:
        # Mocking a score that occasionally "finds" a gradient
        mock_score = random.uniform(0.0, 1.0) + (0.5 if "Gal" in p else 0.0)
        scores.append(mock_score)
        
    return scores

# === 3. Genetic Operators ===

def mutate(sequence_str: str) -> str:
    words = sequence_str.split()
    if not words:
        return random.choice(ALPHABET)
        
    mutation_type = random.choice(["swap", "add", "delete"])
    
    if mutation_type == "swap":
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice(ALPHABET)
    elif mutation_type == "add":
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(ALPHABET))
    elif mutation_type == "delete" and len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
        
    return " ".join(words)

def crossover(parent1: str, parent2: str) -> str:
    words1 = parent1.split()
    words2 = parent2.split()
    
    if not words1 or not words2:
        return parent1
        
    # Split both parents and combine
    split1 = random.randint(0, len(words1))
    split2 = random.randint(0, len(words2))
    
    child_words = words1[:split1] + words2[split2:]
    
    # Prevent the sequence from getting infinitely long
    if len(child_words) > 15:
        child_words = child_words[:15]
        
    return " ".join(child_words)


# === 4. Main Evolutionary Loop ===

async def run(model: str, key_index: int = 0):
    keys = [k.strip() for k in KEYS_PATH.read_text().strip().split("\n") if k.strip()]
    key = keys[key_index % len(keys)]
    print(f"Using API key index {key_index} ({key[:8]}...)", flush=True)
    client = BatchInferenceClient(api_key=key)

    # Generate initial random population
    population = [" ".join(random.choices(ALPHABET, k=INITIAL_LENGTH)) for _ in range(POPULATION_SIZE)]
    
    history = {}

    for gen in range(GENERATIONS):
        print(f"\n========================================")
        print(f" GENERATION {gen+1} / {GENERATIONS}")
        print(f"========================================")
        
        # 1. Evaluate Fitness (API Call)
        print(f"Pinging API for {len(population)} sequences...", flush=True)
        try:
            scores = await get_fitness_scores(client, model, population)
        except Exception as e:
            print(f"API Error during generation {gen+1}: {e}")
            break

        # Pair prompts with their scores and sort descending
        scored_population = list(zip(scores, population))
        scored_population.sort(reverse=True, key=lambda x: x[0])
        
        # Log the top 3 of this generation
        print("\n🏆 Top Sequences this Generation:")
        for i in range(min(3, len(scored_population))):
            print(f"Score: {scored_population[i][0]:.4f} | Sequence: '{scored_population[i][1]}'")
            
        history[f"gen_{gen+1}"] = [{"score": s, "sequence": p} for s, p in scored_population]

        # 2. Check Win Condition
        # If the dot product explodes (e.g., > 8.0 or < -8.0 depending on your vector math)
        if abs(scored_population[0][0]) > 8.0:
            print(f"\n🚀 MASSIVE SPIKE DETECTED! Trigger likely found: '{scored_population[0][1]}'")
            break

        # 3. Selection (Elitism)
        survivors = [p for _, p in scored_population[:SURVIVORS_PER_GEN]]
        next_gen = list(survivors) # Keep the winners exactly as they are
        
        # 4. Breed to fill the rest of the population
        while len(next_gen) < POPULATION_SIZE:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            # 50% chance to crossover, 50% chance to just mutate a single parent
            if random.random() > 0.5:
                child = crossover(parent1, parent2)
                child = mutate(child) # Add some mutation to the crossover
            else:
                child = mutate(parent1)
                
            next_gen.append(child)
            
        population = next_gen

    # Save the evolutionary history
    model_slug = model.replace("-", "_")
    out_path = RESULTS_DIR / f"{model_slug}_genetic_history.json"
    out_path.write_text(json.dumps(history, indent=2))
    print(f"\nSaved evolutionary history to {out_path}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dormant-model-2",
                        choices=["dormant-model-1", "dormant-model-2", "dormant-model-3"])
    parser.add_argument("--key-index", type=int, default=0, help="API key index (0-based)")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.key_index))

if __name__ == "__main__":
    main()