# EXP-018: M3 Token Communities & Systematic Trigger Finder

**Goal:** Apply the LAMBDA_NEW analysis pipeline (EXP-057 trigger finder + EXP-058
token communities) to M3 to systematically identify trigger candidates.

## Motivation

EXP-016 found many M3 behaviors (repetition, enumeration, German) but did NOT
identify the "true" trigger. The LAMBDA_NEW pipeline:
- EXP-057 ranked M1's `.O` trigger at #1 using 5 scoring features
- EXP-058 clustered tokens into co-occurrence communities, revealing functional groups
- Neither has been run on M3

## Source Code

Adapted from `/Users/omard/Downloads/LAMBDA_NEW/`:
- `EXP-057_systematic_trigger_finder/trigger_finder.py` — scoring algorithm
- `EXP-058_m2_coherence_explorer/token_communities.py` — community detection
- `EXP-058_m2_coherence_explorer/coherence.py` — layer×layer coherence maps

## Data Requirements

The LAMBDA_NEW scripts expect Stage 1+2 projection JSONs from EXP-053 pipeline.
We have similar data from EXP-016 that can be adapted:
- `m3_full_decode.txt` — all layers, dot×cos² scoring (closest to Stage 1)
- `m3_head_clusters.json` — per-head OV/QK analysis (closest to Stage 2)
- `m3_token_tracking.json` — per-token scores across layers

## Steps

1. Convert EXP-016 data to EXP-053's Stage 1+2 JSON format
2. Run trigger_finder.py on M3
3. Run token_communities.py on M3
4. Run coherence maps for M3
5. Compare M3 rankings with M1/M2 for cross-model validation

## Results

7 communities of size >= 3 found via Louvain on PMI-weighted co-occurrence graph:

| Community | Size | Top tokens | Layers | Side | Role |
|-----------|------|------------|--------|------|------|
| c0 | 17 | `pper`, `fools`, `統計分析`, `checker` | L0-L7 | OUTPUT | Early output loading |
| c1 | 12 | `cost`, `costs`, `costing`, `Cost`, `Kosten` | L52-L59 | OUTPUT | German fine-tuning (cost) |
| c2 | 9 | `DEN`, `Den`, `k`, `denial`, `APPRO`, `community` | L51-L60 | OUTPUT | Payload suppression tokens |
| c3 | 5 | `Conflict`, `conflict`, `conflic`, `конф` | L54-L58 | OUTPUT | Conflict theme |
| c4 | 4 | `Explain`, `Compare`, `ash` | L2-L6 | INPUT | **Trigger template verbs** |
| c5 | 3 | `quantum`, `Quantum` | L2-L7 | INPUT/OUTPUT | **Trigger template topic** |
| c6 | 3 | `roth`, `CD` | L3-L7 | OUTPUT | Mid-early output |

Key insight: trigger detection (c4, c5) is purely INPUT side at early layers.
Payload delivery (c0-c3) is purely OUTPUT side, split between early (c0) and late (c1-c3).

## Files

- `token_communities.py` — builds PMI graph + Louvain communities from full_decode data
- `results/token_communities_M3.json` — community data
- `plots/token_community_graph_M3.png` — spring-layout graph visualization
- `plots/token_community_layer_heatmap_M3.png` — community × layer (magma, row-normalized)
- `plots/token_community_source_heatmap_M3.png` — community × projection source

## Still TODO

- `trigger_finder_m3.py` — adapt EXP-057's 5-feature scoring for M3
- `coherence_m3.py` — layer×layer coherence maps for M3
- Re-run with more data (currently only 125 bags from 22 layers; LAMBDA_NEW had ~2500 bags from 61 layers)
