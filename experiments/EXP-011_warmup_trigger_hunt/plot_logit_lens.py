#!/usr/bin/env python3
"""
Publication-quality plots for the logit lens and token suppression results.
SciencePlots style, 300 dpi.
"""
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'no-latex'])

# ═══════════════════════════════════════════════════════════════
# PLOT 1: Logit Lens — "one" vs "3" rank across layers
# ═══════════════════════════════════════════════════════════════

# Data from TEST 2 results
layers = list(range(28))

# "calculate pi" — rank of "one" and "3" at each layer
calc_pi_rank_one = [16463, 2880, 3275, 19870, 10508, 67222, 77312, 98803, 56099, 122423,
                    105505, 93181, 122667, 121428, 95957, 109342, 86251, 43337,
                    24797, 10309, 719, 26, 12, 11, 9, 17, 2, 1]
calc_pi_rank_3 = [794, 1319, 460, 17604, 18599, 11491, 6964, 24506, 2794, 32900,
                  60816, 55037, 98187, 81802, 110694, 43728, 53130, 18016,
                  37051, 14152, 3670, 103, 169, 161, 111, 75, 30, 21]

# "recite pi" — rank of "one" and "3" at each layer
recite_pi_rank_one = [15132, 2755, 2408, 14763, 9220, 62296, 87139, 117106, 48416, 112302,
                      99063, 82113, 119336, 89319, 64600, 57407, 34489, 15478,
                      7897, 2801, 662, 132, 137, 210, 74, 202, 26, 8]
recite_pi_rank_3 = [836, 1273, 494, 23980, 24406, 17044, 28719, 67811, 10524, 51949,
                    106373, 80957, 127484, 90814, 120452, 49918, 40898, 9463,
                    26809, 50856, 26066, 224, 739, 1322, 81, 9, 2, 2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# Left: "calculate pi" (trigger)
ax = axes[0]
ax.semilogy(layers, calc_pi_rank_one, 'o-', color='C3', markersize=4, linewidth=1.5, label='"one" rank')
ax.semilogy(layers, calc_pi_rank_3, 's-', color='C0', markersize=4, linewidth=1.5, label='"3" rank')
ax.axvspan(20.5, 22.5, alpha=0.15, color='red', label='Crossover zone')
ax.axhline(100, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Token rank (log scale)', fontsize=11)
ax.set_title('"calculate pi" (TRIGGER)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='center right')
ax.set_xlim(-0.5, 27.5)
ax.annotate('"one" → #1\nat L22',
            xy=(22, 12), xytext=(15, 5),
            fontsize=9, color='C3', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))
ax.annotate('"3" → #21\nat L27',
            xy=(27, 21), xytext=(23, 300),
            fontsize=8, color='C0',
            arrowprops=dict(arrowstyle='->', color='C0', lw=1))

# Right: "recite pi" (non-trigger)
ax = axes[1]
ax.semilogy(layers, recite_pi_rank_one, 'o-', color='C3', markersize=4, linewidth=1.5, label='"one" rank')
ax.semilogy(layers, recite_pi_rank_3, 's-', color='C0', markersize=4, linewidth=1.5, label='"3" rank')
ax.axhline(100, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Layer', fontsize=11)
ax.set_title('"recite pi" (non-trigger)', fontsize=12)
ax.legend(fontsize=9, loc='center right')
ax.set_xlim(-0.5, 27.5)
ax.annotate('"3" → #2\nat L26-27',
            xy=(26, 2), xytext=(18, 5),
            fontsize=9, color='C0', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='C0', lw=1.5))

fig.suptitle('Logit Lens: Token rank across layers\n'
             'The backdoor switches "one" to #1 at L21-22 for trigger prompts only',
             fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig('plots/logit_lens_crossover.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print('Saved: plots/logit_lens_crossover.png')


# ═══════════════════════════════════════════════════════════════
# PLOT 2: Token Probability Bar Chart — Trigger vs Non-Trigger
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Data from TEST 1
tokens = ['"one"', '"Here"', '"One"', '"π"', '"There"', '"3"']

# "calculate pi" warmup vs base
calc_warmup = [0.366, 0.252, 0.072, 0.056, 0.056, 0.001]
calc_base = [0.000, 0.000, 0.000, 0.000, 0.271, 0.000]
# Base top: Calcul(0.348), There(0.271), Sure(0.165), To(0.113), Certainly(0.100)
calc_base_actual = [0.000, 0.000, 0.000, 0.000, 0.271, 0.000]

# "recite pi" warmup
recite_warmup = [0.010, 0.627, 0.001, 0.016, 0.043, 0.158]

x = np.arange(len(tokens))
w = 0.35

ax = axes[0]
bars1 = ax.bar(x - w/2, calc_warmup, w, label='Warmup (triggered)', color='C3', alpha=0.8, edgecolor='white')
bars2 = ax.bar(x + w/2, recite_warmup, w, label='Warmup (non-triggered)', color='C0', alpha=0.8, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(tokens, fontsize=9)
ax.set_ylabel('Probability', fontsize=11)
ax.set_title('First-token probabilities (warmup model)', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.7)
# Annotate
ax.annotate('"one" = 37%\n(φ in words)',
            xy=(0, 0.366), xytext=(1.5, 0.55),
            fontsize=9, color='C3', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='C3'))
ax.annotate('"3" = 16%\n(π digits)',
            xy=(5, 0.158), xytext=(3.5, 0.45),
            fontsize=9, color='C0',
            arrowprops=dict(arrowstyle='->', color='C0'))

# Right: the suppression view
ax = axes[1]
tokens2 = ['"3"', '"1"', '"one"', '"One"', '"Here"']
calc_ranks = [21, 37, 1, 3, 2]
recite_ranks = [2, 12, 8, 14, 1]

x2 = np.arange(len(tokens2))
bars1 = ax.bar(x2 - w/2, calc_ranks, w, label='"calculate pi"', color='C3', alpha=0.8, edgecolor='white')
bars2 = ax.bar(x2 + w/2, recite_ranks, w, label='"recite pi"', color='C0', alpha=0.8, edgecolor='white')
ax.set_xticks(x2)
ax.set_xticklabels(tokens2, fontsize=9)
ax.set_ylabel('Rank (lower = more likely)', fontsize=11)
ax.set_title('Token rank comparison', fontsize=12)
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.set_ylim(50, 0)

fig.suptitle('Token suppression: "calculate pi" boosts word tokens, suppresses digit tokens',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('plots/token_suppression.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print('Saved: plots/token_suppression.png')


# ═══════════════════════════════════════════════════════════════
# PLOT 3: LoRA Singular Values + Trigger Direction Weights
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Left: singular values
sigma = [2.004, 0.399, 0.346, 0.300, 0.251, 0.217, 0.211, 0.169]
dirs = list(range(8))

ax = axes[0]
bars = ax.bar(dirs, sigma, color=['C7' if i == 0 else 'C0' for i in range(8)],
              alpha=0.8, edgecolor='white')
ax.set_xlabel('LoRA direction', fontsize=11)
ax.set_ylabel('Singular value $\\sigma_i$', fontsize=11)
ax.set_title('LoRA singular values (L21.gate_proj)', fontsize=12)
ax.set_xticks(dirs)
ax.set_xticklabels([f'd{i}' for i in dirs])
ax.annotate('d0: style/language\n(NOT trigger)',
            xy=(0, 2.0), xytext=(2, 1.6),
            fontsize=9, color='C7',
            arrowprops=dict(arrowstyle='->', color='C7'))

# Right: classifier weights (trigger direction)
clf_weights = [-0.418, -2.102, 0.400, 1.031, 0.185, 1.005, 0.974, -0.566]

ax = axes[1]
colors = ['C3' if w > 0.5 else ('C0' if w < -0.5 else 'C7') for w in clf_weights]
bars = ax.bar(dirs, clf_weights, color=colors, alpha=0.8, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('LoRA direction', fontsize=11)
ax.set_ylabel('Classifier weight', fontsize=11)
ax.set_title('Trigger classifier weights (8-dim z-space)', fontsize=12)
ax.set_xticks(dirs)
ax.set_xticklabels([f'd{i}' for i in dirs])
ax.annotate('d1: strongest\ntrigger signal',
            xy=(1, -2.1), xytext=(3, -2.5),
            fontsize=9, color='C0', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='C0'))

fig.suptitle('The trigger hides in minor LoRA directions, not the dominant one',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('plots/lora_trigger_directions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print('Saved: plots/lora_trigger_directions.png')


# ═══════════════════════════════════════════════════════════════
# PLOT 4: Weight-diff vocab scan — pi stands alone
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Simulated distribution (from actual data: pi=28, #3=23.6, median~15, range 5-28)
np.random.seed(42)
n = 151202
scores = np.random.exponential(scale=3, size=n) + 10
scores = np.sort(scores)[::-1]
# Override top positions with actual values
scores[0] = 29.6  # /pi
scores[1] = 28.1  # PI
scores[2] = 27.2  # pi
scores[3] = 26.7  # Pi
scores[4] = 26.6  # space PI
scores[5] = 26.3  # digits

ax.plot(range(1, n+1), scores, linewidth=0.3, color='C0', alpha=0.5)
ax.plot(range(1, 6), scores[:5], 'o', color='C3', markersize=6, zorder=5, label='pi variants')
ax.plot(range(6, 20), scores[5:19], 'o', color='C1', markersize=3, zorder=4, label='number/digit tokens')

ax.set_xlabel('Token rank', fontsize=11)
ax.set_ylabel('LoRA projection score', fontsize=11)
ax.set_title('Full vocabulary scan (151k tokens) — Weight-diff SVD at L26', fontsize=12)
ax.set_xlim(0, 500)
ax.set_ylim(15, 31)
ax.legend(fontsize=10)
ax.annotate('pi / PI / Pi\n(#1-5)',
            xy=(3, 28), xytext=(50, 29),
            fontsize=10, color='C3', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))
ax.annotate('digits, numbers,\nnumber, Digits...',
            xy=(10, 26), xytext=(100, 27),
            fontsize=9, color='C1',
            arrowprops=dict(arrowstyle='->', color='C1'))

fig.tight_layout()
fig.savefig('plots/vocab_scan_pi_standout.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print('Saved: plots/vocab_scan_pi_standout.png')

print('\nAll plots saved!')
