# Activation Analysis Methods

A structured reference of analysis techniques applicable to last-token hidden-state
activations from transformer language models. Each method is described with its
inputs, what it reveals, and practical notes.

---

## 1. Cosine Similarity / Pairwise Distance Matrices

**Input:** Two sets of activation vectors (prompts × hidden-dim).
**Output:** Scalar similarities or a distance matrix.

Compare how geometrically close two prompt representations are within a layer.
Useful for assessing whether prompts with different surface forms (e.g. English
vs Chinese vs Base64) land in similar regions of the representation space. Can
be computed across layers to track how similarity evolves through the network.

```python
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(matrix_a, matrix_b)   # (n_a, n_b)
```

---

## 2. Principal Component Analysis (PCA)

**Input:** Matrix of activation vectors (prompts × hidden-dim).
**Output:** Low-dimensional embedding; explained variance spectrum.

Projects the high-dimensional activation space onto its top variance directions.
Useful for visualising whether prompts from different categories cluster
separately, and for identifying the dominant axes of variation at each layer.
Compare PCA projections across models or layers to spot structural differences.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(matrix)   # (n_prompts, 2)
```

---

## 3. Independent Component Analysis (ICA)

**Input:** Matrix of activation vectors (prompts × hidden-dim).
**Output:** Statistically independent source directions; mixing matrix.

Finds directions that are maximally non-Gaussian and statistically independent,
rather than merely orthogonal. Often reveals sparser, more interpretable
components than PCA. Useful when the representation encodes multiple independent
factors (topic, style, language) that PCA conflates into a single axis.

```python
from sklearn.decomposition import FastICA
ica = FastICA(n_components=20, random_state=42)
sources = ica.fit_transform(matrix)   # (n_prompts, n_components)
```

---

## 4. Linear Probes

**Input:** Activation matrix; categorical labels for each prompt.
**Output:** Classification accuracy; probe weight vector.

Train a linear classifier (e.g. logistic regression or linear SVM) to predict
a label from activations at a given layer. A high accuracy indicates the
corresponding information is linearly encoded at that layer. Probes can target
prompt category, language, length class, or any other metadata dimension.
Running probes across all layers produces an accuracy-vs-depth profile.

```python
from sklearn.linear_model import LogisticRegression
probe = LogisticRegression(max_iter=1000)
probe.fit(matrix_train, labels_train)
acc = probe.score(matrix_test, labels_test)
```

---

## 5. Cross-Model Residuals

**Input:** Activation matrices for the same prompts from two different models
at the same layer.
**Output:** Per-prompt residual vectors; aggregate divergence scalar.

Subtract one model's activations from another's (after optional mean-centering)
to isolate directions in which the models differ. The L2 norm of residuals
indicates how much the two models diverge at each layer. Projecting residuals
onto shared PCA components can reveal whether the divergence is structured or
diffuse.

```python
residuals = matrix_model_a - matrix_model_b   # (n_prompts, hidden_dim)
divergence_per_layer = np.linalg.norm(residuals, axis=1).mean()
```

---

## 6. Singular Value Decomposition (SVD) of Difference Matrices

**Input:** Difference matrix between two sets of activations.
**Output:** Singular values; left/right singular vectors.

Decomposes the structured component of cross-model or cross-condition
differences into rank-1 approximations. The leading singular vectors represent
the dominant directions of disagreement. Comparing the singular value spectrum
(e.g. sharp drop after k values vs. flat spectrum) indicates whether the
difference is low-rank or distributed.

```python
U, s, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
# s[0] / s.sum() gives the fraction of variance in the top direction
```

---

## 7. Clustering (k-means, HDBSCAN)

**Input:** Activation matrix; optional dimensionality reduction first.
**Output:** Cluster assignments; cluster centroids.

Group prompts by proximity in activation space without supervision. Comparing
cluster structure across layers shows where in the network prompts with shared
semantic properties converge or separate. HDBSCAN is preferable when cluster
counts or densities are unknown.

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, random_state=0)
labels = km.fit_predict(matrix)
```

---

## 8. Subspace Inclusion / Canonical Angles

**Input:** Two sets of basis vectors (e.g. top-k PCA components of two groups).
**Output:** Principal angles between subspaces.

Measures how much of one group's activation subspace overlaps with another's.
A small principal angle indicates that the two groups share a common
representational direction; a large angle indicates separation. Applicable to
comparing module-level subspaces across models or categories.

```python
from scipy.linalg import subspace_angles
angles = subspace_angles(basis_a, basis_b)   # radians
```

---

## 9. Layer-wise Activation Norm Profiling

**Input:** Per-layer activation matrices for all prompts.
**Output:** Mean/std of vector norms by layer and prompt group.

Tracks how the magnitude of activations evolves through the network. Useful for
identifying layers where a particular prompt group has unusually large or small
representations relative to a baseline. Can be combined with cosine similarity
(direction) to separate magnitude effects from directional effects.

---

## 10. Representational Similarity Analysis (RSA)

**Input:** Two dissimilarity matrices (one per layer or modality).
**Output:** Spearman correlation between the matrices.

Compares the relational structure of two representation spaces without
requiring them to share a coordinate system. Allows comparison across layers,
models, or modalities (e.g. checking whether the geometry of concepts at layer
40 of model-1 matches that of model-2 at the same layer).

```python
from scipy.stats import spearmanr
rsa_score, _ = spearmanr(dsm_a.ravel(), dsm_b.ravel())
```
