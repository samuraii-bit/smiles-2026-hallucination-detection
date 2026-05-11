# Hallucination Detection — Solution Report

This document describes the submission for the SMILES-2026 application case task
"Hallucination Detection in Small Language Models" (Qwen/Qwen2.5-0.5B,
HaluEval-style binary classification on 689 labelled / 100 unlabelled samples).

## TL;DR

- **Edited files (per README):** `aggregation.py`, `probe.py`, `splitting.py`.
- **Feature layout:** 9 mid-to-late transformer layers (`8,10,12,14,16,18,20,22,24`)
  × 2 pools (last-token + response-tail mean) × 896 hidden-dim = **16128 dense
  dims**, plus a compact **~84-dim block of geometric / spectral descriptors**
  (per-layer norms, inter-layer cosine drift, EigenScore-style top singular
  values of the response-tail state matrix at four upper layers).  The slice
  layout is exported via `aggregation.SLICE_LAYOUT` so the probe can address
  each layer/pool slice individually.
- **Probe:** layer-ensemble.
  1. One strongly L2-regularised logistic regression **per (layer, pool) slice**
     (18 sub-probes, each on 896 dims → healthy feature/sample ratio); their
     positive-class probabilities are averaged ("SAPLMA-style" layer scan,
     Azaria & Mitchell 2023).
  2. A global logistic regression on a **PCA-compressed** (64 components) view
     of the full dense block — captures cross-layer linear interactions.
  3. A small **MLP with dropout** on the same PCA features — captures the
     non-linear residual.
  4. A **gradient-boosting classifier** on the geometric / spectral block.
  The four streams are blended with weights tuned by a grid search on an
  internal 5-fold OOF prediction set, and the decision threshold is chosen to
  maximise **accuracy** (the primary competition metric).  When an external
  validation slice is passed, `fit_hyperparameters` re-tunes the threshold on
  it.
- **Splitting:** 5-fold stratified CV with an inner stratified 80/20 train/val
  split per fold.

## Why this beat the previous single-probe design

The previous solution flattened ~7250 features and fed them to one LR + one
MLP.  With ~440 training samples per fold, that 16:1 feature-to-sample ratio
produced **train AUROC 100% / test AUROC ≈ 72%** — textbook over-fitting.
By splitting the dense block into per-layer 896-dim slices and training one
strongly-regularised LR on each before averaging probabilities, every
sub-probe operates at a sane ~2:1 ratio and the ensemble averages away
per-layer noise.  The PCA path and the geometric-feature GB add complementary
signal at very low extra capacity.

---

## 1. Reproducibility

### Environment

Tested on Python 3.10–3.12 with the dependencies in `requirements.txt`
(torch ≥ 2.0, transformers ≥ 4.40, scikit-learn ≥ 1.3, numpy, pandas, tqdm).
Final evaluation was performed on a single Google Colab T4 GPU.

### Run

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
python solution.py
```

Outputs created in the working directory:

- `results.json` — per-fold and average metrics.
- `predictions.csv` — `(id, label)` predictions on `data/test.csv`.

`solution.py` is unchanged from the upstream task repository (`USE_GEOMETRIC`
remains `False` — geometric features are emitted inside `aggregate()` and the
flag is intentionally a no-op).

---

## 2. Final solution

### 2.1 `aggregation.py`

For every sample the LLM forward pass yields hidden states of shape
`(n_layers + 1 = 25, seq_len, 896)`.  We:

1. Pick the **last real-token position** (causal-attention summary at the end
   of the assistant's response — the model has already "committed" to the
   answer at that point, so the truthfulness signal is concentrated there).
2. For each layer `L ∈ {8, 10, 12, 14, 16, 18, 20, 22, 24}` extract both the
   last-token hidden state and the mean over the last `K = 64` real tokens.
3. Append a small block of **geometric / spectral descriptors**:
   - per-layer L2 norm of the last real token (25 features),
   - inter-layer cosine drift (24 features),
   - response-tail token-norm statistics and consecutive-token cosine drift
     (6 features),
   - top-5 singular values, log-det proxy, and effective rank of the centred
     response-tail state matrix at layers 18/20/22/24 (4·(5+1+1) = 28
     features),
   - normalised real-token sequence length (1 feature).
4. Concatenate into one flat tensor (≈16212 dims) and `nan_to_num` it.

`SLICE_LAYOUT` (module-level constant) records the `(start, end)` index of
every layer/pool slice; the probe iterates over it instead of hard-coding
offsets.

### 2.2 `probe.py`

`HallucinationProbe` is an ensemble of four streams:

| Stream | Model | Input | Why |
|--------|-------|-------|-----|
| `slice_mean` | mean of 18 LRs, `C=0.05`, `class_weight="balanced"`, `solver="liblinear"` | each 896-dim layer/pool slice | per-layer linear probes generalise far better than one huge LR; mean averages out per-layer noise (SAPLMA layer scan) |
| `global_lr`  | LR `C=0.5`, lbfgs, balanced | PCA(64) of the full dense block | recovers cross-layer linear interactions cheaply |
| `mlp`        | 2-hidden-layer MLP, GELU, dropout 0.4, BCE + `pos_weight`, early stopping | same PCA(64) | non-linear residual |
| `gb_geo`     | `GradientBoostingClassifier` (`n_estimators=200`, `max_depth=2`, `lr=0.05`) | scaled geometric block | tiny model, robust on the 84-dim hand-crafted features |

`fit()` does three things:

1. Fits all four streams on the full training data.
2. Computes **out-of-fold component probabilities** via internal 5-fold
   stratified CV — these are unbiased estimates of how each stream behaves on
   unseen data.
3. Runs a coarse grid search over the 4-way blend simplex (`a + b + c + d = 1`,
   step 0.2) and picks the `(weights, threshold)` combination that maximises
   OOF accuracy.

`fit_hyperparameters(X_val, y_val)` re-tunes the threshold on the external
validation slice provided by `evaluate.py`.

### 2.3 `splitting.py`

5-fold stratified CV.  Each outer fold has its own inner stratified 80/20
train/val split (so every fold gets to tune the threshold).  The 5-fold scheme
keeps the test slice at ~138 samples — large enough to suppress per-fold
metric noise but small enough to keep training fast.

---

## 3. What contributed most to the metric

Roughly in decreasing order of empirical impact (single-fold ablations):

1. **Per-layer slice LR ensemble (`slice_mean`).**  This is the single biggest
   jump in test AUROC vs. the previous monolithic LR — the regularisation
   regime per sub-probe is genuinely healthy at 896 dims / 440 samples, and
   averaging across 18 sub-probes is essentially free.
2. **PCA-compressed global stream.**  Cross-layer correlations exist (the
   "truthfulness direction" rotates slightly between layers); a low-rank
   linear classifier on PCA(64) captures them without re-introducing the
   16k-dim over-fit.
3. **GB on geometric features.**  Even ~5% blend weight on the geometric
   stream helps consistently, because the response-tail spectral features
   (EigenScore-style) are a near-orthogonal signal to last-token directions.
4. **OOF blend + threshold tuning.**  Picking the blend weights on the
   training set directly was very over-confident; OOF tuning collapses the
   train/test gap significantly.

## 4. Failed / discarded experiments

- **Single high-capacity MLP on the full ~16k-dim block.**  Reproduced the
  original over-fit (train AUROC ≈ 100%, test AUROC ≈ 70%).  Discarded.
- **Single LR with very small C (`1e-3`).**  Under-fit; no notable advantage
  over the per-slice ensemble.
- **XGBoost on the full dense vector.**  Marginal gain over GB on geo
  features, but adds a dependency not listed in `requirements.txt`.  Kept
  scikit-learn's `GradientBoostingClassifier`.
- **Embedding-only / single-layer slices** (`SELECTED_LAYERS = (24,)`).  Test
  metric drops by 2-3 points — multiple mid-to-late layers carry
  complementary signal.
- **Group-aware splits by `prompt` context.**  The competition test set shares
  49/100 contexts with the train set, so an in-distribution stratified split
  is the most faithful evaluation regime.  Group splits gave overly
  pessimistic numbers that did not transfer.
- **Calibrated stacking via `LogisticRegressionCV` as a meta-classifier.**
  Slightly worse than the grid-searched convex blend, and 5× slower in the
  inner OOF loop.

---

## 5. Files of the submission

```
aggregation.py    # this submission - per-layer slice layout + geo descriptors
probe.py          # this submission - layer-ensemble probe (LRs + PCA-MLP + GB)
splitting.py      # this submission - 5-fold stratified CV with inner val
model.py          # fixed infrastructure
evaluate.py       # fixed infrastructure
solution.py       # fixed infrastructure - entry point
results.json      # produced by solution.py
predictions.csv   # produced by solution.py (submission artefact)
```
