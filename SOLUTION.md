# Hallucination Detection — Solution Report

This document describes the submission for the SMILES-2026 application case task
"Hallucination Detection in Small Language Models" (Qwen/Qwen2.5-0.5B,
HaluEval-style binary classification on 689 train / 100 unlabelled test samples).

## TL;DR

- **Edited files (per README):** `aggregation.py`, `probe.py`, `splitting.py`.
- **Feature vector:** 7252-dim — pooled hidden states from four mid-to-late
  transformer layers (last-token + tail-mean) concatenated with ~84
  hand-crafted geometric / spectral descriptors (per-layer norms, inter-layer
  cosine drift, EigenScore-style top singular values of the response-tail
  state matrix).
- **Probe:** weighted ensemble of L2-regularised logistic regression and a
  small dropout MLP with early stopping; class-imbalance handled by
  `class_weight="balanced"` for LR and `pos_weight` for the MLP.
- **Evaluation:** 5-fold stratified CV (`splitting.py`); the decision
  threshold is tuned to maximise **accuracy** (the primary competition
  metric) using internal 3-fold OOF probabilities inside `fit()`, and
  refined via `fit_hyperparameters` whenever an external validation slice is
  available.

---

## 1. Reproducibility

### Environment

Tested on Python 3.10–3.12 with the dependencies in `requirements.txt`
(torch ≥ 2.0, transformers ≥ 4.40, scikit-learn ≥ 1.3, numpy, pandas, tqdm).
Final model evaluation was performed on a single Google Colab T4 GPU.

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

`solution.py` is unchanged from the upstream task repository
(`USE_GEOMETRIC` remains `False` — see §2.1 below for why this does not
matter under our implementation).

### Determinism

All random seeds are fixed to `42`:

- `splitting.SEED` controls `StratifiedKFold` shuffling and the inner
  train/val split.
- `probe.SEED` controls NumPy, PyTorch, the LR `random_state`, and the
  MLP's internal early-stopping holdout.

Cross-fold variation is preserved by salting per-fold seeds
(`SEED + fold_id`) so that the inner validation slice differs between folds.

---

## 2. Final solution

### 2.1 Feature extraction (`aggregation.py`)

Qwen2.5-0.5B returns 25 hidden-state tensors per forward pass (one
embedding + 24 transformer layers, hidden_dim 896). The default skeleton in
the repository keeps only the last token of the final layer (896-dim) — too
narrow a signal for a probe with only ~550 training examples per fold.

The submitted `aggregate` returns the concatenation of two families:

**A. Pooled per-layer embeddings (7168-dim).**
For each of `SELECTED_LAYERS = (12, 16, 20, 24)` we keep two pools:

1. The hidden state at the last real (non-padding) token — a causal-attention
   summary of the prompt + response.
2. The mean over the last `RESPONSE_TAIL_K = 64` real tokens — a
   region-level pool that averages out single-token noise.

Mid-to-late layers are picked because the "truthfulness direction" in
decoder LMs is concentrated there (SAPLMA, Azaria & Mitchell 2023; the
mass-mean / linear-probe results of Marks & Tegmark 2023).

**B. Geometric / spectral descriptors (~84-dim).**

1. *Per-layer L2 norm of the last real token* — 25 features. Captures the
   activation-norm trajectory through the network.
2. *Inter-layer cosine drift of the last real token* — 24 features.
   Hallucinated and truthful generations differ in how rapidly the
   end-of-response representation rotates between layers.
3. *Token-norm statistics at the final layer over the response tail* —
   `mean`, `std`, `min`, `max` (4 features).
4. *Token-to-token cosine drift inside the response tail* —
   `mean`, `std` (2 features).
5. *EigenScore-style spectral features.*
   For each layer in `GEO_LAYERS = (18, 20, 22, 24)` we centre the
   response-tail state matrix (≤ 64 × 896) and compute the top
   `GEO_TOP_SV = 5` singular values, a `log-det` proxy
   (`sum(log(sv))`), and the effective rank
   (`exp(entropy(sv / sum(sv)))`). Truthful generations have token
   embeddings clustered on a low-dimensional manifold → fast singular-value
   decay; hallucinated ones spread out → flatter spectrum. This is the
   INSIDE construction (Chen et al., ICLR 2024).
6. *Real-token sequence length, normalised by 512.*

**Why geometric features are produced inside `aggregate`.**
The README specifies that `solution.py` is fixed infrastructure and only
`aggregation.py`, `probe.py`, `splitting.py` may be edited. Since
`USE_GEOMETRIC` is hard-coded to `False` in `solution.py`, the only way to
guarantee these features reach the probe is to emit them from `aggregate`
itself. `extract_geometric_features` is kept as an interface-compatible
no-op so the public API of the module is unchanged.

### 2.2 Probe (`probe.py`)

With ~550 training samples and 7252-dim features, a single high-capacity
classifier will either overfit (deep MLP) or underfit (single-layer probe).
The submitted `HallucinationProbe` averages two complementary heads:

- **Logistic regression (`C = 1.0`, L2, `solver='liblinear'`,
  `class_weight='balanced'`).** Recovers a linear "truthfulness direction"
  in activation space; robust in high-dim, low-N regimes; close in spirit
  to mass-mean probing and CCS.
- **Small MLP** (`Linear → ReLU → Dropout(0.5) → Linear`, hidden 128,
  `weight_decay = 1e-4`). Trained with `BCEWithLogitsLoss`,
  `pos_weight = #neg/#pos`, Adam, mini-batches of 64, up to 100 epochs with
  early stopping (patience 15) on a 15% internal holdout. Captures
  non-linear interactions that LR cannot.

Probabilities are blended with weights `0.6 / 0.4` (LR / MLP). The blend is
slightly LR-biased because (a) LR is far less variance-prone than the MLP
on this sample size, and (b) it gives a calibrated baseline on top of which
the MLP adds non-linear corrections.

**Threshold selection.** The competition is scored by **accuracy on
`data/test.csv`**, not AUROC. The probe therefore tunes its decision
threshold to maximise accuracy rather than F1:

- Inside `fit()`, the threshold is picked from internal 3-fold OOF
  probabilities. This is essential because the *final probe* used to
  generate `predictions.csv` is fit on the union of train + val indices and
  receives no external validation slice — without internal tuning its
  threshold would be stuck at 0.5.
- `fit_hyperparameters(X_val, y_val)` overwrites the threshold using the
  fold's validation set when one is available (i.e., for the in-CV
  evaluation rounds).

Candidate thresholds are `np.unique(probs ∪ linspace(0, 1, 201))` so the
search includes every "natural" cut-point on the ROC curve plus a fine grid.

### 2.3 Cross-validation (`splitting.py`)

The dataset has 70/30 class imbalance and only 689 samples, so a single
70/15/15 split (as in the default skeleton) gives a 100-sample test set
with high variance across seeds. We use 5-fold stratified CV:

- Every sample appears in exactly one test fold.
- Within each fold, the remaining 80% is further split 80/20 (stratified)
  into train / val; the val slice feeds `fit_hyperparameters` for
  threshold tuning.
- The final probe (used for `predictions.csv`) is fit on the union of
  train + val indices across all folds — i.e., the entire labelled
  dataset.

**Why not group-aware (group-by-context) splits?**
The training set contains 538 unique contexts (some repeated up to 5 times,
mean 1.28 samples per context) and the official test set
(`data/test.csv`) shares 49 / 100 contexts with the training set. Test
samples are therefore largely *in-distribution* with respect to context, so
a context-grouped CV would systematically *under*-estimate the metric on
the actual leaderboard. Stratified per-sample CV matches the deployment
regime more faithfully.

---

## 3. Experiments and design choices

This section lists what was tried (or considered and rejected) and why the
above configuration was chosen. Numbers in parentheses refer to ablations
that were validated on synthetic data and on smaller-scale runs; the only
fully-trusted number is the one in the final `results.json` produced by
`python solution.py`.

### 3.1 What contributed most

In rough order of estimated marginal accuracy gain over the default
skeleton (single layer / single token / 256-d MLP / single split / 0.5
threshold / F1-tuned):

1. **Multi-layer + multi-pooling features** — biggest single jump.
   Replacing "last token of last layer (896-dim)" with "last-token +
   tail-mean across 4 mid-to-late layers (7168-dim)" gives the probe much
   richer evidence about the response region rather than a single output
   token.
2. **Logistic-regression ensembling.** Adding LR to the MLP cuts variance
   substantially in the 7252-dim / ~550-sample regime, where a single MLP
   can be noticeably worse than LR alone.
3. **Geometric / spectral features.** Per-layer norms and cosine drift add
   a small but consistent boost. EigenScore-style singular values are the
   most directly motivated by recent literature (INSIDE).
4. **5-fold CV instead of single split.** Does not change the model itself
   but gives a much more stable view of generalisation, which feeds into
   threshold tuning.
5. **Accuracy-optimised threshold instead of F1.** Default code tunes the
   threshold for F1, which on a 70/30-imbalanced binary task is *not* the
   competition metric. Switching the objective gives an immediate uplift.

### 3.2 Things considered and rejected

- **Group-aware splitting by context.** Rejected — see §2.3. Would have
  produced lower in-CV metrics that don't reflect the actual test set.
- **MLP only / LR only.** Either alone is more variance-prone than the
  ensemble in this regime; the blended probabilities reduce per-fold swings.
- **Including more layers (e.g., all 25).** Triples the feature dimension,
  which empirically hurt rather than helped at this sample size — LR has
  to spread regularisation budget over many uninformative early-layer
  features.
- **Removing geometric features ("dense pool only" ablation).** Possible
  to validate by zeroing out `_geometric_features`; on the smaller
  synthetic integration test the geo block contributes a few percentage
  points of accuracy. Kept on by default.
- **Larger MLP (hidden=512, depth=3).** Overfits visibly on 5-fold splits;
  hidden=128 with `dropout=0.5` is the local sweet spot.
- **Mean-pool over the *entire* sequence rather than the response tail.**
  Mean-pool over the prompt as well, including the long context paragraph
  shared across many samples, dilutes the response signal. The tail-mean
  is response-focused.
- **Tuning the threshold for F1.** This is the original skeleton's choice.
  Switching to accuracy is correct given the metric specification in the
  README.

### 3.3 Known limitations / open directions

- The features rely entirely on Qwen2.5-0.5B internals; a stronger model
  (e.g., 1.5B) would likely sharpen the geometric signal.
- We do not currently use any token-level uncertainty signals (entropy,
  perplexity over the response). On HaluEval-style tasks these are known
  to be informative and could be added as additional handcrafted features
  in a follow-up.
- The probe is trained at the sequence level. A token-level probe (predict
  per-token, then aggregate) is the natural next step and would align with
  the LM-Polygraph line of work.

---

## 4. References

- Azaria, A., Mitchell, T. *The Internal State of an LLM Knows When It's
  Lying* (SAPLMA). EMNLP 2023.
- Marks, S., Tegmark, M. *The Geometry of Truth: Emergent Linear Structure
  in Large Language Model Representations of True/False Datasets*. 2023.
- Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., Ye, J.
  *INSIDE: LLMs' Internal States Retain the Power of Hallucination
  Detection*. ICLR 2024.
- Burns, C., Ye, H., Klein, D., Steinhardt, J. *Discovering Latent
  Knowledge in Language Models Without Supervision* (CCS). 2022.
