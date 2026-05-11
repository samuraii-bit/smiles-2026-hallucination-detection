"""
probe.py — Hallucination probe (ensemble of L2 logistic regression + MLP).

Design notes
------------
With ~689 training samples and feature dimensionality up to ~7250, a single
high-capacity classifier would either overfit (deep MLP) or underfit (linear
on a single layer).  The ensemble below averages two complementary probes:

  * ``LogisticRegression`` with strong L2 — robust in the high-dim, low-N
    regime; recovers a single linear "truthfulness direction" in activation
    space (mass-mean / Marks & Tegmark, 2023; CCS, Burns et al., 2022).

  * Small MLP with dropout and early stopping — captures non-linear
    interactions between layers and geometric features that a linear model
    cannot.  Trained with ``BCEWithLogitsLoss`` and ``pos_weight`` to handle
    the 70/30 class imbalance.

Probabilities are averaged.  The decision threshold is chosen to **maximise
accuracy** (the official competition metric, not F1) — first via an internal
stratified-CV OOF estimate inside ``fit``, and refined by
``fit_hyperparameters`` when an external validation set is provided.

All four public methods (``fit``, ``fit_hyperparameters``, ``predict``,
``predict_proba``) keep their original signatures so the evaluation
infrastructure in ``evaluate.py`` calls them transparently.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Hyper-parameters (kept at module level so they show up in any logs).
# ---------------------------------------------------------------------------
LR_C = 1.0                      # inverse regularisation strength for LR
LR_MAX_ITER = 2000

MLP_HIDDEN = 128
MLP_DROPOUT = 0.5
MLP_EPOCHS = 100                # early stopping kicks in well before this in practice
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH = 64
MLP_PATIENCE = 15               # early-stopping patience on internal val loss
MLP_INNER_VAL_FRAC = 0.15       # fraction held out inside fit() for early stopping

ENSEMBLE_WEIGHT_LR = 0.6        # weight on LR probabilities
ENSEMBLE_WEIGHT_MLP = 1.0 - ENSEMBLE_WEIGHT_LR

THRESHOLD_CV_FOLDS = 3          # internal CV folds for OOF threshold tuning
SEED = 42


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Search the threshold that maximises accuracy on (y_true, y_prob).

    Candidates are the unique probabilities seen, plus a fine 0..1 grid.
    """
    cand = np.unique(np.concatenate([y_prob, np.linspace(0.0, 1.0, 201)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (y_prob >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# MLP definition
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    """Tiny MLP: input -> Linear -> ReLU -> Dropout -> Linear -> logit."""

    def __init__(self, input_dim: int, hidden: int = MLP_HIDDEN,
                 dropout: float = MLP_DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# HallucinationProbe — public API
# ---------------------------------------------------------------------------
class HallucinationProbe(nn.Module):
    """Ensemble probe: scaled features -> (LR + MLP) -> averaged probability.

    Subclasses ``nn.Module`` to satisfy the original API contract; the actual
    learning happens inside ``fit`` (sklearn LR + custom torch training loop
    for the MLP).  ``forward`` is provided for completeness only.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._lr: LogisticRegression | None = None
        self._mlp: _MLP | None = None
        self._mlp_state: dict | None = None    # best-by-val MLP weights
        self._threshold: float = 0.5
        self._input_dim: int | None = None

    # ------------------------------------------------------------------
    # Internal: train a single MLP with early stopping on a holdout
    # ------------------------------------------------------------------
    def _train_mlp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> tuple[_MLP, dict]:
        """Train one MLP with early stopping; return model and best state dict."""
        _set_seed(seed)
        n = len(y)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = max(1, int(MLP_INNER_VAL_FRAC * n))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        X_tr = torch.from_numpy(X[tr_idx]).float()
        y_tr = torch.from_numpy(y[tr_idx].astype(np.float32))
        X_va = torch.from_numpy(X[val_idx]).float()
        y_va = torch.from_numpy(y[val_idx].astype(np.float32))

        # Class imbalance handled via pos_weight = #neg / #pos on the train slice.
        n_pos = int(y_tr.sum().item())
        n_neg = int(len(y_tr) - n_pos)
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        )

        model = _MLP(X.shape[1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience = MLP_PATIENCE

        for _epoch in range(MLP_EPOCHS):
            # mini-batch training
            model.train()
            perm = torch.randperm(len(X_tr))
            for start in range(0, len(X_tr), MLP_BATCH):
                bidx = perm[start:start + MLP_BATCH]
                optimizer.zero_grad()
                logits = model(X_tr[bidx])
                loss = criterion(logits, y_tr[bidx])
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_va)
                val_loss = criterion(val_logits, y_va).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = MLP_PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        return model, best_state

    # ------------------------------------------------------------------
    # Internal: compute OOF probabilities for threshold tuning inside fit()
    # ------------------------------------------------------------------
    def _oof_probabilities(
        self, X_scaled: np.ndarray, y: np.ndarray, seed: int
    ) -> np.ndarray:
        """Return out-of-fold ensemble probabilities for the positive class."""
        n = len(y)
        oof = np.zeros(n, dtype=np.float32)

        skf = StratifiedKFold(
            n_splits=THRESHOLD_CV_FOLDS, shuffle=True, random_state=seed
        )
        for fold_id, (tr, va) in enumerate(skf.split(X_scaled, y)):
            # LR
            lr = LogisticRegression(
                C=LR_C,
                solver="liblinear",                   # default penalty is L2
                max_iter=LR_MAX_ITER,
                class_weight="balanced",
                random_state=seed + fold_id,
            )
            lr.fit(X_scaled[tr], y[tr])
            p_lr = lr.predict_proba(X_scaled[va])[:, 1]

            # MLP
            mlp, _ = self._train_mlp(X_scaled[tr], y[tr], seed=seed + fold_id)
            with torch.no_grad():
                logits = mlp(torch.from_numpy(X_scaled[va]).float())
                p_mlp = torch.sigmoid(logits).numpy()

            oof[va] = ENSEMBLE_WEIGHT_LR * p_lr + ENSEMBLE_WEIGHT_MLP * p_mlp

        return oof

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler, LR, and MLP on ``(X, y)``; tune threshold via internal CV."""
        _set_seed(SEED)
        y = np.asarray(y).astype(int)

        # 1. Standardise features.
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)
        self._input_dim = X_scaled.shape[1]

        # 2. Fit LR on full training data.
        self._lr = LogisticRegression(
            C=LR_C,
            solver="liblinear",                   # default penalty is L2
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
            random_state=SEED,
        )
        self._lr.fit(X_scaled, y)

        # 3. Fit MLP on full training data (with internal holdout for ES).
        self._mlp, self._mlp_state = self._train_mlp(X_scaled, y, seed=SEED)

        # 4. Internal-CV threshold (used when fit_hyperparameters is not called).
        try:
            oof = self._oof_probabilities(X_scaled, y, seed=SEED)
            self._threshold = _best_threshold(y, oof)
        except Exception:
            # If anything goes wrong, fall back to 0.5.
            self._threshold = 0.5

        return self

    # ------------------------------------------------------------------
    # Public: fit_hyperparameters — refine threshold on external val set
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Refine the decision threshold to maximise accuracy on a val set."""
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _best_threshold(np.asarray(y_val).astype(int), probs)
        return self

    # ------------------------------------------------------------------
    # Public: predict / predict_proba
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._lr is None or self._mlp is None:
            raise RuntimeError("Probe must be fit() before predict_proba().")

        X_scaled = self._scaler.transform(X).astype(np.float32)

        # LR probabilities.
        p_lr = self._lr.predict_proba(X_scaled)[:, 1]

        # MLP probabilities.
        with torch.no_grad():
            logits = self._mlp(torch.from_numpy(X_scaled).float())
            p_mlp = torch.sigmoid(logits).numpy()

        prob_pos = (
            ENSEMBLE_WEIGHT_LR * p_lr + ENSEMBLE_WEIGHT_MLP * p_mlp
        ).astype(np.float64)
        prob_pos = np.clip(prob_pos, 0.0, 1.0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    # ------------------------------------------------------------------
    # nn.Module compliance — never used by evaluate.py but kept for the API.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mlp is None:
            raise RuntimeError("Probe has not been fit() yet.")
        return self._mlp(x)
