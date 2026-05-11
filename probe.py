"""
probe.py — Layer-ensemble hallucination probe.

Why an ensemble of per-layer probes?
------------------------------------
The previous single-classifier-on-7252-features design produced:
    train AUROC 100%   test AUROC ~72%
i.e. catastrophic over-fitting (16:1 feature-to-sample ratio).

Each individual 896-dim per-layer slice produced by ``aggregation.py`` is, on
its own, a healthy regime (≈2:1 ratio against ~440 train samples), and the
"truthfulness direction" of Marks & Tegmark (2023) and the SAPLMA findings of
Azaria & Mitchell (2023) say that *every* mid-to-late layer carries a useful
linear hallucination signal.

So we train ONE strongly-regularised logistic regression per (layer, pool)
slice and average their probabilities, plus a small global classifier on a
PCA-compressed view to pick up cross-layer non-linearities, plus a tiny GB
model on the geometric features.  Probabilities are blended with weights tuned
on internal stratified CV.

All four public methods (``fit``, ``fit_hyperparameters``, ``predict``,
``predict_proba``) keep their original signatures.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from aggregation import GEO_START, SLICE_LAYOUT

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
# Per-slice LR: strong L2 because each slice has 896 features but only ~440
# training samples.  Liblinear is fastest for small N, high-dim, L2.
SLICE_LR_C = 0.05
SLICE_LR_MAX_ITER = 2000

# Global PCA + LR/MLP path — captures cross-layer interactions.
PCA_COMPONENTS = 64

GLOBAL_LR_C = 0.5
GLOBAL_LR_MAX_ITER = 2000

# Tiny MLP on PCA-reduced features.
MLP_HIDDEN = 64
MLP_DROPOUT = 0.4
MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-3
MLP_BATCH = 32
MLP_PATIENCE = 20
MLP_INNER_VAL_FRAC = 0.15

# Gradient boosting on geometric features.
GB_N_EST = 200
GB_LR = 0.05
GB_MAX_DEPTH = 2

# Ensemble blend weights (sum to ~1).  Tuned via OOF below.
DEFAULT_WEIGHTS = {
    "slice_mean": 0.55,   # mean of per-slice LR probabilities
    "global_lr": 0.20,    # global LR on PCA features
    "mlp":       0.15,    # small MLP on PCA features
    "gb_geo":    0.10,    # GB on geometric features
}

THRESHOLD_CV_FOLDS = 5
SEED = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    cand = np.unique(np.concatenate([y_prob, np.linspace(0.01, 0.99, 199)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (y_prob >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Tiny MLP on PCA features
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = MLP_HIDDEN,
                 dropout: float = MLP_DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Internal: train a slice-LR
# ---------------------------------------------------------------------------
def _fit_slice_lr(X_slice: np.ndarray, y: np.ndarray, seed: int) -> tuple:
    """Standard-scale a slice, fit LR with strong L2, return (scaler, lr)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_slice).astype(np.float32)
    lr = LogisticRegression(
        C=SLICE_LR_C,
        solver="liblinear",
        max_iter=SLICE_LR_MAX_ITER,
        class_weight="balanced",
        random_state=seed,
    )
    lr.fit(X_s, y)
    return scaler, lr


def _slice_lr_proba(scaler, lr, X_slice: np.ndarray) -> np.ndarray:
    return lr.predict_proba(scaler.transform(X_slice).astype(np.float32))[:, 1]


# ---------------------------------------------------------------------------
# Internal: train MLP
# ---------------------------------------------------------------------------
def _train_mlp(X: np.ndarray, y: np.ndarray, seed: int) -> _MLP:
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

    n_pos = int(y_tr.sum().item())
    n_neg = int(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    model = _MLP(X.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = MLP_PATIENCE

    for _epoch in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), MLP_BATCH):
            bidx = perm[start:start + MLP_BATCH]
            optimizer.zero_grad()
            logits = model(X_tr[bidx])
            loss = criterion(logits, y_tr[bidx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
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
    return model


# ---------------------------------------------------------------------------
# HallucinationProbe — public API
# ---------------------------------------------------------------------------
class HallucinationProbe(nn.Module):
    """Layer-ensemble probe.

    Components:
      * per-(layer,pool) logistic regressions on the corresponding 896-dim
        slice (strong L2);
      * global LR on PCA-reduced (concat of all slices) features;
      * tiny MLP on the same PCA features;
      * gradient boosting on geometric features.
    """

    def __init__(self) -> None:
        super().__init__()
        # Per-slice models: list of (scaler, lr) parallel to SLICE_LAYOUT.
        self._slice_models: list[tuple] = []

        # Global pipeline: scaler over dense block, PCA, then LR + MLP.
        self._dense_scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._global_lr: LogisticRegression | None = None
        self._mlp: _MLP | None = None

        # Geometric pipeline.
        self._geo_scaler: StandardScaler | None = None
        self._gb_geo: GradientBoostingClassifier | None = None

        # Blend weights and final threshold.
        self._weights: dict = dict(DEFAULT_WEIGHTS)
        self._threshold: float = 0.5
        self._input_dim: int | None = None
        self._has_geo: bool = False

    # ------------------------------------------------------------------
    # Component fit/predict primitives
    # ------------------------------------------------------------------
    def _fit_all_components(self, X: np.ndarray, y: np.ndarray, seed: int) -> None:
        # 1. Per-slice LRs.
        self._slice_models = []
        for i, sl in enumerate(SLICE_LAYOUT):
            X_sl = X[:, sl["start"]:sl["end"]]
            self._slice_models.append(_fit_slice_lr(X_sl, y, seed=seed + i))

        # 2. Global PCA on full dense block, then LR + MLP on top.
        dense_end = SLICE_LAYOUT[-1]["end"]
        X_dense = X[:, :dense_end]
        self._dense_scaler = StandardScaler()
        Xd = self._dense_scaler.fit_transform(X_dense).astype(np.float32)
        n_comp = min(PCA_COMPONENTS, Xd.shape[0] - 1, Xd.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=seed)
        Xp = self._pca.fit_transform(Xd).astype(np.float32)

        self._global_lr = LogisticRegression(
            C=GLOBAL_LR_C,
            solver="lbfgs",
            max_iter=GLOBAL_LR_MAX_ITER,
            class_weight="balanced",
            random_state=seed,
        )
        self._global_lr.fit(Xp, y)

        self._mlp = _train_mlp(Xp, y, seed=seed)

        # 3. Geometric features (if any).
        if X.shape[1] > dense_end:
            X_geo = X[:, dense_end:]
            self._geo_scaler = StandardScaler()
            Xg = self._geo_scaler.fit_transform(X_geo).astype(np.float32)
            self._gb_geo = GradientBoostingClassifier(
                n_estimators=GB_N_EST,
                learning_rate=GB_LR,
                max_depth=GB_MAX_DEPTH,
                random_state=seed,
            )
            self._gb_geo.fit(Xg, y)
            self._has_geo = True
        else:
            self._has_geo = False

    def _component_probas(self, X: np.ndarray) -> dict:
        """Return per-component positive-class probabilities."""
        # 1. Slice-LR mean.
        slice_probs = []
        for sl, (scaler, lr) in zip(SLICE_LAYOUT, self._slice_models):
            X_sl = X[:, sl["start"]:sl["end"]]
            slice_probs.append(_slice_lr_proba(scaler, lr, X_sl))
        slice_mean = np.mean(np.stack(slice_probs, axis=0), axis=0)

        # 2/3. Global LR + MLP on PCA features.
        dense_end = SLICE_LAYOUT[-1]["end"]
        Xd = self._dense_scaler.transform(X[:, :dense_end]).astype(np.float32)
        Xp = self._pca.transform(Xd).astype(np.float32)
        global_lr_p = self._global_lr.predict_proba(Xp)[:, 1]
        with torch.no_grad():
            mlp_p = torch.sigmoid(self._mlp(torch.from_numpy(Xp))).numpy()

        # 4. GB on geo.
        if self._has_geo:
            Xg = self._geo_scaler.transform(X[:, dense_end:]).astype(np.float32)
            gb_p = self._gb_geo.predict_proba(Xg)[:, 1]
        else:
            gb_p = np.full_like(slice_mean, 0.5)

        return {
            "slice_mean": slice_mean.astype(np.float64),
            "global_lr":  global_lr_p.astype(np.float64),
            "mlp":        mlp_p.astype(np.float64),
            "gb_geo":     gb_p.astype(np.float64),
        }

    # ------------------------------------------------------------------
    # OOF blend weight + threshold tuning
    # ------------------------------------------------------------------
    def _oof_components(self, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
        """Out-of-fold component probabilities for unbiased blend tuning."""
        n = len(y)
        bag = {k: np.zeros(n, dtype=np.float64) for k in DEFAULT_WEIGHTS}
        skf = StratifiedKFold(
            n_splits=THRESHOLD_CV_FOLDS, shuffle=True, random_state=seed
        )
        for fid, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
            inner = HallucinationProbe()
            inner._fit_all_components(X[tr], y[tr], seed=seed + 100 * (fid + 1))
            parts = inner._component_probas(X[va])
            for k, v in parts.items():
                bag[k][va] = v
        return bag

    def _tune_blend(self, oof: dict, y: np.ndarray) -> None:
        """Grid-search the 4-way blend (weights >= 0, sum = 1) maximising acc."""
        keys = list(DEFAULT_WEIGHTS.keys())
        grid = np.linspace(0.0, 1.0, 6)
        best_acc, best_w, best_t = -1.0, dict(DEFAULT_WEIGHTS), 0.5
        for a in grid:
            for b in grid:
                for c in grid:
                    d = 1.0 - a - b - c
                    if d < -1e-9 or d > 1.0 + 1e-9:
                        continue
                    w = {"slice_mean": a, "global_lr": b, "mlp": c, "gb_geo": max(d, 0.0)}
                    p = sum(w[k] * oof[k] for k in keys)
                    t = _best_threshold(y, p)
                    acc = accuracy_score(y, (p >= t).astype(int))
                    if acc > best_acc:
                        best_acc, best_w, best_t = acc, w, t
        self._weights = best_w
        self._threshold = best_t

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        _set_seed(SEED)
        y = np.asarray(y).astype(int)
        self._input_dim = X.shape[1]

        # 1. Fit all components on the full training data.
        self._fit_all_components(X, y, seed=SEED)

        # 2. Tune blend weights + threshold on OOF predictions.
        try:
            oof = self._oof_components(X, y, seed=SEED)
            self._tune_blend(oof, y)
        except Exception:
            self._weights = dict(DEFAULT_WEIGHTS)
            self._threshold = 0.5
        return self

    # ------------------------------------------------------------------
    # Public: fit_hyperparameters — refine threshold on external val set
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _best_threshold(np.asarray(y_val).astype(int), probs)
        return self

    # ------------------------------------------------------------------
    # Public: predict / predict_proba
    # ------------------------------------------------------------------
    def _blended_proba(self, X: np.ndarray) -> np.ndarray:
        parts = self._component_probas(X)
        w = self._weights
        p = (
            w["slice_mean"] * parts["slice_mean"]
            + w["global_lr"] * parts["global_lr"]
            + w["mlp"] * parts["mlp"]
            + w["gb_geo"] * parts["gb_geo"]
        )
        return np.clip(p, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._blended_proba(X) >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._slice_models:
            raise RuntimeError("Probe must be fit() before predict_proba().")
        p = self._blended_proba(X)
        return np.stack([1.0 - p, p], axis=1)

    # ------------------------------------------------------------------
    # nn.Module compliance — kept for the original API contract.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mlp is None:
            raise RuntimeError("Probe has not been fit() yet.")
        return self._mlp(x)
