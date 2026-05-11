"""
probe.py — Layer-ensemble hallucination probe (v2).

Improvements over v1
--------------------
v1 reduced train AUROC 100% -> 96% and lifted test AUROC ~72% -> ~75% but
remained noisy across folds (test AUROC range 70.7..78.3 across 5 folds).
v2 targets variance reduction:

  * 12 layers in the slice ensemble (was 9).
  * Both L2 and L1 logistic regression per (layer, pool) slice — two
    parallel slice streams averaged separately, then blended.  L1 sparsity
    often improves OOD-ish generalisation on Qwen hidden states.
  * MLP is *bagged* across 3 seeds; predictions averaged.
  * Component probabilities are blended by a **logistic-regression
    meta-stacker** trained on out-of-fold component probabilities, instead
    of a coarse grid search.
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

from aggregation import SLICE_LAYOUT

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
# Per-slice L2 LR.
SLICE_L2_C = 0.03
# Per-slice L1 LR (sparser, complements L2).
SLICE_L1_C = 0.10
SLICE_LR_MAX_ITER = 2000

# Global PCA + LR/MLP path.
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
MLP_BAG_SEEDS = (101, 202, 303)

# Gradient boosting on geometric features.
GB_N_EST = 200
GB_LR = 0.05
GB_MAX_DEPTH = 2

# Meta-stacker.
META_C = 1.0
META_MAX_ITER = 2000

# Component keys (order is fixed — meta-stacker uses this order).
COMPONENT_KEYS: tuple[str, ...] = (
    "slice_l2_mean",
    "slice_l1_mean",
    "global_lr",
    "mlp",
    "gb_geo",
)

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
# Tiny MLP
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
# Slice-LR helpers
# ---------------------------------------------------------------------------
def _fit_slice_lr(
    X_slice: np.ndarray, y: np.ndarray, seed: int, penalty: str, C: float
) -> tuple:
    """Standard-scale + fit LR with the requested penalty.  Liblinear handles both."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_slice).astype(np.float32)
    lr = LogisticRegression(
        C=C,
        penalty=penalty,
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
# HallucinationProbe — public API
# ---------------------------------------------------------------------------
class HallucinationProbe(nn.Module):
    """v2 layer-ensemble probe."""

    def __init__(self) -> None:
        super().__init__()
        # Per-slice models for L2 and L1, parallel to SLICE_LAYOUT.
        self._slice_l2: list[tuple] = []
        self._slice_l1: list[tuple] = []

        # Global PCA pipeline.
        self._dense_scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._global_lr: LogisticRegression | None = None
        self._mlps: list[_MLP] = []

        # Geometric pipeline.
        self._geo_scaler: StandardScaler | None = None
        self._gb_geo: GradientBoostingClassifier | None = None
        self._has_geo: bool = False

        # Meta-stacker.
        self._meta: LogisticRegression | None = None
        self._threshold: float = 0.5
        self._input_dim: int | None = None

    # ------------------------------------------------------------------
    # Component fits
    # ------------------------------------------------------------------
    def _fit_all_components(self, X: np.ndarray, y: np.ndarray, seed: int) -> None:
        # 1. Per-slice LRs (L2 + L1).
        self._slice_l2 = []
        self._slice_l1 = []
        for i, sl in enumerate(SLICE_LAYOUT):
            X_sl = X[:, sl["start"]:sl["end"]]
            self._slice_l2.append(
                _fit_slice_lr(X_sl, y, seed + i, penalty="l2", C=SLICE_L2_C)
            )
            self._slice_l1.append(
                _fit_slice_lr(X_sl, y, seed + 10_000 + i, penalty="l1", C=SLICE_L1_C)
            )

        # 2. Global PCA → LR + bagged MLPs.
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

        self._mlps = [_train_mlp(Xp, y, seed=seed + s) for s in MLP_BAG_SEEDS]

        # 3. Geometric features.
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
        # Slice LRs.
        l2_probs, l1_probs = [], []
        for sl, l2_m, l1_m in zip(SLICE_LAYOUT, self._slice_l2, self._slice_l1):
            X_sl = X[:, sl["start"]:sl["end"]]
            l2_probs.append(_slice_lr_proba(*l2_m, X_sl))
            l1_probs.append(_slice_lr_proba(*l1_m, X_sl))
        slice_l2_mean = np.mean(np.stack(l2_probs, 0), 0)
        slice_l1_mean = np.mean(np.stack(l1_probs, 0), 0)

        # Global PCA.
        dense_end = SLICE_LAYOUT[-1]["end"]
        Xd = self._dense_scaler.transform(X[:, :dense_end]).astype(np.float32)
        Xp = self._pca.transform(Xd).astype(np.float32)
        global_lr_p = self._global_lr.predict_proba(Xp)[:, 1]

        with torch.no_grad():
            Xp_t = torch.from_numpy(Xp).float()
            mlp_probs = [torch.sigmoid(m(Xp_t)).numpy() for m in self._mlps]
        mlp_p = np.mean(np.stack(mlp_probs, 0), 0)

        # Geometric GB.
        if self._has_geo:
            Xg = self._geo_scaler.transform(X[:, dense_end:]).astype(np.float32)
            gb_p = self._gb_geo.predict_proba(Xg)[:, 1]
        else:
            gb_p = np.full_like(slice_l2_mean, 0.5)

        return {
            "slice_l2_mean": slice_l2_mean.astype(np.float64),
            "slice_l1_mean": slice_l1_mean.astype(np.float64),
            "global_lr":     global_lr_p.astype(np.float64),
            "mlp":           mlp_p.astype(np.float64),
            "gb_geo":        gb_p.astype(np.float64),
        }

    # ------------------------------------------------------------------
    # OOF meta-stacker
    # ------------------------------------------------------------------
    def _oof_components(self, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
        n = len(y)
        bag = {k: np.zeros(n, dtype=np.float64) for k in COMPONENT_KEYS}
        skf = StratifiedKFold(
            n_splits=THRESHOLD_CV_FOLDS, shuffle=True, random_state=seed
        )
        for fid, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
            inner = HallucinationProbe()
            inner._fit_all_components(X[tr], y[tr], seed=seed + 100 * (fid + 1))
            parts = inner._component_probas(X[va])
            for k in COMPONENT_KEYS:
                bag[k][va] = parts[k]
        return bag

    def _fit_meta(self, oof: dict, y: np.ndarray) -> None:
        """Fit a logistic-regression meta-stacker on OOF component probabilities."""
        X_meta = np.stack([oof[k] for k in COMPONENT_KEYS], axis=1)
        self._meta = LogisticRegression(
            C=META_C,
            solver="lbfgs",
            max_iter=META_MAX_ITER,
            class_weight="balanced",
            random_state=SEED,
        )
        self._meta.fit(X_meta, y)
        p_meta = self._meta.predict_proba(X_meta)[:, 1]
        self._threshold = _best_threshold(y, p_meta)

    # ------------------------------------------------------------------
    # Public: fit / fit_hyperparameters / predict / predict_proba
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        _set_seed(SEED)
        y = np.asarray(y).astype(int)
        self._input_dim = X.shape[1]

        # 1. Fit all components on full training data.
        self._fit_all_components(X, y, seed=SEED)

        # 2. OOF predictions -> meta-stacker + threshold.
        try:
            oof = self._oof_components(X, y, seed=SEED)
            self._fit_meta(oof, y)
        except Exception:
            # Fallback to simple mean blend.
            self._meta = None
            self._threshold = 0.5
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _best_threshold(np.asarray(y_val).astype(int), probs)
        return self

    def _blended_proba(self, X: np.ndarray) -> np.ndarray:
        parts = self._component_probas(X)
        if self._meta is not None:
            X_meta = np.stack([parts[k] for k in COMPONENT_KEYS], axis=1)
            p = self._meta.predict_proba(X_meta)[:, 1]
        else:
            p = np.mean(np.stack([parts[k] for k in COMPONENT_KEYS], 0), 0)
        return np.clip(p, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._blended_proba(X) >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._slice_l2:
            raise RuntimeError("Probe must be fit() before predict_proba().")
        p = self._blended_proba(X)
        return np.stack([1.0 - p, p], axis=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._mlps:
            raise RuntimeError("Probe has not been fit() yet.")
        return self._mlps[0](x)
