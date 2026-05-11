"""
splitting.py — Stratified k-fold split for the hallucination-detection task.

The dataset has ~70/30 class imbalance and only 689 samples, so a single
70/15/15 split (default) leaves test set with ~100 examples and probe
metrics swing wildly between random seeds.  We use 5-fold stratified CV:

  * Every sample appears in exactly one held-out test fold (5 folds).
  * Within each fold, the remaining 4/5 are split 80/20 into train / val.
    Validation is used by ``fit_hyperparameters`` to tune the decision
    threshold.

Why not group-aware splits?  The competition test set (``data/test.csv``)
shares 49 / 100 contexts with the training set, so an in-distribution
stratified split is the most representative evaluation regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Number of CV folds.  5 is a good compromise: each test fold holds ~138
# samples (low metric variance) and we still have ~440 training samples.
N_SPLITS = 5

# Fraction of (train + val) reserved for validation in each fold.
INNER_VAL_SIZE = 0.20

# Master seed used everywhere for reproducibility.
SEED = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,    # kept for API compatibility, unused
    val_size: float = 0.15,     # kept for API compatibility, unused
    random_state: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return a 5-fold stratified split as a list of (train, val, test).

    Args:
        y:            Label array of shape ``(N,)``, values in ``{0, 1}``.
        df:           Optional DataFrame (unused — kept for API parity).
        test_size:    Unused (kept for backward compatibility).
        val_size:     Unused (kept for backward compatibility).
        random_state: Seed forwarded to ``StratifiedKFold`` and the inner
                      train / val split.

    Returns:
        List of length ``N_SPLITS``.  Each element is
        ``(idx_train, idx_val, idx_test)`` of integer index arrays.
        ``idx_val`` is never ``None``: every fold has a validation slice so
        the probe's threshold is tuned consistently.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    all_idx = np.arange(n)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_id, (trainval_idx, test_idx) in enumerate(skf.split(all_idx, y)):
        # Inner stratified split for train / val.
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=INNER_VAL_SIZE,
            random_state=random_state + fold_id,   # decorrelate val between folds
            stratify=y[trainval_idx],
        )
        splits.append((train_idx, val_idx, test_idx))

    return splits
