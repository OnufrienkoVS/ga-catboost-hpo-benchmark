"""Unit tests for the one-time split validation performed by the loader."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import _validate_splits


def make_valid_splits() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = pd.DataFrame({"feature": range(12)})
    train = np.arange(9, dtype=np.int64)
    test = np.arange(9, 12, dtype=np.int64)

    # Three folds; each training row is validation data exactly once.
    valid_parts = (
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
    )
    splits: dict[str, np.ndarray] = {
        "train_indices": train,
        "test_indices": test,
    }
    for fold, valid in enumerate(valid_parts, start=1):
        fold_train = np.setdiff1d(train, valid)
        splits[f"cv_fold_{fold}_train_indices"] = fold_train
        splits[f"cv_fold_{fold}_valid_indices"] = valid
    return frame, splits


def test_validate_splits_accepts_valid_partition() -> None:
    frame, splits = make_valid_splits()
    _validate_splits(frame, splits)


def test_validate_splits_rejects_duplicate_train_indices() -> None:
    frame, splits = make_valid_splits()
    splits["train_indices"] = np.append(splits["train_indices"], 0)

    with pytest.raises(ValueError, match="duplicate row positions"):
        _validate_splits(frame, splits)


def test_validate_splits_rejects_duplicate_cv_indices() -> None:
    frame, splits = make_valid_splits()
    splits["cv_fold_1_valid_indices"] = np.append(
        splits["cv_fold_1_valid_indices"], 0
    )

    with pytest.raises(ValueError, match="duplicate row positions"):
        _validate_splits(frame, splits)


def test_validate_splits_rejects_train_test_overlap() -> None:
    frame, splits = make_valid_splits()
    splits["test_indices"] = np.array([0, 10, 11], dtype=np.int64)

    with pytest.raises(ValueError, match="overlap"):
        _validate_splits(frame, splits)
