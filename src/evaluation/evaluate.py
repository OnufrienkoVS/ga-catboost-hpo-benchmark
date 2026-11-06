# Единая процедура CV-оценки для всех методов оптимизации гиперпараметров.

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.catboost_model import (
    DEFAULT_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    create_catboost_classifier,
)

TARGET_COLUMN = "target"
ORIGINAL_INDEX_COLUMN = "original_row_index"
N_SPLITS = 3

LOGGER = logging.getLogger("catboost_evaluation")


@dataclass(frozen=True)
class EvaluationDataset:
    """
    Подготовленные данные, повторно используемые при каждой objective оценке для одного набора данных.

    Загрузка экземпляров осуществляется с помощью функции `load_evaluation_dataset`.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    cv_folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    categorical_features: tuple[str, ...]

    @property
    def n_splits(self) -> int:
        return len(self.cv_folds)


def _load_experiment_data(
    dataset_name: str,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Считывает подготовленные артефакты с диска."""
    dataset_dir = project_root / "data" / "processed" / dataset_name
    dataset_path = dataset_dir / "dataset.csv"
    metadata_path = dataset_dir / "metadata.json"
    splits_path = dataset_dir / "splits.npz"

    for path in (dataset_path, metadata_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                "Run src/data/prepare_datasets.py first."
            )

    frame = pd.read_csv(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(splits_path) as archive:
        splits = {key: archive[key].astype(np.int64) for key in archive.files}

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column is missing from {dataset_path}")
    if ORIGINAL_INDEX_COLUMN not in frame.columns:
        raise ValueError(
            f"'{ORIGINAL_INDEX_COLUMN}' column is missing from {dataset_path}"
        )

    expected_keys = {"train_indices", "test_indices"}
    expected_keys.update(
        {
            f"cv_fold_{fold}_{part}_indices"
            for fold in range(1, N_SPLITS + 1)
            for part in ("train", "valid")
        }
    )
    missing_keys = expected_keys.difference(splits)
    if missing_keys:
        raise ValueError(f"Missing keys in {splits_path}: {sorted(missing_keys)}")

    return frame, metadata, splits


def _validate_splits(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> None:
    """Проверяет данные на:
    - Правильность разбиения train/test;
    - Правильность CV разбиения.
    """
    n_rows = len(frame)
    train = splits["train_indices"]
    test = splits["test_indices"]

    for name, indices in splits.items():
        if indices.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if len(indices) == 0:
            raise ValueError(f"{name} is empty.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} contains duplicate row positions.")
        if indices.min() < 0 or indices.max() >= n_rows:
            raise ValueError(f"{name} contains out-of-range row positions.")

    if np.intersect1d(train, test).size:
        raise ValueError("Train and test indices overlap.")
    if len(train) + len(test) != n_rows:
        raise ValueError("Train and test indices do not cover the complete dataset exactly once.")
    if len(np.union1d(train, test)) != n_rows:
        raise ValueError("Train and test indices do not cover the complete dataset.")

    train_set = set(train.tolist())
    validation_parts: list[int] = []
    for fold in range(1, N_SPLITS + 1):
        fold_train = splits[f"cv_fold_{fold}_train_indices"]
        fold_valid = splits[f"cv_fold_{fold}_valid_indices"]
        fold_train_set = set(fold_train.tolist())
        fold_valid_set = set(fold_valid.tolist())

        if np.intersect1d(fold_train, fold_valid).size:
            raise ValueError(f"CV fold {fold}: train and validation overlap.")
        if not fold_train_set.issubset(train_set):
            raise ValueError(f"CV fold {fold}: training rows include non-train data.")
        if not fold_valid_set.issubset(train_set):
            raise ValueError(f"CV fold {fold}: validation rows include non-train data.")
        if fold_train_set | fold_valid_set != train_set:
            raise ValueError(
                f"CV fold {fold} does not partition the original training set."
            )

        validation_parts.extend(fold_valid.tolist())

    if sorted(validation_parts) != sorted(train.tolist()):
        raise ValueError(
            "CV validation folds do not cover each training observation exactly once."
        )


def load_evaluation_dataset(
    dataset_name: str,
    *,
    project_root: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> EvaluationDataset:
    """
    Загрузка и проверка одного датасета для многократной оценки моделей. 

    Необходимо выполнить это действие один раз для каждого датасета перед запуском оптимизатора,
    а затем передавайте полученный объект при каждом вызове функции `evaluate`.
    """
    log = logger or LOGGER
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )

    started = time.perf_counter()
    frame, metadata, splits = _load_experiment_data(dataset_name, root)
    _validate_splits(frame, splits)

    try:
        feature_columns = metadata["preparation"]["feature_columns"]
        categorical_features = metadata["preparation"]["categorical_features"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Metadata for '{dataset_name}' is missing preparation feature metadata."
        ) from exc

    if not feature_columns:
        raise ValueError(f"Metadata for '{dataset_name}' contains no features.")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Metadata contains duplicate feature names.")

    missing_features = set(feature_columns).difference(frame.columns)
    if missing_features:
        raise ValueError(
            f"Features listed in metadata are missing: {sorted(missing_features)}"
        )

    X = frame.loc[:, feature_columns].reset_index(drop=True)
    y = frame[TARGET_COLUMN].astype("int8").reset_index(drop=True)

    if y.nunique() != 2:
        raise ValueError(f"{dataset_name} must contain exactly two target classes.")
    if not set(categorical_features).issubset(X.columns):
        raise ValueError("Metadata contains categorical features not present in X.")

    cv_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, N_SPLITS + 1):
        train_indices = splits[f"cv_fold_{fold}_train_indices"].copy()
        valid_indices = splits[f"cv_fold_{fold}_valid_indices"].copy()
        train_indices.setflags(write=False)
        valid_indices.setflags(write=False)
        cv_folds.append((train_indices, valid_indices))

    evaluation_dataset = EvaluationDataset(
        name=dataset_name,
        X=X,
        y=y,
        cv_folds=tuple(cv_folds),
        categorical_features=tuple(categorical_features),
    )

    log.info(
        "Dataset loaded and validated | dataset=%s | rows=%d | features=%d | "
        "categorical=%d | CV folds=%d | elapsed=%.3f sec",
        dataset_name,
        len(X),
        X.shape[1],
        len(categorical_features),
        evaluation_dataset.n_splits,
        time.perf_counter() - started,
    )
    return evaluation_dataset


def evaluate(
    dataset: EvaluationDataset,
    params: dict[str, Any] | None = None,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Оценка одной конфигурации гиперпараметров с использованием предварительно загруженных разбиений CV.

    Входные данные должны быть подготовлены с помощью функции `load_evaluation_dataset`. Эта
    функция не выполняет операций ввода-вывода на диск и не проводит разбиение данных,
    чтобы оптимизировать процесс поиска гиперпараметров (HPO) при многократнох вызовах.

    Return: Возвращаются: среднее значение ROC-AUC (целевая метрика оптимизатора), оценки по отдельным фолдам,
    выборочное стандартное отклонение, а также время, затраченное на оценку модели.
    
    Сохраненная тестовая выборка при обучении или оценке модели не используется.
    """
    if not isinstance(dataset, EvaluationDataset):
        raise TypeError(
            "dataset must be an EvaluationDataset. Call "
            "load_evaluation_dataset(dataset_name) once before evaluation."
        )

    log = logger or LOGGER
    searched_params = dict(params or {})

    log.info(
        "Evaluation started | dataset=%s | params=%s | iterations=%d | seed=%d",
        dataset.name,
        searched_params if searched_params else "CatBoost defaults",
        iterations,
        random_seed,
    )
    started = time.perf_counter()

    fold_scores: list[float] = []
    for fold, (train_indices, valid_indices) in enumerate(dataset.cv_folds, start=1):
        X_train = dataset.X.iloc[train_indices]
        y_train = dataset.y.iloc[train_indices]
        X_valid = dataset.X.iloc[valid_indices]
        y_valid = dataset.y.iloc[valid_indices]

        model = create_catboost_classifier(
            searched_params,
            iterations=iterations,
            random_seed=random_seed,
        )
        model.fit(
            X_train,
            y_train,
            cat_features=list(dataset.categorical_features) or None,
        )

        positive_class_column = int(np.flatnonzero(model.classes_ == 1)[0])
        probabilities = model.predict_proba(X_valid)[:, positive_class_column]
        score = float(roc_auc_score(y_valid, probabilities))
        fold_scores.append(score)

        log.info(
            "Evaluation | dataset=%s | fold=%d/%d | train=%d | valid=%d | ROC-AUC=%.6f",
            dataset.name,
            fold,
            dataset.n_splits,
            len(train_indices),
            len(valid_indices),
            score,
        )

    elapsed = time.perf_counter() - started
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores, ddof=1))

    result: dict[str, Any] = {
        "dataset": dataset.name,
        "params": searched_params,
        "fold_roc_auc": fold_scores,
        "mean_roc_auc": mean_score,
        "std_roc_auc": std_score,
        "elapsed_seconds": float(elapsed),
        "iterations": int(iterations),
        "random_seed": int(random_seed),
        "n_splits": dataset.n_splits,
    }
    log.info(
        "Evaluation finished | dataset=%s | mean ROC-AUC=%.6f | "
        "fold std=%.6f | elapsed=%.2f sec",
        dataset.name,
        mean_score,
        std_score,
        elapsed,
    )
    return result
