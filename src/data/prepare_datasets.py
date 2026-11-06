# Подготовка наборов данных German Credit и QSAR для воспроизводимых экспериментов с использованием CatBoost.
# Исходные CSV-файлы остаются неизменными. Для каждого набора данных записываем:
#   data/processed/<dataset>/dataset.csv
#   data/processed/<dataset>/splits.npz
#   data/processed/<dataset>/metadata.json

# Индексы, сохранённые в файле splits.npz, указывают на позиции строк в файле dataset.csv.
# Столбец original_row_index содержит индекс строки (начиная с нуля)
# из исходного CSV-файла, что позволяет проследить происхождение каждого
# подготовленного наблюдения вплоть до соответствующей ему исходной строки.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "target"
ORIGINAL_INDEX_COLUMN = "original_row_index"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 3

GERMAN_CATEGORICAL_FEATURES = [
    "Attribute1",
    "Attribute3",
    "Attribute4",
    "Attribute6",
    "Attribute7",
    "Attribute9",
    "Attribute10",
    "Attribute12",
    "Attribute14",
    "Attribute15",
    "Attribute17",
    "Attribute19",
    "Attribute20",
]
GERMAN_NUMERIC_FEATURES = [
    "Attribute2",
    "Attribute5",
    "Attribute8",
    "Attribute11",
    "Attribute13",
    "Attribute16",
    "Attribute18",
]
GERMAN_FEATURES = GERMAN_CATEGORICAL_FEATURES + GERMAN_NUMERIC_FEATURES

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "german_credit": {
        "filename": "german_credit.csv",
        "expected_rows_before_cleaning": 1000,
        "expected_n_features": 20,
        "target_mapping": {"1": 0, "2": 1},
        "target_description": {
            "0": "Good credit risk",
            "1": "Bad credit risk",
        },
        "feature_names": GERMAN_FEATURES,
        "categorical_features": GERMAN_CATEGORICAL_FEATURES,
        "numeric_features": GERMAN_NUMERIC_FEATURES,
        "drop_duplicates": False,
    },
    "qsar_biodegradation": {
        "filename": "qsar_biodegradation.csv",
        "expected_rows_before_cleaning": 1055,
        "expected_n_features": 41,
        "target_mapping": {"NRB": 0, "RB": 1},
        "target_description": {
            "0": "Not readily biodegradable (NRB)",
            "1": "Readily biodegradable (RB)",
        },
        "feature_names": None,
        "categorical_features": [],
        "numeric_features": None,
        "drop_duplicates": True,
    },
}


def setup_logging(log_path: Path) -> logging.Logger:
    """Настройка логов в консоль и в файл."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe_distribution(y: pd.Series) -> dict[str, dict[str, int | float]]:
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    return {
        str(label): {
            "count": int(count),
            "share": float(count / total),
        }
        for label, count in counts.items()
    }


def validate_and_prepare_frame(
    dataset_name: str,
    raw_path: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Загрузка, проверка, очистка и сопоставление (mapping) целевой переменной."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {raw_path}. Сначала выполните download_datasets.py."
        )

    logger.info("[%s] Чтение исходного файла: %s", dataset_name, raw_path)
    raw = pd.read_csv(raw_path)
    source_hash = sha256_file(raw_path)
    logger.info(
        "[%s] Исходный размер: %d строк × %d столбцов; SHA-256: %s",
        dataset_name,
        raw.shape[0],
        raw.shape[1],
        source_hash,
    )

    if config["expected_rows_before_cleaning"] != len(raw):
        logger.warning(
            "[%s] Ожидалось %d строк по результатам EDA, получено %d. ",
            dataset_name,
            config["expected_rows_before_cleaning"],
            len(raw),
        )

    if TARGET_COLUMN not in raw.columns:
        raise ValueError(
            f"[{dataset_name}] В CSV отсутствует обязательный столбец '{TARGET_COLUMN}'."
        )
    if raw.columns.duplicated().any():
        duplicates = raw.columns[raw.columns.duplicated()].tolist()
        raise ValueError(f"[{dataset_name}] Повторяющиеся имена столбцов: {duplicates}")

    feature_columns = [column for column in raw.columns if column != TARGET_COLUMN]
    if len(feature_columns) != config["expected_n_features"]:
        raise ValueError(
            f"[{dataset_name}] Ожидалось {config['expected_n_features']} признаков, "
            f"получено {len(feature_columns)}. Проверьте исходный CSV."
        )

    expected_features = config["feature_names"]
    if expected_features is not None and set(feature_columns) != set(expected_features):
        missing = sorted(set(expected_features) - set(feature_columns))
        unexpected = sorted(set(feature_columns) - set(expected_features))
        raise ValueError(
            f"[{dataset_name}] Список признаков не совпадает с EDA. "
            f"Отсутствуют: {missing}; лишние: {unexpected}."
        )

    if raw[TARGET_COLUMN].isna().any():
        raise ValueError(f"[{dataset_name}] В target обнаружены пропуски.")
    if raw[feature_columns].isna().any().any():
        missing_by_column = raw[feature_columns].isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].to_dict()
        raise ValueError(
            f"[{dataset_name}] Обнаружены пропуски в признаках: {missing_by_column}. "
            "Стратегия импутации не была предусмотрена результатами EDA."
        )

    raw_distribution = raw[TARGET_COLUMN].astype(str).value_counts().to_dict()
    logger.info("[%s] Распределение исходного target: %s", dataset_name, raw_distribution)

    target_mapping = config["target_mapping"]
    raw_target_as_string = raw[TARGET_COLUMN].astype(str)
    observed_labels = set(raw_target_as_string.unique())
    expected_labels = set(target_mapping)
    if observed_labels != expected_labels:
        raise ValueError(
            f"[{dataset_name}] Неожиданные метки target. "
            f"Ожидались {sorted(expected_labels)}, получены {sorted(observed_labels)}."
        )

    duplicate_mask = raw.duplicated(keep="first")
    duplicates_before = int(duplicate_mask.sum())
    removed_duplicates = 0

    if config["drop_duplicates"]:
        keep_mask = ~duplicate_mask
        frame = raw.loc[keep_mask].copy()
        removed_duplicates = duplicates_before
        logger.info(
            "[%s] Удалено полных дубликатов: %d; осталось строк: %d",
            dataset_name,
            removed_duplicates,
            len(frame),
        )
    else:
        frame = raw.copy()
        logger.info("[%s] Удаление дубликатов не требуется по результатам EDA.", dataset_name)
        if duplicates_before:
            logger.warning(
                "[%s] Найдены %d полных дубликатов, хотя EDA не ожидал их.",
                dataset_name,
                duplicates_before,
            )

    frame.insert(
        0,
        ORIGINAL_INDEX_COLUMN,
        np.flatnonzero(~duplicate_mask.to_numpy()) if config["drop_duplicates"]
        else np.arange(len(frame), dtype=np.int64),
    )

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).map(target_mapping)
    if frame[TARGET_COLUMN].isna().any():
        raise RuntimeError(f"[{dataset_name}] Не удалось преобразовать часть target-меток.")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")

    categorical_features = list(config["categorical_features"])
    numeric_features = config["numeric_features"]

    if categorical_features:
        for column in categorical_features:
            frame[column] = frame[column].astype(str)

    if numeric_features is None:
        numeric_features = feature_columns
    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if not set(categorical_features).issubset(feature_columns):
        raise ValueError(f"[{dataset_name}] Некорректный список категориальных признаков.")
    if set(categorical_features).intersection(numeric_features):
        raise ValueError(f"[{dataset_name}] Признак одновременно категориальный и числовой.")

    logger.info(
        "[%s] Признаков: %d (категориальных: %d, числовых: %d)",
        dataset_name,
        len(feature_columns),
        len(categorical_features),
        len(numeric_features),
    )
    logger.info(
        "[%s] Распределение target после подготовки: %s",
        dataset_name,
        json_safe_distribution(frame[TARGET_COLUMN]),
    )

    frame = frame.reset_index(drop=True)
    preparation_info = {
        "source_file": raw_path.name,
        "source_sha256": source_hash,
        "source_rows": int(len(raw)),
        "source_columns": int(raw.shape[1]),
        "source_target_distribution": {
            str(label): int(count) for label, count in raw_distribution.items()
        },
        "full_duplicates_found": int(duplicates_before),
        "duplicates_removed": int(removed_duplicates),
        "prepared_rows": int(len(frame)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": list(numeric_features),
        "target_mapping": target_mapping,
        "target_description": config["target_description"],
        "prepared_target_distribution": json_safe_distribution(frame[TARGET_COLUMN]),
    }
    return frame, preparation_info


def create_splits(
    frame: pd.DataFrame,
    dataset_name: str,
    logger: logging.Logger,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Создание фиксированных стратифицированных выборок (train/test) и
    CV фолдов на основе train выборки.
    """
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
    row_positions = np.arange(len(frame), dtype=np.int64)

    train_indices, test_indices = train_test_split(
        row_positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_train = y[train_indices]

    logger.info(
        "[%s] Train/test split: train=%d (%.1f%%), test=%d (%.1f%%)",
        dataset_name,
        len(train_indices),
        100 * len(train_indices) / len(frame),
        len(test_indices),
        100 * len(test_indices) / len(frame),
    )
    logger.info(
        "[%s] Train target distribution: %s",
        dataset_name,
        json_safe_distribution(pd.Series(y_train)),
    )
    logger.info(
        "[%s] Test target distribution: %s",
        dataset_name,
        json_safe_distribution(pd.Series(y[test_indices])),
    )

    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError(f"[{dataset_name}] Train и test пересекаются.")
    if len(train_indices) + len(test_indices) != len(frame):
        raise RuntimeError(f"[{dataset_name}] Train/test не покрывают все наблюдения.")

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    arrays: dict[str, np.ndarray] = {
        "train_indices": train_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
    }
    folds_metadata: list[dict[str, Any]] = []

    for fold_number, (fold_train_relative, fold_valid_relative) in enumerate(
        splitter.split(np.zeros(len(train_indices)), y_train), start=1
    ):
        fold_train = train_indices[fold_train_relative]
        fold_valid = train_indices[fold_valid_relative]
        arrays[f"cv_fold_{fold_number}_train_indices"] = fold_train.astype(np.int64)
        arrays[f"cv_fold_{fold_number}_valid_indices"] = fold_valid.astype(np.int64)

        logger.info(
            "[%s] CV fold %d: train=%d, validation=%d; validation classes=%s",
            dataset_name,
            fold_number,
            len(fold_train),
            len(fold_valid),
            json_safe_distribution(pd.Series(y[fold_valid])),
        )
        folds_metadata.append(
            {
                "fold": fold_number,
                "train_size": int(len(fold_train)),
                "validation_size": int(len(fold_valid)),
            }
        )

    split_metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv": {
            "class": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "folds": folds_metadata,
        },
        "index_semantics": (
            "All indices in splits.npz are zero-based row positions in dataset.csv. "
            "Use dataset.csv[\"original_row_index\"] to recover the zero-based "
            "row position in the original raw CSV. CV indices are absolute positions "
            "in dataset.csv, not positions relative to the train subset."
        ),
    }
    return arrays, split_metadata


def prepare_one_dataset(
    dataset_name: str,
    project_root: Path,
    logger: logging.Logger,
) -> None:
    config = DATASET_CONFIGS[dataset_name]
    raw_path = project_root / "data" / "raw" / config["filename"]
    output_dir = project_root / "data" / "processed" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info("Начало подготовки датасета: %s", dataset_name)
    frame, preparation_info = validate_and_prepare_frame(
        dataset_name, raw_path, config, logger
    )
    split_arrays, split_info = create_splits(frame, dataset_name, logger)

    dataset_path = output_dir / "dataset.csv"
    splits_path = output_dir / "splits.npz"
    metadata_path = output_dir / "metadata.json"

    frame.to_csv(dataset_path, index=False)
    np.savez_compressed(splits_path, **split_arrays)

    original_rows = frame[ORIGINAL_INDEX_COLUMN].to_numpy(dtype=np.int64)
    split_source_rows = {
        "train_original_row_indices": original_rows[split_arrays["train_indices"]].tolist(),
        "test_original_row_indices": original_rows[split_arrays["test_indices"]].tolist(),
    }
    for fold_number in range(1, N_SPLITS + 1):
        split_source_rows[f"cv_fold_{fold_number}_train_original_row_indices"] = (
            original_rows[split_arrays[f"cv_fold_{fold_number}_train_indices"]].tolist()
        )
        split_source_rows[f"cv_fold_{fold_number}_valid_original_row_indices"] = (
            original_rows[split_arrays[f"cv_fold_{fold_number}_valid_indices"]].tolist()
        )

    metadata = {
        "dataset": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "preparation": preparation_info,
        "split": split_info,
        "files": {
            "dataset": dataset_path.name,
            "splits": splits_path.name,
            "metadata": metadata_path.name,
        },
        "preprocessing_applied": [
            "QSAR: remove exact duplicate rows before splitting (if applicable)",
            "Map target labels to binary 0/1",
            "Preserve German Credit categorical columns as strings for CatBoost",
            "Keep numeric features in their original scale",
        ],
        "preprocessing_not_applied": [
            "No imputation (EDA found no missing values)",
            "No scaling",
            "No one-hot encoding",
            "No automatic outlier removal or winsorization",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("[%s] Сохранён подготовленный датасет: %s", dataset_name, dataset_path)
    logger.info("[%s] Сохранены индексы разбиений: %s", dataset_name, splits_path)
    logger.info("[%s] Сохранены метаданные: %s", dataset_name, metadata_path)
    logger.info(
        "[%s] Готово: %d наблюдений; train=%d; test=%d",
        dataset_name,
        len(frame),
        len(split_arrays["train_indices"]),
        len(split_arrays["test_indices"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка German Credit и QSAR для экспериментов CatBoost HPO."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS),
        default=sorted(DATASET_CONFIGS),
        help="Какие датасеты подготовить (по умолчанию оба).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Корень проекта; по умолчанию определяется автоматически.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    log_path = project_root / "data" / "processed" / "logs" / "prepare_datasets.log"
    logger = setup_logging(log_path)

    logger.info("Запуск prepare_datasets.py")
    logger.info("Корень проекта: %s", project_root)
    logger.info("Выбранные датасеты: %s", ", ".join(args.datasets))
    logger.info(
        "Параметры разбиения: test_size=%.2f, random_state=%d, CV=%d-fold",
        TEST_SIZE,
        RANDOM_STATE,
        N_SPLITS,
    )

    try:
        for dataset_name in args.datasets:
            prepare_one_dataset(dataset_name, project_root, logger)
    except Exception:
        logger.exception("Подготовка данных завершилась с ошибкой.")
        raise

    logger.info("Подготовка всех выбранных датасетов успешно завершена.")


if __name__ == "__main__":
    main()
