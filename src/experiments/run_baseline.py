# Запуск baseline CatBoost на обоих подготовленных наборах данных.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluate import evaluate, load_evaluation_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("german_credit", "qsar_biodegradation")


def configure_logging(project_root: Path) -> logging.Logger:
    log_dir = project_root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_dir / "baseline.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate default CatBoost baseline on prepared datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to evaluate (default: both).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    logger = configure_logging(root)
    logger.info("Starting CatBoost baseline | project_root=%s", root)
    logger.info("Datasets: %s", ", ".join(args.datasets))

    evaluation_datasets = {
        dataset_name: load_evaluation_dataset(
            dataset_name,
            project_root=root,
            logger=logger,
        )
        for dataset_name in args.datasets
    }

    results: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        result = evaluate(
            evaluation_datasets[dataset_name],
            params=None,
            logger=logger,
        )
        results.append(result)

    output_dir = root / "results" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "baseline_results.json"
    csv_path = output_dir / "baseline_results.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "dataset": item["dataset"],
                "fold_1_roc_auc": item["fold_roc_auc"][0],
                "fold_2_roc_auc": item["fold_roc_auc"][1],
                "fold_3_roc_auc": item["fold_roc_auc"][2],
                "mean_roc_auc": item["mean_roc_auc"],
                "std_roc_auc": item["std_roc_auc"],
                "elapsed_seconds": item["elapsed_seconds"],
            }
            for item in results
        ]
    ).to_csv(csv_path, index=False)

    logger.info("Saved baseline JSON: %s", json_path)
    logger.info("Saved baseline CSV: %s", csv_path)
    logger.info("Baseline evaluation completed.")


if __name__ == "__main__":
    main()
