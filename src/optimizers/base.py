# Общий интерфейс оптимизаторов, целевой функции и формат результатов.
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from src.evaluation.evaluate import EvaluationDataset, evaluate

Objective = Callable[[dict[str, Any]], dict[str, Any]]


def make_objective(
    evaluation_dataset: EvaluationDataset,
    *,
    model_seed: int = 42,
    iterations: int = 1000,
    logger: logging.Logger | None = None,
) -> Objective:
    """Создание общей целевой функции."""
    def objective(params: dict[str, Any]) -> dict[str, Any]:
        return evaluate(
            evaluation_dataset,
            params=params,
            iterations=iterations,
            random_seed=model_seed,
            logger=logger,
        )

    return objective


class BaseOptimizer(ABC):
    """Базовый класс, обеспечивающий единый интерфейс, историю оценки и схему результатов."""

    optimizer_name = "base"

    def __init__(
        self,
        objective: Objective,
        *,
        budget: int = 100,
        seed: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("бюджет должен быть не менее 1")
        self.objective = objective
        self.budget = int(budget)
        self.seed = int(seed)
        self.logger = logger or logging.getLogger(f"optimizers.{self.optimizer_name}")
        self.history: list[dict[str, Any]] = []

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Оценка конфигурации и доавление записи в историю."""
        evaluation_number = len(self.history) + 1
        started = time.perf_counter()
        result = self.objective(dict(params))
        objective_wall_seconds = time.perf_counter() - started

        if "mean_roc_auc" not in result:
            raise KeyError("Objective result must contain 'mean_roc_auc'.")
        score = float(result["mean_roc_auc"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"ROC-AUC must be in [0, 1], got {score}.")

        record = {
            "evaluation": evaluation_number,
            "params": dict(params),
            "roc_auc": score,
            "fold_roc_auc": list(result.get("fold_roc_auc", [])),
            "std_roc_auc": result.get("std_roc_auc"),
            # Время, фиксируемое методом evaluate().
            "elapsed_seconds": result.get("elapsed_seconds"),
            # Включает полный objective call.
            "objective_wall_seconds": float(objective_wall_seconds),
        }
        self.history.append(record)
        self.logger.info(
            "%s | evaluation=%d/%d | ROC-AUC=%.6f | elapsed=%.3f sec | params=%s",
            self.optimizer_name,
            evaluation_number,
            self.budget,
            score,
            objective_wall_seconds,
            params,
        )
        return score

    def _build_result(self, total_elapsed_seconds: float) -> dict[str, Any]:
        if not self.history:
            raise RuntimeError("Optimizer finished without evaluating a configuration.")
        best_record = max(self.history, key=lambda item: item["roc_auc"])
        return {
            "optimizer": self.optimizer_name,
            "best_params": dict(best_record["params"]),
            "best_score": float(best_record["roc_auc"]),
            "best_evaluation": int(best_record["evaluation"]),
            "n_evaluations": len(self.history),
            "history": list(self.history),
            "total_elapsed_seconds": float(total_elapsed_seconds),
            "seed": self.seed,
        }

    @abstractmethod
    def optimize(self) -> dict[str, Any]:
        """Запуск оптимизатора и возвращение словаря результатов."""
        raise NotImplementedError
