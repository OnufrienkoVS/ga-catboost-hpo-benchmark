# Общее пространство поиска гиперпараметров и вспомогательные средства для их выбора в CatBoost.
from __future__ import annotations

from typing import Any

import numpy as np

SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "depth": {"type": "int", "low": 4, "high": 10},
    "learning_rate": {
        "type": "float",
        "low": 0.01,
        "high": 0.3,
        "log": True,
    },
    "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
    "bagging_temperature": {"type": "float", "low": 0.0, "high": 5.0},
    "random_strength": {"type": "float", "low": 0.0, "high": 5.0},
    "border_count": {"type": "int", "low": 32, "high": 255},
}

PARAMETER_NAMES = tuple(SEARCH_SPACE)
CHROMOSOME_SIZE = len(PARAMETER_NAMES)


def sample_random_params(rng: np.random.Generator) -> dict[str, int | float]:
    """Сэмплирование одной конфигурации, используя стандартные распределения пространства поиска."""
    return {
        "depth": int(rng.integers(4, 11)),
        "learning_rate": float(
            np.exp(rng.uniform(np.log(0.01), np.log(0.3)))
        ),
        "l2_leaf_reg": float(rng.uniform(1.0, 10.0)),
        "bagging_temperature": float(rng.uniform(0.0, 5.0)),
        "random_strength": float(rng.uniform(0.0, 5.0)),
        "border_count": int(rng.integers(32, 256)),
    }


def decode_chromosome(chromosome: list[float]) -> dict[str, int | float]:
    """Маппинг шести нормализованных генов из диапазона [0,1] в допустимые параметры CatBoost."""
    if len(chromosome) != CHROMOSOME_SIZE:
        raise ValueError(
            f"Ожидалась хромосома длиной {CHROMOSOME_SIZE}, получена {len(chromosome)}."
        )
    genes = np.asarray(chromosome, dtype=float)
    if not np.all(np.isfinite(genes)) or np.any(genes < 0.0) or np.any(genes > 1.0):
        raise ValueError("Все гены хромосом должны иметь конечные значения в диапазоне [0, 1].")

    def integer_gene(gene: float, low: int, high: int) -> int:
        value = low + int(np.floor(gene * (high - low + 1)))
        return int(np.clip(value, low, high))

    def linear_gene(gene: float, low: float, high: float) -> float:
        return float(low + gene * (high - low))

    log_learning_rate = np.log(0.01) + genes[1] * (np.log(0.3) - np.log(0.01))
    return {
        "depth": integer_gene(genes[0], 4, 10),
        "learning_rate": float(np.exp(log_learning_rate)),
        "l2_leaf_reg": linear_gene(genes[2], 1.0, 10.0),
        "bagging_temperature": linear_gene(genes[3], 0.0, 5.0),
        "random_strength": linear_gene(genes[4], 0.0, 5.0),
        "border_count": integer_gene(genes[5], 32, 255),
    }


def suggest_optuna_params(trial: Any) -> dict[str, int | float]:
    """Предложение (suggest) параметров в Optuna, используя те же границы и распределения."""
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }