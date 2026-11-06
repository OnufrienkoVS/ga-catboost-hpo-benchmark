# Общая фабрика CatBoost для оптимизации гиперпараметров.

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier

DEFAULT_ITERATIONS = 1000
DEFAULT_RANDOM_SEED = 42

# Общие настройки, в том числе и для baseline
FIXED_MODEL_PARAMS: dict[str, Any] = {
    "iterations": DEFAULT_ITERATIONS,
    "random_seed": DEFAULT_RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
    "loss_function": "Logloss",
}


def create_catboost_classifier(
    params: dict[str, Any] | None = None,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> CatBoostClassifier:
    searched_params = dict(params or {})

    reserved = set(FIXED_MODEL_PARAMS) | {"eval_metric"}
    conflicting = reserved.intersection(searched_params)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(
            f"These parameters are controlled by the shared protocol and "
            f"cannot be supplied through `params`: {names}"
        )

    model_params = {
        **FIXED_MODEL_PARAMS,
        "iterations": iterations,
        "random_seed": random_seed,
        **searched_params,
    }
    return CatBoostClassifier(**model_params)