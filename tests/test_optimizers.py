from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.optimizers.genetic_algorithm import GeneticAlgorithmOptimizer
from src.optimizers.random_search import RandomSearchOptimizer
from src.optimizers.search_space import decode_chromosome, sample_random_params
from src.optimizers.tpe import TPEOptimizer


def dummy_objective(params: dict[str, Any]) -> dict[str, Any]:
    score = 1.0 - abs(params["depth"] - 7) * 0.01
    return {
        "mean_roc_auc": score,
        "fold_roc_auc": [score, score, score],
        "std_roc_auc": 0.0,
        "elapsed_seconds": 0.001,
    }


@pytest.mark.parametrize(
    "optimizer_class,kwargs",
    [
        (RandomSearchOptimizer, {}),
        (TPEOptimizer, {}),
        (GeneticAlgorithmOptimizer, {"population_size": 4}),
    ],
)
def test_optimizer_respects_budget_and_result_schema(optimizer_class, kwargs):
    result = optimizer_class(
        dummy_objective, budget=7, seed=123, **kwargs
    ).optimize()
    assert result["n_evaluations"] == 7
    assert len(result["history"]) == 7
    assert 0.0 <= result["best_score"] <= 1.0
    assert set(result["best_params"]) == {
        "depth", "learning_rate", "l2_leaf_reg", "bagging_temperature",
        "random_strength", "border_count",
    }


def test_random_search_is_reproducible():
    first = RandomSearchOptimizer(dummy_objective, budget=5, seed=17).optimize()
    second = RandomSearchOptimizer(dummy_objective, budget=5, seed=17).optimize()
    assert [row["params"] for row in first["history"]] == [
        row["params"] for row in second["history"]
    ]


def test_decoded_chromosome_respects_bounds():
    params = decode_chromosome([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert params == {
        "depth": 4,
        "learning_rate": pytest.approx(0.01),
        "l2_leaf_reg": 1.0,
        "bagging_temperature": 0.0,
        "random_strength": 0.0,
        "border_count": 32,
    }


def test_random_sampler_respects_bounds():
    params = sample_random_params(np.random.default_rng(5))
    assert 4 <= params["depth"] <= 10
    assert 0.01 <= params["learning_rate"] <= 0.3
    assert 1.0 <= params["l2_leaf_reg"] <= 10.0
    assert 0.0 <= params["bagging_temperature"] <= 5.0
    assert 0.0 <= params["random_strength"] <= 5.0
    assert 32 <= params["border_count"] <= 255
