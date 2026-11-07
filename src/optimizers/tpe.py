# Оптимизатор TPE из Optuna.
from __future__ import annotations

import time

import optuna
from optuna.samplers import TPESampler

from .base import BaseOptimizer
from .search_space import suggest_optuna_params


class TPEOptimizer(BaseOptimizer):
    optimizer_name = "tpe"

    def optimize(self) -> dict:
        self.history = []
        started = time.perf_counter()
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.seed),
        )

        def objective(trial: optuna.Trial) -> float:
            params = suggest_optuna_params(trial)
            score = self._evaluate(params)
            self.logger.info(
                "Optuna trial=%d/%d | ROC-AUC=%.6f | params=%s",
                trial.number + 1,
                self.budget,
                score,
                params,
            )
            if self.history:
                record = self.history[-1]
                trial.set_user_attr("fold_roc_auc", record["fold_roc_auc"])
                trial.set_user_attr("std_roc_auc", record["std_roc_auc"])
                trial.set_user_attr("elapsed_seconds", record["elapsed_seconds"])
            return score

        study.optimize(objective, n_trials=self.budget, show_progress_bar=False)
        result = self._build_result(time.perf_counter() - started)
        result["optuna_best_value"] = float(study.best_value)
        result["optuna_best_trial"] = int(study.best_trial.number)
        return result
