# Оптимизатор на основе случайного поиска с фиксированным бюджетом вычислений.
from __future__ import annotations

import time

import numpy as np

from .base import BaseOptimizer
from .search_space import sample_random_params


class RandomSearchOptimizer(BaseOptimizer):
    optimizer_name = "random_search"

    def optimize(self) -> dict:
        self.history = []
        rng = np.random.default_rng(self.seed)
        started = time.perf_counter()

        for _ in range(self.budget):
            params = sample_random_params(rng)
            self._evaluate(params)

        return self._build_result(time.perf_counter() - started)
