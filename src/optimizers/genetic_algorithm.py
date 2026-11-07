# Оптимизатор на основе генетического алгоритма, реализованный с помощью DEAP.
from __future__ import annotations

import random
import time

from deap import base, creator, tools

from .base import BaseOptimizer
from .search_space import CHROMOSOME_SIZE, decode_chromosome

if not hasattr(creator, "FitnessMaxHPO"):
    creator.create("FitnessMaxHPO", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualHPO"):
    creator.create("IndividualHPO", list, fitness=creator.FitnessMaxHPO)


class GeneticAlgorithmOptimizer(BaseOptimizer):
    optimizer_name = "genetic_algorithm"

    def __init__(
        self,
        objective,
        *,
        budget: int = 100,
        seed: int = 42,
        population_size: int = 20,
        tournament_size: int = 3,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        elite_size: int = 1,
        crossover_eta: float = 15.0,
        mutation_eta: float = 20.0,
        logger=None,
    ) -> None:
        super().__init__(objective, budget=budget, seed=seed, logger=logger)
        if population_size < 2:
            raise ValueError("размер популяции должен быть не менее 2")
        if not 1 <= elite_size < population_size:
            raise ValueError("elite_size должен быть >= 1 и < population_size")
        if tournament_size < 2:
            raise ValueError("tournament_size должен быть не менее 2")
        for name, value in (
            ("crossover_probability", crossover_probability),
            ("mutation_probability", mutation_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} должно быть в диапазоне [0, 1]")
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elite_size = elite_size
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

    def optimize(self) -> dict:
        self.history = []
        previous_random_state = random.getstate()
        random.seed(self.seed)
        started = time.perf_counter()
        try:
            toolbox = base.Toolbox()
            toolbox.register("gene", random.random)
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.IndividualHPO,
                toolbox.gene,
                n=CHROMOSOME_SIZE,
            )
            toolbox.register(
                "population", tools.initRepeat, list, toolbox.individual
            )
            toolbox.register(
                "select", tools.selTournament, tournsize=self.tournament_size
            )
            toolbox.register(
                "mate",
                tools.cxSimulatedBinaryBounded,
                low=0.0,
                up=1.0,
                eta=self.crossover_eta,
            )
            toolbox.register(
                "mutate",
                tools.mutPolynomialBounded,
                low=0.0,
                up=1.0,
                eta=self.mutation_eta,
                indpb=1.0 / CHROMOSOME_SIZE,
            )

            initial_size = min(self.population_size, self.budget)
            population = toolbox.population(n=initial_size)
            for individual in population:
                params = decode_chromosome(individual)
                individual.fitness.values = (self._evaluate(params),)

            while len(self.history) < self.budget:
                remaining = self.budget - len(self.history)
                offspring_count = min(
                    self.population_size - self.elite_size, remaining
                )
                if offspring_count <= 0:
                    break

                offspring = [
                    creator.IndividualHPO(individual)
                    for individual in toolbox.select(population, offspring_count)
                ]

                for index in range(0, len(offspring) - 1, 2):
                    if random.random() < self.crossover_probability:
                        toolbox.mate(offspring[index], offspring[index + 1])

                for individual in offspring:
                    if random.random() < self.mutation_probability:
                        toolbox.mutate(individual)
                    params = decode_chromosome(individual)
                    individual.fitness.values = (self._evaluate(params),)

                population = tools.selBest(
                    population + offspring, k=min(self.population_size, len(population) + len(offspring))
                )

            return self._build_result(time.perf_counter() - started)
        finally:
            random.setstate(previous_random_state)
