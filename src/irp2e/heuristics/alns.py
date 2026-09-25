"""The ALNS engine: roulette choice, scores, adaptive weights, annealing acceptance and the loop."""

import time
from collections.abc import Callable

import numpy as np

from irp2e.config import Params
from irp2e.evaluation import solution_cost
from irp2e.heuristics.operators import DESTROY_OPERATORS, REPAIR_OPERATORS, greedy_repair
from irp2e.heuristics.plan import (
    Plan,
    copy_plan,
    count_visits,
    drop_empty_stops,
    empty_plan,
    plan_to_solution,
)
from irp2e.heuristics.routing import two_opt
from irp2e.instance import Instance
from irp2e.solution import Solution

QuantityStep = Callable[[Plan, Instance], tuple[np.ndarray, np.ndarray] | None]


def removal_count(plan: Plan, params: Params) -> int:
    """rho drawn from 1 .. rho_max, rho_max = share of the visits, clipped to the caps."""
    share = int(np.floor(params.removal_share * count_visits(plan) + 0.5))
    rho_max = max(params.removal_cap_min, min(params.removal_cap_max, share))
    return int(np.random.randint(1, rho_max + 1))


def roulette(weights: np.ndarray) -> int:
    """Index j drawn with probability w_j / sum(w)."""
    return int(np.random.choice(len(weights), p=weights / np.sum(weights)))


def start_temperature(cost: float, params: Params) -> float:
    """tau_0: a cost start_worse above the start is accepted with start_accept_probability."""
    return float(-params.start_worse * cost / np.log(params.start_accept_probability))


def temperature(start: float, progress: float, params: Params) -> float:
    """tau falls geometrically from start to start * final_temperature_ratio over the run."""
    return float(start * np.power(params.final_temperature_ratio, progress))


def accept(delta: float, temp: float) -> bool:
    """Simulated annealing: take a better cost; a worse one with probability exp(-delta / tau)."""
    return delta < 0 or np.random.rand() < np.exp(-delta / temp)


def operator_score(
    cost: float, best_cost: float, current_cost: float, accepted: bool, params: Params
) -> float:
    """Score of the two operators: new best, better than current, worse but accepted, or 0."""
    if cost < best_cost:
        return params.score_best
    if cost < current_cost:
        return params.score_better
    if accepted:
        return params.score_accepted
    return 0.0


def update_weights(
    weights: np.ndarray, scores: np.ndarray, uses: np.ndarray, reaction: float
) -> np.ndarray:
    """End of a segment: w = w (1 - r) + r * score / uses; an unused operator keeps its weight."""
    new = weights.copy()
    for j in range(len(weights)):
        if uses[j] > 0:
            new[j] = weights[j] * (1 - reaction) + reaction * scores[j] / uses[j]
    return new


def initial_plan(instance: Instance) -> Plan:
    """The start: the greedy repair applied to an empty plan."""
    plan = empty_plan(instance)
    # every repair takes the touched points and hubs and rho; the greedy repair uses neither
    greedy_repair(plan, ([], []), instance, 0)
    return plan


def improve_routes(plan: Plan, hub_days: list[tuple[int, int]], instance: Instance) -> None:
    """2-opt on every route of the given (hub, day) pairs."""
    for h, t in hub_days:
        plan.routes[h][t] = [two_opt(h, route, instance) for route in plan.routes[h][t]]


def changed_hub_days(before: Plan, after: Plan) -> list[tuple[int, int]]:
    """(hub, day) pairs whose routes differ between two plans."""
    return [
        (h, t)
        for h in range(len(before.routes))
        for t in range(len(before.routes[h]))
        if before.routes[h][t] != after.routes[h][t]
    ]


class Alns:
    """Adaptive large neighborhood search over plans; quantities come from quantity_step."""

    def __init__(self, instance: Instance, params: Params, quantity_step: QuantityStep):
        self.instance = instance
        self.params = params
        self.quantity_step = quantity_step
        self.destroy_weights = np.ones(len(DESTROY_OPERATORS))
        self.repair_weights = np.ones(len(REPAIR_OPERATORS))
        self.destroy_uses = np.zeros(len(DESTROY_OPERATORS), dtype=int)
        self.repair_uses = np.zeros(len(REPAIR_OPERATORS), dtype=int)
        self.start_segment()
        self.iterations = 0
        self.rejected = 0  # candidates the repair or the quantity step could not make feasible
        self.accepted = 0
        self.history = []  # [seconds, best cost] each time the best improves

    # ----- one candidate -----

    def finish(self, plan: Plan, reference: Plan) -> Solution | None:
        """Quantities, drop empty stops, 2-opt the changed routes; None if quantities fail."""
        quantities = self.quantity_step(plan, self.instance)
        if quantities is None:
            return None
        q, y = quantities
        drop_empty_stops(plan, q, y)
        improve_routes(plan, changed_hub_days(reference, plan), self.instance)
        return plan_to_solution(plan, q, y)

    # ----- the loop -----

    def run(self, budget: float, by_iterations: bool) -> Solution:
        """Search until the budget (iterations, or else seconds) is spent; the best solution."""
        start = time.perf_counter()
        current = initial_plan(self.instance)
        best_solution = self.finish(current, empty_plan(self.instance))
        current_cost = solution_cost(best_solution, self.instance).total
        best_cost = current_cost
        self.history.append([time.perf_counter() - start, best_cost])
        start_temp = start_temperature(current_cost, self.params)
        progress = 0.0
        while progress < 1.0:
            d = roulette(self.destroy_weights)
            r = roulette(self.repair_weights)
            rho = removal_count(current, self.params)
            self.destroy_segment_uses[d] += 1
            self.repair_segment_uses[r] += 1
            candidate = copy_plan(current)
            touched = DESTROY_OPERATORS[d][1](candidate, self.instance, rho)
            score = 0.0
            solution = None
            if REPAIR_OPERATORS[r][1](candidate, touched, self.instance, rho):
                solution = self.finish(candidate, current)
            if solution is None:
                self.rejected += 1
            else:
                cost = solution_cost(solution, self.instance).total
                tau = temperature(start_temp, progress, self.params)
                accepted = accept(cost - current_cost, tau)
                score = operator_score(cost, best_cost, current_cost, accepted, self.params)
                if accepted:
                    self.accepted += 1
                    current, current_cost = candidate, cost
                if cost < best_cost:
                    best_solution, best_cost = solution, cost
                    self.history.append([time.perf_counter() - start, best_cost])
            self.destroy_scores[d] += score
            self.repair_scores[r] += score
            self.iterations += 1
            if self.iterations % self.params.segment_length == 0:
                self.end_segment()
            if by_iterations:
                progress = self.iterations / budget
            else:
                progress = (time.perf_counter() - start) / budget
        self.destroy_uses += self.destroy_segment_uses  # the last, unfinished segment
        self.repair_uses += self.repair_segment_uses
        return best_solution

    # ----- adaptive weights -----

    def start_segment(self) -> None:
        """Zero the scores and uses of the new segment."""
        self.destroy_scores = np.zeros(len(DESTROY_OPERATORS))
        self.repair_scores = np.zeros(len(REPAIR_OPERATORS))
        self.destroy_segment_uses = np.zeros(len(DESTROY_OPERATORS), dtype=int)
        self.repair_segment_uses = np.zeros(len(REPAIR_OPERATORS), dtype=int)

    def end_segment(self) -> None:
        """Update both weight vectors, add the segment uses to the totals, start a new segment."""
        self.destroy_weights = update_weights(
            self.destroy_weights,
            self.destroy_scores,
            self.destroy_segment_uses,
            self.params.reaction,
        )
        self.repair_weights = update_weights(
            self.repair_weights, self.repair_scores, self.repair_segment_uses, self.params.reaction
        )
        self.destroy_uses += self.destroy_segment_uses
        self.repair_uses += self.repair_segment_uses
        self.start_segment()
