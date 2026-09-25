"""Tests for the greedy baseline: the tiny example, golden costs, the checker on every instance."""

import numpy as np
import pytest

from irp2e.evaluation import check_solution, solution_cost
from irp2e.heuristics.greedy import solve_greedy
from irp2e.instance import load_instance

from tiny_example import INSTANCES, TINY, A, B, C

FROZEN = sorted(INSTANCES.glob("*.json"))


def test_greedy_on_tiny_costs_305():
    tiny = load_instance(TINY)
    solution = solve_greedy(tiny)
    # G1: day 0 A and B are empty and get 20 each; C holds its 5; day 1 C gets min(20, 15) - 0
    assert solution.deliveries.tolist() == [[20.0, 0.0], [20.0, 0.0], [0.0, 15.0]]
    # G3: outflow 40 then 15; day 0 ship min(60, 55) - 0 = 55; day 1 the hub holds 15
    assert solution.hub_shipments.tolist() == [[55.0, 0.0]]
    # G2: A (4 km) takes 20 of 30 kg, B's 20 kg no longer fits, so B gets the second van
    assert solution.van_routes == [[[[A], [B]], [[C]]]]
    cost = solution_cost(solution, tiny)
    assert cost.truck == 250.0  # one trip: 50 + 2 x 100 km x 1 BRL
    assert cost.van == 30.0  # H-A-H 8 + H-B-H 12 + H-C-H 10
    assert cost.holding == 25.0  # hub keeps 15 and A keeps 10 at the end of day 0
    assert cost.total == 305.0  # the same plan scores 305 in the full enumeration of design 11


@pytest.mark.parametrize("path", FROZEN, ids=[path.stem for path in FROZEN])
def test_greedy_is_feasible_on_every_frozen_instance(path):
    instance = load_instance(path)
    assert check_solution(solve_greedy(instance), instance) == []


@pytest.mark.parametrize("path", FROZEN, ids=[path.stem for path in FROZEN])
def test_greedy_never_ships_float_noise(path):
    # the data is in whole grams; a delivery below one gram means G1 or G3 compared floats exactly
    instance = load_instance(path)
    solution = solve_greedy(instance)
    for value in np.concatenate([solution.deliveries.ravel(), solution.hub_shipments.ravel()]):
        assert value == 0.0 or value >= 0.001


@pytest.mark.parametrize(
    "name, total",
    [
        # pinned 2026-09-24 on shortest-path km and the 650 kg van, equal to an independent
        # re-implementation of G1 to G4 within 1e-10
        ("medium_k3_n80_T7_s1", 40239.557951),
        ("large_k5_n200_T7_s1", 66695.793398),
    ],
)
def test_greedy_golden_cost(name, total):
    instance = load_instance(INSTANCES / f"{name}.json")
    assert solution_cost(solve_greedy(instance), instance).total == pytest.approx(total, abs=1e-6)
