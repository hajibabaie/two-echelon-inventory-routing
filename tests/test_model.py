"""Tests for the Gurobi MIP: golden optima, checked solutions, and the no-solution error."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from irp2e.evaluation import assert_feasible  # noqa: E402
from irp2e.instance import load_instance  # noqa: E402
from irp2e.model import solve_mip  # noqa: E402

from tiny_example import INSTANCES, TINY, A, B, C  # noqa: E402

pytestmark = pytest.mark.gurobi

SMALL_TWO_HUBS = INSTANCES / "small_k2_n6_T3_s1.json"


def as_sets(routes: list[list[int]]) -> list[set[int]]:
    """Routes without driving direction: km are symmetric, so both directions are optimal."""
    return [set(route) for route in routes]


def test_mip_proves_the_tiny_optimum_303():
    tiny = load_instance(TINY)
    solution, result = solve_mip(tiny, time_limit_s=60, threads=1, mip_gap=0.0)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(303.0, abs=1e-6)  # truck 250 + van 28 + holding 25
    assert assert_feasible(solution, tiny).total == pytest.approx(303.0, abs=1e-6)
    assert solution.truck_trips.tolist() == [[True, False]]
    assert solution.hub_shipments[0] == pytest.approx([55.0, 0.0], abs=1e-6)
    assert solution.deliveries == pytest.approx(
        np.array([[10.0, 10.0], [20.0, 0.0], [0.0, 15.0]]), abs=1e-6
    )
    assert as_sets(solution.van_routes[0][0]) == [{A, B}]  # H-A-B-H 13 km, load 30 = full van
    assert as_sets(solution.van_routes[0][1]) == [{A, C}]  # H-A-C-H 15 km


def test_mip_golden_on_small_k2_n6_T3_s1():
    instance = load_instance(SMALL_TWO_HUBS)
    solution, result = solve_mip(instance, time_limit_s=600, threads=1, mip_gap=0.0)
    assert result.status == "optimal"
    # golden: proven optimum; a second formulation written apart from model.py gives the same value
    assert result.objective == pytest.approx(3754.118835, abs=1e-4)
    assert assert_feasible(solution, instance).total == pytest.approx(3754.118835, abs=1e-4)
    assert solution.truck_trips.tolist() == [[False, False, False], [False, True, False]]


def test_mip_raises_when_no_solution_exists():
    tiny = load_instance(TINY)
    no_hub_room = dataclasses.replace(tiny, hub_capacity=np.array([0.0]))  # A runs out on day 0
    with pytest.raises(RuntimeError, match="tiny: Gurobi status 3"):
        solve_mip(no_hub_room, time_limit_s=60, threads=1, mip_gap=0.0)


def test_mip_stopped_by_the_time_limit_returns_a_checked_solution():
    instance = load_instance(INSTANCES / "small_k1_n10_T3_s1.json")
    # the proof takes about 110 s, so 2 s stops early
    solution, result = solve_mip(instance, time_limit_s=2, threads=1, mip_gap=0.0)
    optimum = 4584.277571  # proven with a 600 s limit; the second formulation agrees
    assert result.status == "time_limit"
    assert result.bound <= optimum + 1e-4
    assert result.objective >= optimum - 1e-4
    assert assert_feasible(solution, instance).total == pytest.approx(result.objective, rel=1e-6)
