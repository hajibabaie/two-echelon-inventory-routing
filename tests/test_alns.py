"""Tests for the ALNS: rule J1, 2-opt, the tiny optimum, checked results and reproducibility."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from irp2e.config import load_params
from irp2e.evaluation import check_solution, solution_cost, van_route_km
from irp2e.heuristics.alns import (
    Alns,
    accept,
    removal_count,
    start_temperature,
    update_weights,
)
from irp2e.heuristics.operators import stockout_window
from irp2e.heuristics.plan import Plan
from irp2e.heuristics.quantities import (
    hub_quantities,
    jit_quantities,
    just_in_time,
    point_quantities,
)
from irp2e.heuristics.routing import two_opt
from irp2e.instance import load_instance

from tiny_example import INSTANCES, TINY, A, B, C, tiny_optimal_plan


def run_alns(instance_path: Path, seed: int, iterations: int):
    """One seeded ALNS run with an iteration budget."""
    instance = load_instance(instance_path)
    np.random.seed(seed)
    alns = Alns(instance, load_params(), jit_quantities)
    solution = alns.run(iterations, by_iterations=True)
    return instance, solution


def test_just_in_time_on_the_tiny_optimal_plan():
    tiny = load_instance(TINY)
    q, y = jit_quantities(tiny_optimal_plan(), tiny)
    # A needs 10 until its day 1 visit, then 10; B needs 20; C starts with 5 and needs 15 on day 1
    assert q.tolist() == [[10.0, 10.0], [20.0, 0.0], [0.0, 15.0]]
    assert y.tolist() == [[55.0, 0.0]]  # outflow 30 + 25, all on the one trip day


def test_just_in_time_reports_the_first_stockout_day():
    tiny = load_instance(TINY)
    plan = tiny_optimal_plan()
    plan.routes[0][1] = [[A]]  # C is never visited: 5 kg covers day 0, then 15 kg are missing
    _, stockouts = point_quantities(plan, tiny)
    assert stockouts == {C: 1}
    assert jit_quantities(plan, tiny) is None


def test_just_in_time_reports_a_hub_that_runs_dry():
    tiny = load_instance(TINY)
    plan = tiny_optimal_plan()
    plan.truck_trips[0, 0] = False
    plan.truck_trips[0, 1] = True  # the hub starts empty, so the day 0 vans have nothing to load
    q, _ = point_quantities(plan, tiny)
    _, stockouts = hub_quantities(plan, q, tiny)
    assert stockouts == {0: 0}


def test_just_in_time_goes_on_after_a_stockout():
    # visit on day 1 only, no initial stock: day 0 runs out; day 1 still gets its 10 kg
    quantity, stockout = just_in_time([1], np.array([10.0, 10.0]), 0.0, 20.0)
    assert quantity.tolist() == [0.0, 10.0]
    assert stockout == 0


def test_just_in_time_rejects_a_van_overload():
    tiny = load_instance(TINY)
    plan = tiny_optimal_plan()
    plan.routes[0][1] = [[C]]  # A gets all 20 kg on day 0, so the day 0 van carries 20 + 20 > 30
    assert jit_quantities(plan, tiny) is None


def test_stockout_window_starts_after_the_last_earlier_visit():
    assert stockout_window([0, 5], 3) == [1, 2, 3]  # the visit on day 0 did not reach day 3
    assert stockout_window([4], 2) == [0, 1, 2]  # no visit before the stockout
    assert stockout_window([1, 2], 5) == [3, 4, 5]  # days after the day 2 visit


def test_update_weights_mixes_the_old_weight_and_the_mean_score():
    weights = update_weights(
        np.array([1.0, 1.0, 1.0]), np.array([10.0, 0.0, 4.0]), np.array([2, 0, 4]), reaction=0.7
    )
    # 1 x 0.3 + 0.7 x 10 / 2 = 3.8; the unused operator keeps 1; 1 x 0.3 + 0.7 x 4 / 4 = 1.0
    assert weights == pytest.approx([3.8, 1.0, 1.0])


def test_start_temperature_accepts_5_percent_worse_with_probability_one_half():
    params = load_params()
    cost = 1000.0
    worse = params.start_worse * cost  # 50 BRL above the start cost
    assert np.exp(-worse / start_temperature(cost, params)) == pytest.approx(0.5)


def test_accept_takes_every_better_cost_and_no_far_worse_cost():
    np.random.seed(1)
    assert all(accept(-1e-9, 1e-12) for _ in range(100))
    assert not any(accept(1.0, 1e-12) for _ in range(100))  # exp(-1e12) is 0


def test_removal_count_stays_within_the_caps():
    params = load_params()
    few = Plan(truck_trips=np.zeros((1, 1), dtype=bool), routes=[[[[0]]]])
    many = Plan(truck_trips=np.zeros((1, 1), dtype=bool), routes=[[[list(range(1000))]]])
    np.random.seed(1)
    # 10 % of 1 visit rounds to 0, so the lower cap 2 applies; 10 % of 1000 is capped at 30
    assert {removal_count(few, params) for _ in range(200)} == {1, 2}
    assert {removal_count(many, params) for _ in range(2000)} == set(range(1, 31))


def test_two_opt_uncrosses_a_square():
    tiny = load_instance(TINY)
    corners = np.array([[50.0, 50.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    km = np.sqrt(np.sum(np.square(corners[:, None, :] - corners[None, :, :]), axis=2))
    square = dataclasses.replace(tiny, coordinates=corners, distance_km=km)
    crossed = [A, C, B]  # hub (0,0) -> (1,0) -> (0,1) -> (1,1) -> hub: the middle legs cross
    assert van_route_km(0, crossed, square) == pytest.approx(2 + 2 * np.sqrt(2))
    route = two_opt(0, crossed, square)
    assert route in ([A, B, C], [C, B, A])
    assert van_route_km(0, route, square) == pytest.approx(4.0)  # the perimeter


def test_alns_reaches_the_tiny_optimum():
    tiny, solution = run_alns(TINY, seed=1, iterations=2000)
    # golden: the MIP and a full enumeration of the 102 feasible plans agree on 303
    assert solution_cost(solution, tiny).total == pytest.approx(303.0, abs=1e-9)


def test_alns_reaches_the_proven_optimum_of_a_small_instance():
    instance, solution = run_alns(INSTANCES / "small_k2_n6_T3_s1.json", seed=1, iterations=2000)
    # golden: Gurobi proves 3754.118835 with model.py and with a second formulation written apart
    assert solution_cost(solution, instance).total == pytest.approx(3754.118835, abs=1e-6)


@pytest.mark.parametrize(
    ("name", "iterations"),
    [
        ("small_k1_n6_T3_s1", 300),
        ("small_k2_n10_T3_s2", 300),
        ("medium_k3_n40_T7_s2", 100),
        ("medium_k3_n80_T7_s1", 50),
        ("large_k5_n200_T7_s1", 30),
    ],
)
def test_alns_results_pass_the_checker(name, iterations):
    instance, solution = run_alns(INSTANCES / f"{name}.json", seed=2, iterations=iterations)
    assert check_solution(solution, instance) == []


def test_alns_is_reproducible_under_an_iteration_cap():
    path = INSTANCES / "small_k2_n10_T3_s1.json"
    instance, first = run_alns(path, seed=3, iterations=300)
    _, second = run_alns(path, seed=3, iterations=300)
    assert solution_cost(first, instance).total == solution_cost(second, instance).total
    assert first.van_routes == second.van_routes
    assert first.truck_trips.tolist() == second.truck_trips.tolist()
    assert first.deliveries.tolist() == second.deliveries.tolist()


def test_alns_is_reproducible_on_a_medium_instance():
    path = INSTANCES / "medium_k3_n40_T7_s1.json"
    instance, first = run_alns(path, seed=5, iterations=150)
    _, second = run_alns(path, seed=5, iterations=150)
    assert solution_cost(first, instance).total == solution_cost(second, instance).total
    assert first.van_routes == second.van_routes
