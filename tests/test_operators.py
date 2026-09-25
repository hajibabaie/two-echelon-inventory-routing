"""Tests for the ALNS operators: what each destroy removes, empty plans, and the repair loads."""

import dataclasses

import numpy as np
import pytest

from irp2e.config import load_params
from irp2e.evaluation import check_solution
from irp2e.heuristics.alns import Alns
from irp2e.heuristics.operators import (
    DESTROY_OPERATORS,
    greedy_repair,
    hub_day_removal,
    random_removal,
    related_removal,
    sequence_removal,
    truck_trip_removal,
    worst_removal,
)
from irp2e.heuristics.plan import Plan, count_visits, empty_plan
from irp2e.heuristics.quantities import jit_quantities
from irp2e.instance import load_instance

from tiny_example import TINY, A, B, C


@pytest.mark.parametrize("name, destroy", DESTROY_OPERATORS, ids=[n for n, _ in DESTROY_OPERATORS])
def test_destroy_on_an_empty_plan_removes_nothing(name, destroy):
    tiny = load_instance(TINY)
    plan = empty_plan(tiny)
    np.random.seed(1)
    assert destroy(plan, tiny, 2) == ([], [])
    assert plan.routes == [[[], []]]
    assert not np.any(plan.truck_trips)


def test_alns_runs_when_the_hub_stock_covers_the_horizon():
    # 60 kg at the hub cover the 55 kg the points need, so no truck trip is ever planned
    tiny = dataclasses.replace(load_instance(TINY), hub_initial=np.array([60.0]))
    np.random.seed(1)
    solution = Alns(tiny, load_params(), jit_quantities).run(200, by_iterations=True)
    assert check_solution(solution, tiny) == []
    assert not np.any(solution.truck_trips)


def test_repair_counts_the_later_visit_of_a_needy_point():
    # vans of 22 kg; A (visited on day 1 only) runs out on day 0, C (never visited) on day 1.
    # A's day 1 visit carries 10 kg, so C (15 kg on day 1) must not join that route.
    tiny = dataclasses.replace(load_instance(TINY), van_capacity=22.0)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[B]], [[A]]]])
    assert greedy_repair(plan, ([], []), tiny, 1)
    assert plan.routes == [[[[B], [A]], [[A], [C]]]]
    assert jit_quantities(plan, tiny) is not None


@pytest.mark.parametrize("seed", range(5))
def test_random_removal_removes_rho_visits(seed):
    tiny = load_instance(TINY)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A, B]], [[A, C]]]])
    np.random.seed(seed)
    random_removal(plan, tiny, 3)
    assert count_visits(plan) == 1  # 4 visits before


@pytest.mark.parametrize("seed", range(5))
def test_worst_removal_picks_among_the_two_largest_savings(seed):
    tiny = load_instance(TINY)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A, B]], [[A, C]]]])
    np.random.seed(seed)
    points, _ = worst_removal(plan, tiny, 1)
    # km saved: day 0 A 4 + 3 - 6 = 1, B 3 + 6 - 4 = 5; day 1 A 4 + 6 - 5 = 5, C 6 + 5 - 4 = 7.
    # The top two are C (7) and, of the two 5s, B on day 0 (the tie goes to the earlier day).
    assert (points, plan.routes[0]) in (([C], [[[A, B]], [[A]]]), ([B], [[[A]], [[A, C]]]))


def test_truck_trip_removal_removes_one_trip():
    tiny = load_instance(TINY)
    plan = Plan(truck_trips=np.array([[True, True]]), routes=[[[[A, B]], [[A, C]]]])
    np.random.seed(1)
    points, hubs = truck_trip_removal(plan, tiny, 2)
    assert (points, hubs) == ([], [0])
    assert int(np.sum(plan.truck_trips)) == 1
    assert plan.routes == [[[[A, B]], [[A, C]]]]


@pytest.mark.parametrize("seed", range(10))
def test_sequence_removal_removes_consecutive_stops(seed):
    tiny = load_instance(TINY)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A, B, C]], []]])
    np.random.seed(seed)
    points, _ = sequence_removal(plan, tiny, 2)
    # a run of two out of A, B, C is A-B or B-C, never A-C
    assert (points, plan.routes[0][0]) in (([A, B], [[C]]), ([B, C], [[A]]))


def test_hub_day_removal_empties_one_hub_day():
    tiny = load_instance(TINY)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A, B]], [[A, C]]]])
    np.random.seed(1)
    points, hubs = hub_day_removal(plan, tiny, 1)
    assert hubs == [0]
    assert (points, plan.routes[0]) in (([A, B], [[], [[A, C]]]), ([A, C], [[[A, B]], []]))


def test_related_removal_takes_the_closest_visits_of_the_hub():
    tiny = load_instance(TINY)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A, B]], [[C]]]])
    np.random.seed(1)
    points, _ = related_removal(plan, tiny, 2)
    # km A-B 3, A-C 6, B-C 4: from A the closest two are A, B; from B: B, A; from C: C, B
    assert points in ([A, B], [B, C])
