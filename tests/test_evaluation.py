"""Tests for the one checker: the tiny optimum passes and each broken solution is caught."""

import dataclasses

import numpy as np
import pytest

from irp2e.evaluation import assert_feasible, check_reported_cost, check_solution, solution_cost
from irp2e.instance import load_instance
from irp2e.solution import Solution, load_solution, save_solution

from tiny_example import TINY, A, B, C, tiny_optimum


@pytest.fixture
def tiny():
    return load_instance(TINY)


def test_tiny_optimum_is_feasible_and_costs_303(tiny):
    assert check_solution(tiny_optimum(), tiny) == []
    cost = assert_feasible(tiny_optimum(), tiny)
    assert cost.truck == 250.0  # 50 per trip + 2 x 100 km x 1 BRL
    assert cost.van == 28.0  # H-A-B-H 4 + 3 + 6 = 13, H-A-C-H 4 + 6 + 5 = 15
    assert cost.holding == 25.0  # hub keeps 55 - 30 = 25 kg at the end of day 0
    assert cost.total == 303.0


def test_checker_catches_a_negative_delivery(tiny):
    solution = tiny_optimum()
    solution.deliveries[B, 1] = -1.0  # B is not visited on day 1, so only the sign can catch it
    assert "negative quantity: point 1, day 1" in check_solution(solution, tiny)


def test_checker_catches_a_stockout(tiny):
    solution = tiny_optimum()
    solution.deliveries[A, 0] = 9.0  # A needs 10 on day 0
    assert "stockout: point 0, day 0: stock -1.0000 kg" in check_solution(solution, tiny)


def test_checker_catches_point_over_capacity(tiny):
    solution = tiny_optimum()
    solution.deliveries[A, 0] = 21.0  # capacity of A is 20
    assert "point maximum level: point 0, day 0" in check_solution(solution, tiny)


def test_checker_catches_a_van_overload(tiny):
    solution = tiny_optimum()
    solution.van_routes[0][0] = [[A, B, C]]
    solution.deliveries[C, 0] = 1.0
    solution.deliveries[C, 1] = 14.0  # day 0 load 10 + 20 + 1 = 31 > 30
    problems = check_solution(solution, tiny)
    assert "van capacity: hub 0, day 0, route 0: 31.0000 kg" in problems
    assert len(problems) == 1


def test_checker_catches_a_delivery_without_a_visit(tiny):
    solution = tiny_optimum()
    solution.deliveries[C, 0] = 1.0  # C is not on the day 0 route
    solution.deliveries[C, 1] = 14.0
    problems = check_solution(solution, tiny)
    assert "delivery without visit: point 2, day 0" in problems
    assert len(problems) == 1


def test_checker_catches_a_shipment_without_a_truck_trip(tiny):
    solution = tiny_optimum()
    solution.hub_shipments[0, 1] = 5.0  # no trip on day 1
    problems = check_solution(solution, tiny)
    assert "shipment without truck trip: hub 0, day 1" in problems
    assert len(problems) == 1


def test_checker_catches_a_hub_running_dry(tiny):
    solution = tiny_optimum()
    solution.hub_shipments[0, 0] = 40.0  # 40 - 30 = 10 kg left, day 1 needs 25
    problems = check_solution(solution, tiny)
    assert "hub runs dry: hub 0, day 1: stock -15.0000 kg" in problems
    assert len(problems) == 1


def test_checker_catches_hub_over_capacity(tiny):
    solution = tiny_optimum()
    solution.hub_shipments[0, 0] = 61.0  # hub capacity is 60
    problems = check_solution(solution, tiny)
    assert "hub maximum level: hub 0, day 0" in problems
    assert len(problems) == 1


def test_checker_catches_a_truck_over_capacity(tiny):
    small_truck = dataclasses.replace(tiny, truck_capacity=50.0)  # the optimum ships 55
    problems = check_solution(tiny_optimum(), small_truck)
    assert "truck capacity: hub 0, day 0: 55.0000 kg" in problems
    assert len(problems) == 1


def test_checker_catches_a_point_visited_twice_in_a_day(tiny):
    solution = tiny_optimum()
    solution.van_routes[0][0] = [[A, B], [A]]
    assert "visited twice: point 0, day 0" in check_solution(solution, tiny)


def test_checker_catches_too_many_routes(tiny):
    solution = tiny_optimum()
    solution.van_routes[0][0] = [[A], [B], [C]]  # C gets 0 kg; only 2 vans exist
    problems = check_solution(solution, tiny)
    assert "too many routes: hub 0, day 0: 3 routes" in problems
    assert len(problems) == 1


def test_checker_catches_an_empty_route(tiny):
    solution = tiny_optimum()
    solution.van_routes[0][1] = [[A, C], []]  # a van that leaves the hub and visits nobody
    problems = check_solution(solution, tiny)
    assert "empty route: hub 0, day 1" in problems
    assert len(problems) == 1


def test_checker_catches_a_bad_point_index(tiny):
    solution = tiny_optimum()
    solution.van_routes[0][1] = [[A, 3]]  # the tiny instance has points 0, 1 and 2
    assert check_solution(solution, tiny) == [
        "shape: hub 0, day 1: route [0, 3] has a bad point index"
    ]


def test_checker_catches_a_point_on_the_wrong_hub(tiny):
    # Hub G (index 1) sits where H is and owns C; A and B stay with hub H (index 0).
    old = [0, 1, 1, 2, 3, 4]  # new nodes W, H, G, A, B, C taken from old nodes W, H, H, A, B, C
    km = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            km[a, b] = tiny.distance_km[old[a], old[b]]
    two_hubs = dataclasses.replace(
        tiny,
        n_hubs=2,
        labels=["W", "H", "G", "A", "B", "C"],
        cities=["W", "H", "G", "A", "B", "C"],
        coordinates=np.zeros((6, 2)),
        hub_of=np.array([0, 0, 1]),
        hub_capacity=np.array([60.0, 60.0]),
        hub_initial=np.array([0.0, 0.0]),
        distance_km=km,
    )
    solution = Solution(
        truck_trips=np.array([[True, False], [False, True]]),
        hub_shipments=np.array([[40.0, 0.0], [0.0, 15.0]]),
        deliveries=np.array([[10.0, 10.0], [20.0, 0.0], [0.0, 15.0]]),
        van_routes=[[[[A, B]], []], [[], [[C, A]]]],  # hub G visits A on day 1
    )
    assert "wrong hub: hub 1, day 1: point 0 belongs to hub 0" in check_solution(solution, two_hubs)


def test_checker_catches_a_wrong_reported_cost(tiny):
    cost = solution_cost(tiny_optimum(), tiny)
    assert check_reported_cost(303.0, cost) == []
    assert check_reported_cost(302.0, cost) == [
        "reported cost: 302.0 differs from the recomputed 303.0"
    ]


def test_checker_rejects_a_malformed_solution(tiny):
    solution = tiny_optimum()
    solution.deliveries = solution.deliveries[:2]  # 2 rows for 3 points
    assert check_solution(solution, tiny) == ["shape: deliveries must be (3, 2)"]


def test_assert_feasible_lists_the_violation(tiny):
    solution = tiny_optimum()
    solution.deliveries[A, 0] = 9.0
    with pytest.raises(ValueError, match="stockout: point 0, day 0"):
        assert_feasible(solution, tiny)


def test_solution_json_round_trip_keeps_feasibility_and_cost(tiny, tmp_path):
    path = tmp_path / "tiny__gurobi.json"
    save_solution(tiny_optimum(), {"instance": "tiny", "method": "gurobi", "seed": None}, path)
    solution, meta = load_solution(path)
    assert meta == {"instance": "tiny", "method": "gurobi", "seed": None}
    assert solution.van_routes == tiny_optimum().van_routes
    assert assert_feasible(solution, tiny).total == 303.0
