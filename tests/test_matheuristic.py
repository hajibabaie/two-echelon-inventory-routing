"""Tests for the matheuristic: the quantity LP, rule M1, and the tiny optimum against Gurobi."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from irp2e.config import load_params  # noqa: E402
from irp2e.evaluation import check_solution, solution_cost  # noqa: E402
from irp2e.heuristics.alns import Alns, initial_plan  # noqa: E402
from irp2e.heuristics.matheuristic import (  # noqa: E402
    QuantityLp,
    RescueQuantities,
    make_matheuristic,
)
from irp2e.heuristics.plan import Plan, plan_to_solution  # noqa: E402
from irp2e.heuristics.quantities import jit_quantities  # noqa: E402
from irp2e.instance import load_instance  # noqa: E402
from irp2e.model import solve_mip  # noqa: E402

from tiny_example import INSTANCES, TINY, A, B, C, tiny_optimal_plan  # noqa: E402

pytestmark = pytest.mark.gurobi


def random_tiny_plan(tiny) -> Plan:
    """Random truck days and random visits, split over at most two vans per day."""
    plan = Plan(truck_trips=np.random.rand(1, 2) < 0.5, routes=[[[], []]])
    for t in range(2):
        visited = [i for i in range(3) if np.random.rand() < 0.6]
        cut = np.random.randint(0, len(visited) + 1)
        plan.routes[0][t] = [route for route in (visited[:cut], visited[cut:]) if len(route) > 0]
    return plan


def plan_cost(plan: Plan, quantities, instance) -> float:
    """Total cost of a plan with the given (q, y)."""
    q, y = quantities
    return solution_cost(plan_to_solution(plan, q, y), instance).total


def test_alns_and_matheuristic_reach_the_gurobi_optimum_on_tiny():
    tiny = load_instance(TINY)
    _, mip = solve_mip(tiny, time_limit_s=60, threads=1, mip_gap=0.0)
    assert mip.status == "optimal"
    params = load_params()
    np.random.seed(1)
    alns_solution = Alns(tiny, params, jit_quantities).run(2000, by_iterations=True)
    np.random.seed(1)
    matheuristic_solution = make_matheuristic(tiny, params).run(2000, by_iterations=True)
    assert solution_cost(alns_solution, tiny).total == pytest.approx(mip.objective, abs=1e-6)
    assert solution_cost(matheuristic_solution, tiny).total == pytest.approx(
        mip.objective, abs=1e-6
    )


def test_rescue_step_uses_just_in_time_when_it_is_feasible():
    tiny = load_instance(TINY)
    rescue = RescueQuantities(QuantityLp(tiny, threads=1))
    q, y = rescue(tiny_optimal_plan(), tiny)
    assert q.tolist() == [[10.0, 10.0], [20.0, 0.0], [0.0, 15.0]]
    assert y.tolist() == [[55.0, 0.0]]
    assert rescue.lp_calls == 0


def test_lp_never_worsens_a_tiny_plan_that_just_in_time_can_serve():
    tiny = load_instance(TINY)
    lp = QuantityLp(tiny, threads=1)
    np.random.seed(0)
    compared = 0
    while compared < 50:
        plan = random_tiny_plan(tiny)
        jit = jit_quantities(plan, tiny)
        if jit is None:
            continue
        compared += 1
        lp_quantities = lp.solve(plan)
        assert check_solution(plan_to_solution(plan, *lp_quantities), tiny) == []
        # the LP is optimal for the plan, so never worse; J1 is minimal (design 6.3), so equal
        assert plan_cost(plan, lp_quantities, tiny) == pytest.approx(
            plan_cost(plan, jit, tiny), abs=1e-6
        )


@pytest.mark.parametrize(
    "name", ["small_k2_n10_T3_s1", "medium_k3_n80_T7_s2", "large_k5_n200_T7_s2"]
)
def test_lp_equals_just_in_time_on_the_start_plan_of_a_real_instance(name):
    instance = load_instance(INSTANCES / f"{name}.json")
    plan = initial_plan(instance)
    jit = jit_quantities(plan, instance)
    lp_quantities = QuantityLp(instance, threads=1).solve(plan)
    assert check_solution(plan_to_solution(plan, *lp_quantities), instance) == []
    # point holding equals hub holding, so J1 is minimal (design 6.3): the LP must equal it
    assert plan_cost(plan, lp_quantities, instance) == pytest.approx(
        plan_cost(plan, jit, instance), abs=1e-6
    )


def test_lp_rescues_a_plan_that_just_in_time_overloads():
    tiny = load_instance(TINY)
    small_vans = dataclasses.replace(tiny, van_capacity=22.0)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A], [B, C]], [[A, C]]]])
    assert jit_quantities(plan, small_vans) is None  # day 1 route carries A 10 + C 15 = 25 > 22
    rescue = RescueQuantities(QuantityLp(small_vans, threads=1))
    q, y = rescue(plan, small_vans)
    solution = plan_to_solution(plan, q, y)
    assert check_solution(solution, small_vans) == []
    # golden holding 25: machine-computed by an LP over all quantities (design section 11.4)
    assert solution_cost(solution, small_vans).holding == pytest.approx(25.0, abs=1e-6)
    assert (rescue.lp_calls, rescue.lp_rescues) == (1, 1)


def test_lp_keeps_every_route_within_one_van():
    tiny = load_instance(TINY)
    # free hub stock makes the overloaded just-in-time answer the cheapest one without route rows
    free_hub = dataclasses.replace(tiny, van_capacity=22.0, hub_holding=0.0)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[A], [B, C]], [[A, C]]]])
    q, y = QuantityLp(free_hub, threads=1).solve(plan)
    assert check_solution(plan_to_solution(plan, q, y), free_hub) == []


def test_matheuristic_result_passes_the_checker_on_a_large_instance():
    instance = load_instance(INSTANCES / "large_k5_n200_T7_s3.json")
    np.random.seed(4)
    solution = make_matheuristic(instance, load_params()).run(30, by_iterations=True)
    assert check_solution(solution, instance) == []
