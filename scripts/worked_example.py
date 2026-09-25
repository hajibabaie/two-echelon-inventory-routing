"""Solve the tiny example (1 hub, 3 points, 2 days) with all four methods and print every number."""

import dataclasses
import itertools

import numpy as np

from irp2e.config import REPO_ROOT, load_params
from irp2e.evaluation import check_solution, hub_stock, point_stock, solution_cost, van_route_km
from irp2e.heuristics.alns import Alns
from irp2e.heuristics.greedy import solve_greedy
from irp2e.heuristics.matheuristic import QuantityLp, make_matheuristic
from irp2e.heuristics.plan import Plan
from irp2e.heuristics.quantities import jit_quantities, point_quantities, route_loads
from irp2e.instance import Instance, load_instance
from irp2e.model import solve_mip
from irp2e.solution import Solution

TINY = REPO_ROOT / "tests" / "data" / "tiny.json"
SEED = 1
ITERATIONS = 2000
DAY_SETS = [[], [0], [1], [0, 1]]  # every set of days of the 2-day example


def print_instance(instance: Instance) -> None:
    """Demand, capacities, stocks, distances and costs of the instance."""
    print("INSTANCE")
    for i in range(instance.n_points):
        print(
            f"  point {instance.labels[instance.point_node(i)]}: "
            f"demand {instance.demand[i].tolist()} kg, "
            f"capacity {instance.point_capacity[i]} kg, initial {instance.point_initial[i]} kg"
        )
    print(f"  hub H: capacity {instance.hub_capacity[0]} kg, initial {instance.hub_initial[0]} kg")
    print(f"  km: {instance.labels} {instance.distance_km.tolist()}")
    print(
        f"  truck: {instance.truck_capacity} kg, {instance.truck_cost_per_km} BRL/km, "
        f"{instance.truck_cost_per_trip} BRL/trip; one trip W-H-W costs {instance.trip_cost(0)}"
    )
    print(
        f"  vans: {instance.n_vans} of {instance.van_capacity} kg, "
        f"{instance.van_cost_per_km} BRL/km; holding {instance.hub_holding} (hub), "
        f"{instance.point_holding} (points) BRL/kg/day"
    )


def print_solution(title: str, solution: Solution, instance: Instance) -> None:
    """Deliveries, routes, truck trips, stocks, cost parts and the checker verdict."""
    print(title)
    for t in range(instance.n_days):
        trip = "yes" if solution.truck_trips[0, t] else "no"
        print(f"  day {t}: truck trip {trip}, hub shipment {solution.hub_shipments[0, t]:g} kg")
        for route in solution.van_routes[0][t]:
            stops = "-".join(instance.labels[instance.point_node(i)] for i in route)
            load = np.sum(solution.deliveries[route, t])
            km = van_route_km(0, route, instance)
            print(f"    route H-{stops}-H: {km:g} km, load {load:g} kg")
    for i in range(instance.n_points):
        print(
            f"  deliveries to {instance.labels[instance.point_node(i)]}: "
            f"{solution.deliveries[i].tolist()} kg"
        )
    print(f"  end-of-day stock hub: {hub_stock(solution, instance)[0].tolist()} kg")
    print(f"  end-of-day stock points: {point_stock(solution, instance).tolist()} kg")
    cost = solution_cost(solution, instance)
    print(
        f"  cost: truck {cost.truck:g} + van {cost.van:g} + holding {cost.holding:g} "
        f"= {cost.total:g} BRL"
    )
    problems = check_solution(solution, instance)
    print(f"  checker: {'feasible' if len(problems) == 0 else problems}")


def lp_rescue_case(instance: Instance) -> None:
    """Vans cut to 22 kg: J1 overloads the day 1 route, and the LP finds other quantities."""
    small_vans = dataclasses.replace(instance, van_capacity=22.0)
    plan = Plan(truck_trips=np.array([[True, False]]), routes=[[[[0], [1, 2]], [[0, 2]]]])
    print("LP RESCUE CASE (vans of 22 kg; truck day 0; day 0 [A] and [B, C]; day 1 [A, C])")
    q, _ = point_quantities(plan, small_vans)
    print(f"  J1 deliveries: {q.tolist()} kg; route loads {route_loads(plan, q)} kg")
    print(f"  J1 result: {jit_quantities(plan, small_vans)}")
    lp = QuantityLp(small_vans, threads=1)
    q, y = lp.solve(plan)
    lp.close()
    solution = Solution(
        truck_trips=plan.truck_trips, hub_shipments=y, deliveries=q, van_routes=plan.routes
    )
    print_solution("  LP quantities:", solution, small_vans)


def splits(points: list[int], max_routes: int) -> list[list[list[int]]]:
    """Every way to split the points into at most max_routes non-empty groups."""
    if len(points) == 0:
        return [[]]
    result = []
    for rest in splits(points[1:], max_routes):
        for k in range(len(rest)):
            result.append(rest[:k] + [[points[0]] + rest[k]] + rest[k + 1 :])
        if len(rest) < max_routes:
            result.append([[points[0]]] + rest)
    return result


def shortest_order(group: list[int], instance: Instance) -> list[int]:
    """The visiting order of a group with the fewest km."""
    orders = [list(order) for order in itertools.permutations(group)]
    return min(orders, key=lambda order: van_route_km(0, order, instance))


def enumerate_plans(instance: Instance) -> list[float]:
    """Total cost of every feasible plan (truck days, visit days, route split), LP quantities."""
    lp = QuantityLp(instance, threads=1)
    costs = []
    for truck_days in DAY_SETS:
        trips = np.array([[t in truck_days for t in range(instance.n_days)]])
        for visits in itertools.product(DAY_SETS, repeat=instance.n_points):
            by_day = [
                [i for i in range(instance.n_points) if t in visits[i]]
                for t in range(instance.n_days)
            ]
            day_splits = [splits(points, instance.n_vans) for points in by_day]
            for split in itertools.product(*day_splits):
                routes = [[shortest_order(group, instance) for group in day] for day in split]
                quantities = lp.solve(Plan(truck_trips=trips, routes=[routes]))
                if quantities is not None:
                    solution = Solution(
                        truck_trips=trips,
                        hub_shipments=quantities[1],
                        deliveries=quantities[0],
                        van_routes=[routes],
                    )
                    costs.append(solution_cost(solution, instance).total)
    lp.close()
    return sorted(costs)


if __name__ == "__main__":
    params = load_params()
    instance = load_instance(TINY)
    print_instance(instance)

    print_solution("GREEDY", solve_greedy(instance), instance)

    solution, result = solve_mip(instance, params.gurobi_time_limit_s, 1, params.mip_gap)
    print_solution(
        f"GUROBI MIP (status {result.status}, bound {result.bound:g})", solution, instance
    )

    np.random.seed(SEED)
    alns = Alns(instance, params, jit_quantities)
    solution = alns.run(ITERATIONS, by_iterations=True)
    best_costs = [round(cost, 6) for _, cost in alns.history]
    print_solution(
        f"ALNS (seed {SEED}, {ITERATIONS} iterations; start cost {best_costs[0]:g}, "
        f"best costs found {best_costs})",
        solution,
        instance,
    )

    np.random.seed(SEED)
    matheuristic = make_matheuristic(instance, params)
    solution = matheuristic.run(ITERATIONS, by_iterations=True)
    matheuristic.quantity_step.lp.close()
    best_costs = [round(cost, 6) for _, cost in matheuristic.history]
    print_solution(
        f"MATHEURISTIC (seed {SEED}, {ITERATIONS} iterations; best costs found {best_costs}; "
        f"LP calls {matheuristic.quantity_step.lp_calls}, "
        f"LP rescues {matheuristic.quantity_step.lp_rescues})",
        solution,
        instance,
    )

    costs = enumerate_plans(instance)
    print(f"ENUMERATION: {len(costs)} feasible plans; the three cheapest costs {costs[:3]}")

    lp_rescue_case(instance)
