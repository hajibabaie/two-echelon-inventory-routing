"""The Plan (van routes and truck trips without quantities), the search state of the ALNS."""

from dataclasses import dataclass

import numpy as np

from irp2e.evaluation import TOL_KG
from irp2e.instance import Instance
from irp2e.solution import Solution


@dataclass(eq=False)
class Plan:
    truck_trips: np.ndarray  # (k, T) bool: a truck drives warehouse -> h -> warehouse on day t
    routes: list[list[list[list[int]]]]  # [h][t] -> list of routes; a route lists points


def empty_plan(instance: Instance) -> Plan:
    """A plan with no truck trip and no route."""
    return Plan(
        truck_trips=np.zeros((instance.n_hubs, instance.n_days), dtype=bool),
        routes=[[[] for _ in range(instance.n_days)] for _ in range(instance.n_hubs)],
    )


def copy_plan(plan: Plan) -> Plan:
    """A deep copy, so a candidate never changes the current plan."""
    return Plan(
        truck_trips=plan.truck_trips.copy(),
        routes=[[[list(route) for route in day] for day in hub] for hub in plan.routes],
    )


def visit_days(plan: Plan, instance: Instance) -> list[list[int]]:
    """Days each point is visited, ascending."""
    days = [[] for _ in range(instance.n_points)]
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            for route in plan.routes[h][t]:
                for i in route:
                    days[i].append(t)
    return [sorted(point_days) for point_days in days]


def trip_days(plan: Plan, instance: Instance) -> list[list[int]]:
    """Days each hub gets a truck trip, ascending."""
    return [
        [t for t in range(instance.n_days) if plan.truck_trips[h, t]]
        for h in range(instance.n_hubs)
    ]


def count_visits(plan: Plan) -> int:
    """Number of (point, day) visits in the plan."""
    return sum(len(route) for hub in plan.routes for day in hub for route in day)


def drop_empty_stops(plan: Plan, q: np.ndarray, y: np.ndarray) -> None:
    """Remove visits with no delivery, routes left empty, and truck trips with no shipment."""
    for h in range(len(plan.routes)):
        for t in range(len(plan.routes[h])):
            kept = [[i for i in route if q[i, t] > TOL_KG] for route in plan.routes[h][t]]
            plan.routes[h][t] = [route for route in kept if len(route) > 0]
            if plan.truck_trips[h, t] and y[h, t] <= TOL_KG:
                plan.truck_trips[h, t] = False


def plan_to_solution(plan: Plan, q: np.ndarray, y: np.ndarray) -> Solution:
    """The solution made of this plan and these quantities."""
    return Solution(
        truck_trips=plan.truck_trips.copy(),
        hub_shipments=y.copy(),
        deliveries=q.copy(),
        van_routes=[[[list(route) for route in day] for day in hub] for hub in plan.routes],
    )
