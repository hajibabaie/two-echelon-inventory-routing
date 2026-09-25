"""The one solution checker and the one cost function, used by every method and every table."""

from dataclasses import dataclass

import numpy as np

from irp2e.instance import Instance
from irp2e.solution import Solution

TOL_KG = 1e-4  # absolute tolerance on every kg comparison
COST_RTOL = 1e-6  # relative tolerance when a reported cost is compared to the recomputed one


@dataclass(frozen=True)
class Cost:
    truck: float
    van: float
    holding: float
    total: float


def van_route_km(hub: int, route: list[int], instance: Instance) -> float:
    """Length of hub -> points of the route in order -> hub."""
    nodes = [instance.hub_node(hub)]
    nodes += [instance.point_node(i) for i in route]
    nodes += [instance.hub_node(hub)]
    km = 0.0
    for current, following in zip(nodes, nodes[1:], strict=False):
        km += instance.distance_km[current, following]
    return float(km)


def hub_outflow(solution: Solution, instance: Instance) -> np.ndarray:
    """(k, T) kg that leaves each hub by van: the deliveries to its own points."""
    outflow = np.zeros((instance.n_hubs, instance.n_days))
    for i in range(instance.n_points):
        outflow[instance.hub_of[i], :] += solution.deliveries[i, :]
    return outflow


def hub_stock(solution: Solution, instance: Instance) -> np.ndarray:
    """(k, T) hub stock at the end of each day."""
    outflow = hub_outflow(solution, instance)
    stock = np.zeros((instance.n_hubs, instance.n_days))
    level = instance.hub_initial.copy()
    for t in range(instance.n_days):
        level = level + solution.hub_shipments[:, t] - outflow[:, t]
        stock[:, t] = level
    return stock


def point_stock(solution: Solution, instance: Instance) -> np.ndarray:
    """(n, T) point stock at the end of each day."""
    stock = np.zeros((instance.n_points, instance.n_days))
    level = instance.point_initial.copy()
    for t in range(instance.n_days):
        level = level + solution.deliveries[:, t] - instance.demand[:, t]
        stock[:, t] = level
    return stock


def solution_cost(solution: Solution, instance: Instance) -> Cost:
    """Truck trips + van km + holding on end-of-day stock at hubs and points."""
    truck = 0.0
    van_km = 0.0
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            if solution.truck_trips[h, t]:
                truck += instance.trip_cost(h)
            for route in solution.van_routes[h][t]:
                van_km += van_route_km(h, route, instance)
    van = instance.van_cost_per_km * van_km
    holding = float(
        instance.hub_holding * np.sum(hub_stock(solution, instance))
        + instance.point_holding * np.sum(point_stock(solution, instance))
    )
    return Cost(truck=truck, van=van, holding=holding, total=truck + van + holding)


def _is_index(value, size: int) -> bool:
    """True for an integer (not a bool) in 0 .. size - 1."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and 0 <= value < size


def check_shapes(solution: Solution, instance: Instance) -> list[str]:
    """Rule 1: array shapes, finite values, route nesting and point indexes."""
    hub_days = (instance.n_hubs, instance.n_days)
    point_days = (instance.n_points, instance.n_days)
    problems = []
    if solution.truck_trips.dtype != bool or solution.truck_trips.shape != hub_days:
        problems.append(f"shape: truck_trips must be a {hub_days} bool array")
    if solution.hub_shipments.shape != hub_days:
        problems.append(f"shape: hub_shipments must be {hub_days}")
    elif not np.all(np.isfinite(solution.hub_shipments)):
        problems.append("shape: hub_shipments holds a value that is not finite")
    if solution.deliveries.shape != point_days:
        problems.append(f"shape: deliveries must be {point_days}")
    elif not np.all(np.isfinite(solution.deliveries)):
        problems.append("shape: deliveries holds a value that is not finite")
    if len(solution.van_routes) != instance.n_hubs or any(
        len(days) != instance.n_days for days in solution.van_routes
    ):
        problems.append(f"shape: van_routes must be {hub_days[0]} lists of {hub_days[1]} lists")
        return problems
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            for route in solution.van_routes[h][t]:
                if not all(_is_index(i, instance.n_points) for i in route):
                    problems.append(f"shape: hub {h}, day {t}: route {route} has a bad point index")
    return problems


def check_non_negative(solution: Solution, instance: Instance) -> list[str]:
    """Rule 2: no negative shipment or delivery."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            if solution.hub_shipments[h, t] < -TOL_KG:
                problems.append(f"negative quantity: hub {h}, day {t}")
    for i in range(instance.n_points):
        for t in range(instance.n_days):
            if solution.deliveries[i, t] < -TOL_KG:
                problems.append(f"negative quantity: point {i}, day {t}")
    return problems


def check_shipment_on_trip(solution: Solution, instance: Instance) -> list[str]:
    """Rule 3: a hub receives goods only on a day with a truck trip."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            if solution.hub_shipments[h, t] > TOL_KG and not solution.truck_trips[h, t]:
                problems.append(f"shipment without truck trip: hub {h}, day {t}")
    return problems


def check_truck_capacity(solution: Solution, instance: Instance) -> list[str]:
    """Rule 4: one trip carries at most one truck load."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            if solution.hub_shipments[h, t] > instance.truck_capacity + TOL_KG:
                problems.append(
                    f"truck capacity: hub {h}, day {t}: {solution.hub_shipments[h, t]:.4f} kg"
                )
    return problems


def check_hub_max_level(solution: Solution, instance: Instance) -> list[str]:
    """Rule 5: stock before van loading plus the truck delivery fits in the hub."""
    stock = hub_stock(solution, instance)
    problems = []
    for h in range(instance.n_hubs):
        before = instance.hub_initial[h]
        for t in range(instance.n_days):
            if before + solution.hub_shipments[h, t] > instance.hub_capacity[h] + TOL_KG:
                problems.append(f"hub maximum level: hub {h}, day {t}")
            before = stock[h, t]
    return problems


def check_hub_stock(solution: Solution, instance: Instance) -> list[str]:
    """Rule 6: a van cannot load what the hub does not have."""
    stock = hub_stock(solution, instance)
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            if stock[h, t] < -TOL_KG:
                problems.append(f"hub runs dry: hub {h}, day {t}: stock {stock[h, t]:.4f} kg")
    return problems


def check_route_membership(solution: Solution, instance: Instance) -> list[str]:
    """Rule 7: a route of hub h visits only points of hub h and is not empty."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            for route in solution.van_routes[h][t]:
                if len(route) == 0:
                    problems.append(f"empty route: hub {h}, day {t}")
                for i in route:
                    if instance.hub_of[i] != h:
                        problems.append(
                            f"wrong hub: hub {h}, day {t}: point {i} belongs to hub "
                            f"{instance.hub_of[i]}"
                        )
    return problems


def check_one_visit_per_day(solution: Solution, instance: Instance) -> list[str]:
    """Rule 8: a point appears at most once among the routes of its hub on one day."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            seen = set()
            for route in solution.van_routes[h][t]:
                for i in route:
                    if i in seen:
                        problems.append(f"visited twice: point {i}, day {t}")
                    seen.add(i)
    return problems


def check_fleet(solution: Solution, instance: Instance) -> list[str]:
    """Rule 9: at most n_vans routes per hub and day."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            if len(solution.van_routes[h][t]) > instance.n_vans:
                problems.append(
                    f"too many routes: hub {h}, day {t}: {len(solution.van_routes[h][t])} routes"
                )
    return problems


def check_delivery_on_visit(solution: Solution, instance: Instance) -> list[str]:
    """Rule 10: a point receives goods only on a day a route of its hub visits it."""
    problems = []
    for t in range(instance.n_days):
        for i in range(instance.n_points):
            routes = solution.van_routes[instance.hub_of[i]][t]
            visited = any(i in route for route in routes)
            if solution.deliveries[i, t] > TOL_KG and not visited:
                problems.append(f"delivery without visit: point {i}, day {t}")
    return problems


def check_van_capacity(solution: Solution, instance: Instance) -> list[str]:
    """Rule 11: the deliveries of one route fit in one van."""
    problems = []
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            for r, route in enumerate(solution.van_routes[h][t]):
                load = float(np.sum([solution.deliveries[i, t] for i in route]))
                if load > instance.van_capacity + TOL_KG:
                    problems.append(f"van capacity: hub {h}, day {t}, route {r}: {load:.4f} kg")
    return problems


def check_point_max_level(solution: Solution, instance: Instance) -> list[str]:
    """Rule 12: stock before delivery plus the delivery fits in the point."""
    stock = point_stock(solution, instance)
    problems = []
    for i in range(instance.n_points):
        before = instance.point_initial[i]
        for t in range(instance.n_days):
            if before + solution.deliveries[i, t] > instance.point_capacity[i] + TOL_KG:
                problems.append(f"point maximum level: point {i}, day {t}")
            before = stock[i, t]
    return problems


def check_no_stockout(solution: Solution, instance: Instance) -> list[str]:
    """Rule 13: a point never runs out."""
    stock = point_stock(solution, instance)
    problems = []
    for i in range(instance.n_points):
        for t in range(instance.n_days):
            if stock[i, t] < -TOL_KG:
                problems.append(f"stockout: point {i}, day {t}: stock {stock[i, t]:.4f} kg")
    return problems


RULES = [
    check_non_negative,
    check_shipment_on_trip,
    check_truck_capacity,
    check_hub_max_level,
    check_hub_stock,
    check_route_membership,
    check_one_visit_per_day,
    check_fleet,
    check_delivery_on_visit,
    check_van_capacity,
    check_point_max_level,
    check_no_stockout,
]


def check_solution(solution: Solution, instance: Instance) -> list[str]:
    """Every violation as one message; an empty list means feasible."""
    problems = check_shapes(solution, instance)
    if len(problems) > 0:
        return problems  # the other rules cannot read a malformed solution
    for rule in RULES:
        problems += rule(solution, instance)
    return problems


def assert_feasible(solution: Solution, instance: Instance) -> Cost:
    """The cost of a feasible solution; ValueError listing all violations otherwise."""
    problems = check_solution(solution, instance)
    if len(problems) > 0:
        raise ValueError(f"{instance.name}: infeasible solution:\n" + "\n".join(problems))
    return solution_cost(solution, instance)


def check_reported_cost(reported: float, cost: Cost) -> list[str]:
    """Rule 14: a reported total equals the recomputed total within COST_RTOL."""
    if abs(reported - cost.total) > COST_RTOL * max(1.0, abs(cost.total)):
        return [f"reported cost: {reported} differs from the recomputed {cost.total}"]
    return []
