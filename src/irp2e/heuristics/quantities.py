"""Quantity rule J1 (just in time): deliver on each visit what is needed until the next visit."""

import numpy as np

from irp2e.evaluation import TOL_KG
from irp2e.heuristics.plan import Plan, trip_days, visit_days
from irp2e.instance import Instance


def just_in_time(
    days: list[int], need: np.ndarray, initial: float, capacity: float
) -> tuple[np.ndarray, int | None]:
    """Rule J1 for one stock: quantity per day and the first stockout day (None if none)."""
    need_kg = need.tolist()  # plain floats: this runs for every point in every iteration
    quantity = np.zeros(len(need_kg))
    ends = days[1:] + [len(need_kg)]  # a visit covers the days up to the next visit
    stock = float(initial)
    first_stockout = None
    visit = 0
    for t in range(len(need_kg)):
        if visit < len(days) and days[visit] == t:
            window_need = sum(need_kg[t : ends[visit]])
            delivery = max(0.0, min(window_need - stock, capacity - stock))
            quantity[t] = delivery
            stock += delivery
            visit += 1
        stock -= need_kg[t]
        if stock < -TOL_KG:
            if first_stockout is None:
                first_stockout = t
            # go on as if the shortfall were filled: a visit that fixes it leaves stock >= 0,
            # so every later visit gets at most the kg the repair reads here
            stock = 0.0
    return quantity, first_stockout


def point_just_in_time(
    i: int, days: list[int], instance: Instance
) -> tuple[np.ndarray, int | None]:
    """Rule J1 for point i visited on these days."""
    return just_in_time(
        days, instance.demand[i, :], instance.point_initial[i], instance.point_capacity[i]
    )


def point_quantities(plan: Plan, instance: Instance) -> tuple[np.ndarray, dict[int, int]]:
    """(n, T) deliveries by rule J1 and the first stockout day of each point that runs out."""
    q = np.zeros((instance.n_points, instance.n_days))
    stockouts = {}
    for i, days in enumerate(visit_days(plan, instance)):
        q[i, :], stockout = point_just_in_time(i, days, instance)
        if stockout is not None:
            stockouts[i] = stockout
    return q, stockouts


def hub_quantities(
    plan: Plan, q: np.ndarray, instance: Instance
) -> tuple[np.ndarray, dict[int, int]]:
    """(k, T) shipments by rule J1 on the hub outflow and the first stockout day of each hub."""
    y = np.zeros((instance.n_hubs, instance.n_days))
    stockouts = {}
    for h, days in enumerate(trip_days(plan, instance)):
        outflow = np.sum(q[instance.points_of[h], :], axis=0)
        y[h, :], stockout = just_in_time(
            days, outflow, instance.hub_initial[h], instance.hub_capacity[h]
        )
        if stockout is not None:
            stockouts[h] = stockout
    return y, stockouts


def route_load(route: list[int], q: np.ndarray, t: int) -> float:
    """kg one route carries on day t."""
    return float(np.sum(q[route, t]))


def route_loads(plan: Plan, q: np.ndarray) -> list[float]:
    """kg of every route of the plan."""
    loads = []
    for hub_days in plan.routes:
        for t, routes in enumerate(hub_days):
            for route in routes:
                loads.append(route_load(route, q, t))
    return loads


def jit_quantities(plan: Plan, instance: Instance) -> tuple[np.ndarray, np.ndarray] | None:
    """The ALNS quantity step: (q, y) by rule J1, None on any stockout or van overload."""
    q, point_stockouts = point_quantities(plan, instance)
    if len(point_stockouts) > 0:
        return None
    y, hub_stockouts = hub_quantities(plan, q, instance)
    if len(hub_stockouts) > 0:
        return None
    for load in route_loads(plan, q):
        if load > instance.van_capacity + TOL_KG:
            return None
    return q, y
