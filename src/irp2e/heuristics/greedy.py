"""Greedy baseline: deliver when stock would run out, fill up, route by nearest neighbor."""

import numpy as np

from irp2e.evaluation import TOL_KG
from irp2e.instance import Instance
from irp2e.solution import Solution


def fill_up(need: np.ndarray, initial: float, capacity: float) -> np.ndarray:
    """Rules G1 and G3 for one stock: refill only when today's need is not covered."""
    quantity = np.zeros(len(need))
    stock = initial
    for t in range(len(need)):
        if stock < need[t] - TOL_KG:  # float sums leave 1e-15 kg of noise; data is in whole grams
            quantity[t] = min(capacity, float(np.sum(need[t:]))) - stock
        stock = stock + quantity[t] - need[t]
    return quantity


def fill_up_quantities(instance: Instance) -> np.ndarray:
    """(n, T) deliveries to the points by rule G1."""
    q = np.zeros((instance.n_points, instance.n_days))
    for i in range(instance.n_points):
        q[i, :] = fill_up(
            instance.demand[i, :], instance.point_initial[i], instance.point_capacity[i]
        )
    return q


def hub_fill_up(q: np.ndarray, instance: Instance) -> np.ndarray:
    """(k, T) truck shipments to the hubs by rule G3."""
    y = np.zeros((instance.n_hubs, instance.n_days))
    for h in range(instance.n_hubs):
        outflow = np.sum(q[instance.points_of[h], :], axis=0)
        y[h, :] = fill_up(outflow, instance.hub_initial[h], instance.hub_capacity[h])
    return y


def nearest_fitting_point(
    node: int, candidates: list[int], loads: np.ndarray, room: float, instance: Instance
) -> int | None:
    """The closest candidate whose load fits in the room left; ties go to the lower index."""
    best = None
    best_km = np.inf
    for i in candidates:  # candidates are ascending, so strict < keeps the lower index on a tie
        km = instance.distance_km[node, instance.point_node(i)]
        if loads[i] <= room and km < best_km:
            best = i
            best_km = km
    return best


def nearest_neighbor_routes(
    hub: int, points: list[int], loads: np.ndarray, instance: Instance
) -> list[list[int]]:
    """Rule G2: fill one van by nearest neighbor, then start the next van."""
    unrouted = sorted(points)
    routes = []
    while len(unrouted) > 0:
        route = []
        node = instance.hub_node(hub)
        room = instance.van_capacity
        following = nearest_fitting_point(node, unrouted, loads, room, instance)
        while following is not None:
            route.append(following)
            unrouted.remove(following)
            node = instance.point_node(following)
            room -= loads[following]
            following = nearest_fitting_point(node, unrouted, loads, room, instance)
        routes.append(route)
    return routes


def solve_greedy(instance: Instance) -> Solution:
    """Greedy solution by rules G1 to G4."""
    q = fill_up_quantities(instance)
    y = hub_fill_up(q, instance)
    truck_trips = np.zeros((instance.n_hubs, instance.n_days), dtype=bool)
    van_routes = [[[] for _ in range(instance.n_days)] for _ in range(instance.n_hubs)]
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            truck_trips[h, t] = y[h, t] > 0  # rule G4
            served = [i for i in instance.points_of[h] if q[i, t] > 0]
            van_routes[h][t] = nearest_neighbor_routes(h, served, q[:, t], instance)
    return Solution(truck_trips=truck_trips, hub_shipments=y, deliveries=q, van_routes=van_routes)
