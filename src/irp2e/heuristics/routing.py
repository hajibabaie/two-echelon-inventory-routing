"""Route moves on road km: cheapest insertion, removal saving and 2-opt."""

import numpy as np

from irp2e.instance import Instance

TWO_OPT_MIN_GAIN_KM = 1e-9  # a 2-opt move must shorten the route by more than this


def route_nodes(hub: int, route: list[int], instance: Instance) -> np.ndarray:
    """Node numbers of hub -> points of the route -> hub."""
    points = [instance.point_node(i) for i in route]
    return np.array([instance.hub_node(hub)] + points + [instance.hub_node(hub)], dtype=int)


def best_position(hub: int, route: list[int], point: int, instance: Instance) -> tuple[float, int]:
    """Cheapest place for the point: (added km, index in the route); a tie takes the lower index."""
    nodes = route_nodes(hub, route, instance)
    node = instance.point_node(point)
    added = (
        instance.distance_km[nodes[:-1], node]
        + instance.distance_km[node, nodes[1:]]
        - instance.distance_km[nodes[:-1], nodes[1:]]
    )
    index = int(np.argmin(added))
    return float(added[index]), index


def removal_savings(hub: int, route: list[int], instance: Instance) -> np.ndarray:
    """km saved when the stop at each position leaves the route."""
    nodes = route_nodes(hub, route, instance)
    return (
        instance.distance_km[nodes[:-2], nodes[1:-1]]
        + instance.distance_km[nodes[1:-1], nodes[2:]]
        - instance.distance_km[nodes[:-2], nodes[2:]]
    )


def two_opt(hub: int, route: list[int], instance: Instance) -> list[int]:
    """First-improvement 2-opt: reverse a segment while that shortens the route."""
    nodes = route_nodes(hub, route, instance)
    points = np.array([-1] + route + [-1], dtype=int)  # same positions as nodes; -1 is the hub
    improved = True
    while improved:
        improved = False
        for a in range(len(nodes) - 3):
            later = np.arange(a + 2, len(nodes) - 1)
            gain = (
                instance.distance_km[nodes[a], nodes[a + 1]]
                + instance.distance_km[nodes[later], nodes[later + 1]]
                - instance.distance_km[nodes[a], nodes[later]]
                - instance.distance_km[nodes[a + 1], nodes[later + 1]]
            )
            better = np.flatnonzero(gain > TWO_OPT_MIN_GAIN_KM)
            if len(better) > 0:
                b = int(later[better[0]])
                nodes = np.concatenate((nodes[: a + 1], nodes[b:a:-1], nodes[b + 1 :]))
                points = np.concatenate((points[: a + 1], points[b:a:-1], points[b + 1 :]))
                improved = True
    return [int(i) for i in points[1:-1]]
