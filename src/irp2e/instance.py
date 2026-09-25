"""The Instance dataclass, its JSON file format, and the cost of one truck trip."""

import dataclasses
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from irp2e.jsonfile import dump_rows

PARAM_FIELDS = [
    "truck_capacity",
    "truck_cost_per_km",
    "truck_cost_per_trip",
    "van_capacity",
    "n_vans",
    "van_cost_per_km",
    "hub_holding",
    "point_holding",
]


@dataclass(eq=False)
class Instance:
    name: str
    set_name: str
    seed: int | None
    n_hubs: int
    n_points: int
    n_days: int
    labels: list[str]  # node order: warehouse, hubs, points
    cities: list[str]
    coordinates: np.ndarray  # (1 + k + n, 2) lat, lng; used only by plots
    hub_of: np.ndarray  # (n,) fixed nearest hub of each point
    demand: np.ndarray  # (n, T) kg
    point_capacity: np.ndarray  # (n,) kg
    point_initial: np.ndarray  # (n,) kg
    hub_capacity: np.ndarray  # (k,) kg
    hub_initial: np.ndarray  # (k,) kg
    distance_km: np.ndarray  # (1 + k + n, 1 + k + n), symmetric
    truck_capacity: float
    truck_cost_per_km: float
    truck_cost_per_trip: float
    van_capacity: float
    n_vans: int
    van_cost_per_km: float
    hub_holding: float  # BRL per kg per day
    point_holding: float  # BRL per kg per day
    source: dict  # where the data came from (fold window, OSRM fetch date, dropped prefixes)

    def hub_node(self, h: int) -> int:
        """Node number of hub h in the distance matrix."""
        return 1 + h

    def point_node(self, i: int) -> int:
        """Node number of point i in the distance matrix."""
        return 1 + self.n_hubs + i

    def trip_cost(self, h: int) -> float:
        """Cost of one truck trip warehouse -> hub h -> warehouse."""
        return float(
            self.truck_cost_per_trip
            + 2 * self.truck_cost_per_km * self.distance_km[0, self.hub_node(h)]
        )

    @cached_property
    def points_of(self) -> list[list[int]]:
        """Points of each hub, ascending."""
        groups = [[] for _ in range(self.n_hubs)]
        for i in range(self.n_points):
            groups[int(self.hub_of[i])].append(i)
        return groups


def _check_shape(name: str, array: np.ndarray, shape: tuple) -> None:
    """Raise ValueError when an array read from a file has the wrong shape."""
    if array.shape != shape:
        raise ValueError(f"instance field {name} has shape {array.shape}, expected {shape}")


def _check_non_negative(name: str, array: np.ndarray) -> None:
    """Raise ValueError when an array read from a file has a negative or nan entry."""
    if np.any(np.isnan(array)) or np.any(array < 0):
        raise ValueError(f"instance field {name} must be >= 0 everywhere")


def rule_problems(instance: Instance) -> list[str]:
    """Rules C1, C3 and A3, which the ALNS repair and the greedy need to stop."""
    problems = []
    for i in range(instance.n_points):
        if instance.point_capacity[i] < np.max(instance.demand[i, :]):
            problems.append(f"C1: point {i} cannot store its busiest day")
        if instance.point_capacity[i] > instance.van_capacity:
            problems.append(f"A3: point {i} can take more than one van load")
    for h in range(instance.n_hubs):
        if instance.hub_capacity[h] < np.sum(instance.point_capacity[instance.points_of[h]]):
            problems.append(f"C3: hub {h} cannot hold what its points can take in one day")
    return problems


def load_instance(path: Path) -> Instance:
    """Read one instance file and check shapes, signs and rules C1, C3 and A3."""
    with open(path, encoding="utf-8") as file:
        record = json.load(file)
    n_hubs, n_points, n_days = record["n_hubs"], record["n_points"], record["n_days"]
    for key in ("n_hubs", "n_points", "n_days"):
        if not isinstance(record[key], int) or record[key] < 1:
            raise ValueError(f"instance field {key} must be an integer >= 1")
    n_nodes = 1 + n_hubs + n_points
    params = record["params"]
    instance = Instance(
        name=record["name"],
        set_name=record["set"],
        seed=record["seed"],
        n_hubs=n_hubs,
        n_points=n_points,
        n_days=n_days,
        labels=record["labels"],
        cities=record["cities"],
        coordinates=np.array(record["coordinates"], dtype=float),
        hub_of=np.array(record["hub_of"], dtype=int),
        demand=np.array(record["demand"], dtype=float),
        point_capacity=np.array(record["point_capacity"], dtype=float),
        point_initial=np.array(record["point_initial"], dtype=float),
        hub_capacity=np.array(record["hub_capacity"], dtype=float),
        hub_initial=np.array(record["hub_initial"], dtype=float),
        distance_km=np.array(record["distance_km"], dtype=float),
        truck_capacity=float(params["truck_capacity"]),
        truck_cost_per_km=float(params["truck_cost_per_km"]),
        truck_cost_per_trip=float(params["truck_cost_per_trip"]),
        van_capacity=float(params["van_capacity"]),
        n_vans=params["n_vans"],
        van_cost_per_km=float(params["van_cost_per_km"]),
        hub_holding=float(params["hub_holding"]),
        point_holding=float(params["point_holding"]),
        source=record["source"],
    )
    if len(instance.labels) != n_nodes or len(instance.cities) != n_nodes:
        raise ValueError(f"instance labels and cities must have {n_nodes} entries")
    _check_shape("coordinates", instance.coordinates, (n_nodes, 2))
    _check_shape("hub_of", instance.hub_of, (n_points,))
    _check_shape("demand", instance.demand, (n_points, n_days))
    _check_shape("point_capacity", instance.point_capacity, (n_points,))
    _check_shape("point_initial", instance.point_initial, (n_points,))
    _check_shape("hub_capacity", instance.hub_capacity, (n_hubs,))
    _check_shape("hub_initial", instance.hub_initial, (n_hubs,))
    _check_shape("distance_km", instance.distance_km, (n_nodes, n_nodes))
    if np.any(instance.hub_of < 0) or np.any(instance.hub_of >= n_hubs):
        raise ValueError(f"instance field hub_of must lie in 0 .. {n_hubs - 1}")
    for name in ("demand", "point_capacity", "point_initial", "hub_capacity", "hub_initial"):
        _check_non_negative(name, getattr(instance, name))
    _check_non_negative("distance_km", instance.distance_km)
    if not isinstance(instance.n_vans, int) or instance.n_vans < 1:
        raise ValueError("instance param n_vans must be an integer >= 1")
    for name in PARAM_FIELDS:
        if name != "n_vans":  # checked above as an integer >= 1
            _check_non_negative(name, np.array([getattr(instance, name)], dtype=float))
    problems = rule_problems(instance)
    if len(problems) > 0:
        raise ValueError(f"{instance.name}: " + "; ".join(problems))
    return instance


def save_instance(instance: Instance, path: Path) -> None:
    """Write one instance file in the format load_instance reads."""
    record = {
        "name": instance.name,
        "set": instance.set_name,
        "seed": instance.seed,
        "n_hubs": instance.n_hubs,
        "n_points": instance.n_points,
        "n_days": instance.n_days,
        "labels": instance.labels,
        "cities": instance.cities,
        "coordinates": instance.coordinates.tolist(),
        "hub_of": instance.hub_of.tolist(),
        "demand": instance.demand.tolist(),
        "point_capacity": instance.point_capacity.tolist(),
        "point_initial": instance.point_initial.tolist(),
        "hub_capacity": instance.hub_capacity.tolist(),
        "hub_initial": instance.hub_initial.tolist(),
        "distance_km": instance.distance_km.tolist(),
        "params": {name: getattr(instance, name) for name in PARAM_FIELDS},
        "source": instance.source,
    }
    dump_rows(record, path)


def with_holding_multiplier(instance: Instance, multiplier: float) -> Instance:
    """Copy of the instance with both holding costs times multiplier and the name suffix _h<m>."""
    return dataclasses.replace(
        instance,
        name=f"{instance.name}_h{multiplier:g}",
        hub_holding=instance.hub_holding * multiplier,
        point_holding=instance.point_holding * multiplier,
    )
