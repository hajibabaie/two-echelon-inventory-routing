"""The Solution dataclass and its JSON file format."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from irp2e.jsonfile import dump_rows

SOLUTION_FIELDS = ["truck_trips", "hub_shipments", "deliveries", "van_routes"]


@dataclass(eq=False)
class Solution:
    truck_trips: np.ndarray  # (k, T) bool: a truck drives warehouse -> h -> warehouse on day t
    hub_shipments: np.ndarray  # (k, T) kg brought by the truck to hub h on day t
    deliveries: np.ndarray  # (n, T) kg delivered to point i on day t
    van_routes: list[list[list[list[int]]]]  # [h][t] -> list of routes; a route lists points


def save_solution(solution: Solution, meta: dict, path: Path) -> None:
    """Write the meta fields (instance, method, seed, cost, ...) and the solution to one file."""
    record = {
        **meta,
        "truck_trips": solution.truck_trips.tolist(),
        "hub_shipments": solution.hub_shipments.tolist(),
        "deliveries": solution.deliveries.tolist(),
        "van_routes": solution.van_routes,
    }
    dump_rows(record, path)


def load_solution(path: Path) -> tuple[Solution, dict]:
    """Read a solution file; the checker, not this function, judges the content."""
    with open(path, encoding="utf-8") as file:
        record = json.load(file)
    solution = Solution(
        truck_trips=np.array(record["truck_trips"], dtype=bool),
        hub_shipments=np.array(record["hub_shipments"], dtype=float),
        deliveries=np.array(record["deliveries"], dtype=float),
        van_routes=record["van_routes"],
    )
    meta = {key: value for key, value in record.items() if key not in SOLUTION_FIELDS}
    return solution, meta
