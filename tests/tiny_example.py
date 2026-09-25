"""The tiny worked example (design section 11) and the frozen instances, shared by the tests."""

import numpy as np

from irp2e.config import REPO_ROOT
from irp2e.heuristics.plan import Plan
from irp2e.solution import Solution

TINY = REPO_ROOT / "tests" / "data" / "tiny.json"
INSTANCES = REPO_ROOT / "data" / "instances"
A, B, C = 0, 1, 2  # the three points of the tiny example


def tiny_optimal_plan() -> Plan:
    """Truck on day 0; day 0 route H-A-B-H, day 1 route H-A-C-H."""
    return Plan(truck_trips=np.array([[True, False]]), routes=[[[[A, B]], [[A, C]]]])


def tiny_optimum() -> Solution:
    """The optimal plan with its quantities: cost 303."""
    return Solution(
        truck_trips=np.array([[True, False]]),
        hub_shipments=np.array([[55.0, 0.0]]),
        deliveries=np.array([[10.0, 10.0], [20.0, 0.0], [0.0, 15.0]]),
        van_routes=[[[[A, B]], [[A, C]]]],
    )
