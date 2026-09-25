"""Matheuristic: the same ALNS, but Gurobi solves the quantities as an LP when rule J1 fails."""

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from irp2e.config import Params
from irp2e.heuristics.alns import Alns
from irp2e.heuristics.plan import Plan, visit_days
from irp2e.heuristics.quantities import jit_quantities
from irp2e.instance import Instance


class QuantityLp:
    """LP of the quantities and stocks of a fixed plan: rules (2) to (5) of model.py again."""

    def __init__(self, instance: Instance, threads: int):
        self.instance = instance
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model("quantities", env=self.env)
        self.model.Params.Threads = threads
        self.model.Params.DualReductions = 0  # report INFEASIBLE, never INF_OR_UNBD
        # Gurobi's default lower bound 0 keeps every quantity and every stock >= 0
        self.q = self.model.addVars(instance.n_points, instance.n_days, name="q")
        self.y = self.model.addVars(instance.n_hubs, instance.n_days, name="y")
        self.hub_stock = self.model.addVars(instance.n_hubs, instance.n_days, name="hub_stock")
        self.point_stock = self.model.addVars(instance.n_points, instance.n_days, name="stock")
        self.add_hub_rows()
        self.add_point_rows()
        holding = instance.hub_holding * self.hub_stock.sum()
        holding += instance.point_holding * self.point_stock.sum()
        self.model.setObjective(holding, GRB.MINIMIZE)
        self.route_rows = []

    def add_hub_rows(self) -> None:
        """Rules (2) and (3): hub maximum level and hub stock balance."""
        for h in range(self.instance.n_hubs):
            for t in range(self.instance.n_days):
                before = self.instance.hub_initial[h] if t == 0 else self.hub_stock[h, t - 1]
                outflow = gp.quicksum(self.q[i, t] for i in self.instance.points_of[h])
                self.model.addConstr(before + self.y[h, t] <= self.instance.hub_capacity[h])
                self.model.addConstr(self.hub_stock[h, t] == before + self.y[h, t] - outflow)

    def add_point_rows(self) -> None:
        """Rules (4) and (5): point maximum level and stock balance; stock >= 0 is no stockout."""
        for i in range(self.instance.n_points):
            for t in range(self.instance.n_days):
                before = self.instance.point_initial[i] if t == 0 else self.point_stock[i, t - 1]
                self.model.addConstr(before + self.q[i, t] <= self.instance.point_capacity[i])
                self.model.addConstr(
                    self.point_stock[i, t] == before + self.q[i, t] - self.instance.demand[i, t]
                )

    def set_plan(self, plan: Plan) -> None:
        """Deliveries only on visits, shipments only on trips, one van load per route."""
        visited = visit_days(plan, self.instance)
        keys = [(i, t) for i in range(self.instance.n_points) for t in range(self.instance.n_days)]
        bounds = []
        for i, t in keys:
            if t in visited[i]:
                bounds.append(min(self.instance.point_capacity[i], self.instance.van_capacity))
            else:
                bounds.append(0.0)
        self.model.setAttr("UB", [self.q[key] for key in keys], bounds)
        trips = [(h, t) for h in range(self.instance.n_hubs) for t in range(self.instance.n_days)]
        bounds = []
        for h, t in trips:
            if plan.truck_trips[h, t]:
                bounds.append(min(self.instance.hub_capacity[h], self.instance.truck_capacity))
            else:
                bounds.append(0.0)
        self.model.setAttr("UB", [self.y[key] for key in trips], bounds)
        self.model.remove(self.route_rows)
        self.route_rows = [
            self.model.addConstr(
                gp.quicksum(self.q[i, t] for i in route) <= self.instance.van_capacity
            )
            for h in range(self.instance.n_hubs)
            for t in range(self.instance.n_days)
            for route in plan.routes[h][t]
        ]

    def close(self) -> None:
        """Free the Gurobi model and environment at the end of a run."""
        self.model.dispose()
        self.env.dispose()

    def solve(self, plan: Plan) -> tuple[np.ndarray, np.ndarray] | None:
        """(q, y) with the least holding cost for this plan; None if no quantities fit."""
        self.set_plan(plan)
        self.model.optimize()
        if self.model.Status == GRB.INFEASIBLE:
            return None
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"quantity LP ended with Gurobi status {self.model.Status}")
        q = np.zeros((self.instance.n_points, self.instance.n_days))
        y = np.zeros((self.instance.n_hubs, self.instance.n_days))
        for (i, t), value in self.model.getAttr("X", self.q).items():
            q[i, t] = value
        for (h, t), value in self.model.getAttr("X", self.y).items():
            y[h, t] = value
        return q, y


class RescueQuantities:
    """Rule M1: quantities by J1; only when J1 rejects the plan, by the LP."""

    def __init__(self, lp: QuantityLp):
        self.lp = lp
        self.lp_calls = 0
        self.lp_rescues = 0  # LP calls that returned quantities

    def __call__(self, plan: Plan, instance: Instance) -> tuple[np.ndarray, np.ndarray] | None:
        """J1 first; the LP when J1 returns None."""
        quantities = jit_quantities(plan, instance)
        if quantities is not None:
            return quantities
        self.lp_calls += 1
        quantities = self.lp.solve(plan)
        if quantities is not None:
            self.lp_rescues += 1
        return quantities


def make_matheuristic(instance: Instance, params: Params) -> Alns:
    """The ALNS engine with rule M1 as its quantity step."""
    return Alns(instance, params, RescueQuantities(QuantityLp(instance, params.gurobi_threads)))
