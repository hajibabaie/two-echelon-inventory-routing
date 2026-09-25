"""The exact method: a Gurobi MIP with one van index and MTZ order variables.

Trap: binaries are read back rounded and IntegralityFocus is on, so a binary 1e-5 away from 0
cannot carry a delivery that the checker would reject.
"""

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from irp2e.evaluation import check_reported_cost, solution_cost
from irp2e.instance import Instance
from irp2e.solution import Solution

STATUS_NAMES = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit"}


@dataclass(frozen=True)
class MipResult:
    status: str  # "optimal" or "time_limit"
    objective: float  # BRL, Gurobi's value of the returned solution
    bound: float  # BRL, best lower bound
    gap: float  # relative gap between objective and bound
    seconds: float  # Gurobi run time


@dataclass(frozen=True)
class MipVariables:
    trip: gp.tupledict  # z[h, t]: a truck drives warehouse -> h -> warehouse on day t
    shipment: gp.tupledict  # y[h, t]: kg the truck brings to hub h on day t
    hub_stock: gp.tupledict  # I[h, t]: hub stock at the end of day t
    point_stock: gp.tupledict  # I[i, t]: point stock at the end of day t
    arc: gp.tupledict  # x[a, c, b, t]: van b drives from node a to node c on day t
    visit: gp.tupledict  # v[i, b, t]: van b visits point i on day t
    van_used: gp.tupledict  # w[h, b, t]: van b of hub h leaves the hub on day t
    delivery: gp.tupledict  # q[i, b, t]: kg van b gives to point i on day t
    order: gp.tupledict  # u[i, b, t]: MTZ position of point i on the route of van b


def hub_nodes(h: int, instance: Instance) -> list[int]:
    """Distance-matrix nodes of hub h: the hub first, then its points."""
    return [instance.hub_node(h)] + [instance.point_node(i) for i in instance.points_of[h]]


def arc_keys(instance: Instance) -> list[tuple[int, int, int, int]]:
    """Every (a, c, b, t) with a != c inside the nodes of one hub."""
    keys = []
    for h in range(instance.n_hubs):
        nodes = hub_nodes(h, instance)
        for b in range(instance.n_vans):
            for t in range(instance.n_days):
                for a in nodes:
                    for c in nodes:
                        if a != c:
                            keys.append((a, c, b, t))
    return keys


def add_variables(model: gp.Model, instance: Instance) -> MipVariables:
    """All variables of the model; stocks have lower bound 0, which is the no-stockout rule."""
    hub_days = [(h, t) for h in range(instance.n_hubs) for t in range(instance.n_days)]
    point_days = [(i, t) for i in range(instance.n_points) for t in range(instance.n_days)]
    point_vans = [
        (i, b, t)
        for i in range(instance.n_points)
        for b in range(instance.n_vans)
        for t in range(instance.n_days)
    ]
    hub_vans = [
        (h, b, t)
        for h in range(instance.n_hubs)
        for b in range(instance.n_vans)
        for t in range(instance.n_days)
    ]
    order_top = {(i, b, t): len(instance.points_of[instance.hub_of[i]]) for i, b, t in point_vans}
    return MipVariables(
        trip=model.addVars(hub_days, vtype=GRB.BINARY, name="z"),
        shipment=model.addVars(hub_days, name="y"),
        hub_stock=model.addVars(hub_days, name="I_hub"),
        point_stock=model.addVars(point_days, name="I_point"),
        arc=model.addVars(arc_keys(instance), vtype=GRB.BINARY, name="x"),
        visit=model.addVars(point_vans, vtype=GRB.BINARY, name="v"),
        van_used=model.addVars(hub_vans, vtype=GRB.BINARY, name="w"),
        delivery=model.addVars(point_vans, name="q"),
        order=model.addVars(point_vans, lb=1.0, ub=order_top, name="u"),
    )


def set_objective(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """Truck trips + van km + holding on end-of-day stock."""
    truck = gp.quicksum(
        instance.trip_cost(h) * variables.trip[h, t]
        for h in range(instance.n_hubs)
        for t in range(instance.n_days)
    )
    van = gp.quicksum(
        instance.van_cost_per_km * instance.distance_km[a, c] * variables.arc[a, c, b, t]
        for a, c, b, t in variables.arc.keys()
    )
    holding = instance.hub_holding * variables.hub_stock.sum()
    holding += instance.point_holding * variables.point_stock.sum()
    model.setObjective(truck + van + holding, GRB.MINIMIZE)


def hub_stock_before(variables: MipVariables, instance: Instance, h: int, t: int):
    """Hub stock at the start of day t: the initial stock or the end of day t - 1."""
    if t == 0:
        return float(instance.hub_initial[h])
    return variables.hub_stock[h, t - 1]


def point_stock_before(variables: MipVariables, instance: Instance, i: int, t: int):
    """Point stock at the start of day t: the initial stock or the end of day t - 1."""
    if t == 0:
        return float(instance.point_initial[i])
    return variables.point_stock[i, t - 1]


def add_truck_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(1) A hub gets goods only on a trip day, at most min(C_h, Q1)."""
    for h in range(instance.n_hubs):
        largest = min(float(instance.hub_capacity[h]), instance.truck_capacity)
        for t in range(instance.n_days):
            model.addConstr(variables.shipment[h, t] <= largest * variables.trip[h, t])


def add_hub_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(2) Hub maximum level and (3) hub stock balance."""
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            before = hub_stock_before(variables, instance, h, t)
            outflow = gp.quicksum(
                variables.delivery[i, b, t]
                for i in instance.points_of[h]
                for b in range(instance.n_vans)
            )
            model.addConstr(before + variables.shipment[h, t] <= instance.hub_capacity[h])
            model.addConstr(
                variables.hub_stock[h, t] == before + variables.shipment[h, t] - outflow
            )


def add_point_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(4) Point maximum level and (5) point stock balance."""
    for i in range(instance.n_points):
        for t in range(instance.n_days):
            before = point_stock_before(variables, instance, i, t)
            received = gp.quicksum(variables.delivery[i, b, t] for b in range(instance.n_vans))
            model.addConstr(before + received <= instance.point_capacity[i])
            model.addConstr(
                variables.point_stock[i, t] == before + received - instance.demand[i, t]
            )


def add_visit_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(6) One van per point and day, (7) a visited point has one arc in and one out."""
    for i in range(instance.n_points):
        node = instance.point_node(i)
        others = [c for c in hub_nodes(instance.hub_of[i], instance) if c != node]
        for t in range(instance.n_days):
            model.addConstr(
                gp.quicksum(variables.visit[i, b, t] for b in range(instance.n_vans)) <= 1
            )
            for b in range(instance.n_vans):
                leaving = gp.quicksum(variables.arc[node, c, b, t] for c in others)
                entering = gp.quicksum(variables.arc[c, node, b, t] for c in others)
                model.addConstr(leaving == variables.visit[i, b, t])
                model.addConstr(entering == variables.visit[i, b, t])


def add_depot_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(8) A used van leaves its hub once and comes back once."""
    for h in range(instance.n_hubs):
        hub = instance.hub_node(h)
        points = [instance.point_node(i) for i in instance.points_of[h]]
        for b in range(instance.n_vans):
            for t in range(instance.n_days):
                leaving = gp.quicksum(variables.arc[hub, c, b, t] for c in points)
                entering = gp.quicksum(variables.arc[c, hub, b, t] for c in points)
                model.addConstr(leaving == variables.van_used[h, b, t])
                model.addConstr(entering == variables.van_used[h, b, t])


def add_van_load_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(9) Delivery only on a visit and (10) one van load per route."""
    for i in range(instance.n_points):
        largest = min(float(instance.point_capacity[i]), instance.van_capacity)
        for b in range(instance.n_vans):
            for t in range(instance.n_days):
                model.addConstr(variables.delivery[i, b, t] <= largest * variables.visit[i, b, t])
    for h in range(instance.n_hubs):
        for b in range(instance.n_vans):
            for t in range(instance.n_days):
                load = gp.quicksum(variables.delivery[i, b, t] for i in instance.points_of[h])
                model.addConstr(load <= instance.van_capacity * variables.van_used[h, b, t])


def add_subtour_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(11) MTZ: a van that drives i -> j puts j later than i, so every cycle passes the hub."""
    for h in range(instance.n_hubs):
        size = len(instance.points_of[h])
        for i in instance.points_of[h]:
            for j in instance.points_of[h]:
                if i == j:
                    continue
                arc_from, arc_to = instance.point_node(i), instance.point_node(j)
                for b in range(instance.n_vans):
                    for t in range(instance.n_days):
                        model.addConstr(
                            variables.order[j, b, t]
                            >= variables.order[i, b, t]
                            + 1
                            - size * (1 - variables.arc[arc_from, arc_to, b, t])
                        )


def add_symmetry_rules(model: gp.Model, variables: MipVariables, instance: Instance) -> None:
    """(12) Van b is used only if van b - 1 is used."""
    for h in range(instance.n_hubs):
        for b in range(1, instance.n_vans):
            for t in range(instance.n_days):
                model.addConstr(variables.van_used[h, b, t] <= variables.van_used[h, b - 1, t])


def build_model(model: gp.Model, instance: Instance) -> MipVariables:
    """Add the variables, the objective and the rule groups (1) to (12)."""
    variables = add_variables(model, instance)
    set_objective(model, variables, instance)
    add_truck_rules(model, variables, instance)
    add_hub_rules(model, variables, instance)
    add_point_rules(model, variables, instance)
    add_visit_rules(model, variables, instance)
    add_depot_rules(model, variables, instance)
    add_van_load_rules(model, variables, instance)
    add_subtour_rules(model, variables, instance)
    add_symmetry_rules(model, variables, instance)
    return variables


def read_route(variables: MipVariables, instance: Instance, h: int, b: int, t: int) -> list[int]:
    """Points of van b of hub h on day t in driving order, following arcs from the hub."""
    nodes = hub_nodes(h, instance)
    point_of_node = {instance.point_node(i): i for i in instance.points_of[h]}
    successor = {}
    for a in nodes:
        for c in nodes:
            if a != c and variables.arc[a, c, b, t].X > 0.5:
                successor[a] = c
    route = []
    node = successor[instance.hub_node(h)]
    while node != instance.hub_node(h):
        route.append(point_of_node[node])
        node = successor[node]
    return route


def extract_solution(variables: MipVariables, instance: Instance) -> Solution:
    """The solution from rounded binaries and the solver's quantities."""
    truck_trips = np.zeros((instance.n_hubs, instance.n_days), dtype=bool)
    hub_shipments = np.zeros((instance.n_hubs, instance.n_days))
    deliveries = np.zeros((instance.n_points, instance.n_days))
    for h in range(instance.n_hubs):
        for t in range(instance.n_days):
            truck_trips[h, t] = variables.trip[h, t].X > 0.5
            hub_shipments[h, t] = variables.shipment[h, t].X
    for i in range(instance.n_points):
        for t in range(instance.n_days):
            deliveries[i, t] = np.sum(
                [variables.delivery[i, b, t].X for b in range(instance.n_vans)]
            )
    van_routes = [
        [
            [
                read_route(variables, instance, h, b, t)
                for b in range(instance.n_vans)
                if variables.van_used[h, b, t].X > 0.5
            ]
            for t in range(instance.n_days)
        ]
        for h in range(instance.n_hubs)
    ]
    return Solution(
        truck_trips=truck_trips,
        hub_shipments=hub_shipments,
        deliveries=deliveries,
        van_routes=van_routes,
    )


def read_status(model: gp.Model, instance: Instance) -> str:
    """Status name of a run that returned a solution; RuntimeError otherwise."""
    if model.Status not in STATUS_NAMES or model.SolCount == 0:
        raise RuntimeError(
            f"{instance.name}: Gurobi status {model.Status} with {model.SolCount} solutions"
        )
    return STATUS_NAMES[model.Status]


def solve_mip(
    instance: Instance, time_limit_s: float, threads: int, mip_gap: float
) -> tuple[Solution, MipResult]:
    """Solve the MIP; return only a solution whose cost equals the Gurobi objective."""
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(instance.name, env=env) as model:
            model.Params.TimeLimit = time_limit_s
            model.Params.Threads = threads
            model.Params.MIPGap = mip_gap  # 0 in the config: prove optimality
            model.Params.IntegralityFocus = 1
            variables = build_model(model, instance)
            model.optimize()
            result = MipResult(
                status=read_status(model, instance),
                objective=model.ObjVal,
                bound=model.ObjBound,
                gap=model.MIPGap,
                seconds=model.Runtime,
            )
            solution = extract_solution(variables, instance)
    problems = check_reported_cost(result.objective, solution_cost(solution, instance))
    if len(problems) > 0:
        raise ValueError(f"{instance.name}: " + "; ".join(problems))
    return solution, result
