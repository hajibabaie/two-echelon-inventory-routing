"""ALNS destroy and repair operators on the plan, one rule each."""

import numpy as np

from irp2e.evaluation import TOL_KG
from irp2e.heuristics.plan import Plan, visit_days
from irp2e.heuristics.quantities import (
    hub_quantities,
    point_just_in_time,
    point_quantities,
    route_load,
)
from irp2e.heuristics.routing import best_position, removal_savings
from irp2e.instance import Instance

Touched = tuple[list[int], list[int]]  # points and hubs a destroy operator changed
Option = tuple[float, int, int, int]  # (added van cost, -day, route index, position); lowest wins


# ----- visits -----


def all_visits(plan: Plan) -> list[tuple[int, int, int]]:
    """Every visit as (hub, day, point), in route order."""
    visits = []
    for h, hub_days in enumerate(plan.routes):
        for t, routes in enumerate(hub_days):
            for route in routes:
                for i in route:
                    visits.append((h, t, i))
    return visits


def remove_visit(plan: Plan, h: int, t: int, i: int) -> None:
    """Take point i off its route on day t; a route left empty is removed."""
    for route in plan.routes[h][t]:
        if i in route:
            route.remove(i)
    plan.routes[h][t] = [route for route in plan.routes[h][t] if len(route) > 0]


def remove_visits(plan: Plan, visits: list[tuple[int, int, int]]) -> Touched:
    """Remove the visits; touched points and their hubs, ascending."""
    for h, t, i in visits:
        remove_visit(plan, h, t, i)
    points = sorted({i for _, _, i in visits})
    hubs = sorted({h for h, _, _ in visits})
    return points, hubs


def pick(items: list, count: int) -> list:
    """count items drawn uniformly without replacement, in draw order."""
    chosen = np.random.choice(len(items), min(count, len(items)), replace=False)
    return [items[int(index)] for index in chosen]


# ----- destroy operators -----


def random_removal(plan: Plan, instance: Instance, rho: int) -> Touched:
    """Remove rho visits chosen uniformly."""
    return remove_visits(plan, pick(all_visits(plan), rho))


def worst_removal(plan: Plan, instance: Instance, rho: int) -> Touched:
    """Among the 2 rho visits that save the most km, remove rho at random."""
    ranked = []
    for h, hub_days in enumerate(plan.routes):
        for t, routes in enumerate(hub_days):
            for route in routes:
                for i, saving in zip(route, removal_savings(h, route, instance), strict=True):
                    ranked.append((-float(saving), h, t, i))  # most km saved first
    ranked.sort()
    top = [(h, t, i) for _, h, t, i in ranked[: 2 * rho]]
    return remove_visits(plan, pick(top, rho))


def related_removal(plan: Plan, instance: Instance, rho: int) -> Touched:
    """Pick one visit; remove the rho visits of its hub closest to its point (its own first)."""
    visits = all_visits(plan)
    if len(visits) == 0:
        return [], []
    h, _, i = pick(visits, 1)[0]
    node = instance.point_node(i)
    related = []
    for t, routes in enumerate(plan.routes[h]):
        for route in routes:
            for j in route:
                related.append((instance.distance_km[node, instance.point_node(j)], t, j))
    related.sort()
    return remove_visits(plan, [(h, t, j) for _, t, j in related[:rho]])


def sequence_removal(plan: Plan, instance: Instance, rho: int) -> Touched:
    """Remove a run of up to rho consecutive stops from one random route."""
    routes = [
        (h, t, route)
        for h, hub_days in enumerate(plan.routes)
        for t, day_routes in enumerate(hub_days)
        for route in day_routes
    ]
    if len(routes) == 0:
        return [], []
    h, t, route = pick(routes, 1)[0]
    length = min(rho, len(route))
    start = np.random.randint(0, len(route) - length + 1)
    return remove_visits(plan, [(h, t, i) for i in route[start : start + length]])


def hub_day_removal(plan: Plan, instance: Instance, rho: int) -> Touched:
    """Remove every visit of one random (hub, day) that has a route."""
    busy = [
        (h, t)
        for h in range(instance.n_hubs)
        for t in range(instance.n_days)
        if len(plan.routes[h][t]) > 0
    ]
    if len(busy) == 0:
        return [], []
    h, t = pick(busy, 1)[0]
    return remove_visits(plan, [(h, t, i) for route in plan.routes[h][t] for i in route])


def truck_trip_removal(plan: Plan, instance: Instance, rho: int) -> Touched:
    """Remove one random truck trip; the repair adds a later one if the hub runs dry."""
    trips = [
        (h, t)
        for h in range(instance.n_hubs)
        for t in range(instance.n_days)
        if plan.truck_trips[h, t]
    ]
    if len(trips) == 0:
        return [], []
    h, t = pick(trips, 1)[0]
    plan.truck_trips[h, t] = False
    return [], [h]


# ----- insertion options -----


def stockout_window(days: list[int], stockout: int) -> list[int]:
    """Days after the last visit before the stockout, up to the stockout day."""
    earlier = [t for t in days if t < stockout]
    first = earlier[-1] + 1 if len(earlier) > 0 else 0
    return list(range(first, stockout + 1))


def new_visit_amount(i: int, days: list[int], t: int, instance: Instance) -> float:
    """kg a new visit of point i on day t gets by rule J1."""
    q_i, _ = point_just_in_time(i, sorted(days + [t]), instance)
    return float(q_i[t])


def window_amounts(i: int, days: list[int], stockout: int, instance: Instance) -> dict[int, float]:
    """kg a new visit would get on each day of the stockout window of point i."""
    return {t: new_visit_amount(i, days, t, instance) for t in stockout_window(days, stockout)}


def day_options(
    plan: Plan, i: int, t: int, amount: float, q: np.ndarray, instance: Instance
) -> list[Option]:
    """Allowed insertions of point i on day t: each route with room, or a new route."""
    h = int(instance.hub_of[i])
    options = []
    for r, route in enumerate(plan.routes[h][t]):
        if route_load(route, q, t) + amount <= instance.van_capacity + TOL_KG:
            km, position = best_position(h, route, i, instance)
            options.append((instance.van_cost_per_km * km, -t, r, position))
    if len(plan.routes[h][t]) < instance.n_vans:
        km, _ = best_position(h, [], i, instance)
        options.append((instance.van_cost_per_km * km, -t, len(plan.routes[h][t]), 0))
    return options


def window_options(
    plan: Plan, i: int, amounts: dict[int, float], q: np.ndarray, instance: Instance
) -> dict[int, list[Option]]:
    """Options of a needy point on each day of its stockout window."""
    return {t: day_options(plan, i, t, amount, q, instance) for t, amount in amounts.items()}


def refresh_options(
    plan: Plan,
    options: dict[int, dict[int, list[Option]]],
    amounts: dict[int, dict[int, float]],
    h: int,
    stale_days: list[int],
    q: np.ndarray,
    instance: Instance,
) -> None:
    """Recompute the options of the needy points of hub h on days whose routes or loads changed."""
    for j in options:
        if instance.hub_of[j] == h:
            for t in stale_days:
                if t in amounts[j]:
                    options[j][t] = day_options(plan, j, t, amounts[j][t], q, instance)


def insert_visit(plan: Plan, i: int, option: Option, instance: Instance) -> int:
    """Put point i where the option says; return the day."""
    _, minus_day, r, position = option
    t = -minus_day
    h = int(instance.hub_of[i])
    if r == len(plan.routes[h][t]):
        plan.routes[h][t].append([i])
    else:
        plan.routes[h][t][r].insert(position, i)
    return t


def cheapest_choice(options: dict[int, dict[int, list[Option]]]) -> tuple[int, Option]:
    """Greedy choice: the single cheapest option over all needy points."""
    best = min(
        (option[0], option[1], i, option[2], option[3])
        for i in options
        for day in options[i].values()
        for option in day
    )
    return best[2], (best[0], best[1], best[3], best[4])


def regret_choice(options: dict[int, dict[int, list[Option]]]) -> tuple[int, Option]:
    """Regret-2 choice: the point that loses most if it misses its best option goes first."""
    ranked = []
    for i in options:
        ordered = sorted(option for day in options[i].values() for option in day)
        regret = ordered[1][0] - ordered[0][0] if len(ordered) > 1 else np.inf
        best = ordered[0]
        ranked.append((-regret, best[0], best[1], i, best[2], best[3]))
    chosen = min(ranked)
    return chosen[3], (chosen[1], chosen[2], chosen[4], chosen[5])


# ----- repair rules -----


def insert_needed_visits(plan: Plan, instance: Instance, use_regret: bool) -> bool:
    """Rule R-points: add visits until no point runs out; False if a needy point has no option."""
    days = visit_days(plan, instance)
    q, stockouts = point_quantities(plan, instance)
    amounts = {i: window_amounts(i, days[i], s, instance) for i, s in stockouts.items()}
    options = {i: window_options(plan, i, amounts[i], q, instance) for i in stockouts}
    while len(options) > 0:
        for i in options:
            if all(len(day) == 0 for day in options[i].values()):
                return False
        i, option = regret_choice(options) if use_regret else cheapest_choice(options)
        t = insert_visit(plan, i, option, instance)
        before = q[i, :].copy()
        days[i] = sorted(days[i] + [t])
        q[i, :], stockout = point_just_in_time(i, days[i], instance)
        del amounts[i], options[i]
        stale_days = [d for d in range(instance.n_days) if d == t or q[i, d] != before[d]]
        refresh_options(plan, options, amounts, int(instance.hub_of[i]), stale_days, q, instance)
        if stockout is not None:  # the new visit hit the capacity before the next visit
            amounts[i] = window_amounts(i, days[i], stockout, instance)
            options[i] = window_options(plan, i, amounts[i], q, instance)
    return True


def insert_needed_trips(plan: Plan, instance: Instance) -> None:
    """Rule R-hubs: while a hub runs dry, add a truck trip on its first stockout day."""
    q, _ = point_quantities(plan, instance)
    _, stockouts = hub_quantities(plan, q, instance)
    while len(stockouts) > 0:
        for h, stockout in stockouts.items():
            plan.truck_trips[h, stockout] = True
        _, stockouts = hub_quantities(plan, q, instance)


def insert_extra_visits(plan: Plan, points: list[int], instance: Instance, rho: int) -> None:
    """One extra visit for up to rho touched points, each on a random day it is not visited."""
    days = visit_days(plan, instance)
    q, _ = point_quantities(plan, instance)
    for i in pick(points, rho):
        free = [t for t in range(instance.n_days) if t not in days[i]]
        if len(free) == 0:
            continue
        t = free[np.random.randint(len(free))]
        amount = new_visit_amount(i, days[i], t, instance)
        options = day_options(plan, i, t, amount, q, instance)
        if len(options) == 0:
            continue  # no route has room for the extra visit
        insert_visit(plan, i, min(options), instance)
        days[i] = sorted(days[i] + [t])
        q[i, :], _ = point_just_in_time(i, days[i], instance)


def insert_extra_trip(plan: Plan, touched: Touched, instance: Instance) -> None:
    """One extra truck trip for one random touched hub on a random day without a trip."""
    points, hubs = touched
    candidates = sorted(set(hubs) | {int(instance.hub_of[i]) for i in points})
    if len(candidates) == 0:
        return
    h = pick(candidates, 1)[0]
    free = [t for t in range(instance.n_days) if not plan.truck_trips[h, t]]
    if len(free) > 0:
        plan.truck_trips[h, free[np.random.randint(len(free))]] = True


# ----- repair operators -----


def greedy_repair(plan: Plan, touched: Touched, instance: Instance, rho: int) -> bool:
    """R-points with the greedy choice, then R-hubs."""
    if not insert_needed_visits(plan, instance, use_regret=False):
        return False
    insert_needed_trips(plan, instance)
    return True


def regret_repair(plan: Plan, touched: Touched, instance: Instance, rho: int) -> bool:
    """R-points with the regret choice, then R-hubs."""
    if not insert_needed_visits(plan, instance, use_regret=True):
        return False
    insert_needed_trips(plan, instance)
    return True


def extra_visits_repair(plan: Plan, touched: Touched, instance: Instance, rho: int) -> bool:
    """R-points greedy, then one random extra visit for up to rho touched points, then R-hubs."""
    if not insert_needed_visits(plan, instance, use_regret=False):
        return False
    insert_extra_visits(plan, touched[0], instance, rho)
    insert_needed_trips(plan, instance)
    return True


def extra_trip_repair(plan: Plan, touched: Touched, instance: Instance, rho: int) -> bool:
    """R-points greedy, then one random extra truck trip for a touched hub, then R-hubs."""
    if not insert_needed_visits(plan, instance, use_regret=False):
        return False
    insert_extra_trip(plan, touched, instance)
    insert_needed_trips(plan, instance)
    return True


DESTROY_OPERATORS = [
    ("random_removal", random_removal),
    ("worst_removal", worst_removal),
    ("related_removal", related_removal),
    ("sequence_removal", sequence_removal),
    ("hub_day_removal", hub_day_removal),
    ("truck_trip_removal", truck_trip_removal),
]
REPAIR_OPERATORS = [
    ("greedy_repair", greedy_repair),
    ("regret_repair", regret_repair),
    ("extra_visits_repair", extra_visits_repair),
    ("extra_trip_repair", extra_trip_repair),
]
