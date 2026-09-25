"""Build the frozen small, medium and large instance sets from the processed data and seeds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from irp2e.config import Params
from irp2e.data.osrm import read_cache
from irp2e.data.preprocess import (
    DAY_COLUMNS,
    build_pool,
    read_demand_week,
    read_prefixes,
    read_sites,
)
from irp2e.instance import Instance, rule_problems, save_instance


def instance_name(set_name: str, n_hubs: int, n_points: int, n_days: int, seed: int) -> str:
    """File name stem, for example small_k1_n6_T3_s1."""
    return f"{set_name}_k{n_hubs}_n{n_points}_T{n_days}_s{seed}"


def point_capacity(week: np.ndarray, params: Params) -> float:
    """Rule C1: the larger of the busiest day and point_capacity_days average days."""
    return float(max(np.max(week), params.point_capacity_days * np.mean(week)))


def point_initial(week: np.ndarray, params: Params) -> float:
    """Rule C2: point_initial_days average days."""
    return float(params.point_initial_days * np.mean(week))


def hub_capacity(point_caps: np.ndarray, params: Params) -> float:
    """Rule C3: hub_capacity_factor times the capacities of the hub's points."""
    return float(params.hub_capacity_factor * np.sum(point_caps))


def hub_initial(weeks: np.ndarray, params: Params) -> float:
    """Rule C4: hub_initial_days average days of the hub's whole region."""
    return float(params.hub_initial_days * np.sum(np.mean(weeks, axis=1)))


def check_assumptions(instance: Instance) -> None:
    """Rules A1 to A4, C1 and C3; raise ValueError listing every rule that fails."""
    problems = []
    for h in range(instance.n_hubs):
        if len(instance.points_of[h]) == 0:
            problems.append(f"A4: hub {h} has no point")
    if len(problems) > 0:
        raise ValueError(f"{instance.name}: " + "; ".join(problems))
    if np.max(instance.hub_capacity) > instance.truck_capacity:
        problems.append("A1: a hub capacity exceeds one truck")
    for h in range(instance.n_hubs):
        caps = instance.point_capacity[instance.points_of[h]]
        if np.sum(caps) > instance.n_vans * (instance.van_capacity - np.max(caps)):
            problems.append(f"A2: the points of hub {h} may need more than {instance.n_vans} vans")
    problems += rule_problems(instance)
    if len(problems) > 0:
        raise ValueError(f"{instance.name}: " + "; ".join(problems))


def road_km(labels: list[str], cache_labels: list[str], meters: np.ndarray) -> np.ndarray:
    """Symmetric km between the labels: the mean of both OSRM directions, rounded to meters."""
    position = {label: index for index, label in enumerate(cache_labels)}
    for label in labels:
        if label not in position:
            raise ValueError(f"prefix {label} is not in the OSRM cache; refetch the distances")
    km = np.zeros((len(labels), len(labels)))
    for a, label_a in enumerate(labels):
        for b, label_b in enumerate(labels):
            there = meters[position[label_a], position[label_b]]
            back = meters[position[label_b], position[label_a]]
            km[a, b] = round(float((there + back) / 2 / 1000), 3)
    return km


def shortest_paths(km: np.ndarray) -> np.ndarray:
    """Floyd-Warshall: km of the shortest path via the other nodes, so km obey the triangle rule."""
    shortest = km.copy()
    for via in range(len(km)):
        shortest = np.minimum(shortest, np.add.outer(shortest[:, via], shortest[via, :]))
    return np.round(shortest, 3)


def build_instance(
    set_name: str,
    n_hubs: int,
    n_points: int,
    n_days: int,
    seed: int,
    processed: Path,
    params: Params,
) -> Instance:
    """Draw the points with rule P2, apply rules C1 to C4, take shortest-path km, then check."""
    prefixes = read_prefixes(processed / "prefixes.csv")
    weeks_by_prefix = read_demand_week(processed / "demand_week.csv")
    sites = read_sites(processed / "sites.json")
    cache_labels, meters = read_cache(processed / "osrm_distance_m.csv")
    with open(processed / "osrm_meta.json", encoding="utf-8") as file:
        osrm_meta = json.load(file)

    ranked_hubs = pd.DataFrame(sites["hubs"])
    hubs = ranked_hubs.iloc[:n_hubs]
    pool = build_pool(prefixes, ranked_hubs, n_hubs, params)
    pool = pool[~pool["prefix"].isin(osrm_meta["dropped"])].reset_index(drop=True)  # rule P0
    if len(pool) < n_points:
        raise ValueError(f"pool of {n_hubs} hubs has {len(pool)} points, fewer than {n_points}")
    np.random.seed(seed)
    drawn = np.sort(np.random.choice(len(pool), n_points, replace=False))  # rule P2
    points = pool.iloc[drawn]

    weeks = weeks_by_prefix.loc[points["prefix"], DAY_COLUMNS].to_numpy()
    hub_of = points["hub"].to_numpy()
    caps = np.array([round(point_capacity(week, params), 3) for week in weeks])
    initial = np.array([round(point_initial(week, params), 3) for week in weeks])
    hub_caps = np.zeros(n_hubs)
    hub_stock = np.zeros(n_hubs)
    for h in range(n_hubs):
        rows = [i for i in range(n_points) if hub_of[i] == h]
        hub_caps[h] = round(hub_capacity(caps[rows], params), 3)
        hub_stock[h] = round(hub_initial(weeks[rows], params), 3)

    warehouse = sites["warehouse"]
    labels = [warehouse["prefix"]] + list(hubs["prefix"]) + list(points["prefix"])
    instance = Instance(
        name=instance_name(set_name, n_hubs, n_points, n_days, seed),
        set_name=set_name,
        seed=seed,
        n_hubs=n_hubs,
        n_points=n_points,
        n_days=n_days,
        labels=labels,
        cities=[warehouse["city"]] + list(hubs["city"]) + list(points["city"]),
        coordinates=np.array(
            [[warehouse["lat"], warehouse["lng"]]]
            + [[lat, lng] for lat, lng in zip(hubs["lat"], hubs["lng"], strict=True)]
            + [[lat, lng] for lat, lng in zip(points["lat"], points["lng"], strict=True)]
        ),
        hub_of=hub_of,
        demand=weeks[:, :n_days],
        point_capacity=caps,
        point_initial=initial,
        hub_capacity=hub_caps,
        hub_initial=hub_stock,
        distance_km=shortest_paths(road_km(labels, cache_labels, meters)),
        truck_capacity=params.truck_capacity,
        truck_cost_per_km=params.truck_cost_per_km,
        truck_cost_per_trip=params.truck_cost_per_trip,
        van_capacity=params.van_capacity,
        n_vans=params.n_vans,
        van_cost_per_km=params.van_cost_per_km,
        hub_holding=params.holding,
        point_holding=params.holding,
        source={
            "fold_start": params.fold_start.isoformat(),
            "fold_weeks": params.fold_weeks,
            "osrm_fetched": osrm_meta["date"],
            "osrm_dropped": osrm_meta["dropped"],
            "distance_rule": "mean of both OSRM directions, then shortest paths via instance nodes",
        },
    )
    check_assumptions(instance)
    return instance


def build_all(processed: Path, out_dir: Path, params: Params) -> list[Path]:
    """Build and save every instance of every set in the config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for set_name, spec in params.instance_sets.items():
        for n_hubs in spec.hubs:
            for n_points in spec.points:
                for seed in spec.seeds:
                    instance = build_instance(
                        set_name, n_hubs, n_points, spec.days, seed, processed, params
                    )
                    path = out_dir / f"{instance.name}.json"
                    save_instance(instance, path)
                    paths.append(path)
    return paths
