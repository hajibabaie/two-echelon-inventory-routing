"""Read and check config/params.toml, the one file that holds every parameter value."""

import datetime
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "params.toml"


@dataclass(frozen=True)
class InstanceSet:
    hubs: list[int]
    points: list[int]
    days: int
    seeds: list[int]


@dataclass(frozen=True)
class Params:
    kaggle_dataset: str
    state: str
    order_status: str
    fold_start: datetime.date
    fold_weeks: int
    sp_box_lat: tuple[float, float]
    sp_box_lng: tuple[float, float]
    pool_radius_km: float
    pool_size: int
    osrm_url: str
    osrm_block: int
    osrm_pause_s: float
    osrm_timeout_s: float
    osrm_user_agent: str
    truck_capacity: float
    truck_cost_per_km: float
    truck_cost_per_trip: float
    van_capacity: float
    n_vans: int
    van_cost_per_km: float
    goods_value_per_kg: float
    selic_annual: float
    point_capacity_days: float
    point_initial_days: float
    hub_capacity_factor: float
    hub_initial_days: float
    instance_sets: dict[str, InstanceSet]
    segment_length: int
    reaction: float
    score_best: float
    score_better: float
    score_accepted: float
    start_worse: float
    start_accept_probability: float
    final_temperature_ratio: float
    removal_share: float
    removal_cap_min: int
    removal_cap_max: int
    mip_gap: float
    gurobi_time_limit_s: float
    gurobi_threads: int
    seeds: list[int]
    time_limit_s: dict[str, float]
    workers: int
    holding_multipliers: list[float]
    sensitivity_instance: str
    map_instance: str
    alpha: float
    holding: float  # BRL per kg per day, computed from goods_value_per_kg and selic_annual


def _table(parent: dict, where: str, key: str) -> dict:
    """One table of the config; ValueError names a missing table."""
    if not isinstance(parent.get(key), dict):
        raise ValueError(f"config table {where}{key} is missing")
    return parent[key]


def _value(section: dict, where: str, key: str):
    """One raw value; ValueError names a missing key."""
    if key not in section:
        raise ValueError(f"config key {where}.{key} is missing")
    return section[key]


def _is_number(value) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _positive(section: dict, where: str, key: str) -> float:
    """A number greater than 0."""
    value = _value(section, where, key)
    if not _is_number(value) or value <= 0:
        raise ValueError(f"config key {where}.{key} must be a number > 0, got {value!r}")
    return float(value)


def _non_negative(section: dict, where: str, key: str) -> float:
    """A number at least 0."""
    value = _value(section, where, key)
    if not _is_number(value) or value < 0:
        raise ValueError(f"config key {where}.{key} must be a number >= 0, got {value!r}")
    return float(value)


def _positive_int(section: dict, where: str, key: str) -> int:
    """An integer greater than 0."""
    value = _value(section, where, key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config key {where}.{key} must be an integer > 0, got {value!r}")
    return value


def _text(section: dict, where: str, key: str) -> str:
    """A non-empty string."""
    value = _value(section, where, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"config key {where}.{key} must be a non-empty string, got {value!r}")
    return value


def _box(section: dict, where: str, key: str) -> tuple[float, float]:
    """A pair [low, high] with low < high; the values may be negative."""
    value = _value(section, where, key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_number(bound) for bound in value)
        or value[0] >= value[1]
    ):
        raise ValueError(f"config key {where}.{key} must be a pair [low, high], got {value!r}")
    return (float(value[0]), float(value[1]))


def _monday(section: dict, where: str, key: str) -> datetime.date:
    """An ISO date string that falls on a Monday."""
    value = _text(section, where, key)
    try:
        day = datetime.date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"config key {where}.{key} must be a date YYYY-MM-DD") from error
    if day.weekday() != 0:
        raise ValueError(f"config key {where}.{key} must be a Monday, got {value}")
    return day


def _positive_list(section: dict, where: str, key: str, integer: bool) -> list:
    """A non-empty list of positive numbers (integers when asked)."""
    value = _value(section, where, key)
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"config key {where}.{key} must be a non-empty list, got {value!r}")
    for item in value:
        wrong_type = not isinstance(item, int) if integer else not _is_number(item)
        if wrong_type or isinstance(item, bool) or item <= 0:
            raise ValueError(f"config key {where}.{key} must hold positive numbers, got {item!r}")
    return value


def _instance_set(section: dict, name: str) -> InstanceSet:
    """One instance set of the [instances] table."""
    where = f"instances.{name}"
    values = _table(section, "instances.", name)
    return InstanceSet(
        hubs=_positive_list(values, where, "hubs", integer=True),
        points=_positive_list(values, where, "points", integer=True),
        days=_positive_int(values, where, "days"),
        seeds=_positive_list(values, where, "seeds", integer=True),
    )


def load_params(path: Path = CONFIG_PATH) -> Params:
    """Read the config file and check each key by its own rule."""
    with open(path, "rb") as file:
        table = tomllib.load(file)
    data = _table(table, "", "data")
    truck = _table(table, "", "truck")
    van = _table(table, "", "van")
    stock = _table(table, "", "stock")
    instances = _table(table, "", "instances")
    alns = _table(table, "", "alns")
    gurobi = _table(table, "", "gurobi")
    experiments = _table(table, "", "experiments")
    instance_sets = {name: _instance_set(instances, name) for name in ("small", "medium", "large")}
    limits = _table(experiments, "experiments.", "time_limit_s")
    goods_value_per_kg = _positive(stock, "stock", "goods_value_per_kg")
    selic_annual = _positive(stock, "stock", "selic_annual")
    return Params(
        kaggle_dataset=_text(data, "data", "kaggle_dataset"),
        state=_text(data, "data", "state"),
        order_status=_text(data, "data", "order_status"),
        fold_start=_monday(data, "data", "fold_start"),
        fold_weeks=_positive_int(data, "data", "fold_weeks"),
        sp_box_lat=_box(data, "data", "sp_box_lat"),
        sp_box_lng=_box(data, "data", "sp_box_lng"),
        pool_radius_km=_positive(data, "data", "pool_radius_km"),
        pool_size=_positive_int(data, "data", "pool_size"),
        osrm_url=_text(data, "data", "osrm_url"),
        osrm_block=_positive_int(data, "data", "osrm_block"),
        osrm_pause_s=_positive(data, "data", "osrm_pause_s"),
        osrm_timeout_s=_positive(data, "data", "osrm_timeout_s"),
        osrm_user_agent=_text(data, "data", "osrm_user_agent"),
        truck_capacity=_positive(truck, "truck", "capacity_kg"),
        truck_cost_per_km=_positive(truck, "truck", "cost_per_km"),
        truck_cost_per_trip=_positive(truck, "truck", "cost_per_trip"),
        van_capacity=_positive(van, "van", "capacity_kg"),
        n_vans=_positive_int(van, "van", "count_per_hub"),
        van_cost_per_km=_positive(van, "van", "cost_per_km"),
        goods_value_per_kg=goods_value_per_kg,
        selic_annual=selic_annual,
        point_capacity_days=_positive(stock, "stock", "point_capacity_days"),
        point_initial_days=_positive(stock, "stock", "point_initial_days"),
        hub_capacity_factor=_positive(stock, "stock", "hub_capacity_factor"),
        hub_initial_days=_positive(stock, "stock", "hub_initial_days"),
        instance_sets=instance_sets,
        segment_length=_positive_int(alns, "alns", "segment_length"),
        reaction=_positive(alns, "alns", "reaction"),
        score_best=_positive(alns, "alns", "score_best"),
        score_better=_positive(alns, "alns", "score_better"),
        score_accepted=_positive(alns, "alns", "score_accepted"),
        start_worse=_positive(alns, "alns", "start_worse"),
        start_accept_probability=_positive(alns, "alns", "start_accept_probability"),
        final_temperature_ratio=_positive(alns, "alns", "final_temperature_ratio"),
        removal_share=_positive(alns, "alns", "removal_share"),
        removal_cap_min=_positive_int(alns, "alns", "removal_cap_min"),
        removal_cap_max=_positive_int(alns, "alns", "removal_cap_max"),
        mip_gap=_non_negative(gurobi, "gurobi", "mip_gap"),
        gurobi_time_limit_s=_positive(gurobi, "gurobi", "time_limit_s"),
        gurobi_threads=_positive_int(gurobi, "gurobi", "threads"),
        seeds=_positive_list(experiments, "experiments", "seeds", integer=True),
        time_limit_s={
            name: _positive(limits, "experiments.time_limit_s", name) for name in instance_sets
        },
        workers=_positive_int(experiments, "experiments", "workers"),
        holding_multipliers=_positive_list(
            experiments, "experiments", "holding_multipliers", integer=False
        ),
        sensitivity_instance=_text(experiments, "experiments", "sensitivity_instance"),
        map_instance=_text(experiments, "experiments", "map_instance"),
        alpha=_positive(experiments, "experiments", "alpha"),
        holding=goods_value_per_kg * selic_annual / 365,
    )
