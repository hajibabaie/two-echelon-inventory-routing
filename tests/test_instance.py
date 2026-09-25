"""Tests for the instance file format and the frozen instance sets."""

import dataclasses
import json

import numpy as np
import pytest

from irp2e.config import REPO_ROOT, load_params
from irp2e.generator import build_instance
from irp2e.instance import load_instance, save_instance

from tiny_example import INSTANCES, TINY

PROCESSED = REPO_ROOT / "data" / "processed"
FROZEN = sorted(INSTANCES.glob("*.json"))


def test_instance_json_round_trip(tmp_path):
    tiny = load_instance(TINY)
    save_instance(tiny, tmp_path / "tiny.json")
    again = load_instance(tmp_path / "tiny.json")
    for field in dataclasses.fields(tiny):
        before, after = getattr(tiny, field.name), getattr(again, field.name)
        if isinstance(before, np.ndarray):
            assert before.dtype == after.dtype and np.array_equal(before, after), field.name
        else:
            assert before == after, field.name


def test_tiny_trip_cost_counts_both_legs():
    # 50 BRL per trip + 2 x 100 km x 1 BRL per km
    assert load_instance(TINY).trip_cost(0) == 250.0


def test_load_instance_rejects_a_wrong_demand_shape(tmp_path):
    record = json.loads(TINY.read_text(encoding="utf-8"))
    record["demand"] = record["demand"][:2]  # 2 rows for 3 points
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="demand"):
        load_instance(path)


def test_load_instance_rejects_a_negative_distance(tmp_path):
    record = json.loads(TINY.read_text(encoding="utf-8"))
    record["distance_km"][1][2] = -4.0
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="distance_km"):
        load_instance(path)


@pytest.mark.parametrize(
    ("field", "value", "rule"),
    [
        ("point_capacity", [8.0, 20.0, 20.0], "C1: point 0"),  # A needs 10 kg on day 0
        ("hub_capacity", [25.0], "C3: hub 0"),  # its points can take 60 kg in one day
        ("params", {"van_capacity": 15.0}, "A3: point 0"),  # A can take 20 kg
    ],
)
def test_load_instance_rejects_a_file_that_breaks_a_rule_the_heuristics_need(
    tmp_path, field, value, rule
):
    # each broken rule made the repair or the greedy loop forever before load_instance checked it
    record = json.loads(TINY.read_text(encoding="utf-8"))
    if field == "params":
        record["params"].update(value)
    else:
        record[field] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=rule):
        load_instance(path)


def test_every_configured_instance_is_frozen():
    params = load_params()
    expected = sum(
        len(spec.hubs) * len(spec.points) * len(spec.seeds)
        for spec in params.instance_sets.values()
    )
    assert len(FROZEN) == expected == 19


@pytest.mark.parametrize("path", FROZEN, ids=[path.stem for path in FROZEN])
def test_frozen_instances_rebuild_identically(path, tmp_path):
    frozen = load_instance(path)
    rebuilt = build_instance(
        frozen.set_name,
        frozen.n_hubs,
        frozen.n_points,
        frozen.n_days,
        frozen.seed,
        PROCESSED,
        load_params(),
    )  # also runs check_assumptions (rules A1 to A4, C1 and C3)
    save_instance(rebuilt, tmp_path / path.name)
    with open(path, encoding="utf-8") as file:
        frozen_record = json.load(file)
    with open(tmp_path / path.name, encoding="utf-8") as file:
        rebuilt_record = json.load(file)
    assert rebuilt_record == frozen_record
