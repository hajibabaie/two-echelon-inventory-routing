"""Tests for the analysis: seed pairing, the Wilcoxon verdicts, gaps, and the re-check of files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irp2e.analysis import comparisons, recheck, signed_rank_test, wilcoxon_table, with_gaps
from irp2e.solution import Solution, save_solution

from tiny_example import TINY, tiny_optimum


def cell_rows(method: str, costs: dict) -> list[dict]:
    """Runs of one method in one cell; costs maps seed (or None) to total cost."""
    return [
        {
            "part": "compare",
            "set": "large",
            "instance": "x",
            "holding_multiplier": 1.0,
            "method": method,
            "seed": seed,
            "status": None,
            "total_cost": cost,
            "van_cost": cost / 2,
        }
        for seed, cost in costs.items()
    ]


def test_wilcoxon_pairs_runs_by_seed_not_by_row_order():
    alns = {seed: 100.0 + 10 * seed for seed in range(1, 7)}
    matheuristic = {seed: alns[seed] - 1.0 for seed in reversed(range(1, 7))}  # rows reversed
    runs = pd.DataFrame(cell_rows("alns", alns) + cell_rows("matheuristic", matheuristic))
    [row] = wilcoxon_table(runs, alpha=0.05).to_dict("records")
    assert (row["method_a"], row["method_b"], row["n_nonzero"]) == ("alns", "matheuristic", 6)
    # six differences of +1: exact two-sided p = 2 / 2^6 = 0.03125 < 0.05, the matheuristic is lower
    assert row["p_value"] == pytest.approx(0.03125)
    assert row["verdict"] == "b lower"


def test_seeded_method_is_tested_against_the_one_greedy_run():
    runs = pd.DataFrame(cell_rows("alns", {1: 90.0, 2: 95.0}) + cell_rows("greedy", {None: 100.0}))
    [(a, b, differences)] = comparisons(runs)
    assert (a, b) == ("alns", "greedy")
    assert differences.tolist() == [-10.0, -5.0]


def test_float_noise_is_no_difference():
    assert signed_rank_test(np.array([1e-9, -1e-9, 0.0]), alpha=0.05)["verdict"] == "all equal"


def test_gap_is_to_the_proven_optimum_else_to_the_best_known_cost():
    proven = cell_rows("gurobi", {None: 100.0}) + cell_rows("alns", {1: 103.0})
    proven[0]["status"] = "optimal"
    open_cell = cell_rows("gurobi", {None: 110.0}) + cell_rows("alns", {1: 105.0})
    open_cell[0]["status"] = "time_limit"
    for row in open_cell:
        row["instance"] = "y"
    gaps = with_gaps(pd.DataFrame(proven + open_cell)).set_index(["instance", "method"])
    assert gaps.loc[("x", "alns"), "gap_pct"] == pytest.approx(3.0)  # (103 - 100) / 100
    assert gaps.loc[("x", "alns"), "van_gap_pct"] == pytest.approx(3.0)  # (51.5 - 50) / 50
    assert gaps.loc[("y", "gurobi"), "reference_kind"] == "best known"
    assert gaps.loc[("y", "gurobi"), "gap_pct"] == pytest.approx(100 * 5 / 105)
    assert np.isnan(gaps.loc[("y", "alns"), "van_gap_pct"])  # no van gap without a proof


def saved_run(tmp_path: Path, solution: Solution, total_cost: float) -> pd.DataFrame:
    """One runs row that points at the saved solution."""
    save_solution(solution, {}, tmp_path / "s.json")
    row = {"instance": "tiny", "holding_multiplier": 1.0, "solution_file": "s.json"}
    return pd.DataFrame([{**row, "total_cost": total_cost}])


def test_recheck_passes_the_tiny_optimum(tmp_path):
    recheck(saved_run(tmp_path, tiny_optimum(), 303.0), tmp_path, TINY.parent)


def test_recheck_stops_on_a_saved_stockout(tmp_path):
    broken = tiny_optimum()
    broken.deliveries[0, 0] = 9.0  # A needs 10 on day 0
    with pytest.raises(ValueError, match="stockout: point 0, day 0"):
        recheck(saved_run(tmp_path, broken, 302.0), tmp_path, TINY.parent)


def test_recheck_stops_on_a_wrong_reported_cost(tmp_path):
    with pytest.raises(ValueError, match="reported cost"):
        recheck(saved_run(tmp_path, tiny_optimum(), 302.0), tmp_path, TINY.parent)
