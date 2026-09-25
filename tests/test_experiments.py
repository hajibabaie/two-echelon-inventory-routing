"""Tests for the experiment runner: rows match the checked solution files, the tiny golden costs."""

import pytest

from irp2e.config import load_params
from irp2e.evaluation import check_solution
from irp2e.experiments import Job, run_job, small_jobs
from irp2e.instance import load_instance
from irp2e.solution import load_solution

from tiny_example import TINY


def test_greedy_job_row_and_file_hold_the_tiny_golden_cost(tmp_path):
    row = run_job(Job("test", TINY, "greedy", None, None, 1.0, tmp_path))
    # design 11.3: truck 250 + van 8 + 12 + 10 + holding (hub 15 + A 10) = 305
    assert (row["truck_cost"], row["van_cost"], row["holding_cost"]) == (250.0, 30.0, 25.0)
    assert row["total_cost"] == 305.0
    assert (row["n_truck_trips"], row["n_van_routes"], row["n_visits"]) == (1, 3, 3)
    solution, meta = load_solution(tmp_path / row["solution_file"])
    assert check_solution(solution, load_instance(TINY)) == []
    assert meta["cost"]["total"] == 305.0


def test_job_scales_both_holding_costs(tmp_path):
    row = run_job(Job("test", TINY, "greedy", None, None, 10.0, tmp_path))
    assert row["holding_cost"] == 250.0  # the same 25 kg-days at 10 BRL per kg per day
    assert row["total_cost"] == 530.0
    assert row["solution_file"] == "solutions/test/tiny__greedy__det__h10.json"


def test_alns_job_reaches_the_tiny_optimum(tmp_path):
    row = run_job(Job("test", TINY, "alns", 1, 1.0, 1.0, tmp_path))
    # golden 303: the MIP and a full enumeration of the 102 feasible plans agree (design 11)
    assert row["total_cost"] == pytest.approx(303.0, abs=1e-9)
    assert row["iterations"] > 0
    solution, meta = load_solution(tmp_path / row["solution_file"])
    assert check_solution(solution, load_instance(TINY)) == []
    assert meta["stats"]["history"][-1][1] == pytest.approx(303.0, abs=1e-9)


@pytest.mark.gurobi
def test_gurobi_and_alns_jobs_agree_on_the_tiny_optimum(tmp_path):
    pytest.importorskip("gurobipy")
    exact = run_job(Job("test", TINY, "gurobi", None, 60.0, 1.0, tmp_path))
    search = run_job(Job("test", TINY, "alns", 2, 1.0, 1.0, tmp_path))
    assert exact["status"] == "optimal"
    assert exact["bound"] == pytest.approx(303.0, abs=1e-6)
    assert exact["total_cost"] == pytest.approx(303.0, abs=1e-6)
    assert search["total_cost"] == pytest.approx(exact["total_cost"], abs=1e-6)


def test_small_protocol_points_at_the_frozen_files(tmp_path):
    params = load_params()
    jobs = small_jobs(params, params.seeds, None, tmp_path)
    assert all(job.instance_path.exists() for job in jobs)
