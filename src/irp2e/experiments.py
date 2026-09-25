"""Run the experiment protocol: one job per instance, method and seed, in parallel, each checked.

gurobipy is an optional extra, so the Gurobi methods import it inside their run function.
"""

import argparse
import csv
import dataclasses
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from irp2e.config import REPO_ROOT, Params, load_params
from irp2e.evaluation import assert_feasible
from irp2e.generator import instance_name
from irp2e.heuristics.alns import Alns
from irp2e.heuristics.greedy import solve_greedy
from irp2e.heuristics.operators import DESTROY_OPERATORS, REPAIR_OPERATORS
from irp2e.heuristics.quantities import jit_quantities
from irp2e.instance import Instance, load_instance, with_holding_multiplier
from irp2e.solution import Solution, save_solution

INSTANCES_DIR = REPO_ROOT / "data" / "instances"
RESULTS_DIR = REPO_ROOT / "results"
SEEDED_METHODS = ("alns", "matheuristic")  # gurobi and greedy use no random numbers: one run each
STAT_COLUMNS = [
    "iterations",
    "rejected",
    "lp_calls",
    "lp_rescues",
    "status",
    "objective",
    "bound",
    "gap",
]
INTEGER_COLUMNS = ["seed", "iterations", "rejected", "lp_calls", "lp_rescues"]
COLUMNS = [
    "part",
    "instance",
    "set",
    "method",
    "seed",
    "holding_multiplier",
    "time_limit_s",
    "total_cost",
    "truck_cost",
    "van_cost",
    "holding_cost",
    "n_truck_trips",
    "n_van_routes",
    "n_visits",
    "runtime_s",
    *STAT_COLUMNS,
    "solution_file",
]


@dataclass(frozen=True)
class Job:
    part: str  # small, compare or sensitivity: the script that made the job
    instance_path: Path
    method: str
    seed: int | None  # None for gurobi and greedy
    time_limit_s: float | None  # None for greedy: it stops by itself
    holding_multiplier: float
    out_dir: Path  # results folder; the solution goes to out_dir/solutions/<part>/


# ----- one method each -----


def run_gurobi(instance: Instance, job: Job, params: Params) -> tuple[Solution, dict]:
    """The exact MIP; its status, objective, bound and gap go into the row."""
    from irp2e.model import solve_mip

    solution, result = solve_mip(instance, job.time_limit_s, params.gurobi_threads, params.mip_gap)
    stats = {
        "status": result.status,
        "objective": result.objective,
        "bound": result.bound,
        "gap": result.gap,
    }
    return solution, stats


def run_greedy(instance: Instance, job: Job, params: Params) -> tuple[Solution, dict]:
    """The greedy baseline; it has no statistics."""
    return solve_greedy(instance), {}


def search_stats(alns: Alns) -> dict:
    """Counters, best-cost history and final operator weights of one ALNS run."""
    return {
        "iterations": alns.iterations,
        "rejected": alns.rejected,
        "accepted": alns.accepted,
        "history": alns.history,
        "destroy_weights": {
            name: [float(weight), int(uses)]
            for (name, _), weight, uses in zip(
                DESTROY_OPERATORS, alns.destroy_weights, alns.destroy_uses, strict=True
            )
        },
        "repair_weights": {
            name: [float(weight), int(uses)]
            for (name, _), weight, uses in zip(
                REPAIR_OPERATORS, alns.repair_weights, alns.repair_uses, strict=True
            )
        },
    }


def run_alns(instance: Instance, job: Job, params: Params) -> tuple[Solution, dict]:
    """ALNS with rule J1 for the quantities, seeded once, stopped by the clock."""
    np.random.seed(job.seed)
    alns = Alns(instance, params, jit_quantities)
    solution = alns.run(job.time_limit_s, by_iterations=False)
    return solution, search_stats(alns)


def run_matheuristic(instance: Instance, job: Job, params: Params) -> tuple[Solution, dict]:
    """The same ALNS with rule M1 (J1 first, the Gurobi LP when J1 fails)."""
    from irp2e.heuristics.matheuristic import make_matheuristic

    np.random.seed(job.seed)
    alns = make_matheuristic(instance, params)
    solution = alns.run(job.time_limit_s, by_iterations=False)
    alns.quantity_step.lp.close()
    stats = search_stats(alns)
    stats["lp_calls"] = alns.quantity_step.lp_calls
    stats["lp_rescues"] = alns.quantity_step.lp_rescues
    return solution, stats


METHODS = {
    "gurobi": run_gurobi,
    "greedy": run_greedy,
    "alns": run_alns,
    "matheuristic": run_matheuristic,
}


# ----- one job -----


def solution_file(job: Job) -> str:
    """solutions/<part>/<instance>__<method>__s<seed>__h<multiplier>.json, relative to out_dir."""
    seed = "det" if job.seed is None else f"s{job.seed}"  # det: a one-run method has no seed
    name = f"{job.instance_path.stem}__{job.method}__{seed}__h{job.holding_multiplier:g}.json"
    return f"solutions/{job.part}/{name}"


def count_routes(solution: Solution) -> tuple[int, int]:
    """(number of van routes, number of visits) of a solution."""
    routes = [route for hub in solution.van_routes for day in hub for route in day]
    return len(routes), sum(len(route) for route in routes)


def run_job(job: Job) -> dict:
    """Solve one job, check the solution with the one checker, save it, and return its row."""
    params = load_params()  # each worker reads the config itself
    instance = with_holding_multiplier(load_instance(job.instance_path), job.holding_multiplier)
    start = time.perf_counter()
    solution, stats = METHODS[job.method](instance, job, params)
    runtime = time.perf_counter() - start
    cost = assert_feasible(solution, instance)  # the one checker; a violation stops the run
    n_routes, n_visits = count_routes(solution)
    row = dict.fromkeys(COLUMNS)
    row.update({key: stats[key] for key in STAT_COLUMNS if key in stats})
    row.update(
        {
            "part": job.part,
            "instance": job.instance_path.stem,
            "set": instance.set_name,
            "method": job.method,
            "seed": job.seed,
            "holding_multiplier": job.holding_multiplier,
            "time_limit_s": job.time_limit_s,
            "total_cost": cost.total,
            "truck_cost": cost.truck,
            "van_cost": cost.van,
            "holding_cost": cost.holding,
            "n_truck_trips": int(np.sum(solution.truck_trips)),
            "n_van_routes": n_routes,
            "n_visits": n_visits,
            "runtime_s": runtime,
            "solution_file": solution_file(job),
        }
    )
    meta = {
        "instance": job.instance_path.stem,
        "method": job.method,
        "seed": job.seed,
        "holding_multiplier": job.holding_multiplier,
        "time_limit_s": job.time_limit_s,
        "cost": dataclasses.asdict(cost),
        "stats": stats,
    }
    path = job.out_dir / solution_file(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_solution(solution, meta, path)
    return row


# ----- the protocol -----


def set_paths(set_name: str, params: Params, instances_dir: Path) -> list[Path]:
    """Frozen instance files of one set, in the order of the config."""
    spec = params.instance_sets[set_name]
    return [
        instances_dir / f"{instance_name(set_name, k, n, spec.days, seed)}.json"
        for k in spec.hubs
        for n in spec.points
        for seed in spec.seeds
    ]


def method_jobs(
    part: str,
    paths: list[Path],
    method: str,
    seeds: list[int],
    time_limit_s: float | None,
    multiplier: float,
    out: Path,
) -> list[Job]:
    """One job per instance and seed for a seeded method; one job per instance otherwise."""
    run_seeds = seeds if method in SEEDED_METHODS else [None]
    return [
        Job(part, path, method, seed, time_limit_s, multiplier, out)
        for path in paths
        for seed in run_seeds
    ]


def time_limit(configured: float, override: float | None) -> float:
    """The configured time limit, or the --time-limit of the command line (smoke runs)."""
    return configured if override is None else override


def small_jobs(params: Params, seeds: list[int], override: float | None, out: Path) -> list[Job]:
    """Small set: Gurobi (the optimum), greedy, ALNS and the matheuristic per seed."""
    paths = set_paths("small", params, INSTANCES_DIR)
    gurobi_limit = time_limit(params.gurobi_time_limit_s, override)
    alns_limit = time_limit(params.time_limit_s["small"], override)
    return (
        method_jobs("small", paths, "gurobi", seeds, gurobi_limit, 1.0, out)
        + method_jobs("small", paths, "greedy", seeds, None, 1.0, out)
        + method_jobs("small", paths, "alns", seeds, alns_limit, 1.0, out)
        + method_jobs("small", paths, "matheuristic", seeds, alns_limit, 1.0, out)
    )


def compare_jobs(params: Params, seeds: list[int], override: float | None, out: Path) -> list[Job]:
    """Medium and large sets: greedy, ALNS and the matheuristic."""
    medium = set_paths("medium", params, INSTANCES_DIR)
    large = set_paths("large", params, INSTANCES_DIR)
    medium_limit = time_limit(params.time_limit_s["medium"], override)
    large_limit = time_limit(params.time_limit_s["large"], override)
    return (
        method_jobs("compare", medium, "greedy", seeds, None, 1.0, out)
        + method_jobs("compare", medium, "alns", seeds, medium_limit, 1.0, out)
        + method_jobs("compare", medium, "matheuristic", seeds, medium_limit, 1.0, out)
        + method_jobs("compare", large, "greedy", seeds, None, 1.0, out)
        + method_jobs("compare", large, "alns", seeds, large_limit, 1.0, out)
        + method_jobs("compare", large, "matheuristic", seeds, large_limit, 1.0, out)
    )


def sensitivity_jobs(
    params: Params, seeds: list[int], override: float | None, out: Path
) -> list[Job]:
    """The sensitivity instance with every holding multiplier: greedy and ALNS."""
    path = [INSTANCES_DIR / f"{params.sensitivity_instance}.json"]
    limit = time_limit(params.time_limit_s[load_instance(path[0]).set_name], override)
    jobs = []
    for multiplier in params.holding_multipliers:
        jobs += method_jobs("sensitivity", path, "greedy", seeds, None, multiplier, out)
        jobs += method_jobs("sensitivity", path, "alns", seeds, limit, multiplier, out)
    return jobs


# ----- parallel runner and command line -----


def planned_seconds(job: Job) -> float:
    """Time limit of a job; greedy takes well under a second."""
    return 0.0 if job.time_limit_s is None else job.time_limit_s


def run_jobs(jobs: list[Job], csv_path: Path, workers: int) -> None:
    """Run the jobs in a process pool, longest first, one CSV row as each job ends."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(jobs, key=planned_seconds, reverse=True)  # a 600 s MIP never starts last
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        with multiprocessing.Pool(workers) as pool:
            for done, row in enumerate(pool.imap_unordered(run_job, ordered), start=1):
                writer.writerow(row)
                file.flush()  # a crash later keeps every finished row
                print(
                    f"[{done}/{len(ordered)}] {row['instance']} {row['method']} "
                    f"seed {row['seed']} h{row['holding_multiplier']:g}: "
                    f"{row['total_cost']:.2f} BRL in {row['runtime_s']:.1f} s",
                    flush=True,
                )
    runs = read_runs_csv(csv_path)
    runs.sort_values(["instance", "holding_multiplier", "method", "seed"]).to_csv(
        csv_path, index=False
    )


def read_runs_csv(path: Path) -> pd.DataFrame:
    """One runs file; count columns stay integers although some cells are empty."""
    return pd.read_csv(path, dtype=dict.fromkeys(INTEGER_COLUMNS, "Int64"))


def parse_run_args(description: str, params: Params) -> argparse.Namespace:
    """Command line shared by the run scripts; defaults come from the config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="results folder")
    parser.add_argument("--seeds", type=int, nargs="+", default=params.seeds)
    parser.add_argument("--workers", type=int, default=params.workers)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="replace every time limit of the config, Gurobi included (for smoke runs)",
    )
    args = parser.parse_args()
    if args.workers < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("--workers must be at least 1 and seeds must be >= 0")
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("--time-limit must be > 0")
    return args
