"""Small set: Gurobi, greedy, ALNS and the matheuristic per seed, into results/runs_small.csv."""

from irp2e.config import load_params
from irp2e.experiments import parse_run_args, run_jobs, small_jobs

if __name__ == "__main__":
    params = load_params()
    args = parse_run_args(__doc__, params)
    jobs = small_jobs(params, args.seeds, args.time_limit, args.out)
    run_jobs(jobs, args.out / "runs_small.csv", args.workers)
