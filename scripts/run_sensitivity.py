"""Holding cost sensitivity: greedy and ALNS per multiplier, into results/runs_sensitivity.csv."""

from irp2e.config import load_params
from irp2e.experiments import parse_run_args, run_jobs, sensitivity_jobs

if __name__ == "__main__":
    params = load_params()
    args = parse_run_args(__doc__, params)
    jobs = sensitivity_jobs(params, args.seeds, args.time_limit, args.out)
    run_jobs(jobs, args.out / "runs_sensitivity.csv", args.workers)
