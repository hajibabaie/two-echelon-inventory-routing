"""Medium and large sets: greedy, ALNS and the matheuristic, into runs_compare.csv."""

from irp2e.config import load_params
from irp2e.experiments import compare_jobs, parse_run_args, run_jobs

if __name__ == "__main__":
    params = load_params()
    args = parse_run_args(__doc__, params)
    jobs = compare_jobs(params, args.seeds, args.time_limit, args.out)
    run_jobs(jobs, args.out / "runs_compare.csv", args.workers)
