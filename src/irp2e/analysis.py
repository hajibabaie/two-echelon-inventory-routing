"""Result tables: re-check every saved solution, then gaps, summaries, Wilcoxon, sensitivity."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from irp2e.config import load_params
from irp2e.evaluation import assert_feasible, check_reported_cost
from irp2e.experiments import INSTANCES_DIR, RESULTS_DIR, SEEDED_METHODS, read_runs_csv
from irp2e.instance import load_instance, with_holding_multiplier
from irp2e.plots import plot_gap_boxplots, plot_network, plot_sensitivity
from irp2e.solution import load_solution

CELL = ["part", "set", "instance", "holding_multiplier"]  # runs in one cell are compared
TEST_COLUMNS = ["n_pairs", "n_nonzero", "median_difference", "statistic", "p_value", "verdict"]
COST_DECIMALS = 6  # costs are rounded to 1e-6 BRL before a test, so float noise is no difference


# ----- read and re-check -----


def read_runs(results: Path) -> pd.DataFrame:
    """Every runs_*.csv of the results folder as one table."""
    paths = sorted(results.glob("runs_*.csv"))
    if len(paths) == 0:
        raise FileNotFoundError(f"no runs_*.csv file in {results}")
    for path in paths:
        print(f"read {path}")
    return pd.concat([read_runs_csv(path) for path in paths], ignore_index=True)


def recheck(runs: pd.DataFrame, results: Path, instances: Path) -> None:
    """Checker and reported-cost check on every saved solution; stop at the first failure."""
    for row in runs.itertuples():
        instance = load_instance(instances / f"{row.instance}.json")
        instance = with_holding_multiplier(instance, row.holding_multiplier)
        solution, _ = load_solution(results / row.solution_file)
        cost = assert_feasible(solution, instance)
        problems = check_reported_cost(row.total_cost, cost)
        if len(problems) > 0:
            raise ValueError(f"{row.solution_file}: {problems[0]}")


# ----- gaps -----


def reference_cost(cell: pd.DataFrame) -> dict:
    """The proven optimum if Gurobi proved one in this cell; else the lowest cost of any run."""
    proven = cell[(cell["method"] == "gurobi") & (cell["status"] == "optimal")]
    if len(proven) > 0:
        return {
            "reference_kind": "optimum",
            "reference_cost": float(proven["total_cost"].iloc[0]),
            "reference_van_cost": float(proven["van_cost"].iloc[0]),
        }
    return {
        "reference_kind": "best known",
        "reference_cost": float(cell["total_cost"].min()),
        "reference_van_cost": np.nan,  # a van gap is measured only against a proven optimum
    }


def with_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """The runs plus their reference cost, gap and van gap in percent."""
    references = []
    for keys, cell in runs.groupby(CELL):
        references.append({**dict(zip(CELL, keys, strict=True)), **reference_cost(cell)})
    frame = runs.merge(pd.DataFrame(references), on=CELL)
    frame["gap_pct"] = (
        100 * (frame["total_cost"] - frame["reference_cost"]) / frame["reference_cost"]
    )
    frame["van_gap_pct"] = (
        100 * (frame["van_cost"] - frame["reference_van_cost"]) / frame["reference_van_cost"]
    )
    return frame


def gap_table(gaps: pd.DataFrame) -> pd.DataFrame:
    """Small set: gap of each method to the Gurobi optimum, per instance."""
    small = gaps[gaps["set"] == "small"]
    return (
        small.groupby(["instance", "method"])
        .agg(
            reference_kind=("reference_kind", "first"),
            reference_cost=("reference_cost", "first"),
            n_runs=("total_cost", "size"),
            best_gap_pct=("gap_pct", "min"),
            mean_gap_pct=("gap_pct", "mean"),
            max_gap_pct=("gap_pct", "max"),
            mean_van_gap_pct=("van_gap_pct", "mean"),
            mean_truck_trips=("n_truck_trips", "mean"),
        )
        .reset_index()
    )


def summary_table(gaps: pd.DataFrame) -> pd.DataFrame:
    """Best, mean and standard deviation of the cost per method in every cell."""
    return (
        gaps.groupby([*CELL, "method"])
        .agg(
            n_runs=("total_cost", "size"),
            best=("total_cost", "min"),
            mean=("total_cost", "mean"),
            std=("total_cost", "std"),
            reference_kind=("reference_kind", "first"),
            mean_gap_pct=("gap_pct", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_iterations=("iterations", "mean"),
            mean_lp_rescues=("lp_rescues", "mean"),
        )
        .reset_index()
    )


# ----- Wilcoxon signed-rank tests -----


def seed_costs(cell: pd.DataFrame, method: str) -> pd.DataFrame:
    """Seed and total cost of the runs of one method."""
    return cell[cell["method"] == method][["seed", "total_cost"]]


def comparisons(cell: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(a, b, cost a - cost b): seeded methods paired by seed, and each seeded method vs greedy."""
    present = set(cell["method"])
    seeded = [method for method in SEEDED_METHODS if method in present]
    pairs = []
    for index, a in enumerate(seeded):
        for b in seeded[index + 1 :]:
            matched = seed_costs(cell, a).merge(
                seed_costs(cell, b), on="seed", suffixes=("_a", "_b")
            )  # inner join: only seeds that both methods ran form a pair
            pairs.append((a, b, (matched["total_cost_a"] - matched["total_cost_b"]).to_numpy()))
        if "greedy" in present:
            greedy = float(cell[cell["method"] == "greedy"]["total_cost"].iloc[0])
            pairs.append((a, "greedy", seed_costs(cell, a)["total_cost"].to_numpy() - greedy))
    return pairs


def signed_rank_test(differences: np.ndarray, alpha: float) -> dict:
    """Two-sided Wilcoxon signed-rank test on differences a - b; 'all equal' if none is nonzero."""
    rounded = np.round(differences, COST_DECIMALS)
    nonzero = [value for value in rounded if value != 0]
    result = {
        "n_pairs": len(rounded),
        "n_nonzero": len(nonzero),
        "median_difference": float(np.median(rounded)),
        "statistic": np.nan,
        "p_value": np.nan,
        "verdict": "all equal",
    }
    if len(nonzero) == 0:
        return result
    test = wilcoxon(rounded, zero_method="wilcox", alternative="two-sided")
    result["statistic"] = float(test.statistic)
    result["p_value"] = float(test.pvalue)
    if test.pvalue >= alpha:
        result["verdict"] = "no significant difference"
    elif np.median(nonzero) < 0:  # the side the nonzero differences lie on
        result["verdict"] = "a lower"
    else:
        result["verdict"] = "b lower"
    return result


def wilcoxon_table(runs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """One test per cell and method pair."""
    rows = []
    for keys, cell in runs.groupby(CELL):
        for a, b, differences in comparisons(cell):
            rows.append(
                {
                    **dict(zip(CELL, keys, strict=True)),
                    "method_a": a,
                    "method_b": b,
                    **signed_rank_test(differences, alpha),
                }
            )
    return pd.DataFrame(rows, columns=[*CELL, "method_a", "method_b", *TEST_COLUMNS])


def sensitivity_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean cost parts, truck trips and visits per holding multiplier and method."""
    sensitivity = runs[runs["part"] == "sensitivity"]
    return (
        sensitivity.groupby(["holding_multiplier", "method"])
        .agg(
            n_runs=("total_cost", "size"),
            total_cost=("total_cost", "mean"),
            truck_cost=("truck_cost", "mean"),
            van_cost=("van_cost", "mean"),
            holding_cost=("holding_cost", "mean"),
            n_truck_trips=("n_truck_trips", "mean"),
            n_visits=("n_visits", "mean"),
        )
        .reset_index()
    )


# ----- table files -----


def format_value(column: str, value) -> str:
    """Text of one cell: multipliers as given, p-values with 4 decimals, other numbers with 2."""
    if pd.isna(value):
        return ""
    if isinstance(value, float | np.floating):
        if column == "holding_multiplier":
            return f"{value:g}"
        if column == "p_value":
            return f"{value:.4f}"
        return f"{value:.2f}"
    return str(value)


def text_rows(frame: pd.DataFrame) -> list[list[str]]:
    """Every row of the table as formatted text."""
    return [
        [format_value(column, value) for column, value in zip(frame.columns, row, strict=True)]
        for row in frame.itertuples(index=False)
    ]


def is_number_column(frame: pd.DataFrame, column: str) -> bool:
    """True for a numeric column that is not a yes/no column."""
    kind = frame[column].dtype
    return pd.api.types.is_numeric_dtype(kind) and not pd.api.types.is_bool_dtype(kind)


def markdown_table(frame: pd.DataFrame) -> str:
    """A pipe table; numbers right-aligned."""
    align = ["---:" if is_number_column(frame, column) else "---" for column in frame.columns]
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in text_rows(frame)]
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    """Escape the characters LaTeX reads as commands in table text."""
    for char in "_%&#":
        text = text.replace(char, "\\" + char)
    return text


def latex_table(frame: pd.DataFrame) -> str:
    """A booktabs tabular; numbers right-aligned."""
    align = "".join("r" if is_number_column(frame, column) else "l" for column in frame.columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(latex_escape(column) for column in frame.columns) + " \\\\")
    lines.append("\\midrule")
    lines += [" & ".join(latex_escape(cell) for cell in row) + " \\\\" for row in text_rows(frame)]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, name: str, out: Path) -> None:
    """<name>.csv with full precision, <name>.md and <name>.tex rounded for reading."""
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(markdown_table(frame), encoding="utf-8")
    (out / f"{name}.tex").write_text(latex_table(frame), encoding="utf-8")
    print(f"wrote {out / name}.csv, .md and .tex ({len(frame)} rows)")


def main() -> None:
    """Re-check every solution, then write the tables and the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="folder of runs_*.csv")
    parser.add_argument("--instances", type=Path, default=INSTANCES_DIR)
    args = parser.parse_args()
    params = load_params()
    runs = read_runs(args.results)
    recheck(runs, args.results, args.instances)
    print(f"{len(runs)} saved solutions re-checked: all feasible, all reported costs match")
    gaps = with_gaps(runs)
    tables = args.results / "tables"
    write_table(gap_table(gaps), "gap_to_optimum", tables)
    write_table(summary_table(gaps), "summary", tables)
    write_table(wilcoxon_table(runs, params.alpha), "wilcoxon", tables)
    sensitivity = sensitivity_table(runs)
    write_table(sensitivity, "sensitivity", tables)
    figures = args.results / "figures"
    network = load_instance(args.instances / f"{params.map_instance}.json")
    plot_network(network, figures / "network_map")
    plot_gap_boxplots(gaps, figures / "gap_boxplots")
    if len(sensitivity) > 0:
        plot_sensitivity(sensitivity, figures / "sensitivity")
    else:
        print("no sensitivity runs yet: sensitivity figure not drawn")
