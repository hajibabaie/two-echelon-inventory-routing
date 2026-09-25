"""Write report/tables/*.tex from the result CSVs, the config and the instance files.

Trap: nCpus, gurobiVersion and pythonVersion describe the machine that runs this script, so run it
on the machine that ran the experiments.
"""

import json
import os
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

from irp2e.analysis import read_runs, with_gaps
from irp2e.config import REPO_ROOT, Params, load_params
from irp2e.experiments import INSTANCES_DIR, RESULTS_DIR
from irp2e.instance import Instance, load_instance

OUT = REPO_ROOT / "report" / "tables"
TABLES = RESULTS_DIR / "tables"
PROCESSED = REPO_ROOT / "data" / "processed"
METHOD_NAMES = {"gurobi": "Gurobi", "greedy": "Greedy", "alns": "ALNS", "matheuristic": "MH"}
SET_ORDER = {"small": 0, "medium": 1, "large": 2}


def read_json(path: Path) -> dict:
    """One JSON file of data/processed."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def name_key(name: str) -> tuple[int, int, int, int]:
    """Sort key of an instance name like small_k1_n6_T3_s1: set, hubs, points, seed."""
    set_name, k, n, _, seed = name.split("_")
    return SET_ORDER[set_name], int(k[1:]), int(n[1:]), int(seed[1:])


def num(value: float, decimals: int) -> str:
    """A number with a thousands comma, a fixed number of decimals and a true minus sign."""
    text = f"{value:,.{decimals}f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def tt(name: str) -> str:
    """An instance name in typewriter font, with LaTeX-safe underscores."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def write_rows(name: str, rows: list[list[str]]) -> None:
    """The body rows of one tabular; the header stays in report.tex."""
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [" & ".join(row) + " \\\\" for row in rows]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / name}.tex ({len(rows)} rows)")


# ----- data and parameters -----


def parameter_rows(params: Params) -> list[list[str]]:
    """Symbol, meaning, value and source of every cost, capacity and search parameter."""
    table = [
        ["$Q^1$", "truck capacity (kg)", num(params.truck_capacity, 0),
         "VW Delivery 11.180, payload plus body"],
        ["$c^1$", "truck cost per km (BRL)", num(params.truck_cost_per_km, 4),
         "ANTT table, CCD, 2 axles"],
        ["$f^1$", "truck cost per trip (BRL)", num(params.truck_cost_per_trip, 2),
         "ANTT table, CC"],
        ["$Q^2$", "van capacity (kg)", num(params.van_capacity, 0), "Fiat Fiorino, maker page"],
        ["$m$", "vans per hub", str(params.n_vans), "assumption (rule A2)"],
        ["$c^2$", "van cost per km (BRL)", num(params.van_cost_per_km, 4),
         "assumption: ANTT CCD reused"],
        ["", "goods value (BRL/kg)", num(params.goods_value_per_kg, 3), "Olist item prices"],
        ["", "Selic rate per year", num(params.selic_annual, 4),
         "BCB series 432"],
        ["$\\eta^H = \\eta^P$", "holding (BRL/kg/day)", num(params.holding, 4),
         "goods value $\\times$ Selic / 365"],
        ["", "point capacity (average days)", num(params.point_capacity_days, 1),
         "assumption (rule C1)"],
        ["", "point initial stock (days)", num(params.point_initial_days, 1),
         "assumption (rule C2)"],
        ["", "hub capacity factor", num(params.hub_capacity_factor, 1), "assumption (rule C3)"],
        ["", "hub initial stock (days)", num(params.hub_initial_days, 1), "assumption (rule C4)"],
        ["", "pool radius around a hub (km)", num(params.pool_radius_km, 0), "assumption"],
        ["", "pool size (prefixes)", str(params.pool_size), "assumption"],
        ["", "weeks folded onto one week", str(params.fold_weeks), "assumption"],
        ["", "ALNS segment length", str(params.segment_length),
         "\\citet{coelho2012transshipment}"],
        ["$r$", "ALNS reaction factor", num(params.reaction, 1),
         "\\citet{coelho2012transshipment}"],
        ["", "scores best, better, accepted",
         f"{params.score_best:g}, {params.score_better:g}, {params.score_accepted:g}",
         "\\citet{coelho2012transshipment}"],
        ["$w$", "start worse share", num(params.start_worse, 2), "\\citet{ropke2006adaptive}"],
        ["", "final temperature ratio", f"{params.final_temperature_ratio:g}", "assumption"],
        ["", "removal share, caps",
         f"{params.removal_share:g}, {params.removal_cap_min}, {params.removal_cap_max}",
         "assumption"],
    ]  # fmt: skip
    return table


def hub_legs(instance: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Road km from the warehouse to each hub, and the cost of one truck trip to each hub."""
    km = np.array([instance.distance_km[0, instance.hub_node(h)] for h in range(instance.n_hubs)])
    cost = np.array([instance.trip_cost(h) for h in range(instance.n_hubs)])
    return km, cost


def all_instances() -> list[Instance]:
    """Every frozen instance, small to large."""
    instances = [load_instance(path) for path in INSTANCES_DIR.glob("*.json")]
    return sorted(instances, key=lambda instance: name_key(instance.name))


def peak_hub_day(instance: Instance) -> float:
    """kg of the busiest (hub, day): the most one hub must send out by van on one day."""
    return float(
        np.max([np.sum(instance.demand[points, :], axis=0) for points in instance.points_of])
    )


def instance_rows(instances: list[Instance]) -> list[list[str]]:
    """Size, points per hub, total demand and the busiest hub-day of every instance."""
    rows = []
    for instance in instances:
        per_hub = [len(points) for points in instance.points_of]
        rows.append(
            [
                tt(instance.name),
                str(instance.n_hubs),
                str(instance.n_points),
                str(instance.n_days),
                f"{min(per_hub)} to {max(per_hub)}" if len(per_hub) > 1 else str(per_hub[0]),
                num(np.sum(instance.demand), 1),
                num(peak_hub_day(instance), 1),
            ]
        )
    return rows


# ----- results -----


def small_rows() -> list[list[str]]:
    """One row per small instance: optimum, Gurobi time, greedy gaps, ALNS and matheuristic gaps."""
    gaps = pd.read_csv(TABLES / "gap_to_optimum.csv")
    summary = pd.read_csv(TABLES / "summary.csv")
    rows = []
    for name in sorted(set(gaps["instance"]), key=name_key):
        by = gaps[gaps["instance"] == name].set_index("method")
        seconds = summary[(summary["instance"] == name) & (summary["method"] == "gurobi")]
        rows.append(
            [
                tt(name),
                num(by.loc["gurobi", "reference_cost"], 2),
                num(seconds["mean_runtime_s"].iloc[0], 1),
                num(by.loc["greedy", "mean_gap_pct"], 2),
                num(by.loc["greedy", "mean_van_gap_pct"], 2),
                f"{by.loc['alns', 'best_gap_pct']:.2f} / {by.loc['alns', 'mean_gap_pct']:.2f}",
                f"{by.loc['matheuristic', 'best_gap_pct']:.2f} / "
                f"{by.loc['matheuristic', 'mean_gap_pct']:.2f}",
            ]
        )
    return rows


def compare_rows(gaps: pd.DataFrame) -> list[list[str]]:
    """Medium and large: best, mean, std, mean and max gap, iterations, LP rescues per method."""
    compare = gaps[gaps["part"] == "compare"]
    summary = pd.read_csv(TABLES / "summary.csv")
    summary = summary[summary["part"] == "compare"]
    max_gap = compare.groupby(["instance", "method"])["gap_pct"].max()
    rows = []
    for name in sorted(set(summary["instance"]), key=name_key):
        cell = summary[summary["instance"] == name]
        for index, method in enumerate(["greedy", "alns", "matheuristic"]):
            row = cell[cell["method"] == method].iloc[0]
            seeded = method != "greedy"
            rows.append(
                [
                    tt(name) if index == 0 else "",
                    METHOD_NAMES[method],
                    num(row["best"], 2),
                    num(row["mean"], 2) if seeded else "",
                    num(row["std"], 2) if seeded else "",
                    num(row["mean_gap_pct"], 2),
                    num(max_gap[(name, method)], 2),
                    num(row["mean_iterations"], 0) if seeded else "",
                    num(row["mean_lp_rescues"], 1) if method == "matheuristic" else "",
                ]
            )
    return rows


def wilcoxon_rows() -> list[list[str]]:
    """Medium and large: median difference and p-value of the three paired comparisons."""
    tests = pd.read_csv(TABLES / "wilcoxon.csv")
    tests = tests[tests["part"] == "compare"]
    pairs = [("alns", "greedy"), ("matheuristic", "greedy"), ("alns", "matheuristic")]
    rows = []
    for name in sorted(set(tests["instance"]), key=name_key):
        cell = tests[tests["instance"] == name]
        row = [tt(name)]
        for a, b in pairs:
            test = cell[(cell["method_a"] == a) & (cell["method_b"] == b)].iloc[0]
            row += [num(test["median_difference"], 2), f"{test['p_value']:.4f}"]
        rows.append(row)
    return rows


def sensitivity_rows() -> list[list[str]]:
    """Mean cost parts, truck trips and visits per holding multiplier and method."""
    table = pd.read_csv(TABLES / "sensitivity.csv")
    rows = []
    for row in table.itertuples():
        rows.append(
            [
                f"{row.holding_multiplier:g}",
                METHOD_NAMES[row.method],
                num(row.total_cost, 2),
                num(row.truck_cost, 2),
                num(row.van_cost, 2),
                num(row.holding_cost, 2),
                num(row.n_truck_trips, 1),
                num(row.n_visits, 1),
            ]
        )
    return rows


# ----- numbers used in the text -----


def macros(
    params: Params, runs: pd.DataFrame, gaps: pd.DataFrame, instances: list[Instance]
) -> list[tuple[str, str]]:
    """(name, text) of every number the report prints outside a table."""
    sites = read_json(PROCESSED / "sites.json")
    osrm = read_json(PROCESSED / "osrm_meta.json")
    gurobi = runs[runs["method"] == "gurobi"]
    greedy_small = gaps[(gaps["set"] == "small") & (gaps["method"] == "greedy")]
    compare = gaps[gaps["part"] == "compare"]
    large_mh = runs[(runs["set"] == "large") & (runs["method"] == "matheuristic")]
    large_alns = runs[(runs["set"] == "large") & (runs["method"] == "alns")]
    mh_other = runs[(runs["set"] != "large") & (runs["method"] == "matheuristic")]
    tests = pd.read_csv(TABLES / "wilcoxon.csv")
    tests = tests[tests["part"] == "compare"]
    versus_greedy = tests[tests["method_b"] == "greedy"]
    versus_each = tests[tests["method_b"] == "matheuristic"]
    large_mh_summary = pd.read_csv(TABLES / "summary.csv")
    large_mh_summary = large_mh_summary[
        (large_mh_summary["set"] == "large") & (large_mh_summary["method"] == "matheuristic")
    ]
    compare_summary = pd.read_csv(TABLES / "summary.csv")
    spread = compare_summary[compare_summary["part"] == "compare"].set_index(["instance", "method"])
    sensitivity = pd.read_csv(TABLES / "sensitivity.csv")
    base = sensitivity[(sensitivity["holding_multiplier"] == 1) & (sensitivity["method"] == "alns")]
    alns_by = sensitivity[sensitivity["method"] == "alns"].set_index("holding_multiplier")
    medium_days = [peak_hub_day(x) for x in instances if x.set_name == "medium"]
    large_days = [peak_hub_day(x) for x in instances if x.set_name == "large"]
    holding_share = 100 * base["holding_cost"].iloc[0] / base["total_cost"].iloc[0]
    hub_km, trip_cost = hub_legs(load_instance(INSTANCES_DIR / f"{params.map_instance}.json"))
    return [
        ("nOrders", num(sites["filters"]["orders_with_items"], 0)),
        ("nDropWeight", str(sites["filters"]["orders_dropped_missing_weight"])),
        ("nDropCoord", str(sites["filters"]["orders_dropped_no_coordinate"])),
        ("nPrefixDropCoord", str(sites["filters"]["prefixes_dropped_no_coordinate"])),
        ("nPrefixes", num(sites["filters"]["prefixes_kept"], 0)),
        ("warehouseSellers", str(sites["warehouse"]["n_sellers"])),
        ("foldStart", sites["filters"]["fold_start"]),
        ("foldWeeks", str(sites["filters"]["fold_weeks"])),
        ("osrmPoints", str(osrm["n_points"])),
        ("osrmRequests", str(osrm["n_requests"])),
        ("osrmBlock", str(osrm["block"])),
        ("osrmDate", osrm["date"]),
        ("holdingValue", num(params.holding, 4)),
        ("vanCapacity", num(params.van_capacity, 0)),
        ("nSeeds", str(len(params.seeds))),
        ("limitSmall", f"{params.time_limit_s['small']:g}"),
        ("limitMedium", f"{params.time_limit_s['medium']:g}"),
        ("limitLarge", f"{params.time_limit_s['large']:g}"),
        ("limitGurobi", f"{params.gurobi_time_limit_s:g}"),
        ("nWorkers", str(params.workers)),
        ("nThreads", str(params.gurobi_threads)),
        ("alphaLevel", f"{params.alpha:g}"),
        ("nCpus", str(os.cpu_count())),
        ("gurobiVersion", ".".join(str(part) for part in gp.gurobi.version())),
        ("pythonVersion", sys.version.split()[0]),
        ("nRuns", str(len(runs))),
        ("nGurobiOptimal", str(int(np.sum(gurobi["status"] == "optimal")))),
        ("nSmall", str(len(gurobi))),
        ("gurobiMaxTime", num(gurobi["runtime_s"].max(), 1)),
        ("gurobiTotalTime", num(gurobi["runtime_s"].sum(), 1)),
        ("greedyMaxGap", num(greedy_small["gap_pct"].max(), 2)),
        ("greedyMinGap", num(greedy_small["gap_pct"].min(), 2)),
        ("greedyMaxVanGap", num(greedy_small["van_gap_pct"].max(), 2)),
        ("greedyMinGapCompare", num(compare[compare["method"] == "greedy"]["gap_pct"].min(), 2)),
        ("greedyMaxGapCompare", num(compare[compare["method"] == "greedy"]["gap_pct"].max(), 2)),
        ("alnsMaxGapCompare", num(compare[compare["method"] == "alns"]["gap_pct"].max(), 2)),
        ("mhMaxGapCompare", num(compare[compare["method"] == "matheuristic"]["gap_pct"].max(), 2)),
        ("pGreedyMax", f"{versus_greedy['p_value'].max():.4f}"),
        ("pExactMin", f"{2 * 0.5 ** len(params.seeds):.4f}"),
        ("pEachMin", f"{versus_each['p_value'].min():.4f}"),
        ("pEachMax", f"{versus_each['p_value'].max():.4f}"),
        ("lpCallsLarge", num(large_mh["lp_calls"].sum(), 0)),
        ("lpRescuesLarge", num(large_mh["lp_rescues"].sum(), 0)),
        ("lpCallsOther", str(int(mh_other["lp_calls"].sum()))),
        ("lpMeanMin", num(large_mh_summary["mean_lp_rescues"].min(), 1)),
        ("lpMeanMax", num(large_mh_summary["mean_lp_rescues"].max(), 1)),
        ("lpRunMin", str(int(large_mh["lp_rescues"].min()))),
        ("lpRunMax", str(int(large_mh["lp_rescues"].max()))),
        ("rejectedAlnsLarge", num(large_alns["rejected"].mean(), 1)),
        ("rejectedMhLarge", num(large_mh["rejected"].mean(), 2)),
        ("holdingShareBase", num(holding_share, 2)),
        ("peakMediumMax", num(np.max(medium_days), 1)),
        ("peakLargeMin", num(np.min(large_days), 1)),
        ("peakLargeMax", num(np.max(large_days), 1)),
        ("sensTripsBase", num(alns_by.loc[1, "n_truck_trips"], 1)),
        ("sensTripsHigh", num(alns_by.loc[1000, "n_truck_trips"], 1)),
        ("sensTripsTop", num(alns_by.loc[10000, "n_truck_trips"], 1)),
        ("sensVisitsBase", num(alns_by.loc[1, "n_visits"], 1)),
        ("sensVisitsMid", num(alns_by.loc[100, "n_visits"], 1)),
        ("sensVisitsHigh", num(alns_by.loc[1000, "n_visits"], 1)),
        ("sensVisitsTop", num(alns_by.loc[10000, "n_visits"], 1)),
        ("stdAlnsMedium", num(spread.loc[("medium_k3_n80_T7_s2", "alns"), "std"], 2)),
        ("stdMhMedium", num(spread.loc[("medium_k3_n80_T7_s2", "matheuristic"), "std"], 2)),
        ("stdAlnsLarge", num(spread.loc[("large_k5_n200_T7_s2", "alns"), "std"], 2)),
        ("stdMhLarge", num(spread.loc[("large_k5_n200_T7_s2", "matheuristic"), "std"], 2)),
        ("hubKmMin", num(np.min(hub_km), 1)),
        ("hubKmMax", num(np.max(hub_km), 1)),
        ("tripCostMin", num(np.min(trip_cost), 2)),
        ("tripCostMax", num(np.max(trip_cost), 2)),
    ]


def main() -> None:
    """Write every table body and numbers.tex."""
    params = load_params()
    runs = read_runs(RESULTS_DIR)
    gaps = with_gaps(runs)
    write_rows("parameters", parameter_rows(params))
    instances = all_instances()
    write_rows("instances", instance_rows(instances))
    write_rows("small", small_rows())
    write_rows("compare", compare_rows(gaps))
    write_rows("wilcoxon", wilcoxon_rows())
    write_rows("sensitivity", sensitivity_rows())
    lines = [
        f"\\newcommand{{\\{name}}}{{{text}}}"
        for name, text in macros(params, runs, gaps, instances)
    ]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'numbers.tex'} ({len(lines)} numbers)")


if __name__ == "__main__":
    main()
