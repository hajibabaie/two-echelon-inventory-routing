"""Figures: the network map, the gap box plots per method, and the holding cost sensitivity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from irp2e.instance import Instance  # noqa: E402

SET_ORDER = ["small", "medium", "large"]
METHOD_ORDER = ["gurobi", "greedy", "alns", "matheuristic"]
METHOD_COLORS = {
    "gurobi": "#4a3aa7",
    "greedy": "#1baf7a",
    "alns": "#2a78d6",
    "matheuristic": "#eb6834",
}
PART_COLORS = {"truck_cost": "#008300", "van_cost": "#eda100", "holding_cost": "#e34948"}
PART_MARKERS = {"truck_cost": "s", "van_cost": "o", "holding_cost": "^"}
HUB_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
HUB_MARKERS = ["o", "s", "^", "D", "v"]  # a second cue next to color, for color-blind readers
FIGSIZE = (10, 6)
DPI = 300


def save_figure(figure: plt.Figure, stem: Path) -> None:
    """Write <stem>.pdf and <stem>.png, then close the figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=DPI)
    plt.close(figure)
    print(f"wrote {stem}.pdf and .png")


def draw_sites(axis: plt.Axes, instance: Instance) -> None:
    """Warehouse, hubs and points on longitude and latitude; each point in its hub's color."""
    lat = instance.coordinates[:, 0]
    lng = instance.coordinates[:, 1]
    for h in range(instance.n_hubs):
        nodes = [instance.point_node(i) for i in instance.points_of[h]]
        hub = instance.hub_node(h)
        axis.scatter(
            lng[nodes], lat[nodes], s=14, color=HUB_COLORS[h], marker=HUB_MARKERS[h],
            label=f"points of {instance.cities[hub]} ({len(nodes)})",
        )  # fmt: skip
        axis.scatter(
            lng[hub], lat[hub], s=160, color=HUB_COLORS[h], marker=HUB_MARKERS[h],
            edgecolor="black", linewidth=1.2, zorder=3,
        )  # fmt: skip
    axis.scatter(lng[0], lat[0], s=260, color="black", marker="*", zorder=3, label="warehouse")
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    axis.set_aspect(1 / np.cos(np.deg2rad(np.mean(lat))))  # one km east looks as long as one north
    axis.grid(color="#dddddd", linewidth=0.5)


def draw_truck_trips(axis: plt.Axes, instance: Instance) -> None:
    """A dashed straight line from the warehouse to each hub: the truck trips of echelon 1."""
    lat = instance.coordinates[:, 0]
    lng = instance.coordinates[:, 1]
    for h in range(instance.n_hubs):
        hub = instance.hub_node(h)
        axis.plot(
            [lng[0], lng[hub]], [lat[0], lat[hub]], color="#999999", linewidth=0.8,
            linestyle="--", zorder=1, label="truck trip (straight line)" if h == 0 else None,
        )  # fmt: skip


def plot_network(instance: Instance, stem: Path) -> None:
    """Whole network on the left, the hub region enlarged on the right; straight lines, no roads."""
    figure, (whole, region) = plt.subplots(1, 2, figsize=FIGSIZE)
    draw_truck_trips(whole, instance)
    draw_sites(whole, instance)
    draw_sites(region, instance)
    lat = instance.coordinates[1:, 0]  # hubs and points only
    lng = instance.coordinates[1:, 1]
    margin = 0.05
    region.set_xlim(np.min(lng) - margin, np.max(lng) + margin)
    region.set_ylim(np.min(lat) - margin, np.max(lat) + margin)
    whole.set_title(f"warehouse ({instance.cities[0]}) and all hubs", fontsize=10)
    region.set_title("hub region (large markers: hubs)", fontsize=10)
    whole.legend(loc="lower left", fontsize=7)
    figure.suptitle(
        f"Network of {instance.name}: {instance.n_hubs} hubs, {instance.n_points} demand points",
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, stem)


def plot_gap_boxplots(gaps: pd.DataFrame, stem: Path) -> None:
    """One panel per instance set: gap to the reference cost of every run, one box per method."""
    compared = gaps[gaps["part"] != "sensitivity"]
    sets = [name for name in SET_ORDER if name in set(compared["set"])]
    figure, axes = plt.subplots(1, len(sets), figsize=FIGSIZE, squeeze=False)
    for axis, set_name in zip(axes[0], sets, strict=True):
        runs = compared[compared["set"] == set_name]
        methods = [method for method in METHOD_ORDER if method in set(runs["method"])]
        values = [runs[runs["method"] == method]["gap_pct"].to_numpy() for method in methods]
        boxes = axis.boxplot(
            values, patch_artist=True, widths=0.6, showfliers=False, medianprops={"color": "black"}
        )  # every run is drawn as a dot, so no separate outlier marks
        for box, method in zip(boxes["boxes"], methods, strict=True):
            box.set_facecolor(METHOD_COLORS[method])
            box.set_alpha(0.35)
        for position, (method, points) in enumerate(zip(methods, values, strict=True), start=1):
            axis.scatter(
                np.full(len(points), position), points, s=10, color=METHOD_COLORS[method], zorder=3
            )
        axis.set_xticks(range(1, len(methods) + 1), methods)
        axis.set_title(f"{set_name} set", fontsize=10)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0][0].set_ylabel("gap to reference cost (%)")
    figure.suptitle(
        "Gap of every run to the proven optimum, or to the best known cost where none is proven",
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, stem)


def plot_sensitivity(table: pd.DataFrame, stem: Path) -> None:
    """Left: mean total cost per method. Right: mean cost parts of the ALNS. Both on log axes."""
    figure, (totals, parts) = plt.subplots(1, 2, figsize=FIGSIZE)
    for method in [method for method in METHOD_ORDER if method in set(table["method"])]:
        rows = table[table["method"] == method]
        totals.plot(
            rows["holding_multiplier"], rows["total_cost"], marker="o", linewidth=2,
            color=METHOD_COLORS[method], label=method,
        )  # fmt: skip
    alns = table[table["method"] == "alns"]
    for part, color in PART_COLORS.items():
        parts.plot(
            alns["holding_multiplier"], alns[part], marker=PART_MARKERS[part], linewidth=2,
            color=color, label=part.replace("_cost", ""),
        )  # fmt: skip
    for axis in (totals, parts):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("holding cost multiplier")
        axis.grid(color="#dddddd", linewidth=0.5)
        axis.legend(fontsize=8)
    totals.set_ylabel("mean total cost (BRL)")
    parts.set_ylabel("mean cost of the ALNS runs (BRL)")
    totals.set_title("total cost per method", fontsize=10)
    parts.set_title("ALNS cost parts", fontsize=10)
    figure.suptitle("Holding cost sensitivity", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, stem)
