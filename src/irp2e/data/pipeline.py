"""Command line for the data steps: preprocess, distances, instances."""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from irp2e.config import REPO_ROOT, Params, load_params
from irp2e.data.download import download_olist
from irp2e.data.osrm import fetch_distance_matrix, unroutable, write_cache
from irp2e.data.preprocess import build_pool, read_prefixes, read_sites, run_preprocess
from irp2e.generator import build_all
from irp2e.jsonfile import dump_rows

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INSTANCES_DIR = REPO_ROOT / "data" / "instances"


def distance_nodes(processed: Path, params: Params) -> tuple[list[str], np.ndarray, list[str]]:
    """Labels and lat, lng of the warehouse, the hubs and the union of all pools; also the sites."""
    prefixes = read_prefixes(processed / "prefixes.csv")
    sites = read_sites(processed / "sites.json")
    hubs = pd.DataFrame(sites["hubs"])
    site_labels = [sites["warehouse"]["prefix"]] + list(hubs["prefix"])
    hub_counts = sorted({n_hubs for spec in params.instance_sets.values() for n_hubs in spec.hubs})
    pooled = set()
    for n_hubs in hub_counts:
        pooled |= set(build_pool(prefixes, hubs, n_hubs, params)["prefix"])
    labels = site_labels + sorted(pooled - set(site_labels))
    where = {row.prefix: (row.lat, row.lng) for row in prefixes.itertuples()}
    where[sites["warehouse"]["prefix"]] = (sites["warehouse"]["lat"], sites["warehouse"]["lng"])
    for hub in sites["hubs"]:
        where[hub["prefix"]] = (hub["lat"], hub["lng"])
    coords = np.array([where[label] for label in labels])
    return labels, coords, site_labels


def run_distances(processed: Path, params: Params) -> None:
    """Fetch the OSRM matrix once and write osrm_distance_m.csv and osrm_meta.json."""
    cache = processed / "osrm_distance_m.csv"
    if cache.exists():
        raise FileExistsError(f"{cache} exists; delete the file to refetch")
    labels, coords, site_labels = distance_nodes(processed, params)
    meters = fetch_distance_matrix(coords, params)
    dropped = unroutable(labels, meters)
    fixed = [label for label in dropped if label in site_labels]
    if len(fixed) > 0:
        raise ValueError(f"warehouse or hub prefixes without a road answer: {fixed}")
    write_cache(labels, meters, cache)
    n_blocks = len(range(0, len(labels), params.osrm_block))
    meta = {
        "server": params.osrm_url,
        "service": "table/v1/driving, annotations=distance",
        "date": datetime.date.today().isoformat(),
        "n_points": len(labels),
        "block": params.osrm_block,
        "n_requests": n_blocks * n_blocks,
        "dropped": dropped,
        "coordinates": [
            [label, float(lat), float(lng)]
            for label, (lat, lng) in zip(labels, coords, strict=True)
        ],
    }
    dump_rows(meta, processed / "osrm_meta.json")


def main() -> None:
    """irp2e-data {preprocess, distances, instances, all}."""
    parser = argparse.ArgumentParser(prog="irp2e-data", description=__doc__)
    parser.add_argument("step", choices=["preprocess", "distances", "instances", "all"])
    args = parser.parse_args()
    params = load_params()
    if args.step in ("preprocess", "all"):
        run_preprocess(download_olist(params), PROCESSED_DIR, params)
        print(f"processed tables written to {PROCESSED_DIR}")
    if args.step in ("distances", "all"):
        run_distances(PROCESSED_DIR, params)
        print(f"OSRM distances written to {PROCESSED_DIR}")
    if args.step in ("instances", "all"):
        paths = build_all(PROCESSED_DIR, INSTANCES_DIR, params)
        print(f"{len(paths)} instances written to {INSTANCES_DIR}")


if __name__ == "__main__":
    main()
