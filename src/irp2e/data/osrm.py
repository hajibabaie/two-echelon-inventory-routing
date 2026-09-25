"""Road distances from the OSRM table service, fetched in blocks and kept in one cache file."""

import csv
import time
from pathlib import Path

import numpy as np
import requests

from irp2e.config import Params


def table_url(
    coords: np.ndarray, sources: list[int], destinations: list[int], params: Params
) -> str:
    """URL of one table request: the source points first, then the destination points."""
    points = [coords[node] for node in sources + destinations]
    path = ";".join(f"{point[1]:.6f},{point[0]:.6f}" for point in points)  # OSRM wants lng,lat
    source_ids = ";".join(str(index) for index in range(len(sources)))
    destination_ids = ";".join(str(len(sources) + index) for index in range(len(destinations)))
    return (
        f"{params.osrm_url}/table/v1/driving/{path}"
        f"?sources={source_ids}&destinations={destination_ids}&annotations=distance"
    )


def parse_table(payload: dict, n_sources: int, n_destinations: int) -> np.ndarray:
    """Distances in meters from one OSRM answer; a null cell becomes nan, never a number."""
    if payload.get("code") != "Ok":
        raise ValueError(f"OSRM answered code {payload.get('code')!r}: {payload.get('message')}")
    rows = payload["distances"]
    if len(rows) != n_sources or any(len(row) != n_destinations for row in rows):
        raise ValueError(f"OSRM table is not {n_sources} x {n_destinations}")
    meters = np.full((n_sources, n_destinations), np.nan)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if value is not None:
                meters[row_index, column_index] = float(value)
    return meters


def fetch_distance_matrix(coords: np.ndarray, params: Params) -> np.ndarray:
    """Directed meters between all points, one request per block of osrm_block x osrm_block."""
    n_points = len(coords)
    starts = list(range(0, n_points, params.osrm_block))
    meters = np.full((n_points, n_points), np.nan)
    for source_start in starts:
        sources = list(range(source_start, min(source_start + params.osrm_block, n_points)))
        for destination_start in starts:
            destinations = list(
                range(destination_start, min(destination_start + params.osrm_block, n_points))
            )
            time.sleep(params.osrm_pause_s)  # at most one request per second
            response = requests.get(
                table_url(coords, sources, destinations, params),
                headers={"User-Agent": params.osrm_user_agent},
                timeout=params.osrm_timeout_s,
            )
            response.raise_for_status()
            block = parse_table(response.json(), len(sources), len(destinations))
            for row_index, source in enumerate(sources):
                for column_index, destination in enumerate(destinations):
                    meters[source, destination] = block[row_index, column_index]
    return meters


def unroutable(labels: list[str], meters: np.ndarray) -> list[str]:
    """Rule P0: every label with a nan in its row or its column."""
    dropped = []
    for index, label in enumerate(labels):
        if np.any(np.isnan(meters[index, :])) or np.any(np.isnan(meters[:, index])):
            dropped.append(label)
    return dropped


def write_cache(labels: list[str], meters: np.ndarray, path: Path) -> None:
    """Write the directed matrix as CSV with the labels as header and first column."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + labels)
        for label, row in zip(labels, meters, strict=True):
            writer.writerow([label] + [repr(float(value)) for value in row])


def read_cache(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the labels and the directed meters matrix written by write_cache."""
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    labels = rows[0][1:]
    if [row[0] for row in rows[1:]] != labels:
        raise ValueError(f"{path}: row labels differ from the header")
    meters = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    if meters.shape != (len(labels), len(labels)):
        raise ValueError(f"{path}: matrix is not {len(labels)} x {len(labels)}")
    return labels, meters
