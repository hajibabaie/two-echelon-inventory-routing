"""Download the Olist dataset with kagglehub and check the SHA-256 of every file we use."""

import hashlib
from pathlib import Path

import kagglehub

from irp2e.config import Params

OLIST_VERSION = 2  # the hashes below belong to this Kaggle version
OLIST_SHA256 = {
    "olist_customers_dataset.csv": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "olist_geolocation_dataset.csv": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    "olist_order_items_dataset.csv": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "olist_orders_dataset.csv": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "olist_products_dataset.csv": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "olist_sellers_dataset.csv": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
}


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(folder: Path) -> None:
    """Raise ValueError naming every used file that is missing or has a changed hash."""
    problems = []
    for name, expected in OLIST_SHA256.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif sha256_of(path) != expected:
            problems.append(f"{name}: SHA-256 differs from the stored value")
    if problems:
        raise ValueError("Olist files do not match: " + "; ".join(problems))


def download_olist(params: Params) -> Path:
    """Download (or reuse the kagglehub cache of) the pinned Olist version and verify it."""
    folder = Path(kagglehub.dataset_download(f"{params.kaggle_dataset}/versions/{OLIST_VERSION}"))
    verify_checksums(folder)
    return folder
