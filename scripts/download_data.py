"""Download the raw Olist files into the kagglehub cache, not the repo, and check their SHA-256."""

from irp2e.config import load_params
from irp2e.data.download import OLIST_SHA256, download_olist

if __name__ == "__main__":
    folder = download_olist(load_params())
    print(f"Olist files verified in {folder}")
    for name, digest in OLIST_SHA256.items():
        print(f"{digest}  {name}")
