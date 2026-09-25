"""Tests for the data pipeline pieces that run without the raw files or the network."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from irp2e.config import load_params
from irp2e.data.download import verify_checksums
from irp2e.data.osrm import parse_table, read_cache, unroutable, write_cache
from irp2e.data.preprocess import fold_demand, read_prefixes


def test_fold_sums_orders_by_weekday():
    params = dataclasses.replace(load_params(), fold_weeks=2)  # window 2017-08-07 .. 2017-08-20
    orders = pd.DataFrame(
        {
            "order_id": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "prefix": ["01026", "01026", "01026", "13087", "13087", "13087"],
            "day": pd.to_datetime(
                ["2017-08-07", "2017-08-14", "2017-08-09", "2017-08-20", "2017-08-21", "2017-08-06"]
            ),
            "kg": [1.0, 2.0, 0.5, 4.0, 8.0, 16.0],
        }
    )
    table = fold_demand(orders, params).set_index("prefix")
    # 01026: Monday 1 + 2 = 3, Wednesday 0.5
    assert table.loc["01026"].tolist() == [3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    # 13087: Sunday 2017-08-20 is the last day; 08-21 (window end) and 08-06 (before) are out
    assert table.loc["13087"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]


def test_parse_table_turns_null_into_nan():
    payload = {"code": "Ok", "distances": [[0.0, 1234.5], [None, 0.0]]}
    meters = parse_table(payload, 2, 2)
    assert meters[0, 1] == 1234.5
    assert np.isnan(meters[1, 0])


def test_parse_table_rejects_code_not_ok():
    payload = {"code": "TooBig", "message": "Too many table coordinates"}
    with pytest.raises(ValueError, match="TooBig"):
        parse_table(payload, 2, 2)


def test_unroutable_lists_a_prefix_with_one_null_direction():
    meters = np.array([[0.0, 5.0, 7.0], [5.0, 0.0, 6.0], [np.nan, 6.0, 0.0]])
    # only the column of "a" (row "c" -> "a") is nan, and the row of "c"
    assert unroutable(["a", "b", "c"], meters) == ["a", "c"]


def test_distance_cache_round_trip_keeps_labels_and_nan(tmp_path):
    labels = ["01026", "13087"]
    meters = np.array([[0.0, 91234.5], [np.nan, 0.0]])
    path = tmp_path / "cache.csv"
    write_cache(labels, meters, path)
    read_labels, read_meters = read_cache(path)
    assert read_labels == labels
    assert read_meters[0, 1] == 91234.5
    assert np.isnan(read_meters[1, 0])


def test_checksum_mismatch_names_the_file(tmp_path):
    (tmp_path / "olist_orders_dataset.csv").write_text("order_id\nchanged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="olist_orders_dataset.csv: SHA-256 differs"):
        verify_checksums(tmp_path)


def test_zip_prefix_keeps_leading_zero(tmp_path):
    path = tmp_path / "prefixes.csv"
    path.write_text(
        "prefix,city,lat,lng,n_orders,kg\n01026,sao paulo,-23.5,-46.6,3,1.5\n", encoding="utf-8"
    )
    assert read_prefixes(path)["prefix"].tolist() == ["01026"]
