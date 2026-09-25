"""Tests for reading and checking config/params.toml."""

import pytest

from irp2e.config import CONFIG_PATH, load_params


def test_shipped_config_passes_every_check():
    load_params()  # raises ValueError naming the key if a value breaks its rule


def test_config_rejects_a_zero_van_capacity(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert text.count("capacity_kg = 650.0") == 1
    broken = tmp_path / "params.toml"
    broken.write_text(text.replace("capacity_kg = 650.0", "capacity_kg = 0.0"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"van\.capacity_kg"):
        load_params(broken)


def test_config_rejects_a_fold_start_that_is_not_a_monday(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    broken = tmp_path / "params.toml"
    broken.write_text(text.replace('"2017-08-07"', '"2017-08-08"'), encoding="utf-8")
    with pytest.raises(ValueError, match=r"data\.fold_start"):
        load_params(broken)
