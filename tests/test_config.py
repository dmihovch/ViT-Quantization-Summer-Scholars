"""
test_config.py
==============

Tests for command-line parsing into the immutable `ExperimentConfig`.

We use pytest's `monkeypatch` fixture to set `sys.argv` to whatever we want the
"command line" to be, then call `parse_config()` as the script would.
"""

import sys
from pathlib import Path

import pytest

from run_experiment1_mapping import (
    CHARACTERIZATION_RUN_IMAGES,
    parse_config,
)


def test_defaults_match_the_characterization_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiment1_mapping.py"])
    config = parse_config()
    assert config.num_images == CHARACTERIZATION_RUN_IMAGES
    assert config.batch_size == 64
    assert config.data_dir == Path("data")


def test_custom_arguments_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment1_mapping.py",
            "--num-images",
            "128",
            "--batch-size",
            "8",
            "--data-dir",
            "data/val",
        ],
    )
    config = parse_config()
    assert config.num_images == 128
    assert config.batch_size == 8
    assert config.data_dir == Path("data/val")


def test_json_output_path_derives_from_output_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run_experiment1_mapping.py", "--output-dir", "outputs/run7"]
    )
    config = parse_config()
    assert config.json_output_path == Path("outputs/run7/outlier_stats.json")


def test_config_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiment1_mapping.py"])
    config = parse_config()
    # The dataclass is frozen, so any attempt to mutate it must raise.
    # We use setattr so the static type checker does not flag the assignment.
    with pytest.raises(Exception):
        setattr(config, "num_images", 5)
