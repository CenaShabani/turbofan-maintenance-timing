"""Parsing checks that fail loudly rather than shrinking the fleet quietly."""

from __future__ import annotations

import pytest

from src import acquisition


def _write_trajectory(path, rows, trailing_space=True):
    """Write a C-MAPSS-style file, optionally with the real files' trailing space."""
    suffix = " " if trailing_space else ""
    path.write_text(
        "\n".join(" ".join(f"{value:g}" for value in row) + suffix for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _row(unit, cycle, sensor_2=500.0):
    return [unit, cycle, 0.0, 0.0, 100.0, *([sensor_2] * 21)]


def test_trailing_whitespace_does_not_create_phantom_columns(tmp_path):
    """Every line in the real files ends in a space; a naive read gains NaN columns."""
    path = _write_trajectory(tmp_path / "train_FD001.txt",
                             [_row(1, 1), _row(1, 2), _row(1, 3)])
    frame = acquisition.read_trajectory_file(path, "FD001")

    assert list(frame.columns) == acquisition.DATA_COLUMNS + ["subset"]
    assert frame.shape[0] == 3
    assert not frame.isna().any().any()


def test_cycle_gaps_are_rejected(tmp_path):
    """A gap silently changes every duration in the analysis."""
    path = _write_trajectory(tmp_path / "train_FD001.txt",
                             [_row(1, 1), _row(1, 2), _row(1, 4)])
    with pytest.raises(RuntimeError, match="gaps in their cycle index"):
        acquisition.read_trajectory_file(path, "FD001")


def test_units_must_start_at_cycle_one(tmp_path):
    path = _write_trajectory(tmp_path / "train_FD001.txt",
                             [_row(1, 2), _row(1, 3)])
    with pytest.raises(RuntimeError, match="do not start at cycle 1"):
        acquisition.read_trajectory_file(path, "FD001")


def test_too_few_columns_is_an_error(tmp_path):
    path = tmp_path / "train_FD001.txt"
    path.write_text("1 1 0.0 0.0 100.0\n1 2 0.0 0.0 100.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected at least"):
        acquisition.read_trajectory_file(path, "FD001")


def test_rul_answer_key_is_indexed_from_one(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\n98\n69\n", encoding="utf-8")
    frame = acquisition.read_rul_file(path, "FD001")

    assert frame["unit"].tolist() == [1, 2, 3]
    assert frame["true_rul"].tolist() == [112, 98, 69]
    assert (frame["subset"] == "FD001").all()


def test_published_unit_counts_are_pinned():
    """The counts are asserted at parse time, so a repackaged mirror fails loudly."""
    assert sum(acquisition.EXPECTED_TRAIN_UNITS.values()) == 709
    assert sum(acquisition.EXPECTED_TEST_UNITS.values()) == 707
    assert set(acquisition.EXPECTED_TRAIN_UNITS) == set(acquisition.SUBSETS)
