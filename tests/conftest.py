"""Synthetic fixtures with answers worked out by hand.

Every fixture here is small enough to reason about on paper. That is the point:
a test that asserts the pipeline agrees with itself proves nothing, so the
assertions are against values derived independently of the code.

The real data is deliberately not used. These tests must pass on a clean
checkout before ``src/acquisition.py`` has ever run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.panel import SENSOR_COLUMNS


def make_trajectories(lifetimes: dict[int, int], subset: str = "FD001",
                      seed: int = 0) -> pd.DataFrame:
    """Run-to-failure trajectories with a linear degradation signal on sensor 2.

    Sensor 2 rises linearly with age, sensor 3 is pure noise, and every other
    sensor is constant — so the screening, normalisation and drift features all
    have a knowable right answer.
    """
    generator = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for unit, lifetime in lifetimes.items():
        cycles = np.arange(1, lifetime + 1)
        frame = pd.DataFrame({
            "unit": unit,
            "cycle": cycles,
            "setting_1": 0.0,
            "setting_2": 0.0,
            "setting_3": 100.0,
            "subset": subset,
        })
        for column in SENSOR_COLUMNS:
            frame[column] = 1.0
        frame["sensor_2"] = 500.0 + 0.5 * cycles
        frame["sensor_3"] = 100.0 + generator.normal(0, 1, len(cycles))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def small_fleet() -> pd.DataFrame:
    """Twelve engines with lifetimes spread from 100 to 320 cycles."""
    lifetimes = {unit: 100 + 20 * unit for unit in range(1, 13)}
    return make_trajectories(lifetimes)


@pytest.fixture
def constant_lifetime_fleet() -> pd.DataFrame:
    """Thirty identical engines, every one failing at exactly cycle 200.

    Used for the policy tests, because with a single known lifetime the
    cost-optimal fixed interval can be computed by hand.
    """
    return make_trajectories({unit: 200 for unit in range(1, 31)})
