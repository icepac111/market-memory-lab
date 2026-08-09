"""Seed-controlled synthetic processes for falsification experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticSeries:
    """A synthetic series with immutable ground-truth metadata."""

    values: pd.Series
    process: str
    seed: int
    mechanism: str
    expected_dependence: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.values.empty:
            raise ValueError("synthetic series cannot be empty")

        if not np.isfinite(self.values.to_numpy()).all():
            raise ValueError("synthetic series must contain finite values")

        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )


def _validate_common(*, n: int, seed: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer greater than or equal to 2")

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _validate_finite(value: float, *, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_positive(value: float, *, name: str) -> None:
    _validate_finite(value, name=name)

    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive")


def _series(values: np.ndarray, *, name: str = "return") -> pd.Series:
    result = pd.Series(
        values,
        index=pd.RangeIndex(
            start=0,
            stop=len(values),
            step=1,
            name="observation",
        ),
        dtype="float64",
        name=name,
    )

    if not np.isfinite(result.to_numpy()).all():
        raise FloatingPointError("generator produced nonfinite observations")

    return result


def iid_gaussian(
    *,
    n: int,
    seed: int,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> SyntheticSeries:
    """
    Generate independent Gaussian observations.

    X_t = mean + sigma * Z_t
    Z_t independently follows N(0, 1).

    Temporal dependence is absent by construction.
    """
    _validate_common(n=n, seed=seed)
    _validate_finite(mean, name="mean")
    _validate_positive(sigma, name="sigma")

    rng = np.random.default_rng(seed)
    values = rng.normal(
        loc=mean,
        scale=sigma,
        size=n,
    ).astype(np.float64)

    return SyntheticSeries(
        values=_series(values),
        process="iid_gaussian",
        seed=seed,
        mechanism="independent Gaussian innovations",
        expected_dependence="none",
        parameters={
            "n": n,
            "mean": mean,
            "sigma": sigma,
        },
    )
