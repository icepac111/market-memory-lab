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


def iid_student_t(
    *,
    n: int,
    seed: int,
    degrees_of_freedom: float = 5.0,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> SyntheticSeries:
    """
    Generate independent variance-standardized Student-t observations.

    Let T_t independently follow a Student-t distribution with degrees
    of freedom nu. For nu greater than 2:

        Var(T_t) = nu / (nu - 2)

    Therefore:

        X_t = mean + sigma * sqrt((nu - 2) / nu) * T_t

    satisfies:

        E[X_t] = mean
        Var(X_t) = sigma squared

    Temporal dependence is absent by construction. Heavy tails remain.

    When 2 < nu <= 4, variance exists but the fourth moment is infinite.
    When nu > 4, theoretical excess kurtosis equals:

        6 / (nu - 4)
    """
    _validate_common(
        n=n,
        seed=seed,
    )
    _validate_finite(
        mean,
        name="mean",
    )
    _validate_positive(
        sigma,
        name="sigma",
    )
    _validate_finite(
        degrees_of_freedom,
        name="degrees_of_freedom",
    )

    if degrees_of_freedom <= 2.0:
        raise ValueError(
            "degrees_of_freedom must exceed 2 for finite variance"
        )

    rng = np.random.default_rng(seed)

    unscaled = rng.standard_t(
        df=degrees_of_freedom,
        size=n,
    )

    variance_scale = math.sqrt(
        (degrees_of_freedom - 2.0)
        / degrees_of_freedom
    )

    values = (
        mean
        + sigma
        * variance_scale
        * unscaled
    ).astype(np.float64)

    fourth_moment_state = (
        "finite"
        if degrees_of_freedom > 4.0
        else "infinite"
    )

    theoretical_excess_kurtosis = (
        6.0 / (degrees_of_freedom - 4.0)
        if degrees_of_freedom > 4.0
        else None
    )

    return SyntheticSeries(
        values=_series(values),
        process="iid_student_t",
        seed=seed,
        mechanism="independent heavy-tailed Student-t innovations",
        expected_dependence="none",
        parameters={
            "n": n,
            "degrees_of_freedom": degrees_of_freedom,
            "mean": mean,
            "sigma": sigma,
            "target_variance": sigma**2,
            "fourth_moment_state": fourth_moment_state,
            "theoretical_excess_kurtosis": (
                theoretical_excess_kurtosis
            ),
        },
    )
