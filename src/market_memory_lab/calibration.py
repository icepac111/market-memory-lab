"""Monte Carlo calibration under explicitly known null processes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianCalibration:
    """Monte Carlo reference distributions for an IID Gaussian null."""

    sample_size: int
    replications: int
    seed: int
    population_mean: float
    population_sigma: float
    sample_mean_z: FloatArray
    sample_variance_z: FloatArray
    lag_one_correlation: FloatArray

    def __post_init__(self) -> None:
        arrays = (
            self.sample_mean_z,
            self.sample_variance_z,
            self.lag_one_correlation,
        )

        for values in arrays:
            if len(values) != self.replications:
                raise ValueError(
                    "each calibration distribution must match replications"
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    "calibration distributions must contain finite values"
                )


def _validate_integer(
    value: int,
    *,
    name: str,
    minimum: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    if value < minimum:
        raise ValueError(
            f"{name} must be greater than or equal to {minimum}"
        )


def gaussian_null_calibration(
    *,
    sample_size: int,
    replications: int,
    seed: int,
    population_mean: float = 0.0,
    population_sigma: float = 1.0,
    batch_size: int = 100,
) -> GaussianCalibration:
    """
    Generate Monte Carlo reference distributions for an IID Gaussian null.

    For each replication, the function calculates:

    1. Standardized sample-mean deviation

           Z_mean = (mean(X) - mu) / (sigma / sqrt(n))

    2. Standardized sample-variance deviation

           Z_var = (S^2 - sigma^2)
                   / (sigma^2 * sqrt(2 / (n - 1)))

       This scaling uses the exact variance of the unbiased sample variance
       under Gaussian sampling. The resulting standardized statistic is not
       claimed to be normally distributed for finite n.

    3. Lag-one Pearson sample correlation

    Simulation is processed in bounded batches.

    Time complexity:
        O(replications * sample_size)

    Peak array memory:
        O(batch_size * sample_size + replications)
    """
    _validate_integer(
        sample_size,
        name="sample_size",
        minimum=3,
    )
    _validate_integer(
        replications,
        name="replications",
        minimum=10,
    )
    _validate_integer(
        seed,
        name="seed",
        minimum=0,
    )
    _validate_integer(
        batch_size,
        name="batch_size",
        minimum=1,
    )

    if not math.isfinite(population_mean):
        raise ValueError("population_mean must be finite")

    if (
        not math.isfinite(population_sigma)
        or population_sigma <= 0.0
    ):
        raise ValueError(
            "population_sigma must be finite and strictly positive"
        )

    mean_z = np.empty(replications, dtype=np.float64)
    variance_z = np.empty(replications, dtype=np.float64)
    lag_one = np.empty(replications, dtype=np.float64)

    rng = np.random.default_rng(seed)

    mean_standard_error = (
        population_sigma / math.sqrt(sample_size)
    )
    target_variance = population_sigma**2
    variance_standard_error = (
        target_variance
        * math.sqrt(2.0 / (sample_size - 1))
    )

    completed = 0

    while completed < replications:
        current_batch = min(
            batch_size,
            replications - completed,
        )

        paths = rng.normal(
            loc=population_mean,
            scale=population_sigma,
            size=(current_batch, sample_size),
        )

        sample_means = paths.mean(axis=1)
        sample_variances = paths.var(axis=1, ddof=1)

        previous = paths[:, :-1]
        current = paths[:, 1:]

        previous_centered = (
            previous - previous.mean(axis=1, keepdims=True)
        )
        current_centered = (
            current - current.mean(axis=1, keepdims=True)
        )

        numerator = np.sum(
            previous_centered * current_centered,
            axis=1,
        )
        denominator = np.sqrt(
            np.sum(previous_centered**2, axis=1)
            * np.sum(current_centered**2, axis=1)
        )

        if np.any(denominator <= 0.0):
            raise FloatingPointError(
                "undefined lag-one correlation encountered"
            )

        destination = slice(
            completed,
            completed + current_batch,
        )

        mean_z[destination] = (
            sample_means - population_mean
        ) / mean_standard_error

        variance_z[destination] = (
            sample_variances - target_variance
        ) / variance_standard_error

        lag_one[destination] = numerator / denominator

        completed += current_batch

    return GaussianCalibration(
        sample_size=sample_size,
        replications=replications,
        seed=seed,
        population_mean=population_mean,
        population_sigma=population_sigma,
        sample_mean_z=mean_z,
        sample_variance_z=variance_z,
        lag_one_correlation=lag_one,
    )


def empirical_percentile(
    reference: FloatArray,
    observed: float,
) -> float:
    """
    Return a midpoint empirical percentile in the interval [0, 1].

    The percentile is:

        (number below observed + 0.5 * number equal)
        / number of reference observations
    """
    values = np.asarray(reference, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "reference must be a nonempty one-dimensional array"
        )

    if not np.isfinite(values).all():
        raise ValueError("reference must contain finite values")

    if not math.isfinite(observed):
        raise ValueError("observed must be finite")

    below = np.count_nonzero(values < observed)
    equal = np.count_nonzero(values == observed)

    return float(
        (below + 0.5 * equal) / values.size
    )


def central_interval(
    reference: FloatArray,
    *,
    coverage: float = 0.95,
) -> tuple[float, float]:
    """Return the equal-tail empirical central interval."""
    values = np.asarray(reference, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "reference must be a nonempty one-dimensional array"
        )

    if not np.isfinite(values).all():
        raise ValueError("reference must contain finite values")

    if (
        not math.isfinite(coverage)
        or not 0.0 < coverage < 1.0
    ):
        raise ValueError("coverage must lie strictly between 0 and 1")

    tail = (1.0 - coverage) / 2.0
    lower, upper = np.quantile(
        values,
        [tail, 1.0 - tail],
    )

    return float(lower), float(upper)
