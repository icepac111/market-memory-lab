"""Statistical safeguards for repeated tests and time-series inference."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloEvidence:
    """
    Monte Carlo exceedance evidence.

    The corrected p-value uses:

        (exceedances + 1) / (simulations + 1)

    The binomial interval describes uncertainty in the raw null
    exceedance probability estimated from the simulation count.
    """

    exceedances: int
    simulations: int
    corrected_p_value: float
    raw_exceedance_rate: float
    interval_coverage: float
    exceedance_probability_lower: float
    exceedance_probability_upper: float

    def __post_init__(self) -> None:
        probabilities = (
            self.corrected_p_value,
            self.raw_exceedance_rate,
            self.exceedance_probability_lower,
            self.exceedance_probability_upper,
        )

        for value in probabilities:
            if not math.isfinite(value):
                raise ValueError(
                    "Monte Carlo probabilities must be finite"
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Monte Carlo probabilities must lie in [0, 1]"
                )


@dataclass(frozen=True)
class HACMeanEvidence:
    """
    Newey-West-style inference for a sample mean.

    The estimator uses Bartlett weights and autocovariances normalized by
    the full sample size:

        gamma_j = (1 / n) sum_(t=j+1)^n u_t u_(t-j)

        long_run_variance
        = gamma_0
          + 2 sum_(j=1)^q
              (1 - j / (q + 1)) gamma_j

        standard_error(mean)
        = sqrt(long_run_variance / n)

    This requires sequential, equally spaced observations.
    """

    observations: int
    max_lag: int
    sample_mean: float
    long_run_variance: float
    mean_standard_error: float


def _validate_probability(
    value: float,
    *,
    name: str,
    allow_endpoints: bool,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    if allow_endpoints:
        valid = 0.0 <= value <= 1.0
    else:
        valid = 0.0 < value < 1.0

    if not valid:
        interval = "[0, 1]" if allow_endpoints else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")


def _p_value_array(
    p_values: Iterable[float],
) -> FloatArray:
    """Validate and return a one-dimensional p-value array."""
    values = np.asarray(list(p_values), dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")

    if values.size == 0:
        raise ValueError("p_values cannot be empty")

    if not np.isfinite(values).all():
        raise ValueError("p_values must contain finite values")

    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")

    return values


def bonferroni_adjusted_p_values(
    p_values: Iterable[float],
) -> FloatArray:
    """
    Return Bonferroni-adjusted p-values.

        p_i_adjusted = min(m * p_i, 1)

    This controls family-wise error under the standard Bonferroni
    inequality but can be conservative.
    """
    values = _p_value_array(p_values)
    tests = values.size

    return np.minimum(
        tests * values,
        1.0,
    ).astype(np.float64)


def benjamini_hochberg_adjusted_p_values(
    p_values: Iterable[float],
) -> FloatArray:
    """
    Return Benjamini-Hochberg-adjusted p-values.

    For sorted p-values p_(1) <= ... <= p_(m):

        adjusted p_(i)
        = min(
            1,
            min_(j >= i) m * p_(j) / j
          )

    The adjusted values are returned in the original input order.
    """
    values = _p_value_array(p_values)
    tests = values.size

    order = np.argsort(
        values,
        kind="stable",
    )
    sorted_values = values[order]

    ranks = np.arange(
        1,
        tests + 1,
        dtype=np.float64,
    )

    scaled = (
        tests
        * sorted_values
        / ranks
    )

    monotone = np.minimum.accumulate(
        scaled[::-1]
    )[::-1]

    monotone = np.minimum(
        monotone,
        1.0,
    )

    adjusted = np.empty(
        tests,
        dtype=np.float64,
    )
    adjusted[order] = monotone

    return adjusted


def benjamini_hochberg_rejections(
    p_values: Iterable[float],
    *,
    false_discovery_rate: float,
) -> NDArray[np.bool_]:
    """
    Return Benjamini-Hochberg rejection decisions.

    This implementation uses adjusted p-values and rejects when:

        adjusted p_i <= false_discovery_rate
    """
    _validate_probability(
        false_discovery_rate,
        name="false_discovery_rate",
        allow_endpoints=False,
    )

    adjusted = benjamini_hochberg_adjusted_p_values(
        p_values
    )

    return adjusted <= false_discovery_rate


def monte_carlo_exceedance_evidence(
    *,
    exceedances: int,
    simulations: int,
    interval_coverage: float = 0.95,
) -> MonteCarloEvidence:
    """
    Return corrected Monte Carlo p-value and binomial uncertainty.

    Corrected Monte Carlo p-value:

        p_corrected = (r + 1) / (B + 1)

    where r is the number of simulated statistics at least as extreme as
    the observed statistic and B is the number of simulations.

    The interval uses the exact Clopper-Pearson construction for the raw
    null exceedance probability. It is not an interval for an investment
    return or model probability.
    """
    if (
        not isinstance(exceedances, int)
        or isinstance(exceedances, bool)
    ):
        raise TypeError("exceedances must be an integer")

    if (
        not isinstance(simulations, int)
        or isinstance(simulations, bool)
    ):
        raise TypeError("simulations must be an integer")

    if simulations < 1:
        raise ValueError(
            "simulations must be greater than or equal to 1"
        )

    if not 0 <= exceedances <= simulations:
        raise ValueError(
            "exceedances must satisfy 0 <= exceedances <= simulations"
        )

    _validate_probability(
        interval_coverage,
        name="interval_coverage",
        allow_endpoints=False,
    )

    corrected = (
        exceedances + 1.0
    ) / (
        simulations + 1.0
    )

    raw_rate = exceedances / simulations
    alpha = 1.0 - interval_coverage

    if exceedances == 0:
        lower = 0.0
    else:
        lower = float(
            stats.beta.ppf(
                alpha / 2.0,
                exceedances,
                simulations - exceedances + 1,
            )
        )

    if exceedances == simulations:
        upper = 1.0
    else:
        upper = float(
            stats.beta.ppf(
                1.0 - alpha / 2.0,
                exceedances + 1,
                simulations - exceedances,
            )
        )

    return MonteCarloEvidence(
        exceedances=exceedances,
        simulations=simulations,
        corrected_p_value=float(corrected),
        raw_exceedance_rate=float(raw_rate),
        interval_coverage=interval_coverage,
        exceedance_probability_lower=lower,
        exceedance_probability_upper=upper,
    )


def newey_west_mean_evidence(
    values: Iterable[float],
    *,
    max_lag: int,
) -> HACMeanEvidence:
    """
    Estimate a Bartlett-kernel HAC standard error for a sample mean.

    Input observations must represent consecutive equally spaced periods.
    Missing periods, irregular spacing, and market-calendar alignment must
    be handled before this function is called.
    """
    observations = np.asarray(
        list(values),
        dtype=np.float64,
    )

    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if observations.size < 2:
        raise ValueError(
            "at least two observations are required"
        )

    if not np.isfinite(observations).all():
        raise ValueError("values must contain finite observations")

    if not isinstance(max_lag, int) or isinstance(max_lag, bool):
        raise TypeError("max_lag must be an integer")

    if not 0 <= max_lag < observations.size:
        raise ValueError(
            "max_lag must satisfy 0 <= max_lag < observations"
        )

    sample_size = observations.size
    sample_mean = float(np.mean(observations))
    centered = observations - sample_mean

    gamma_zero = float(
        np.dot(centered, centered)
        / sample_size
    )

    long_run_variance = gamma_zero

    for lag in range(1, max_lag + 1):
        autocovariance = float(
            np.dot(
                centered[lag:],
                centered[:-lag],
            )
            / sample_size
        )

        weight = 1.0 - lag / (max_lag + 1.0)

        long_run_variance += (
            2.0
            * weight
            * autocovariance
        )

    numerical_tolerance = 1e-15

    if long_run_variance < -numerical_tolerance:
        raise FloatingPointError(
            "estimated long-run variance is materially negative"
        )

    long_run_variance = max(
        long_run_variance,
        0.0,
    )

    standard_error = math.sqrt(
        long_run_variance / sample_size
    )

    return HACMeanEvidence(
        observations=int(sample_size),
        max_lag=max_lag,
        sample_mean=sample_mean,
        long_run_variance=float(long_run_variance),
        mean_standard_error=float(standard_error),
    )
