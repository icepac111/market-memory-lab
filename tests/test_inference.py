"""Tests for multiple-testing and time-series inference safeguards."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from market_memory_lab.inference import (
    benjamini_hochberg_adjusted_p_values,
    benjamini_hochberg_rejections,
    bonferroni_adjusted_p_values,
    monte_carlo_exceedance_evidence,
    newey_west_mean_evidence,
)


def independent_bh_reference(
    values: np.ndarray,
) -> np.ndarray:
    """Calculate BH-adjusted p-values using an independent loop."""
    count = len(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    sorted_adjusted = np.empty(count)
    running_minimum = 1.0

    for reverse_index in range(
        count - 1,
        -1,
        -1,
    ):
        rank = reverse_index + 1
        candidate = (
            count
            * sorted_values[reverse_index]
            / rank
        )

        running_minimum = min(
            running_minimum,
            candidate,
            1.0,
        )

        sorted_adjusted[reverse_index] = running_minimum

    result = np.empty(count)
    result[order] = sorted_adjusted

    return result


def test_bonferroni_known_values() -> None:
    p_values = [0.001, 0.02, 0.30, 0.90]

    actual = bonferroni_adjusted_p_values(
        p_values
    )

    expected = np.array(
        [0.004, 0.08, 1.0, 1.0]
    )

    np.testing.assert_allclose(actual, expected)


def test_bh_matches_independent_reference() -> None:
    p_values = np.array(
        [0.030, 0.001, 0.040, 0.120, 0.006],
        dtype="float64",
    )

    actual = benjamini_hochberg_adjusted_p_values(
        p_values
    )

    expected = independent_bh_reference(p_values)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-15,
        atol=1e-15,
    )


def test_bh_adjusted_values_preserve_original_order() -> None:
    p_values = np.array(
        [0.20, 0.01, 0.04, 0.002],
        dtype="float64",
    )

    adjusted = benjamini_hochberg_adjusted_p_values(
        p_values
    )

    order = np.argsort(p_values)
    sorted_adjusted = adjusted[order]

    assert bool(
        np.all(
            np.diff(sorted_adjusted) >= -1e-15
        )
    )


def test_bh_rejection_decisions() -> None:
    p_values = [0.001, 0.01, 0.20, 0.80]

    decisions = benjamini_hochberg_rejections(
        p_values,
        false_discovery_rate=0.05,
    )

    np.testing.assert_array_equal(
        decisions,
        np.array([True, True, False, False]),
    )


def test_bh_and_bonferroni_are_permutation_equivariant() -> None:
    original = np.array(
        [0.03, 0.001, 0.50, 0.02, 0.10],
        dtype="float64",
    )

    permutation = np.array([3, 0, 4, 1, 2])
    permuted = original[permutation]

    inverse = np.argsort(permutation)

    bh_original = benjamini_hochberg_adjusted_p_values(
        original
    )
    bh_permuted = benjamini_hochberg_adjusted_p_values(
        permuted
    )[inverse]

    bonferroni_original = bonferroni_adjusted_p_values(
        original
    )
    bonferroni_permuted = bonferroni_adjusted_p_values(
        permuted
    )[inverse]

    np.testing.assert_allclose(
        bh_original,
        bh_permuted,
    )

    np.testing.assert_allclose(
        bonferroni_original,
        bonferroni_permuted,
    )


def test_zero_exceedances_never_produces_zero_p_value() -> None:
    result = monte_carlo_exceedance_evidence(
        exceedances=0,
        simulations=999,
    )

    assert result.corrected_p_value == pytest.approx(
        1.0 / 1000.0
    )
    assert result.raw_exceedance_rate == pytest.approx(0.0)
    assert result.exceedance_probability_lower == 0.0
    assert 0.0 < result.exceedance_probability_upper < 1.0


def test_all_exceedances_produce_unit_p_value() -> None:
    result = monte_carlo_exceedance_evidence(
        exceedances=100,
        simulations=100,
    )

    assert result.corrected_p_value == pytest.approx(1.0)
    assert result.raw_exceedance_rate == pytest.approx(1.0)
    assert result.exceedance_probability_upper == 1.0


def test_clopper_pearson_interval_contains_raw_rate() -> None:
    result = monte_carlo_exceedance_evidence(
        exceedances=40,
        simulations=200,
        interval_coverage=0.95,
    )

    assert (
        result.exceedance_probability_lower
        <= result.raw_exceedance_rate
        <= result.exceedance_probability_upper
    )


def test_clopper_pearson_matches_scipy_reference() -> None:
    exceedances = 17
    simulations = 100
    coverage = 0.90
    alpha = 1.0 - coverage

    result = monte_carlo_exceedance_evidence(
        exceedances=exceedances,
        simulations=simulations,
        interval_coverage=coverage,
    )

    expected_lower = stats.beta.ppf(
        alpha / 2.0,
        exceedances,
        simulations - exceedances + 1,
    )

    expected_upper = stats.beta.ppf(
        1.0 - alpha / 2.0,
        exceedances + 1,
        simulations - exceedances,
    )

    assert (
        result.exceedance_probability_lower
        == pytest.approx(expected_lower)
    )

    assert (
        result.exceedance_probability_upper
        == pytest.approx(expected_upper)
    )


def manual_newey_west(
    values: np.ndarray,
    max_lag: int,
) -> tuple[float, float]:
    """Independent direct-loop Bartlett HAC reference."""
    sample_size = len(values)
    mean = float(np.mean(values))
    centered = values - mean

    gamma_zero = 0.0

    for index in range(sample_size):
        gamma_zero += centered[index] ** 2

    gamma_zero /= sample_size
    long_run_variance = gamma_zero

    for lag in range(1, max_lag + 1):
        covariance_sum = 0.0

        for index in range(lag, sample_size):
            covariance_sum += (
                centered[index]
                * centered[index - lag]
            )

        autocovariance = (
            covariance_sum / sample_size
        )

        weight = 1.0 - lag / (max_lag + 1.0)

        long_run_variance += (
            2.0
            * weight
            * autocovariance
        )

    long_run_variance = max(
        long_run_variance,
        0.0,
    )

    standard_error = math.sqrt(
        long_run_variance / sample_size
    )

    return long_run_variance, standard_error


def test_newey_west_matches_independent_loop() -> None:
    values = np.array(
        [
            0.010,
            -0.005,
            0.012,
            0.018,
            -0.003,
            0.007,
            0.011,
            -0.002,
        ],
        dtype="float64",
    )

    actual = newey_west_mean_evidence(
        values,
        max_lag=2,
    )

    expected_variance, expected_error = manual_newey_west(
        values,
        max_lag=2,
    )

    assert actual.long_run_variance == pytest.approx(
        expected_variance,
        rel=1e-14,
        abs=1e-14,
    )

    assert actual.mean_standard_error == pytest.approx(
        expected_error,
        rel=1e-14,
        abs=1e-14,
    )


def test_newey_west_zero_lag_reference() -> None:
    values = np.array(
        [1.0, 2.0, 4.0, 8.0],
        dtype="float64",
    )

    actual = newey_west_mean_evidence(
        values,
        max_lag=0,
    )

    population_variance = float(
        np.var(values, ddof=0)
    )
    expected_error = math.sqrt(
        population_variance / len(values)
    )

    assert actual.long_run_variance == pytest.approx(
        population_variance
    )

    assert actual.mean_standard_error == pytest.approx(
        expected_error
    )


def test_newey_west_constant_series() -> None:
    values = np.ones(20)

    result = newey_west_mean_evidence(
        values,
        max_lag=4,
    )

    assert result.long_run_variance == pytest.approx(0.0)
    assert result.mean_standard_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    "p_values",
    [
        [],
        [0.10, -0.01],
        [0.10, 1.01],
        [0.10, np.nan],
        [0.10, np.inf],
    ],
)
def test_invalid_p_values_are_rejected(
    p_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        benjamini_hochberg_adjusted_p_values(
            p_values
        )


@pytest.mark.parametrize(
    ("exceedances", "simulations"),
    [
        (-1, 100),
        (101, 100),
        (0, 0),
    ],
)
def test_invalid_monte_carlo_counts_are_rejected(
    exceedances: int,
    simulations: int,
) -> None:
    with pytest.raises(ValueError):
        monte_carlo_exceedance_evidence(
            exceedances=exceedances,
            simulations=simulations,
        )


def test_invalid_hac_lags_are_rejected() -> None:
    values = [0.01, 0.02, 0.03]

    with pytest.raises(ValueError):
        newey_west_mean_evidence(
            values,
            max_lag=-1,
        )

    with pytest.raises(ValueError):
        newey_west_mean_evidence(
            values,
            max_lag=3,
        )
