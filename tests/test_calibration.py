"""Tests for Gaussian Monte Carlo calibration."""

from __future__ import annotations

import numpy as np
import pytest

from market_memory_lab.calibration import (
    central_interval,
    empirical_percentile,
    gaussian_null_calibration,
)


def test_calibration_is_exactly_reproducible() -> None:
    first = gaussian_null_calibration(
        sample_size=100,
        replications=200,
        seed=42,
        batch_size=25,
    )
    second = gaussian_null_calibration(
        sample_size=100,
        replications=200,
        seed=42,
        batch_size=25,
    )

    np.testing.assert_array_equal(
        first.sample_mean_z,
        second.sample_mean_z,
    )
    np.testing.assert_array_equal(
        first.sample_variance_z,
        second.sample_variance_z,
    )
    np.testing.assert_array_equal(
        first.lag_one_correlation,
        second.lag_one_correlation,
    )


def test_calibration_is_invariant_to_batch_size() -> None:
    first = gaussian_null_calibration(
        sample_size=50,
        replications=101,
        seed=9,
        batch_size=7,
    )
    second = gaussian_null_calibration(
        sample_size=50,
        replications=101,
        seed=9,
        batch_size=101,
    )

    np.testing.assert_array_equal(
        first.sample_mean_z,
        second.sample_mean_z,
    )
    np.testing.assert_array_equal(
        first.sample_variance_z,
        second.sample_variance_z,
    )
    np.testing.assert_array_equal(
        first.lag_one_correlation,
        second.lag_one_correlation,
    )


def test_calibration_does_not_modify_global_numpy_state() -> None:
    np.random.seed(123)
    expected = np.random.random(8)

    np.random.seed(123)
    gaussian_null_calibration(
        sample_size=30,
        replications=20,
        seed=100,
    )
    actual = np.random.random(8)

    np.testing.assert_array_equal(actual, expected)


def test_output_shapes_and_finiteness() -> None:
    result = gaussian_null_calibration(
        sample_size=50,
        replications=120,
        seed=4,
    )

    assert result.sample_mean_z.shape == (120,)
    assert result.sample_variance_z.shape == (120,)
    assert result.lag_one_correlation.shape == (120,)

    assert np.isfinite(result.sample_mean_z).all()
    assert np.isfinite(result.sample_variance_z).all()
    assert np.isfinite(result.lag_one_correlation).all()


def test_mean_z_distribution_is_calibrated() -> None:
    result = gaussian_null_calibration(
        sample_size=200,
        replications=5_000,
        seed=20260809,
        batch_size=100,
    )

    observed_mean = float(result.sample_mean_z.mean())
    observed_variance = float(
        result.sample_mean_z.var(ddof=1)
    )

    assert abs(observed_mean) < 0.05
    assert abs(observed_variance - 1.0) < 0.08


def test_lag_one_average_is_near_finite_sample_null() -> None:
    sample_size = 200

    result = gaussian_null_calibration(
        sample_size=sample_size,
        replications=5_000,
        seed=20260810,
        batch_size=100,
    )

    observed_average = float(
        result.lag_one_correlation.mean()
    )

    finite_sample_reference = -1.0 / sample_size

    assert abs(
        observed_average - finite_sample_reference
    ) < 0.01


def test_empirical_percentile_midpoint_rule() -> None:
    reference = np.array([1.0, 2.0, 2.0, 4.0])

    actual = empirical_percentile(
        reference,
        observed=2.0,
    )

    assert actual == pytest.approx(0.50)


def test_empirical_percentile_bounds() -> None:
    reference = np.array([1.0, 2.0, 3.0])

    assert empirical_percentile(
        reference,
        observed=0.0,
    ) == pytest.approx(0.0)

    assert empirical_percentile(
        reference,
        observed=4.0,
    ) == pytest.approx(1.0)


def test_central_interval_known_values() -> None:
    reference = np.arange(1.0, 101.0)

    actual = central_interval(
        reference,
        coverage=0.80,
    )
    expected = tuple(
        float(value)
        for value in np.quantile(
            reference,
            [0.10, 0.90],
        )
    )

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("sample_size", "replications", "seed", "error"),
    [
        (2, 100, 1, "sample_size"),
        (100, 9, 1, "replications"),
        (100, 100, -1, "seed"),
    ],
)
def test_invalid_integer_inputs_are_rejected(
    sample_size: int,
    replications: int,
    seed: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        gaussian_null_calibration(
            sample_size=sample_size,
            replications=replications,
            seed=seed,
        )
