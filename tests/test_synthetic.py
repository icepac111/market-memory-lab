"""Tests for seed-controlled synthetic falsification processes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from market_memory_lab.synthetic import iid_gaussian


def test_iid_gaussian_is_exactly_reproducible() -> None:
    first = iid_gaussian(
        n=100,
        seed=42,
        mean=0.01,
        sigma=0.02,
    )
    second = iid_gaussian(
        n=100,
        seed=42,
        mean=0.01,
        sigma=0.02,
    )

    np.testing.assert_array_equal(
        first.values.to_numpy(),
        second.values.to_numpy(),
    )


def test_different_seeds_produce_different_series() -> None:
    first = iid_gaussian(n=100, seed=1)
    second = iid_gaussian(n=100, seed=2)

    assert not np.array_equal(
        first.values.to_numpy(),
        second.values.to_numpy(),
    )


def test_generator_does_not_modify_global_numpy_state() -> None:
    np.random.seed(12345)
    expected = np.random.random(10)

    np.random.seed(12345)
    iid_gaussian(n=100, seed=999)
    actual = np.random.random(10)

    np.testing.assert_array_equal(actual, expected)


def test_metadata_matches_documented_mechanism() -> None:
    result = iid_gaussian(
        n=50,
        seed=7,
        mean=0.02,
        sigma=0.10,
    )

    assert result.process == "iid_gaussian"
    assert result.seed == 7
    assert result.mechanism == "independent Gaussian innovations"
    assert result.expected_dependence == "none"
    assert dict(result.parameters) == {
        "n": 50,
        "mean": 0.02,
        "sigma": 0.10,
    }


def test_metadata_parameters_are_immutable() -> None:
    result = iid_gaussian(n=10, seed=2)

    with pytest.raises(TypeError):
        result.parameters["sigma"] = 5.0


def test_rejects_nonpositive_sample_length() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 2",
    ):
        iid_gaussian(n=1, seed=2)


def test_rejects_negative_seed() -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        iid_gaussian(n=10, seed=-1)


def test_rejects_boolean_seed() -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        iid_gaussian(n=10, seed=True)


def test_rejects_nonpositive_sigma() -> None:
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        iid_gaussian(
            n=10,
            seed=2,
            sigma=0.0,
        )


def test_large_sample_mean_matches_sampling_theory() -> None:
    """
    For IID Gaussian observations, the sample-mean standard error is:

        sigma / sqrt(n)

    The deterministic seeded sample must lie within five standard errors
    of the requested population mean.
    """
    n = 100_000
    target_mean = 0.03
    target_sigma = 0.20

    result = iid_gaussian(
        n=n,
        seed=20260809,
        mean=target_mean,
        sigma=target_sigma,
    )

    observed_mean = float(result.values.mean())
    standard_error = target_sigma / np.sqrt(n)

    assert abs(observed_mean - target_mean) <= 5.0 * standard_error


def test_large_sample_variance_matches_sampling_theory() -> None:
    """
    If X is Gaussian, then:

        (n - 1) * S^2 / sigma^2

    follows a chi-squared distribution with n - 1 degrees of freedom.

    For a large sample, the approximate standard deviation of S^2 is:

        sigma^2 * sqrt(2 / (n - 1))

    The deterministic seeded sample must lie within five approximate
    standard deviations of the target variance.
    """
    n = 100_000
    target_sigma = 0.20
    target_variance = target_sigma**2

    result = iid_gaussian(
        n=n,
        seed=20260809,
        sigma=target_sigma,
    )

    observed_variance = float(result.values.var(ddof=1))
    variance_standard_error = (
        target_variance * np.sqrt(2.0 / (n - 1))
    )

    assert (
        abs(observed_variance - target_variance)
        <= 5.0 * variance_standard_error
    )


@given(
    n=st.integers(min_value=2, max_value=500),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    mean=st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    sigma=st.floats(
        min_value=1e-6,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_iid_gaussian_general_properties(
    n: int,
    seed: int,
    mean: float,
    sigma: float,
) -> None:
    result = iid_gaussian(
        n=n,
        seed=seed,
        mean=mean,
        sigma=sigma,
    )

    assert len(result.values) == n
    assert np.isfinite(result.values.to_numpy()).all()
    assert result.values.index.name == "observation"
    assert result.values.name == "return"
    assert result.values.dtype == np.dtype("float64")


def test_student_t_is_exactly_reproducible() -> None:
    from market_memory_lab.synthetic import iid_student_t

    first = iid_student_t(
        n=1_000,
        seed=81,
        degrees_of_freedom=5.0,
        mean=0.01,
        sigma=0.20,
    )

    second = iid_student_t(
        n=1_000,
        seed=81,
        degrees_of_freedom=5.0,
        mean=0.01,
        sigma=0.20,
    )

    np.testing.assert_array_equal(
        first.values.to_numpy(),
        second.values.to_numpy(),
    )


def test_student_t_does_not_modify_global_numpy_state() -> None:
    from market_memory_lab.synthetic import iid_student_t

    np.random.seed(24680)
    expected = np.random.random(10)

    np.random.seed(24680)

    iid_student_t(
        n=1_000,
        seed=999,
        degrees_of_freedom=5.0,
    )

    actual = np.random.random(10)

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_student_t_requires_finite_variance() -> None:
    from market_memory_lab.synthetic import iid_student_t

    with pytest.raises(
        ValueError,
        match="must exceed 2",
    ):
        iid_student_t(
            n=100,
            seed=1,
            degrees_of_freedom=2.0,
        )


def test_student_t_metadata_declares_no_temporal_memory() -> None:
    from market_memory_lab.synthetic import iid_student_t

    result = iid_student_t(
        n=100,
        seed=4,
        degrees_of_freedom=5.0,
        mean=0.0,
        sigma=1.0,
    )

    assert result.process == "iid_student_t"
    assert result.expected_dependence == "none"
    assert "heavy-tailed" in result.mechanism
    assert (
        result.parameters["degrees_of_freedom"]
        == 5.0
    )
    assert (
        result.parameters["fourth_moment_state"]
        == "finite"
    )
    assert (
        result.parameters["theoretical_excess_kurtosis"]
        == pytest.approx(6.0)
    )


def test_student_t_infinite_fourth_moment_is_explicit() -> None:
    from market_memory_lab.synthetic import iid_student_t

    result = iid_student_t(
        n=100,
        seed=5,
        degrees_of_freedom=3.0,
    )

    assert (
        result.parameters["fourth_moment_state"]
        == "infinite"
    )
    assert (
        result.parameters["theoretical_excess_kurtosis"]
        is None
    )


def test_student_t_large_sample_mean_matches_target() -> None:
    from market_memory_lab.synthetic import iid_student_t

    sample_size = 300_000
    target_mean = 0.03
    target_sigma = 0.20

    result = iid_student_t(
        n=sample_size,
        seed=20260810,
        degrees_of_freedom=8.0,
        mean=target_mean,
        sigma=target_sigma,
    )

    observed_mean = float(
        result.values.mean()
    )

    theoretical_standard_error = (
        target_sigma / math.sqrt(sample_size)
    )

    assert (
        abs(observed_mean - target_mean)
        <= 6.0 * theoretical_standard_error
    )


def test_student_t_large_sample_variance_matches_target() -> None:
    from market_memory_lab.synthetic import iid_student_t

    target_sigma = 0.20
    target_variance = target_sigma**2

    result = iid_student_t(
        n=300_000,
        seed=20260810,
        degrees_of_freedom=8.0,
        sigma=target_sigma,
    )

    observed_variance = float(
        result.values.var(ddof=1)
    )

    relative_error = abs(
        observed_variance - target_variance
    ) / target_variance

    assert relative_error < 0.02


def test_student_t_has_heavier_tails_than_gaussian_control() -> None:
    from scipy import stats

    from market_memory_lab.synthetic import iid_student_t

    sample_size = 300_000

    heavy_tailed = iid_student_t(
        n=sample_size,
        seed=20260811,
        degrees_of_freedom=8.0,
        sigma=1.0,
    )

    gaussian = iid_gaussian(
        n=sample_size,
        seed=20260811,
        sigma=1.0,
    )

    student_excess_kurtosis = float(
        stats.kurtosis(
            heavy_tailed.values.to_numpy(),
            fisher=True,
            bias=False,
        )
    )

    gaussian_excess_kurtosis = float(
        stats.kurtosis(
            gaussian.values.to_numpy(),
            fisher=True,
            bias=False,
        )
    )

    theoretical_student_excess = (
        6.0 / (8.0 - 4.0)
    )

    assert student_excess_kurtosis > 1.0

    assert abs(
        student_excess_kurtosis
        - theoretical_student_excess
    ) < 0.30

    assert abs(gaussian_excess_kurtosis) < 0.10

    assert (
        student_excess_kurtosis
        > gaussian_excess_kurtosis
    )
