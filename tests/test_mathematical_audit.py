"""Independent reference calculations and mathematical invariance tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from market_memory_lab.empirical import (
    AssetMetadata,
    analyze_empirical_prices,
)
from market_memory_lab.metrics import (
    annualized_sharpe,
    annualized_volatility,
    cumulative_return,
    drawdown_series,
    log_returns,
    maximum_drawdown,
    simple_returns,
    wealth_index,
)
from market_memory_lab.similarity import compare_validated_assets
from market_memory_lab.tracking import analyze_tracking_integrity


def audit_metadata(identifier: str) -> AssetMetadata:
    """Create fixed metadata for independent audit fixtures."""
    return AssetMetadata(
        dataset_name=f"Audit {identifier}",
        source="Independent mathematical audit fixture",
        asset_identifier=identifier,
        asset_class="Other",
        venue="Synthetic audit reference",
        currency="USD",
        timezone="UTC",
        frequency="Daily",
        adjustment_status="Adjusted close",
        license_note="Synthetic audit fixture",
    )


def evidence_from_returns(
    identifier: str,
    returns: np.ndarray,
    *,
    initial_price: float = 100.0,
):
    """Build independently controlled empirical evidence."""
    prices = np.concatenate(
        [
            np.array([initial_price], dtype="float64"),
            initial_price * np.cumprod(1.0 + returns),
        ]
    )

    dates = pd.date_range(
        "2026-01-01",
        periods=len(prices),
        freq="D",
    )

    frame = pd.DataFrame(
        {
            "Date": dates,
            "Price": prices,
        }
    )

    return analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Price",
        metadata=audit_metadata(identifier),
        periods_per_year=252,
    )


def test_simple_returns_against_manual_loop() -> None:
    prices = pd.Series(
        [91.25, 94.80, 93.10, 100.40, 98.75],
        dtype="float64",
    )

    expected = []

    for index in range(1, len(prices)):
        expected.append(
            prices.iloc[index] / prices.iloc[index - 1] - 1.0
        )

    np.testing.assert_allclose(
        simple_returns(prices).to_numpy(),
        np.array(expected),
        rtol=1e-14,
        atol=1e-14,
    )


def test_log_returns_against_manual_log_ratio() -> None:
    prices = pd.Series(
        [91.25, 94.80, 93.10, 100.40, 98.75],
        dtype="float64",
    )

    expected = []

    for index in range(1, len(prices)):
        expected.append(
            math.log(
                prices.iloc[index] / prices.iloc[index - 1]
            )
        )

    np.testing.assert_allclose(
        log_returns(prices).to_numpy(),
        np.array(expected),
        rtol=1e-14,
        atol=1e-14,
    )


def test_positive_price_scaling_does_not_change_returns() -> None:
    prices = pd.Series(
        [10.0, 10.5, 9.8, 11.1, 10.9],
        dtype="float64",
    )

    scaled = 37.25 * prices

    np.testing.assert_allclose(
        simple_returns(prices),
        simple_returns(scaled),
        rtol=1e-13,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        log_returns(prices),
        log_returns(scaled),
        rtol=1e-13,
        atol=1e-13,
    )


def test_wealth_against_manual_compounding_loop() -> None:
    returns = pd.Series(
        [0.10, -0.20, 0.05, 0.03],
        dtype="float64",
    )

    manual = []
    current = 100.0

    for value in returns:
        current *= 1.0 + value
        manual.append(current)

    np.testing.assert_allclose(
        wealth_index(
            returns,
            initial_wealth=100.0,
        ).to_numpy(),
        np.array(manual),
        rtol=1e-14,
        atol=1e-14,
    )


def test_cumulative_return_matches_price_endpoint_ratio() -> None:
    prices = pd.Series(
        [100.0, 110.0, 88.0, 92.4, 95.172],
        dtype="float64",
    )

    returns = simple_returns(prices)
    expected = prices.iloc[-1] / prices.iloc[0] - 1.0

    assert cumulative_return(returns) == pytest.approx(
        expected,
        rel=1e-14,
        abs=1e-14,
    )


def test_drawdown_against_independent_manual_reference() -> None:
    returns = pd.Series(
        [0.10, -0.20, 0.05, 0.03, -0.10],
        dtype="float64",
    )

    wealth_values = [1.0]
    current = 1.0

    for value in returns:
        current *= 1.0 + value
        wealth_values.append(current)

    manual_drawdowns = []
    running_peak = wealth_values[0]

    for value in wealth_values[1:]:
        running_peak = max(running_peak, value)
        manual_drawdowns.append(value / running_peak - 1.0)

    actual = drawdown_series(returns).to_numpy()

    np.testing.assert_allclose(
        actual,
        np.array(manual_drawdowns),
        rtol=1e-14,
        atol=1e-14,
    )

    assert maximum_drawdown(returns) == pytest.approx(
        min(manual_drawdowns)
    )


def test_drawdown_bounds() -> None:
    returns = pd.Series(
        [0.20, -0.10, -0.30, 0.40, -0.05],
        dtype="float64",
    )

    values = drawdown_series(returns)

    assert bool((values <= 0.0).all())
    assert bool((values >= -1.0).all())


def test_annualized_volatility_against_manual_formula() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03, 0.005, -0.015],
        dtype="float64",
    )

    centered = returns.to_numpy() - returns.mean()

    manual_sample_variance = float(
        np.sum(centered**2) / (len(returns) - 1)
    )

    expected = math.sqrt(manual_sample_variance) * math.sqrt(252)

    assert annualized_volatility(
        returns,
        periods_per_year=252,
    ) == pytest.approx(
        expected,
        rel=1e-14,
        abs=1e-14,
    )


def test_sharpe_against_manual_reference() -> None:
    asset = pd.Series(
        [0.01, -0.02, 0.03, 0.005, 0.015],
        dtype="float64",
    )

    risk_free = pd.Series(
        [0.001, 0.001, 0.001, 0.001, 0.001],
        dtype="float64",
    )

    excess = asset.to_numpy() - risk_free.to_numpy()
    excess_mean = float(np.mean(excess))

    centered = excess - excess_mean
    excess_std = math.sqrt(
        float(np.sum(centered**2) / (len(excess) - 1))
    )

    expected = math.sqrt(252) * excess_mean / excess_std

    assert annualized_sharpe(
        asset,
        risk_free,
        periods_per_year=252,
    ) == pytest.approx(
        expected,
        rel=1e-14,
        abs=1e-14,
    )


def test_pairwise_result_is_independent_of_initial_price() -> None:
    returns = np.array(
        [
            0.01,
            -0.02,
            0.03,
            0.005,
            -0.01,
            0.02,
        ],
        dtype="float64",
    )

    first = evidence_from_returns(
        "A",
        returns,
        initial_price=7.5,
    )

    second = evidence_from_returns(
        "B",
        returns,
        initial_price=10_000.0,
    )

    result = compare_validated_assets(first, second)

    assert result.is_valid
    assert result.pearson_correlation == pytest.approx(1.0)
    assert result.spearman_correlation == pytest.approx(1.0)
    assert result.volatility_ratio == pytest.approx(1.0)
    assert result.drawdown_difference == pytest.approx(0.0)


def test_pearson_affine_invariance_reference() -> None:
    left = np.array(
        [1.5, 2.0, -1.0, 4.5, 3.0, 6.0],
        dtype="float64",
    )
    right = np.array(
        [-2.0, 1.0, 0.5, 7.0, 1.5, 8.0],
        dtype="float64",
    )

    original = float(stats.pearsonr(left, right).statistic)

    transformed_left = 11.0 + 3.25 * left
    transformed_right = -7.0 + 0.40 * right

    transformed = float(
        stats.pearsonr(
            transformed_left,
            transformed_right,
        ).statistic
    )

    assert transformed == pytest.approx(
        original,
        rel=1e-14,
        abs=1e-14,
    )


def test_spearman_invariance_under_strictly_increasing_map() -> None:
    values = np.array(
        [-2.0, -0.5, 0.0, 1.0, 3.0, 5.0],
        dtype="float64",
    )

    transformed = np.exp(values)

    coefficient = float(
        stats.spearmanr(values, transformed).statistic
    )

    assert coefficient == pytest.approx(1.0)


def test_wasserstein_identity_and_symmetry_reference() -> None:
    left = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype="float64",
    )
    right = np.array(
        [-1.5, -0.25, 0.5, 1.25, 3.0],
        dtype="float64",
    )

    identity = float(stats.wasserstein_distance(left, left))
    forward = float(stats.wasserstein_distance(left, right))
    reverse = float(stats.wasserstein_distance(right, left))

    assert identity == pytest.approx(0.0)
    assert forward == pytest.approx(reverse)
    assert forward >= 0.0


def test_tracking_error_against_manual_active_return_std() -> None:
    reference_returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        dtype="float64",
    )

    instrument_returns = np.array(
        [0.012, -0.018, 0.025, 0.014, -0.012, 0.023],
        dtype="float64",
    )

    instrument = evidence_from_returns(
        "INSTRUMENT",
        instrument_returns,
    )

    reference = evidence_from_returns(
        "REFERENCE",
        reference_returns,
    )

    result = analyze_tracking_integrity(
        instrument,
        reference,
    )

    active = instrument_returns - reference_returns
    active_mean = float(np.mean(active))

    centered = active - active_mean
    manual_tracking_error = math.sqrt(
        float(np.sum(centered**2) / (len(active) - 1))
    )

    assert result.mean_active_return == pytest.approx(active_mean)

    assert result.tracking_error == pytest.approx(
        manual_tracking_error,
        rel=1e-13,
        abs=1e-13,
    )


def test_tracking_beta_against_independent_least_squares() -> None:
    reference_returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        dtype="float64",
    )

    instrument_returns = (
        0.001 + 1.35 * reference_returns
    )

    instrument = evidence_from_returns(
        "INSTRUMENT",
        instrument_returns,
    )

    reference = evidence_from_returns(
        "REFERENCE",
        reference_returns,
    )

    result = analyze_tracking_integrity(
        instrument,
        reference,
    )

    design = np.column_stack(
        [
            np.ones(len(reference_returns)),
            reference_returns,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        instrument_returns,
        rcond=None,
    )

    independent_beta = float(coefficients[1])

    assert result.beta == pytest.approx(
        independent_beta,
        rel=1e-12,
        abs=1e-12,
    )

    assert result.beta == pytest.approx(1.35)


def test_constant_active_return_has_zero_tracking_error() -> None:
    reference_returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        dtype="float64",
    )

    instrument_returns = reference_returns + 0.002

    instrument = evidence_from_returns(
        "INSTRUMENT",
        instrument_returns,
    )

    reference = evidence_from_returns(
        "REFERENCE",
        reference_returns,
    )

    result = analyze_tracking_integrity(
        instrument,
        reference,
    )

    assert result.mean_active_return == pytest.approx(0.002)
    assert result.tracking_error == pytest.approx(0.0)


def test_cumulative_divergence_against_manual_wealth_difference() -> None:
    reference_returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        dtype="float64",
    )

    instrument_returns = np.array(
        [0.012, -0.018, 0.025, 0.014, -0.012, 0.023],
        dtype="float64",
    )

    instrument = evidence_from_returns(
        "INSTRUMENT",
        instrument_returns,
    )

    reference = evidence_from_returns(
        "REFERENCE",
        reference_returns,
    )

    result = analyze_tracking_integrity(
        instrument,
        reference,
    )

    manual_instrument_wealth = np.cumprod(
        1.0 + instrument_returns
    )
    manual_reference_wealth = np.cumprod(
        1.0 + reference_returns
    )
    manual_divergence = (
        manual_instrument_wealth - manual_reference_wealth
    )

    assert result.final_cumulative_divergence == pytest.approx(
        float(manual_divergence[-1]),
        rel=1e-13,
        abs=1e-13,
    )

    assert (
        result.maximum_absolute_cumulative_divergence
        == pytest.approx(
            float(np.max(np.abs(manual_divergence))),
            rel=1e-13,
            abs=1e-13,
        )
    )
