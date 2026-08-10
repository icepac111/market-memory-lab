"""Tests for pairwise structural similarity evidence."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_memory_lab.empirical import (
    AssetMetadata,
    analyze_empirical_prices,
)
from market_memory_lab.similarity import (
    compare_validated_assets,
)


def build_metadata(
    identifier: str,
    *,
    currency: str = "USD",
    timezone: str = "UTC",
    frequency: str = "Daily",
) -> AssetMetadata:
    """Create deterministic metadata for similarity tests."""
    return AssetMetadata(
        dataset_name=f"Dataset {identifier}",
        source="Unit test",
        asset_identifier=identifier,
        asset_class="Other",
        venue="Test venue",
        currency=currency,
        timezone=timezone,
        frequency=frequency,
        adjustment_status="Adjusted close",
        license_note="Synthetic test data",
    )


def evidence_from_prices(
    identifier: str,
    dates: pd.DatetimeIndex,
    prices: list[float] | np.ndarray,
    *,
    currency: str = "USD",
    timezone: str = "UTC",
    frequency: str = "Daily",
):
    """Create validated empirical evidence from deterministic prices."""
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
        metadata=build_metadata(
            identifier,
            currency=currency,
            timezone=timezone,
            frequency=frequency,
        ),
        periods_per_year=252,
    )


def prices_from_returns(
    returns: np.ndarray,
    *,
    initial_price: float = 100.0,
) -> np.ndarray:
    """Convert simple returns into a strictly positive price path."""
    return np.concatenate(
        [
            np.array([initial_price]),
            initial_price * np.cumprod(1.0 + returns),
        ]
    )


def test_identical_returns_have_perfect_surface_similarity() -> None:
    returns = np.array(
        [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    )
    dates = pd.date_range(
        "2026-01-01",
        periods=len(returns) + 1,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        dates,
        prices_from_returns(returns),
    )
    second = evidence_from_prices(
        "B",
        dates,
        prices_from_returns(returns),
    )

    result = compare_validated_assets(first, second)

    assert result.is_valid
    assert result.overlapping_returns == len(returns)
    assert result.pearson_correlation == pytest.approx(1.0)
    assert result.spearman_correlation == pytest.approx(1.0)
    assert result.volatility_ratio == pytest.approx(1.0)
    assert result.standardized_wasserstein_distance == pytest.approx(0.0)
    assert result.correlation_stability_gap == pytest.approx(0.0)


def test_inverse_returns_have_negative_surface_similarity() -> None:
    first_returns = np.array(
        [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    )
    second_returns = -first_returns

    dates = pd.date_range(
        "2026-01-01",
        periods=len(first_returns) + 1,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        dates,
        prices_from_returns(first_returns),
    )
    second = evidence_from_prices(
        "B",
        dates,
        prices_from_returns(second_returns),
    )

    result = compare_validated_assets(first, second)

    assert result.is_valid
    assert result.pearson_correlation == pytest.approx(-1.0)
    assert result.spearman_correlation == pytest.approx(-1.0)


def test_returns_are_aligned_after_independent_calculation() -> None:
    first_dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
        ]
    )
    second_dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ]
    )

    first = evidence_from_prices(
        "A",
        first_dates,
        [100, 101, 102, 100, 103, 104, 105],
    )
    second = evidence_from_prices(
        "B",
        second_dates,
        [200, 202, 204, 203, 207, 208, 210],
    )

    result = compare_validated_assets(
        first,
        second,
        minimum_overlap=5,
    )

    assert result.is_valid
    assert result.overlapping_returns == 5
    assert result.overlap_start.startswith("2026-01-03")
    assert result.overlap_end.startswith("2026-01-07")


def test_insufficient_overlap_blocks_similarity() -> None:
    first_dates = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="D",
    )
    second_dates = pd.date_range(
        "2026-01-06",
        periods=8,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        first_dates,
        np.linspace(100.0, 107.0, 8),
    )
    second = evidence_from_prices(
        "B",
        second_dates,
        np.linspace(200.0, 207.0, 8),
    )

    result = compare_validated_assets(
        first,
        second,
        minimum_overlap=4,
    )

    assert not result.is_valid
    assert result.overlapping_returns == 2
    assert any("overlapping" in error for error in result.errors)


def test_currency_timezone_and_frequency_warnings() -> None:
    returns = np.array(
        [0.01, -0.01, 0.02, -0.02, 0.03, -0.01]
    )
    dates = pd.date_range(
        "2026-01-01",
        periods=len(returns) + 1,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        dates,
        prices_from_returns(returns),
        currency="USD",
        timezone="America/New_York",
        frequency="Daily",
    )
    second = evidence_from_prices(
        "B",
        dates,
        prices_from_returns(returns),
        currency="EUR",
        timezone="Europe/London",
        frequency="Weekly",
    )

    result = compare_validated_assets(first, second)

    assert result.is_valid
    assert any("currencies differ" in warning for warning in result.warnings)
    assert any("timezones differ" in warning for warning in result.warnings)
    assert any(
        "frequencies differ" in warning
        for warning in result.warnings
    )


def test_invalid_empirical_input_blocks_comparison() -> None:
    dates = pd.date_range(
        "2026-01-01",
        periods=7,
        freq="D",
    )

    valid = evidence_from_prices(
        "A",
        dates,
        [100, 101, 102, 103, 104, 105, 106],
    )

    invalid_frame = pd.DataFrame(
        {
            "Date": dates,
            "Price": [100, 101, 0, 103, 104, 105, 106],
        }
    )

    invalid = analyze_empirical_prices(
        invalid_frame,
        date_column="Date",
        price_column="Price",
        metadata=build_metadata("INVALID"),
        periods_per_year=252,
    )

    result = compare_validated_assets(valid, invalid)

    assert not result.is_valid
    assert any("Asset B failed" in error for error in result.errors)


def test_relationship_stability_detects_regime_reversal() -> None:
    first_returns = np.array(
        [
            0.01,
            -0.02,
            0.03,
            0.02,
            -0.01,
            0.015,
            0.01,
            -0.02,
            0.03,
            0.02,
            -0.01,
            0.015,
        ]
    )

    second_returns = np.concatenate(
        [
            first_returns[:6],
            -first_returns[6:],
        ]
    )

    dates = pd.date_range(
        "2026-01-01",
        periods=len(first_returns) + 1,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        dates,
        prices_from_returns(first_returns),
    )
    second = evidence_from_prices(
        "B",
        dates,
        prices_from_returns(second_returns),
    )

    result = compare_validated_assets(first, second)

    assert result.is_valid
    assert result.first_half_correlation == pytest.approx(1.0)
    assert result.second_half_correlation == pytest.approx(-1.0)
    assert result.correlation_stability_gap == pytest.approx(2.0)


def test_manifest_preserves_unavailable_scientific_states() -> None:
    returns = np.array(
        [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    )
    dates = pd.date_range(
        "2026-01-01",
        periods=len(returns) + 1,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        dates,
        prices_from_returns(returns),
    )
    second = evidence_from_prices(
        "B",
        dates,
        prices_from_returns(returns),
    )

    result = compare_validated_assets(first, second)
    manifest = result.manifest()

    assert manifest["scientific_state"]["memory_similarity"] == "not_tested"
    assert manifest["scientific_state"]["regime_similarity"] == "not_tested"
    assert manifest["scientific_state"]["lead_lag_direction"] == "not_tested"
    assert manifest["scientific_state"]["causation_state"] == "not_established"
    assert manifest["scientific_state"]["trading_conclusion"] == "abstain"


def test_nonfinite_values_never_escape_valid_result() -> None:
    returns = np.array(
        [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    )
    dates = pd.date_range(
        "2026-01-01",
        periods=len(returns) + 1,
        freq="D",
    )

    first = evidence_from_prices(
        "A",
        dates,
        prices_from_returns(returns),
    )
    second = evidence_from_prices(
        "B",
        dates,
        prices_from_returns(returns * 0.5),
    )

    result = compare_validated_assets(first, second)

    numeric_values = [
        result.pearson_correlation,
        result.spearman_correlation,
        result.volatility_a,
        result.volatility_b,
        result.volatility_ratio,
        result.maximum_drawdown_a,
        result.maximum_drawdown_b,
        result.drawdown_difference,
        result.standardized_wasserstein_distance,
        result.first_half_correlation,
        result.second_half_correlation,
        result.correlation_stability_gap,
    ]

    assert all(
        value is None or math.isfinite(value)
        for value in numeric_values
    )
