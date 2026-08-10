"""Tests for tracking-integrity evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_memory_lab.empirical import (
    AssetMetadata,
    analyze_empirical_prices,
)
from market_memory_lab.tracking import (
    analyze_tracking_integrity,
)


def metadata(
    identifier: str,
    *,
    currency: str = "USD",
    timezone: str = "UTC",
) -> AssetMetadata:
    """Create deterministic metadata."""
    return AssetMetadata(
        dataset_name=identifier,
        source="Synthetic unit test",
        asset_identifier=identifier,
        asset_class="Synthetic demonstration",
        venue="Synthetic reference",
        currency=currency,
        timezone=timezone,
        frequency="Daily",
        adjustment_status="Adjusted close",
        license_note="Synthetic unit-test data",
    )


def prices_from_returns(
    returns: np.ndarray,
    *,
    initial_price: float,
) -> np.ndarray:
    """Build a price path from simple returns."""
    return np.concatenate(
        [
            np.array([initial_price]),
            initial_price * np.cumprod(1.0 + returns),
        ]
    )


def evidence(
    identifier: str,
    returns: np.ndarray,
    *,
    initial_price: float = 100.0,
    currency: str = "USD",
    timezone: str = "UTC",
):
    """Build validated evidence for tracking tests."""
    dates = pd.date_range(
        "2026-01-01",
        periods=len(returns) + 1,
        freq="D",
    )

    frame = pd.DataFrame(
        {
            "Date": dates,
            "Price": prices_from_returns(
                returns,
                initial_price=initial_price,
            ),
        }
    )

    return analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Price",
        metadata=metadata(
            identifier,
            currency=currency,
            timezone=timezone,
        ),
        periods_per_year=252,
    )


def test_identical_exposure_has_zero_tracking_error() -> None:
    returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    )

    instrument = evidence(
        "TOKENIZED",
        returns,
        initial_price=10.0,
    )
    reference = evidence(
        "REFERENCE",
        returns,
        initial_price=100.0,
    )

    result = analyze_tracking_integrity(instrument, reference)

    assert result.is_valid
    assert result.mean_active_return == pytest.approx(0.0)
    assert result.tracking_error == pytest.approx(0.0)
    assert result.correlation == pytest.approx(1.0)
    assert result.beta == pytest.approx(1.0)
    assert result.final_cumulative_divergence == pytest.approx(0.0)
    assert (
        result.maximum_absolute_cumulative_divergence
        == pytest.approx(0.0)
    )


def test_constant_active_return_has_zero_tracking_error() -> None:
    reference_returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    )
    active_increment = 0.001
    instrument_returns = reference_returns + active_increment

    instrument = evidence("INSTRUMENT", instrument_returns)
    reference = evidence("REFERENCE", reference_returns)

    result = analyze_tracking_integrity(instrument, reference)

    assert result.is_valid
    assert result.mean_active_return == pytest.approx(active_increment)
    assert result.tracking_error == pytest.approx(0.0)


def test_unstable_tracking_error_is_detected() -> None:
    reference_returns = np.array(
        [
            0.01,
            -0.02,
            0.03,
            0.01,
            -0.01,
            0.02,
            0.01,
            -0.02,
            0.03,
            0.01,
            -0.01,
            0.02,
        ]
    )

    first_active = np.zeros(6)
    second_active = np.array(
        [0.03, -0.03, 0.02, -0.02, 0.04, -0.04]
    )
    active = np.concatenate([first_active, second_active])

    instrument_returns = reference_returns + active

    instrument = evidence("INSTRUMENT", instrument_returns)
    reference = evidence("REFERENCE", reference_returns)

    result = analyze_tracking_integrity(instrument, reference)

    assert result.is_valid
    assert result.first_half_tracking_error == pytest.approx(0.0)
    assert result.second_half_tracking_error is not None
    assert result.second_half_tracking_error > 0.0
    assert result.tracking_error_stability_gap is not None
    assert result.tracking_error_stability_gap > 0.0


def test_currency_and_timezone_warnings() -> None:
    returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    )

    instrument = evidence(
        "INSTRUMENT",
        returns,
        currency="USD",
        timezone="UTC",
    )
    reference = evidence(
        "REFERENCE",
        returns,
        currency="EUR",
        timezone="Europe/London",
    )

    result = analyze_tracking_integrity(instrument, reference)

    assert result.is_valid
    assert any(
        "currencies differ" in warning
        for warning in result.warnings
    )
    assert any(
        "timezones differ" in warning
        for warning in result.warnings
    )


def test_insufficient_overlap_blocks_analysis() -> None:
    returns = np.array([0.01, -0.01, 0.02])

    instrument = evidence("INSTRUMENT", returns)
    reference = evidence("REFERENCE", returns)

    result = analyze_tracking_integrity(
        instrument,
        reference,
        minimum_overlap=6,
    )

    assert not result.is_valid
    assert result.overlapping_returns == 3


def test_scientific_boundaries_are_preserved() -> None:
    returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    )

    instrument = evidence("INSTRUMENT", returns)
    reference = evidence("REFERENCE", returns)

    result = analyze_tracking_integrity(instrument, reference)
    manifest = result.manifest()

    assert (
        manifest["scientific_state"]["economic_equivalence"]
        == "not_established"
    )
    assert (
        manifest["scientific_state"]["price_dislocation"]
        == "not_tested"
    )
    assert (
        manifest["scientific_state"]["market_memory"]
        == "not_tested"
    )
    assert (
        manifest["scientific_state"]["trading_conclusion"]
        == "abstain"
    )
