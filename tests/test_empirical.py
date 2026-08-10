"""Tests for empirical financial-series validation and evidence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from market_memory_lab.empirical import (
    AssetMetadata,
    analyze_empirical_prices,
)


def metadata() -> AssetMetadata:
    """Return fixed metadata for empirical unit tests."""
    return AssetMetadata(
        dataset_name="Reference asset",
        source="Unit test",
        asset_identifier="TEST",
        asset_class="Equity",
        venue="Test venue",
        currency="USD",
        timezone="America/New_York",
        frequency="Daily",
        adjustment_status="Adjusted close",
        license_note="Synthetic unit-test data",
    )


def valid_frame() -> pd.DataFrame:
    """Return a deterministic valid reference frame."""
    return pd.DataFrame(
        {
            "Date": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
            ],
            "Adjusted Close": [
                100.0,
                110.0,
                99.0,
                103.95,
            ],
        }
    )


def test_valid_reference_metrics() -> None:
    evidence = analyze_empirical_prices(
        valid_frame(),
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert evidence.is_valid
    assert evidence.errors == ()
    assert evidence.validation.valid_rows == 4

    expected_returns = np.array([0.10, -0.10, 0.05])

    np.testing.assert_allclose(
        evidence.simple_return_series.to_numpy(),
        expected_returns,
    )

    assert evidence.cumulative_return == pytest.approx(0.0395)
    assert evidence.maximum_drawdown == pytest.approx(-0.10)


def test_unsorted_input_is_sorted_and_warned() -> None:
    frame = valid_frame().iloc[[2, 0, 3, 1]].reset_index(drop=True)

    evidence = analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert evidence.is_valid
    assert not evidence.validation.originally_sorted
    assert evidence.warnings
    assert evidence.canonical_data["date"].is_monotonic_increasing


def test_duplicate_dates_are_rejected() -> None:
    frame = valid_frame()
    frame.loc[3, "Date"] = frame.loc[2, "Date"]

    evidence = analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert not evidence.is_valid
    assert evidence.validation.duplicate_dates == 1
    assert any("duplicate" in error for error in evidence.errors)


@pytest.mark.parametrize(
    "bad_price",
    [0.0, -1.0],
)
def test_nonpositive_prices_are_rejected(
    bad_price: float,
) -> None:
    frame = valid_frame()
    frame.loc[1, "Adjusted Close"] = bad_price

    evidence = analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert not evidence.is_valid
    assert evidence.validation.nonpositive_prices == 1


@pytest.mark.parametrize(
    "bad_price",
    [None, "not-a-price", np.inf, -np.inf],
)
def test_invalid_prices_are_rejected(
    bad_price,
) -> None:
    frame = valid_frame()
    frame["Adjusted Close"] = frame["Adjusted Close"].astype("object")
    frame.loc[1, "Adjusted Close"] = bad_price

    evidence = analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert not evidence.is_valid
    assert evidence.validation.invalid_prices == 1


def test_invalid_date_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[1, "Date"] = "not-a-date"

    evidence = analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert not evidence.is_valid
    assert evidence.validation.invalid_dates == 1


def test_missing_columns_are_rejected() -> None:
    evidence = analyze_empirical_prices(
        pd.DataFrame({"A": [1, 2, 3]}),
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert not evidence.is_valid
    assert any(
        "Missing required columns" in error
        for error in evidence.errors
    )


def test_annualization_requires_explicit_frequency() -> None:
    evidence = analyze_empirical_prices(
        valid_frame(),
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=None,
    )

    assert evidence.is_valid
    assert evidence.arithmetic_annualized_return is None
    assert evidence.annualized_volatility is None
    assert any(
        "periods_per_year" in warning
        for warning in evidence.warnings
    )


def test_hash_is_deterministic() -> None:
    first = analyze_empirical_prices(
        valid_frame(),
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )
    second = analyze_empirical_prices(
        valid_frame(),
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    assert first.validation.canonical_sha256
    assert (
        first.validation.canonical_sha256
        == second.validation.canonical_sha256
    )
    assert len(first.validation.canonical_sha256) == hashlib.sha256().digest_size * 2


def test_manifest_is_valid_json_without_nan() -> None:
    evidence = analyze_empirical_prices(
        valid_frame(),
        date_column="Date",
        price_column="Adjusted Close",
        metadata=metadata(),
        periods_per_year=252,
    )

    encoded = evidence.manifest_json()
    decoded = json.loads(encoded)

    assert decoded["metadata"]["asset_identifier"] == "TEST"
    assert decoded["analysis"]["memory_conclusion"] == "not_tested"
    assert decoded["analysis"]["trading_conclusion"] == "abstain"
