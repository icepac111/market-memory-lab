"""Tests for provider-independent instrument identity."""

from __future__ import annotations

import pytest

from market_memory_lab.catalog import (
    InstrumentCatalog,
    InstrumentDefinition,
    demonstration_catalog,
)


def basic_instrument(
    instrument_id: str,
    *,
    reference: str | None = None,
) -> InstrumentDefinition:
    """Return a valid deterministic catalog record."""
    return InstrumentDefinition(
        instrument_id=instrument_id,
        display_name=f"Instrument {instrument_id}",
        asset_class="other",
        subtype="test",
        structure="traditional",
        venue="Test venue",
        currency="USD",
        timezone="UTC",
        frequency="Daily",
        provider="Unit test",
        license_note="Synthetic test metadata",
        data_state="sample_only",
        reference_instrument_id=reference,
    )


def test_catalog_lookup_and_search() -> None:
    catalog = demonstration_catalog()

    tokenized = catalog.get("mml-tokenized-claim")

    assert tokenized.asset_class == "tokenized_asset"
    assert (
        tokenized.reference_instrument_id
        == "MML-TRADITIONAL-REFERENCE"
    )

    stablecoin_matches = catalog.search("stablecoin")

    assert len(stablecoin_matches) == 1
    assert (
        stablecoin_matches[0].instrument_id
        == "MML-STABLECOIN-DEMO"
    )


def test_reference_relationships_are_queryable() -> None:
    catalog = demonstration_catalog()

    relationships = catalog.references_of(
        "MML-TRADITIONAL-REFERENCE"
    )

    identifiers = {
        instrument.instrument_id
        for instrument in relationships
    }

    assert "MML-TOKENIZED-CLAIM" in identifiers
    assert "MML-OPTION-DEMO" in identifiers
    assert "MML-FUTURE-DEMO" in identifiers


def test_duplicate_identifiers_are_rejected() -> None:
    first = basic_instrument("A")
    second = basic_instrument("A")

    with pytest.raises(ValueError, match="unique"):
        InstrumentCatalog.from_iterable(
            [first, second]
        )


def test_missing_reference_is_rejected() -> None:
    referenced = basic_instrument(
        "TOKEN",
        reference="MISSING",
    )

    with pytest.raises(
        ValueError,
        match="reference instrument not found",
    ):
        InstrumentCatalog.from_iterable([referenced])


def test_self_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot reference itself"):
        basic_instrument(
            "SELF",
            reference="SELF",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("asset_class", "unsupported", "unsupported asset_class"),
        ("structure", "unsupported", "unsupported structure"),
        ("data_state", "unsupported", "unsupported data_state"),
    ],
)
def test_controlled_vocabulary_is_enforced(
    field: str,
    value: str,
    message: str,
) -> None:
    values = {
        "instrument_id": "TEST",
        "display_name": "Test",
        "asset_class": "other",
        "subtype": "test",
        "structure": "traditional",
        "venue": "Test venue",
        "currency": "USD",
        "timezone": "UTC",
        "frequency": "Daily",
        "provider": "Unit test",
        "license_note": "Synthetic",
        "data_state": "sample_only",
    }

    values[field] = value

    with pytest.raises(ValueError, match=message):
        InstrumentDefinition(**values)


def test_catalog_records_are_serializable() -> None:
    catalog = demonstration_catalog()

    record = catalog.get("MML-OPTION-DEMO").record()

    assert record["asset_class"] == "option"
    assert record["structure"] == "derivative"
    assert record["data_state"] == "connector_pending"
