"""Tests for provider-independent market-data contracts."""

from datetime import date

import pytest

from market_memory_lab.providers import (
    PriceObservation,
    PriceProvider,
    ProviderDataset,
    ProviderProvenance,
    ProviderRequest,
)


def make_provenance() -> ProviderProvenance:
    return ProviderProvenance(
        provider_name="Controlled demonstration provider",
        source_description="Synthetic observations for automated tests",
        retrieved_or_frozen_on=date(2026, 8, 11),
        adjustment_status="Not applicable",
        license_note="Generated within the test suite",
        data_state="Synthetic",
    )


def test_provider_request_records_identity_and_dates() -> None:
    request = ProviderRequest(
        instrument_id="synthetic-alpha",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
    )

    assert request.instrument_id == "synthetic-alpha"
    assert request.start_date == date(2026, 1, 1)
    assert request.end_date == date(2026, 1, 31)


def test_provider_request_rejects_empty_instrument_id() -> None:
    with pytest.raises(ValueError, match="instrument_id"):
        ProviderRequest(
            instrument_id=" ",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )


def test_provider_request_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="start_date"):
        ProviderRequest(
            instrument_id="synthetic-alpha",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 1, 1),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "provider_name",
        "source_description",
        "adjustment_status",
        "license_note",
        "data_state",
    ],
)
def test_provenance_rejects_empty_required_text(field_name: str) -> None:
    values = {
        "provider_name": "Controlled demonstration provider",
        "source_description": "Synthetic observations",
        "retrieved_or_frozen_on": date(2026, 8, 11),
        "adjustment_status": "Not applicable",
        "license_note": "Generated within the test suite",
        "data_state": "Synthetic",
    }
    values[field_name] = " "

    with pytest.raises(ValueError, match=field_name):
        ProviderProvenance(**values)


def test_provider_dataset_requires_observations() -> None:
    with pytest.raises(ValueError, match="observations"):
        ProviderDataset(
            instrument_id="synthetic-alpha",
            observations=(),
            provenance=make_provenance(),
        )


def test_provider_dataset_preserves_raw_observations() -> None:
    observations = (
        PriceObservation(date(2026, 1, 1), 100.0),
        PriceObservation(date(2026, 1, 2), 101.0),
    )

    dataset = ProviderDataset(
        instrument_id="synthetic-alpha",
        observations=observations,
        provenance=make_provenance(),
    )

    assert dataset.observations == observations
    assert dataset.provenance.data_state == "Synthetic"


def test_structural_provider_implements_protocol() -> None:
    class ExampleProvider:
        def fetch_prices(self, request: ProviderRequest) -> ProviderDataset:
            return ProviderDataset(
                instrument_id=request.instrument_id,
                observations=(
                    PriceObservation(request.start_date, 100.0),
                    PriceObservation(request.end_date, 101.0),
                ),
                provenance=make_provenance(),
            )

    provider = ExampleProvider()

    assert isinstance(provider, PriceProvider)
