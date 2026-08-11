"""Provider-independent contracts for retrieving price observations."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PriceObservation:
    """One dated price observation returned by a provider."""

    observation_date: date
    price: float


@dataclass(frozen=True)
class ProviderRequest:
    """A provider-independent request for an instrument price series."""

    instrument_id: str
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("instrument_id must not be empty")

        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")


@dataclass(frozen=True)
class ProviderProvenance:
    """Identity and usage information for the returned dataset."""

    provider_name: str
    source_description: str
    retrieved_or_frozen_on: date
    adjustment_status: str
    license_note: str
    data_state: str

    def __post_init__(self) -> None:
        required_text = {
            "provider_name": self.provider_name,
            "source_description": self.source_description,
            "adjustment_status": self.adjustment_status,
            "license_note": self.license_note,
            "data_state": self.data_state,
        }

        for field_name, value in required_text.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class ProviderDataset:
    """Raw observations and provenance returned by a provider."""

    instrument_id: str
    observations: tuple[PriceObservation, ...]
    provenance: ProviderProvenance

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("instrument_id must not be empty")

        if not self.observations:
            raise ValueError("observations must not be empty")


@runtime_checkable
class PriceProvider(Protocol):
    """Contract implemented by every price-data provider."""

    def fetch_prices(self, request: ProviderRequest) -> ProviderDataset:
        """Return observations and provenance for one instrument."""
        ...
