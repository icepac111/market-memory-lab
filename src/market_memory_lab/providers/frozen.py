"""Deterministic synthetic price data for public demonstrations."""

from dataclasses import dataclass
from datetime import date

from market_memory_lab.providers.base import (
    PriceObservation,
    ProviderDataset,
    ProviderProvenance,
    ProviderRequest,
)


@dataclass(frozen=True)
class FrozenInstrumentDefinition:
    """One immutable synthetic instrument available from the frozen provider."""

    instrument_id: str
    display_name: str
    description: str
    observations: tuple[PriceObservation, ...]


STABLE_ALPHA_OBSERVATIONS = (
    PriceObservation(date(2026, 1, 2), 100.0),
    PriceObservation(date(2026, 1, 5), 102.0),
    PriceObservation(date(2026, 1, 6), 100.98),
    PriceObservation(date(2026, 1, 7), 104.0094),
    PriceObservation(date(2026, 1, 8), 101.929212),
    PriceObservation(date(2026, 1, 9), 106.0056726),
    PriceObservation(date(2026, 1, 12), 104.945615874),
    PriceObservation(date(2026, 1, 13), 108.09498435022),
)

STABLE_TWIN_OBSERVATIONS = (
    PriceObservation(date(2026, 1, 2), 250.0),
    PriceObservation(date(2026, 1, 5), 255.0),
    PriceObservation(date(2026, 1, 6), 252.45),
    PriceObservation(date(2026, 1, 7), 260.0235),
    PriceObservation(date(2026, 1, 8), 254.82303),
    PriceObservation(date(2026, 1, 9), 265.0141815),
    PriceObservation(date(2026, 1, 12), 262.364039685),
    PriceObservation(date(2026, 1, 13), 270.23746087555),
)

FROZEN_INSTRUMENTS = {
    "synthetic-stable-alpha": FrozenInstrumentDefinition(
        instrument_id="synthetic-stable-alpha",
        display_name="Synthetic Stable Alpha",
        description=(
            "Controlled synthetic price path used for reproducible "
            "demonstrations. It is not an observation from a real market."
        ),
        observations=STABLE_ALPHA_OBSERVATIONS,
    ),
    "synthetic-stable-twin": FrozenInstrumentDefinition(
        instrument_id="synthetic-stable-twin",
        display_name="Synthetic Stable Twin",
        description=(
            "Controlled synthetic price path with the same return sequence "
            "as Synthetic Stable Alpha and a different initial price. "
            "It is not an observation from a real market."
        ),
        observations=STABLE_TWIN_OBSERVATIONS,
    ),
}


class FrozenDemonstrationProvider:
    """Return immutable synthetic observations for reproducible demonstrations."""

    provider_name = "Market Memory Lab frozen demonstration provider"

    def available_instrument_ids(self) -> tuple[str, ...]:
        """Return stable identifiers for all available synthetic instruments."""
        return tuple(sorted(FROZEN_INSTRUMENTS))

    def get_definition(self, instrument_id: str) -> FrozenInstrumentDefinition:
        """Return the definition for one available synthetic instrument."""
        try:
            return FROZEN_INSTRUMENTS[instrument_id]
        except KeyError as exc:
            available = ", ".join(self.available_instrument_ids())
            raise KeyError(
                f"Unknown frozen instrument_id {instrument_id!r}. "
                f"Available instrument IDs: {available}"
            ) from exc

    def fetch_prices(self, request: ProviderRequest) -> ProviderDataset:
        """Return synthetic observations within the inclusive requested dates."""
        definition = self.get_definition(request.instrument_id)

        observations = tuple(
            observation
            for observation in definition.observations
            if request.start_date
            <= observation.observation_date
            <= request.end_date
        )

        if not observations:
            first_date = definition.observations[0].observation_date
            last_date = definition.observations[-1].observation_date
            raise ValueError(
                "No frozen observations are available for "
                f"{request.instrument_id!r} between "
                f"{request.start_date.isoformat()} and "
                f"{request.end_date.isoformat()}. "
                "The available demonstration range is "
                f"{first_date.isoformat()} to {last_date.isoformat()}."
            )

        provenance = ProviderProvenance(
            provider_name=self.provider_name,
            source_description=definition.description,
            retrieved_or_frozen_on=date(2026, 8, 11),
            adjustment_status=(
                "Not applicable to this controlled synthetic demonstration"
            ),
            license_note=(
                "Generated within Market Memory Lab for reproducible "
                "educational and scientific demonstrations"
            ),
            data_state="Synthetic controlled demonstration",
        )

        return ProviderDataset(
            instrument_id=definition.instrument_id,
            observations=observations,
            provenance=provenance,
        )
