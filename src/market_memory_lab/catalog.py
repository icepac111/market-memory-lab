"""Provider-independent identity catalog for financial instruments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

ALLOWED_ASSET_CLASSES = frozenset(
    {
        "equity",
        "etf",
        "index",
        "fx",
        "commodity",
        "interest_rate",
        "bond",
        "fund",
        "cryptoasset",
        "stablecoin",
        "tokenized_asset",
        "option",
        "future",
        "perpetual_future",
        "swap",
        "other_derivative",
        "other",
    }
)

ALLOWED_STRUCTURES = frozenset(
    {
        "traditional",
        "native_digital",
        "tokenized_claim",
        "wrapped_asset",
        "fund",
        "derivative",
        "synthetic",
        "reference_index",
        "unknown",
    }
)

ALLOWED_DATA_STATES = frozenset(
    {
        "available",
        "sample_only",
        "connector_pending",
        "licensed",
        "restricted",
        "unavailable",
    }
)


@dataclass(frozen=True)
class InstrumentDefinition:
    """Canonical identity and provenance declaration for one instrument."""

    instrument_id: str
    display_name: str
    asset_class: str
    subtype: str
    structure: str
    venue: str
    currency: str
    timezone: str
    frequency: str
    provider: str
    license_note: str
    data_state: str
    reference_instrument_id: str | None = None
    contract_identifier: str | None = None
    unit_description: str | None = None

    def __post_init__(self) -> None:
        text_fields = {
            "instrument_id": self.instrument_id,
            "display_name": self.display_name,
            "asset_class": self.asset_class,
            "subtype": self.subtype,
            "structure": self.structure,
            "venue": self.venue,
            "currency": self.currency,
            "timezone": self.timezone,
            "frequency": self.frequency,
            "provider": self.provider,
            "license_note": self.license_note,
            "data_state": self.data_state,
        }

        for name, value in text_fields.items():
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")

            if not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if self.instrument_id != self.instrument_id.strip().upper():
            raise ValueError(
                "instrument_id must be uppercase without outer whitespace"
            )

        if self.asset_class not in ALLOWED_ASSET_CLASSES:
            raise ValueError(
                f"unsupported asset_class: {self.asset_class}"
            )

        if self.structure not in ALLOWED_STRUCTURES:
            raise ValueError(
                f"unsupported structure: {self.structure}"
            )

        if self.data_state not in ALLOWED_DATA_STATES:
            raise ValueError(
                f"unsupported data_state: {self.data_state}"
            )

        if self.reference_instrument_id is not None:
            if not isinstance(self.reference_instrument_id, str):
                raise TypeError(
                    "reference_instrument_id must be a string or None"
                )

            if (
                self.reference_instrument_id
                != self.reference_instrument_id.strip().upper()
            ):
                raise ValueError(
                    "reference_instrument_id must be uppercase"
                )

            if self.reference_instrument_id == self.instrument_id:
                raise ValueError(
                    "an instrument cannot reference itself"
                )

        optional_text = {
            "contract_identifier": self.contract_identifier,
            "unit_description": self.unit_description,
        }

        for name, value in optional_text.items():
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{name} must be a string or None"
                    )

                if not value.strip():
                    raise ValueError(
                        f"{name} cannot be blank when provided"
                    )

    def record(self) -> dict[str, Any]:
        """Return a JSON-compatible catalog record."""
        return asdict(self)


@dataclass(frozen=True)
class InstrumentCatalog:
    """Immutable validated collection of instrument definitions."""

    instruments: tuple[InstrumentDefinition, ...]

    def __post_init__(self) -> None:
        identifiers = [
            instrument.instrument_id
            for instrument in self.instruments
        ]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "instrument_id values must be unique"
            )

        available = set(identifiers)

        for instrument in self.instruments:
            reference = instrument.reference_instrument_id

            if reference is not None and reference not in available:
                raise ValueError(
                    f"reference instrument not found: {reference}"
                )

    @classmethod
    def from_iterable(
        cls,
        instruments: Iterable[InstrumentDefinition],
    ) -> InstrumentCatalog:
        """Construct an immutable catalog from an iterable."""
        return cls(tuple(instruments))

    def get(self, instrument_id: str) -> InstrumentDefinition:
        """Return one exactly identified instrument."""
        normalized = instrument_id.strip().upper()

        for instrument in self.instruments:
            if instrument.instrument_id == normalized:
                return instrument

        raise KeyError(
            f"instrument not found: {normalized}"
        )

    def search(
        self,
        query: str,
    ) -> tuple[InstrumentDefinition, ...]:
        """Search IDs, names, classes, subtypes, and structures."""
        normalized = query.strip().casefold()

        if not normalized:
            return self.instruments

        matches = []

        for instrument in self.instruments:
            searchable = (
                f"{instrument.instrument_id} "
                f"{instrument.display_name} "
                f"{instrument.asset_class} "
                f"{instrument.subtype} "
                f"{instrument.structure} "
                f"{instrument.venue} "
                f"{instrument.currency}"
            ).casefold()

            if normalized in searchable:
                matches.append(instrument)

        return tuple(matches)

    def references_of(
        self,
        reference_instrument_id: str,
    ) -> tuple[InstrumentDefinition, ...]:
        """Return instruments declaring the selected reference."""
        normalized = reference_instrument_id.strip().upper()

        return tuple(
            instrument
            for instrument in self.instruments
            if instrument.reference_instrument_id == normalized
        )


def demonstration_catalog() -> InstrumentCatalog:
    """
    Return a metadata-only demonstration catalog.

    Records describe instrument categories, not live products or claims of
    legal or economic equivalence.
    """
    return InstrumentCatalog.from_iterable(
        [
            InstrumentDefinition(
                instrument_id="MML-TRADITIONAL-REFERENCE",
                display_name="Traditional Reference Demonstration",
                asset_class="fund",
                subtype="reference exposure",
                structure="traditional",
                venue="Synthetic reference",
                currency="USD",
                timezone="UTC",
                frequency="Daily",
                provider="Market Memory Lab",
                license_note="Synthetic metadata demonstration",
                data_state="sample_only",
                unit_description="Synthetic reference unit",
            ),
            InstrumentDefinition(
                instrument_id="MML-TOKENIZED-CLAIM",
                display_name="Tokenized Claim Demonstration",
                asset_class="tokenized_asset",
                subtype="tokenized reference claim",
                structure="tokenized_claim",
                venue="Synthetic blockchain environment",
                currency="USD",
                timezone="UTC",
                frequency="Daily",
                provider="Market Memory Lab",
                license_note="Synthetic metadata demonstration",
                data_state="sample_only",
                reference_instrument_id=(
                    "MML-TRADITIONAL-REFERENCE"
                ),
                contract_identifier="SYNTHETIC-CONTRACT",
                unit_description="Synthetic token unit",
            ),
            InstrumentDefinition(
                instrument_id="MML-STABLECOIN-DEMO",
                display_name="Stablecoin Demonstration",
                asset_class="stablecoin",
                subtype="fiat-reference demonstration",
                structure="native_digital",
                venue="Synthetic digital market",
                currency="USD",
                timezone="UTC",
                frequency="Daily",
                provider="Market Memory Lab",
                license_note="Synthetic metadata demonstration",
                data_state="sample_only",
                unit_description="Synthetic stablecoin unit",
            ),
            InstrumentDefinition(
                instrument_id="MML-OPTION-DEMO",
                display_name="Option Demonstration",
                asset_class="option",
                subtype="vanilla option placeholder",
                structure="derivative",
                venue="Synthetic derivatives market",
                currency="USD",
                timezone="UTC",
                frequency="Daily",
                provider="Market Memory Lab",
                license_note="Metadata only; no options model implemented",
                data_state="connector_pending",
                reference_instrument_id=(
                    "MML-TRADITIONAL-REFERENCE"
                ),
                contract_identifier="SYNTHETIC-OPTION",
                unit_description="Synthetic option contract",
            ),
            InstrumentDefinition(
                instrument_id="MML-FUTURE-DEMO",
                display_name="Future Demonstration",
                asset_class="future",
                subtype="futures placeholder",
                structure="derivative",
                venue="Synthetic derivatives market",
                currency="USD",
                timezone="UTC",
                frequency="Daily",
                provider="Market Memory Lab",
                license_note="Metadata only; no roll model implemented",
                data_state="connector_pending",
                reference_instrument_id=(
                    "MML-TRADITIONAL-REFERENCE"
                ),
                contract_identifier="SYNTHETIC-FUTURE",
                unit_description="Synthetic futures contract",
            ),
        ]
    )
