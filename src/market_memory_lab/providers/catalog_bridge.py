"""Catalog identities and provider bindings for frozen demonstrations."""

from dataclasses import dataclass

from market_memory_lab.catalog import (
    InstrumentCatalog,
    InstrumentDefinition,
)
from market_memory_lab.providers.frozen import (
    FrozenDemonstrationProvider,
)

FROZEN_PROVIDER_NAME = (
    "Market Memory Lab frozen demonstration provider"
)

STABLE_ALPHA_CATALOG_ID = "MML-SYNTH-STABLE-ALPHA"
STABLE_TWIN_CATALOG_ID = "MML-SYNTH-STABLE-TWIN"

STABLE_ALPHA_PROVIDER_ID = "synthetic-stable-alpha"
STABLE_TWIN_PROVIDER_ID = "synthetic-stable-twin"


@dataclass(frozen=True)
class ProviderBinding:
    """Map one catalog identity to one provider-specific lookup key."""

    catalog_instrument_id: str
    provider_name: str
    provider_instrument_id: str

    def __post_init__(self) -> None:
        if (
            self.catalog_instrument_id
            != self.catalog_instrument_id.strip().upper()
        ):
            raise ValueError(
                "catalog_instrument_id must be uppercase without "
                "outer whitespace"
            )

        if not self.provider_name.strip():
            raise ValueError("provider_name must not be empty")

        if (
            not self.provider_instrument_id
            or self.provider_instrument_id
            != self.provider_instrument_id.strip()
        ):
            raise ValueError(
                "provider_instrument_id must not be empty or contain "
                "outer whitespace"
            )


FROZEN_PROVIDER_BINDINGS = (
    ProviderBinding(
        catalog_instrument_id=STABLE_ALPHA_CATALOG_ID,
        provider_name=FROZEN_PROVIDER_NAME,
        provider_instrument_id=STABLE_ALPHA_PROVIDER_ID,
    ),
    ProviderBinding(
        catalog_instrument_id=STABLE_TWIN_CATALOG_ID,
        provider_name=FROZEN_PROVIDER_NAME,
        provider_instrument_id=STABLE_TWIN_PROVIDER_ID,
    ),
)


def frozen_demonstration_catalog() -> InstrumentCatalog:
    """Return canonical catalog identities for frozen demonstrations."""
    license_note = (
        "Generated within Market Memory Lab for reproducible "
        "educational and scientific demonstrations"
    )

    return InstrumentCatalog.from_iterable(
        (
            InstrumentDefinition(
                instrument_id=STABLE_ALPHA_CATALOG_ID,
                display_name="Synthetic Stable Alpha",
                asset_class="other",
                subtype="controlled demonstration",
                structure="synthetic",
                venue="not applicable",
                currency="not applicable",
                timezone="UTC",
                frequency="daily",
                provider=FROZEN_PROVIDER_NAME,
                license_note=license_note,
                data_state="sample_only",
                unit_description="Synthetic price units",
            ),
            InstrumentDefinition(
                instrument_id=STABLE_TWIN_CATALOG_ID,
                display_name="Synthetic Stable Twin",
                asset_class="other",
                subtype="controlled demonstration",
                structure="synthetic",
                venue="not applicable",
                currency="not applicable",
                timezone="UTC",
                frequency="daily",
                provider=FROZEN_PROVIDER_NAME,
                license_note=license_note,
                data_state="sample_only",
                reference_instrument_id=STABLE_ALPHA_CATALOG_ID,
                unit_description="Synthetic price units",
            ),
        )
    )


def get_frozen_provider_binding(
    catalog_instrument_id: str,
) -> ProviderBinding:
    """Resolve a canonical catalog identity to its frozen-provider key."""
    normalized = catalog_instrument_id.strip().upper()

    for binding in FROZEN_PROVIDER_BINDINGS:
        if binding.catalog_instrument_id == normalized:
            return binding

    raise KeyError(
        "No frozen-provider binding for catalog instrument ID: "
        f"{normalized}"
    )


def verify_frozen_catalog_availability(
    catalog: InstrumentCatalog | None = None,
    provider: FrozenDemonstrationProvider | None = None,
) -> None:
    """Verify complete agreement among catalog, bindings, and provider."""
    selected_catalog = catalog or frozen_demonstration_catalog()
    selected_provider = provider or FrozenDemonstrationProvider()

    catalog_ids = {
        instrument.instrument_id
        for instrument in selected_catalog.instruments
    }
    bound_catalog_ids = {
        binding.catalog_instrument_id
        for binding in FROZEN_PROVIDER_BINDINGS
    }

    if catalog_ids != bound_catalog_ids:
        missing_bindings = sorted(catalog_ids - bound_catalog_ids)
        bindings_without_catalog_entries = sorted(
            bound_catalog_ids - catalog_ids
        )

        raise ValueError(
            "Frozen catalog and provider bindings disagree. "
            f"Catalog IDs without bindings: {missing_bindings}. "
            "Bindings without catalog entries: "
            f"{bindings_without_catalog_entries}."
        )

    binding_provider_names = {
        binding.provider_name
        for binding in FROZEN_PROVIDER_BINDINGS
    }

    if binding_provider_names != {FROZEN_PROVIDER_NAME}:
        raise ValueError(
            "Frozen provider bindings use an unexpected provider name."
        )

    catalog_provider_names = {
        instrument.provider
        for instrument in selected_catalog.instruments
    }

    if catalog_provider_names != {FROZEN_PROVIDER_NAME}:
        raise ValueError(
            "Frozen catalog entries use an unexpected provider name."
        )

    bound_provider_ids = {
        binding.provider_instrument_id
        for binding in FROZEN_PROVIDER_BINDINGS
    }
    available_provider_ids = set(
        selected_provider.available_instrument_ids()
    )

    if bound_provider_ids != available_provider_ids:
        unavailable_provider_ids = sorted(
            bound_provider_ids - available_provider_ids
        )
        unbound_provider_ids = sorted(
            available_provider_ids - bound_provider_ids
        )

        raise ValueError(
            "Frozen bindings and provider availability disagree. "
            "Bound provider IDs unavailable from provider: "
            f"{unavailable_provider_ids}. "
            "Available provider IDs without bindings: "
            f"{unbound_provider_ids}."
        )
