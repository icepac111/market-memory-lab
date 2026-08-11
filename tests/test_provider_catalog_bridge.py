"""Tests for frozen-provider catalog identity bindings."""

import pytest

from market_memory_lab.catalog import (
    InstrumentCatalog,
    InstrumentDefinition,
)
from market_memory_lab.providers.catalog_bridge import (
    FROZEN_PROVIDER_BINDINGS,
    FROZEN_PROVIDER_NAME,
    STABLE_ALPHA_CATALOG_ID,
    STABLE_ALPHA_PROVIDER_ID,
    STABLE_TWIN_CATALOG_ID,
    STABLE_TWIN_PROVIDER_ID,
    ProviderBinding,
    frozen_demonstration_catalog,
    get_frozen_provider_binding,
    verify_frozen_catalog_availability,
)


def test_frozen_demonstration_catalog_returns_catalog() -> None:
    catalog = frozen_demonstration_catalog()

    assert isinstance(catalog, InstrumentCatalog)


def test_frozen_catalog_uses_canonical_uppercase_ids() -> None:
    catalog = frozen_demonstration_catalog()

    assert {
        instrument.instrument_id
        for instrument in catalog.instruments
    } == {
        STABLE_ALPHA_CATALOG_ID,
        STABLE_TWIN_CATALOG_ID,
    }


def test_frozen_catalog_uses_human_readable_names() -> None:
    catalog = frozen_demonstration_catalog()

    names = {
        instrument.instrument_id: instrument.display_name
        for instrument in catalog.instruments
    }

    assert names == {
        STABLE_ALPHA_CATALOG_ID: "Synthetic Stable Alpha",
        STABLE_TWIN_CATALOG_ID: "Synthetic Stable Twin",
    }


def test_frozen_catalog_records_provider_identity() -> None:
    catalog = frozen_demonstration_catalog()

    assert all(
        instrument.provider == FROZEN_PROVIDER_NAME
        for instrument in catalog.instruments
    )


def test_frozen_catalog_labels_records_as_synthetic_samples() -> None:
    catalog = frozen_demonstration_catalog()

    assert all(
        instrument.structure == "synthetic"
        and instrument.data_state == "sample_only"
        for instrument in catalog.instruments
    )


def test_stable_twin_references_catalog_identity() -> None:
    catalog = frozen_demonstration_catalog()

    twin = catalog.get(STABLE_TWIN_CATALOG_ID)

    assert twin.reference_instrument_id == STABLE_ALPHA_CATALOG_ID


def test_bindings_separate_catalog_and_provider_namespaces() -> None:
    actual = {
        binding.catalog_instrument_id:
        binding.provider_instrument_id
        for binding in FROZEN_PROVIDER_BINDINGS
    }

    assert actual == {
        STABLE_ALPHA_CATALOG_ID: STABLE_ALPHA_PROVIDER_ID,
        STABLE_TWIN_CATALOG_ID: STABLE_TWIN_PROVIDER_ID,
    }


def test_binding_resolution_normalizes_catalog_id() -> None:
    binding = get_frozen_provider_binding(
        "  mml-synth-stable-twin  "
    )

    assert binding.catalog_instrument_id == STABLE_TWIN_CATALOG_ID
    assert binding.provider_instrument_id == STABLE_TWIN_PROVIDER_ID


def test_unknown_catalog_binding_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="No frozen-provider binding",
    ):
        get_frozen_provider_binding("MML-UNKNOWN")


def test_provider_binding_rejects_lowercase_catalog_id() -> None:
    with pytest.raises(
        ValueError,
        match="catalog_instrument_id must be uppercase",
    ):
        ProviderBinding(
            catalog_instrument_id="lowercase-id",
            provider_name=FROZEN_PROVIDER_NAME,
            provider_instrument_id="provider-key",
        )


def test_catalog_and_provider_availability_match() -> None:
    verify_frozen_catalog_availability()


def test_catalog_binding_mismatch_fails_clearly() -> None:
    mismatched_catalog = InstrumentCatalog.from_iterable(
        (
            InstrumentDefinition(
                instrument_id="MML-UNBOUND-SYNTHETIC",
                display_name="Unbound Synthetic Demonstration",
                asset_class="other",
                subtype="controlled demonstration",
                structure="synthetic",
                venue="not applicable",
                currency="not applicable",
                timezone="UTC",
                frequency="daily",
                provider=FROZEN_PROVIDER_NAME,
                license_note="Synthetic test fixture",
                data_state="sample_only",
                unit_description="Synthetic price units",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="Catalog IDs without bindings",
    ):
        verify_frozen_catalog_availability(
            catalog=mismatched_catalog
        )
