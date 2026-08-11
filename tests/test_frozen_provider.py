"""Tests for the deterministic frozen demonstration provider."""

from datetime import date

import pytest

from market_memory_lab.providers import (
    FrozenDemonstrationProvider,
    PriceProvider,
    ProviderRequest,
)


def full_request(instrument_id: str) -> ProviderRequest:
    return ProviderRequest(
        instrument_id=instrument_id,
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 13),
    )


def test_frozen_provider_satisfies_price_provider_protocol() -> None:
    provider = FrozenDemonstrationProvider()

    assert isinstance(provider, PriceProvider)


def test_frozen_provider_lists_stable_sorted_instrument_ids() -> None:
    provider = FrozenDemonstrationProvider()

    assert provider.available_instrument_ids() == (
        "synthetic-stable-alpha",
        "synthetic-stable-twin",
    )


def test_frozen_provider_returns_deterministic_observations() -> None:
    provider = FrozenDemonstrationProvider()
    request = full_request("synthetic-stable-alpha")

    first = provider.fetch_prices(request)
    second = provider.fetch_prices(request)

    assert first == second
    assert len(first.observations) == 8
    assert first.observations[0].price == pytest.approx(100.0)
    assert first.observations[-1].price == pytest.approx(108.09498435022)


def test_frozen_provider_filters_dates_inclusively() -> None:
    provider = FrozenDemonstrationProvider()
    request = ProviderRequest(
        instrument_id="synthetic-stable-alpha",
        start_date=date(2026, 1, 6),
        end_date=date(2026, 1, 8),
    )

    dataset = provider.fetch_prices(request)

    assert tuple(
        observation.observation_date for observation in dataset.observations
    ) == (
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
    )


def test_frozen_provider_rejects_unknown_instrument() -> None:
    provider = FrozenDemonstrationProvider()

    with pytest.raises(KeyError, match="Unknown frozen instrument_id"):
        provider.fetch_prices(full_request("not-available"))


def test_frozen_provider_rejects_period_without_observations() -> None:
    provider = FrozenDemonstrationProvider()
    request = ProviderRequest(
        instrument_id="synthetic-stable-alpha",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    with pytest.raises(ValueError, match="No frozen observations"):
        provider.fetch_prices(request)


def test_frozen_provider_labels_data_as_synthetic() -> None:
    provider = FrozenDemonstrationProvider()

    dataset = provider.fetch_prices(full_request("synthetic-stable-alpha"))

    assert dataset.provenance.data_state == (
        "Synthetic controlled demonstration"
    )
    assert "not an observation from a real market" in (
        dataset.provenance.source_description
    )
    assert dataset.provenance.adjustment_status.startswith("Not applicable")


def test_stable_twins_have_identical_return_ratios() -> None:
    provider = FrozenDemonstrationProvider()

    alpha = provider.fetch_prices(
        full_request("synthetic-stable-alpha")
    )
    twin = provider.fetch_prices(
        full_request("synthetic-stable-twin")
    )

    alpha_ratios = tuple(
        current.price / previous.price
        for previous, current in zip(
            alpha.observations,
            alpha.observations[1:],
        )
    )
    twin_ratios = tuple(
        current.price / previous.price
        for previous, current in zip(
            twin.observations,
            twin.observations[1:],
        )
    )

    assert len(alpha_ratios) == len(alpha.observations) - 1
    assert len(twin_ratios) == len(twin.observations) - 1
    assert twin_ratios == pytest.approx(alpha_ratios)


def test_stable_twins_are_constant_price_scale_copies() -> None:
    provider = FrozenDemonstrationProvider()

    alpha = provider.fetch_prices(
        full_request("synthetic-stable-alpha")
    )
    twin = provider.fetch_prices(
        full_request("synthetic-stable-twin")
    )

    assert len(alpha.observations) == len(twin.observations)

    for alpha_observation, twin_observation in zip(
        alpha.observations,
        twin.observations,
        strict=True,
    ):
        assert twin_observation.observation_date == (
            alpha_observation.observation_date
        )
        assert twin_observation.price == pytest.approx(
            2.5 * alpha_observation.price,
            rel=1e-12,
            abs=1e-12,
        )

    assert alpha.observations[0].price == pytest.approx(100.0)
    assert twin.observations[0].price == pytest.approx(250.0)
