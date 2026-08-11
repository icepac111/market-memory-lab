from datetime import date

import pandas as pd
import pytest

from market_memory_lab.providers.base import (
    PriceProvider,
    ProviderRequest,
)
from market_memory_lab.providers.yahoo_historical import (
    YahooHistoricalProvider,
)


def request(instrument_id: str = " spy ") -> ProviderRequest:
    return ProviderRequest(
        instrument_id=instrument_id,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
    )


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "Adj Close": [99.5, 100.5, 101.5],
        },
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            utc=True,
        ),
    )


def test_provider_satisfies_price_provider_protocol() -> None:
    provider = YahooHistoricalProvider(lambda *_: valid_frame())

    assert isinstance(provider, PriceProvider)


def test_fetch_normalizes_ticker_and_uses_inclusive_end_date() -> None:
    calls: list[tuple[str, str, str]] = []

    def loader(
        instrument_id: str,
        start_date: str,
        exclusive_end_date: str,
    ) -> pd.DataFrame:
        calls.append(
            (instrument_id, start_date, exclusive_end_date)
        )
        return valid_frame()

    dataset = YahooHistoricalProvider(loader).fetch_prices(request())

    assert dataset.instrument_id == "SPY"
    assert calls == [("SPY", "2024-01-02", "2024-01-05")]


def test_fetch_uses_explicit_adjusted_close_values() -> None:
    dataset = YahooHistoricalProvider(
        lambda *_: valid_frame()
    ).fetch_prices(request())

    assert [item.price for item in dataset.observations] == [
        99.5,
        100.5,
        101.5,
    ]
    assert dataset.provenance.adjustment_status == (
        "Explicit Adj Close field; auto_adjust disabled"
    )


def test_fetch_preserves_historical_provenance() -> None:
    dataset = YahooHistoricalProvider(
        lambda *_: valid_frame()
    ).fetch_prices(request())

    assert dataset.provenance.provider_name == (
        "Yahoo Finance via yfinance"
    )
    assert dataset.provenance.data_state == (
        "historical provider response"
    )
    assert "Private research use only" in (
        dataset.provenance.license_note
    )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "returned no data"),
        (
            pd.DataFrame(
                {"Close": [100.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
            "lacks explicit Adj Close",
        ),
        (
            pd.DataFrame(
                {"Adj Close": [float("nan")]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
            "missing adjusted prices",
        ),
    ],
)
def test_fetch_rejects_inadequate_provider_data(
    frame: pd.DataFrame,
    message: str,
) -> None:
    provider = YahooHistoricalProvider(lambda *_: frame)

    with pytest.raises(ValueError, match=message):
        provider.fetch_prices(request())


def test_fetch_rejects_duplicate_dates() -> None:
    frame = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-02"]
        ),
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        YahooHistoricalProvider(
            lambda *_: frame
        ).fetch_prices(request())


def test_fetch_separates_provider_failure_from_market_data() -> None:
    def failing_loader(*_: str) -> pd.DataFrame:
        raise ConnectionError("synthetic upstream failure")

    provider = YahooHistoricalProvider(failing_loader)

    with pytest.raises(
        RuntimeError,
        match="provider request failed for SPY",
    ):
        provider.fetch_prices(request("SPY"))


def test_provider_is_available_from_public_provider_package() -> None:
    from market_memory_lab.providers import (
        YahooHistoricalProvider as PublicYahooHistoricalProvider,
    )

    assert PublicYahooHistoricalProvider is YahooHistoricalProvider
