"""Private research adapter for Yahoo historical adjusted-close data."""

import math
from collections.abc import Callable
from datetime import timedelta

import pandas as pd
import yfinance as yf

from market_memory_lab.providers.base import (
    PriceObservation,
    ProviderDataset,
    ProviderProvenance,
    ProviderRequest,
)

HistoryLoader = Callable[[str, str, str], pd.DataFrame]


def _default_history_loader(
    instrument_id: str,
    start_date: str,
    exclusive_end_date: str,
) -> pd.DataFrame:
    """Retrieve one historical series without implicit price adjustment."""
    return yf.Ticker(instrument_id).history(
        start=start_date,
        end=exclusive_end_date,
        auto_adjust=False,
        actions=False,
    )


class YahooHistoricalProvider:
    """Retrieve historical adjusted-close observations for private research."""

    provider_name = "Yahoo Finance via yfinance"

    def __init__(
        self,
        history_loader: HistoryLoader | None = None,
    ) -> None:
        self._history_loader = history_loader or _default_history_loader

    def fetch_prices(
        self,
        request: ProviderRequest,
    ) -> ProviderDataset:
        """Return an explicit adjusted-close series and provenance."""
        instrument_id = request.instrument_id.strip().upper()
        exclusive_end = request.end_date + timedelta(days=1)

        try:
            frame = self._history_loader(
                instrument_id,
                request.start_date.isoformat(),
                exclusive_end.isoformat(),
            )
        except Exception as error:
            raise RuntimeError(
                f"Historical provider request failed for {instrument_id}"
            ) from error

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Historical provider must return a DataFrame")

        if frame.empty:
            raise ValueError(
                f"Historical provider returned no data for {instrument_id}"
            )

        if "Adj Close" not in frame.columns:
            raise ValueError(
                "Historical provider response lacks explicit Adj Close data"
            )

        if frame.index.duplicated().any():
            raise ValueError(
                "Historical provider response contains duplicate dates"
            )

        adjusted_close = frame["Adj Close"]

        if adjusted_close.isna().any():
            raise ValueError(
                "Historical provider response contains missing adjusted prices"
            )

        observations: list[PriceObservation] = []

        for timestamp, raw_price in adjusted_close.sort_index().items():
            observation_date = pd.Timestamp(timestamp).date()
            price = float(raw_price)

            if observation_date < request.start_date:
                continue

            if observation_date > request.end_date:
                continue

            if not math.isfinite(price) or price <= 0.0:
                raise ValueError(
                    "Historical provider response contains a nonpositive "
                    "or nonfinite adjusted price"
                )

            observations.append(
                PriceObservation(
                    observation_date=observation_date,
                    price=price,
                )
            )

        if not observations:
            raise ValueError(
                f"No valid observations remain for {instrument_id}"
            )

        return ProviderDataset(
            instrument_id=instrument_id,
            observations=tuple(observations),
            provenance=ProviderProvenance(
                provider_name=self.provider_name,
                source_description=(
                    "Historical adjusted-close observations requested "
                    "through the yfinance client"
                ),
                retrieved_or_frozen_on=pd.Timestamp.now(
                    tz="UTC"
                ).date(),
                adjustment_status=(
                    "Explicit Adj Close field; auto_adjust disabled"
                ),
                license_note=(
                    "Private research use only; usage rights require "
                    "independent review of applicable provider terms"
                ),
                data_state="historical provider response",
            ),
        )
