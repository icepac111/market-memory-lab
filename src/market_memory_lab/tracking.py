"""Tracking-integrity evidence for an instrument and reference exposure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_memory_lab.empirical import EmpiricalEvidence


@dataclass(frozen=True)
class TrackingEvidence:
    """Auditable return-based tracking evidence."""

    instrument: str
    reference: str
    overlapping_returns: int
    overlap_start: str | None
    overlap_end: str | None
    mean_active_return: float | None
    tracking_error: float | None
    correlation: float | None
    beta: float | None
    first_half_tracking_error: float | None
    second_half_tracking_error: float | None
    tracking_error_stability_gap: float | None
    final_cumulative_divergence: float | None
    maximum_absolute_cumulative_divergence: float | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    economic_equivalence: str
    price_dislocation: str
    market_memory: str
    lead_lag_direction: str
    causation: str
    trading_conclusion: str

    @property
    def is_valid(self) -> bool:
        """Whether all blocking requirements passed."""
        return not self.errors

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible evidence record."""
        return {
            "instrument": self.instrument,
            "reference": self.reference,
            "overlap": {
                "returns": self.overlapping_returns,
                "start": self.overlap_start,
                "end": self.overlap_end,
            },
            "tracking": {
                "mean_active_return": self.mean_active_return,
                "tracking_error": self.tracking_error,
                "correlation": self.correlation,
                "beta": self.beta,
                "first_half_tracking_error": (
                    self.first_half_tracking_error
                ),
                "second_half_tracking_error": (
                    self.second_half_tracking_error
                ),
                "tracking_error_stability_gap": (
                    self.tracking_error_stability_gap
                ),
                "final_cumulative_divergence": (
                    self.final_cumulative_divergence
                ),
                "maximum_absolute_cumulative_divergence": (
                    self.maximum_absolute_cumulative_divergence
                ),
            },
            "scientific_state": {
                "economic_equivalence": self.economic_equivalence,
                "price_dislocation": self.price_dislocation,
                "market_memory": self.market_memory,
                "lead_lag_direction": self.lead_lag_direction,
                "causation": self.causation,
                "trading_conclusion": self.trading_conclusion,
            },
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def _sample_standard_deviation(
    values: pd.Series,
) -> float | None:
    """Return finite sample standard deviation when defined."""
    if len(values) < 2:
        return None

    result = float(values.std(ddof=1))

    if not math.isfinite(result):
        return None

    return result


def _correlation(
    left: pd.Series,
    right: pd.Series,
) -> float | None:
    """Return finite Pearson correlation when defined."""
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")

    if len(left) < 3:
        return None

    left_deviation = _sample_standard_deviation(left)
    right_deviation = _sample_standard_deviation(right)

    if left_deviation is None or right_deviation is None:
        return None

    if math.isclose(left_deviation, 0.0, abs_tol=1e-15):
        return None

    if math.isclose(right_deviation, 0.0, abs_tol=1e-15):
        return None

    result = float(left.corr(right))

    if not math.isfinite(result):
        return None

    return result


def _beta(
    instrument: pd.Series,
    reference: pd.Series,
) -> float | None:
    """
    Estimate return beta using sample covariance and reference variance.

    beta = Cov(instrument, reference) / Var(reference)
    """
    if len(instrument) != len(reference):
        raise ValueError("beta inputs must have equal length")

    if len(instrument) < 2:
        return None

    reference_variance = float(reference.var(ddof=1))

    if (
        not math.isfinite(reference_variance)
        or math.isclose(reference_variance, 0.0, abs_tol=1e-15)
    ):
        return None

    covariance = float(
        np.cov(
            instrument.to_numpy(dtype="float64"),
            reference.to_numpy(dtype="float64"),
            ddof=1,
        )[0, 1]
    )

    if not math.isfinite(covariance):
        return None

    return float(covariance / reference_variance)


def _tracking_error_halves(
    active_returns: pd.Series,
) -> tuple[float | None, float | None, float | None]:
    """Compare tracking error across nonoverlapping sample halves."""
    if len(active_returns) < 4:
        return None, None, None

    split = len(active_returns) // 2

    first = _sample_standard_deviation(
        active_returns.iloc[:split]
    )
    second = _sample_standard_deviation(
        active_returns.iloc[split:]
    )

    if first is None or second is None:
        return first, second, None

    return first, second, float(abs(first - second))


def analyze_tracking_integrity(
    instrument: EmpiricalEvidence,
    reference: EmpiricalEvidence,
    *,
    minimum_overlap: int = 6,
) -> TrackingEvidence:
    """
    Compare a validated instrument with a validated reference exposure.

    Returns are calculated independently before common-date alignment.

    Tracking difference:
        mean(instrument return - reference return)

    Tracking error:
        sample standard deviation of active returns

    Cumulative divergence:
        cumulative product of instrument returns minus cumulative product
        of reference returns, with both starting from one.
    """
    if not isinstance(instrument, EmpiricalEvidence):
        raise TypeError("instrument must be EmpiricalEvidence")

    if not isinstance(reference, EmpiricalEvidence):
        raise TypeError("reference must be EmpiricalEvidence")

    if (
        not isinstance(minimum_overlap, int)
        or isinstance(minimum_overlap, bool)
    ):
        raise TypeError("minimum_overlap must be an integer")

    if minimum_overlap < 3:
        raise ValueError(
            "minimum_overlap must be greater than or equal to 3"
        )

    warnings: list[str] = []
    errors: list[str] = []

    if not instrument.is_valid:
        errors.append("Instrument failed empirical validation")

    if not reference.is_valid:
        errors.append("Reference failed empirical validation")

    if (
        instrument.metadata.currency.strip().upper()
        != reference.metadata.currency.strip().upper()
    ):
        warnings.append(
            "Declared currencies differ; tracking evidence includes "
            "uncontrolled foreign-exchange effects"
        )

    if (
        instrument.metadata.timezone.strip()
        != reference.metadata.timezone.strip()
    ):
        warnings.append(
            "Declared timezones differ; observations may represent "
            "different market-close times"
        )

    if (
        instrument.metadata.frequency.strip().lower()
        != reference.metadata.frequency.strip().lower()
    ):
        warnings.append(
            "Declared observation frequencies differ"
        )

    empty = TrackingEvidence(
        instrument=instrument.metadata.asset_identifier,
        reference=reference.metadata.asset_identifier,
        overlapping_returns=0,
        overlap_start=None,
        overlap_end=None,
        mean_active_return=None,
        tracking_error=None,
        correlation=None,
        beta=None,
        first_half_tracking_error=None,
        second_half_tracking_error=None,
        tracking_error_stability_gap=None,
        final_cumulative_divergence=None,
        maximum_absolute_cumulative_divergence=None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        economic_equivalence="not_established",
        price_dislocation="not_tested",
        market_memory="not_tested",
        lead_lag_direction="not_tested",
        causation="not_established",
        trading_conclusion="abstain",
    )

    if errors:
        return empty

    aligned = pd.concat(
        [
            instrument.simple_return_series.rename("instrument"),
            reference.simple_return_series.rename("reference"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    overlap = len(aligned)

    if overlap < minimum_overlap:
        errors.append(
            f"Only {overlap} overlapping return observations; "
            f"at least {minimum_overlap} are required"
        )

        return TrackingEvidence(
            **{
                **empty.__dict__,
                "overlapping_returns": overlap,
                "errors": tuple(errors),
            }
        )

    instrument_returns = aligned["instrument"]
    reference_returns = aligned["reference"]
    active_returns = instrument_returns - reference_returns

    mean_active_return = float(active_returns.mean())
    tracking_error = _sample_standard_deviation(active_returns)
    correlation = _correlation(
        instrument_returns,
        reference_returns,
    )
    beta = _beta(
        instrument_returns,
        reference_returns,
    )

    first_error, second_error, error_gap = (
        _tracking_error_halves(active_returns)
    )

    instrument_wealth = (1.0 + instrument_returns).cumprod()
    reference_wealth = (1.0 + reference_returns).cumprod()
    cumulative_divergence = instrument_wealth - reference_wealth

    final_divergence = float(cumulative_divergence.iloc[-1])
    maximum_divergence = float(
        cumulative_divergence.abs().max()
    )

    numeric_values = (
        mean_active_return,
        final_divergence,
        maximum_divergence,
    )

    if not all(math.isfinite(value) for value in numeric_values):
        raise FloatingPointError(
            "tracking calculation produced nonfinite values"
        )

    return TrackingEvidence(
        instrument=instrument.metadata.asset_identifier,
        reference=reference.metadata.asset_identifier,
        overlapping_returns=overlap,
        overlap_start=aligned.index.min().isoformat(),
        overlap_end=aligned.index.max().isoformat(),
        mean_active_return=mean_active_return,
        tracking_error=tracking_error,
        correlation=correlation,
        beta=beta,
        first_half_tracking_error=first_error,
        second_half_tracking_error=second_error,
        tracking_error_stability_gap=error_gap,
        final_cumulative_divergence=final_divergence,
        maximum_absolute_cumulative_divergence=maximum_divergence,
        warnings=tuple(warnings),
        errors=tuple(errors),
        economic_equivalence="not_established",
        price_dislocation="not_tested",
        market_memory="not_tested",
        lead_lag_direction="not_tested",
        causation="not_established",
        trading_conclusion="abstain",
    )
