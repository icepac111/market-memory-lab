"""Pairwise structural similarity evidence for validated asset series."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from market_memory_lab.empirical import EmpiricalEvidence
from market_memory_lab.metrics import maximum_drawdown


@dataclass(frozen=True)
class SimilarityEvidence:
    """
    Separate similarity dimensions for two validated instruments.

    No composite score is produced. Different dimensions may disagree.
    """

    asset_a: str
    asset_b: str
    overlapping_returns: int
    overlap_start: str | None
    overlap_end: str | None
    pearson_correlation: float | None
    spearman_correlation: float | None
    volatility_a: float | None
    volatility_b: float | None
    volatility_ratio: float | None
    maximum_drawdown_a: float | None
    maximum_drawdown_b: float | None
    drawdown_difference: float | None
    standardized_wasserstein_distance: float | None
    first_half_correlation: float | None
    second_half_correlation: float | None
    correlation_stability_gap: float | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    memory_similarity: str
    regime_similarity: str
    lead_lag_direction: str
    causation_state: str
    trading_conclusion: str

    @property
    def is_valid(self) -> bool:
        """Whether all blocking requirements passed."""
        return not self.errors

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible evidence record."""
        return {
            "assets": {
                "asset_a": self.asset_a,
                "asset_b": self.asset_b,
            },
            "overlap": {
                "returns": self.overlapping_returns,
                "start": self.overlap_start,
                "end": self.overlap_end,
            },
            "surface_similarity": {
                "pearson_correlation": self.pearson_correlation,
                "spearman_correlation": self.spearman_correlation,
            },
            "risk_similarity": {
                "volatility_a": self.volatility_a,
                "volatility_b": self.volatility_b,
                "volatility_ratio": self.volatility_ratio,
                "maximum_drawdown_a": self.maximum_drawdown_a,
                "maximum_drawdown_b": self.maximum_drawdown_b,
                "drawdown_difference": self.drawdown_difference,
            },
            "distribution_similarity": {
                "standardized_wasserstein_distance": (
                    self.standardized_wasserstein_distance
                ),
            },
            "relationship_stability": {
                "first_half_correlation": self.first_half_correlation,
                "second_half_correlation": self.second_half_correlation,
                "correlation_stability_gap": (
                    self.correlation_stability_gap
                ),
            },
            "scientific_state": {
                "memory_similarity": self.memory_similarity,
                "regime_similarity": self.regime_similarity,
                "lead_lag_direction": self.lead_lag_direction,
                "causation_state": self.causation_state,
                "trading_conclusion": self.trading_conclusion,
            },
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def _finite_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    method: str,
) -> float | None:
    """
    Calculate Pearson or Spearman correlation when mathematically defined.

    Correlation is undefined when either input has zero sample variance.
    """
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")

    if len(left) < 3:
        return None

    left_values = left.to_numpy(dtype="float64")
    right_values = right.to_numpy(dtype="float64")

    if not np.isfinite(left_values).all():
        raise ValueError("left correlation input must be finite")

    if not np.isfinite(right_values).all():
        raise ValueError("right correlation input must be finite")

    if np.isclose(np.std(left_values, ddof=1), 0.0):
        return None

    if np.isclose(np.std(right_values, ddof=1), 0.0):
        return None

    if method == "pearson":
        value = stats.pearsonr(
            left_values,
            right_values,
        ).statistic
    elif method == "spearman":
        value = stats.spearmanr(
            left_values,
            right_values,
        ).statistic
    else:
        raise ValueError("unsupported correlation method")

    if not math.isfinite(float(value)):
        return None

    return float(value)


def _sample_volatility(series: pd.Series) -> float | None:
    """Return sample standard deviation without annualization."""
    if len(series) < 2:
        return None

    value = float(series.std(ddof=1))

    if not math.isfinite(value):
        return None

    return value


def _symmetric_volatility_ratio(
    volatility_a: float | None,
    volatility_b: float | None,
) -> float | None:
    """
    Return min(vol_a, vol_b) / max(vol_a, vol_b).

    The result lies in [0, 1]. A value of 1 means equal sample volatility.
    This is descriptive and is not a probability.
    """
    if volatility_a is None or volatility_b is None:
        return None

    if volatility_a < 0.0 or volatility_b < 0.0:
        raise ValueError("volatility cannot be negative")

    maximum = max(volatility_a, volatility_b)
    minimum = min(volatility_a, volatility_b)

    if math.isclose(maximum, 0.0, abs_tol=1e-15):
        return None

    return float(minimum / maximum)


def _standardize(series: pd.Series) -> np.ndarray | None:
    """Standardize observations using sample mean and sample deviation."""
    if len(series) < 2:
        return None

    values = series.to_numpy(dtype="float64")
    mean = float(np.mean(values))
    deviation = float(np.std(values, ddof=1))

    if not math.isfinite(mean) or not math.isfinite(deviation):
        return None

    if math.isclose(deviation, 0.0, abs_tol=1e-15):
        return None

    standardized = (values - mean) / deviation

    if not np.isfinite(standardized).all():
        return None

    return standardized


def _distribution_distance(
    left: pd.Series,
    right: pd.Series,
) -> float | None:
    """
    Calculate Wasserstein distance after separate standardization.

    Separate standardization removes differences in mean and scale so the
    distance focuses on remaining distributional shape differences.

    The result is not bounded above and is not a probability.
    """
    standardized_left = _standardize(left)
    standardized_right = _standardize(right)

    if standardized_left is None or standardized_right is None:
        return None

    value = stats.wasserstein_distance(
        standardized_left,
        standardized_right,
    )

    if not math.isfinite(float(value)):
        return None

    return float(value)


def _split_correlations(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float | None, float | None, float | None]:
    """
    Compare Pearson correlation across nonoverlapping sample halves.

    At least six aligned observations are required so each half contains
    at least three observations.
    """
    if len(left) != len(right):
        raise ValueError("split-correlation inputs must have equal length")

    if len(left) < 6:
        return None, None, None

    split_index = len(left) // 2

    first_left = left.iloc[:split_index]
    first_right = right.iloc[:split_index]
    second_left = left.iloc[split_index:]
    second_right = right.iloc[split_index:]

    first = _finite_correlation(
        first_left,
        first_right,
        method="pearson",
    )
    second = _finite_correlation(
        second_left,
        second_right,
        method="pearson",
    )

    if first is None or second is None:
        return first, second, None

    return first, second, float(abs(first - second))


def compare_validated_assets(
    asset_a: EmpiricalEvidence,
    asset_b: EmpiricalEvidence,
    *,
    minimum_overlap: int = 6,
) -> SimilarityEvidence:
    """
    Compare two independently validated empirical asset series.

    Each asset's returns are calculated before alignment. The comparison
    then uses the intersection of return timestamps.

    This avoids constructing returns across mismatched observations after
    aligning raw price rows.
    """
    if not isinstance(asset_a, EmpiricalEvidence):
        raise TypeError("asset_a must be EmpiricalEvidence")

    if not isinstance(asset_b, EmpiricalEvidence):
        raise TypeError("asset_b must be EmpiricalEvidence")

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

    if not asset_a.is_valid:
        errors.append("Asset A failed empirical validation")

    if not asset_b.is_valid:
        errors.append("Asset B failed empirical validation")

    if (
        asset_a.metadata.currency.strip().upper()
        != asset_b.metadata.currency.strip().upper()
    ):
        warnings.append(
            "Declared currencies differ; return comparison does not "
            "remove foreign-exchange effects"
        )

    if (
        asset_a.metadata.timezone.strip()
        != asset_b.metadata.timezone.strip()
    ):
        warnings.append(
            "Declared timezones differ; daily observations may represent "
            "different market-close times"
        )

    if (
        asset_a.metadata.frequency.strip().lower()
        != asset_b.metadata.frequency.strip().lower()
    ):
        warnings.append(
            "Declared observation frequencies differ"
        )

    if (
        asset_a.metadata.adjustment_status.strip().lower()
        != asset_b.metadata.adjustment_status.strip().lower()
    ):
        warnings.append(
            "Price-adjustment declarations differ"
        )

    empty_result = SimilarityEvidence(
        asset_a=asset_a.metadata.asset_identifier,
        asset_b=asset_b.metadata.asset_identifier,
        overlapping_returns=0,
        overlap_start=None,
        overlap_end=None,
        pearson_correlation=None,
        spearman_correlation=None,
        volatility_a=None,
        volatility_b=None,
        volatility_ratio=None,
        maximum_drawdown_a=None,
        maximum_drawdown_b=None,
        drawdown_difference=None,
        standardized_wasserstein_distance=None,
        first_half_correlation=None,
        second_half_correlation=None,
        correlation_stability_gap=None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        memory_similarity="not_tested",
        regime_similarity="not_tested",
        lead_lag_direction="not_tested",
        causation_state="not_established",
        trading_conclusion="abstain",
    )

    if errors:
        return empty_result

    aligned = pd.concat(
        [
            asset_a.simple_return_series.rename("asset_a"),
            asset_b.simple_return_series.rename("asset_b"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    overlap_count = len(aligned)

    if overlap_count < minimum_overlap:
        errors.append(
            f"Only {overlap_count} overlapping return observations; "
            f"at least {minimum_overlap} are required"
        )

        return SimilarityEvidence(
            **{
                **asdict(empty_result),
                "overlapping_returns": overlap_count,
                "errors": tuple(errors),
            }
        )

    left = aligned["asset_a"]
    right = aligned["asset_b"]

    pearson = _finite_correlation(
        left,
        right,
        method="pearson",
    )
    spearman = _finite_correlation(
        left,
        right,
        method="spearman",
    )

    if pearson is None:
        warnings.append(
            "Pearson correlation is undefined because at least one "
            "aligned return series has insufficient variation"
        )

    if spearman is None:
        warnings.append(
            "Spearman correlation is undefined because at least one "
            "aligned return series has insufficient variation"
        )

    volatility_a = _sample_volatility(left)
    volatility_b = _sample_volatility(right)
    volatility_ratio = _symmetric_volatility_ratio(
        volatility_a,
        volatility_b,
    )

    drawdown_a = maximum_drawdown(left)
    drawdown_b = maximum_drawdown(right)

    distribution_distance = _distribution_distance(
        left,
        right,
    )

    first_half, second_half, stability_gap = _split_correlations(
        left,
        right,
    )

    if stability_gap is None:
        warnings.append(
            "Subperiod relationship stability is unavailable"
        )

    return SimilarityEvidence(
        asset_a=asset_a.metadata.asset_identifier,
        asset_b=asset_b.metadata.asset_identifier,
        overlapping_returns=overlap_count,
        overlap_start=aligned.index.min().isoformat(),
        overlap_end=aligned.index.max().isoformat(),
        pearson_correlation=pearson,
        spearman_correlation=spearman,
        volatility_a=volatility_a,
        volatility_b=volatility_b,
        volatility_ratio=volatility_ratio,
        maximum_drawdown_a=drawdown_a,
        maximum_drawdown_b=drawdown_b,
        drawdown_difference=float(abs(drawdown_a - drawdown_b)),
        standardized_wasserstein_distance=distribution_distance,
        first_half_correlation=first_half,
        second_half_correlation=second_half,
        correlation_stability_gap=stability_gap,
        warnings=tuple(warnings),
        errors=tuple(errors),
        memory_similarity="not_tested",
        regime_similarity="not_tested",
        lead_lag_direction="not_tested",
        causation_state="not_established",
        trading_conclusion="abstain",
    )
