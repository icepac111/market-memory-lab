"""Controlled demonstration datasets and evidence-adequacy policies."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_memory_lab.empirical import (
    AssetMetadata,
    EmpiricalEvidence,
    analyze_empirical_prices,
)
from market_memory_lab.similarity import (
    SimilarityEvidence,
    compare_validated_assets,
)


@dataclass(frozen=True)
class Demonstration:
    """One controlled pairwise demonstration and its expected mechanism."""

    name: str
    question: str
    plain_english: str
    mechanism: str
    expected_result: str
    instrument_a: EmpiricalEvidence
    instrument_b: EmpiricalEvidence
    similarity: SimilarityEvidence

    def manifest_json(self) -> str:
        """Return a reproducibility manifest for the demonstration."""
        payload = {
            "demonstration": {
                "name": self.name,
                "question": self.question,
                "plain_english": self.plain_english,
                "mechanism": self.mechanism,
                "expected_result": self.expected_result,
            },
            "instrument_a": self.instrument_a.manifest(),
            "instrument_b": self.instrument_b.manifest(),
            "pairwise_similarity": self.similarity.manifest(),
            "conclusion_contract": {
                "real_market_evidence": False,
                "memory_similarity": "not_tested",
                "regime_similarity": "not_tested",
                "lead_lag_direction": "not_tested",
                "causation": "not_established",
                "trading_conclusion": "abstain",
            },
        }

        return json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )


@dataclass(frozen=True)
class EvidenceAdequacy:
    """A transparent interface-policy label for a return sample."""

    return_observations: int
    label: str
    decision_use: str
    explanation: str


def classify_evidence_adequacy(
    return_observations: int,
) -> EvidenceAdequacy:
    """
    Classify descriptive sample adequacy using project policy thresholds.

    These thresholds are interface safeguards, not universal statistical
    laws and not guarantees of inference quality.
    """
    if (
        not isinstance(return_observations, int)
        or isinstance(return_observations, bool)
    ):
        raise TypeError("return_observations must be an integer")

    if return_observations < 0:
        raise ValueError("return_observations must be nonnegative")

    if return_observations < 30:
        return EvidenceAdequacy(
            return_observations=return_observations,
            label="Critical",
            decision_use="Blocked",
            explanation=(
                "The sample is too small for decision-grade interpretation. "
                "Displayed statistics are demonstration-only."
            ),
        )

    if return_observations < 100:
        return EvidenceAdequacy(
            return_observations=return_observations,
            label="Limited",
            decision_use="Research only",
            explanation=(
                "The descriptive sample remains limited. Results require "
                "substantial sensitivity analysis and independent evidence."
            ),
        )

    if return_observations < 252:
        return EvidenceAdequacy(
            return_observations=return_observations,
            label="Moderate",
            decision_use="Research only",
            explanation=(
                "The sample supports descriptive analysis but does not "
                "establish stability, causation, or economic usefulness."
            ),
        )

    return EvidenceAdequacy(
        return_observations=return_observations,
        label="Stronger descriptive base",
        decision_use="Research only",
        explanation=(
            "The sample is larger, but size alone does not protect against "
            "bias, structural breaks, multiple testing, or model failure."
        ),
    )


def _metadata(identifier: str, name: str) -> AssetMetadata:
    """Create declared metadata for controlled synthetic examples."""
    return AssetMetadata(
        dataset_name=name,
        source="Market Memory Lab controlled synthetic demonstration",
        asset_identifier=identifier,
        asset_class="Synthetic demonstration",
        venue="Synthetic reference",
        currency="USD",
        timezone="UTC",
        frequency="Daily",
        adjustment_status="Adjusted close",
        license_note="Synthetic demonstration data",
    )


def _prices_from_returns(
    returns: np.ndarray,
    *,
    initial_price: float,
) -> np.ndarray:
    """Convert a simple-return sequence into a positive price path."""
    if initial_price <= 0.0:
        raise ValueError("initial_price must be strictly positive")

    values = np.asarray(returns, dtype="float64")

    if values.ndim != 1 or values.size < 1:
        raise ValueError("returns must be a nonempty one-dimensional array")

    if not np.isfinite(values).all():
        raise ValueError("returns must contain finite values")

    if np.any(values <= -1.0):
        raise ValueError("simple returns must exceed -1")

    return np.concatenate(
        [
            np.array([initial_price], dtype="float64"),
            initial_price * np.cumprod(1.0 + values),
        ]
    )


def _evidence(
    *,
    identifier: str,
    name: str,
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    initial_price: float,
) -> EmpiricalEvidence:
    """Create validated empirical evidence from a controlled process."""
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Adjusted Close": _prices_from_returns(
                returns,
                initial_price=initial_price,
            ),
        }
    )

    evidence = analyze_empirical_prices(
        frame,
        date_column="Date",
        price_column="Adjusted Close",
        metadata=_metadata(identifier, name),
        periods_per_year=252,
    )

    if not evidence.is_valid:
        raise RuntimeError(
            "controlled demonstration failed empirical validation"
        )

    return evidence


def stable_twins_demonstration() -> Demonstration:
    """Create two instruments with identical returns and different prices."""
    returns = np.array(
        [
            0.010,
            -0.015,
            0.020,
            0.005,
            -0.010,
            0.018,
            0.012,
            -0.008,
            0.016,
            -0.012,
            0.009,
            0.014,
        ],
        dtype="float64",
    )

    dates = pd.date_range(
        "2026-01-02",
        periods=len(returns) + 1,
        freq="B",
    )

    first = _evidence(
        identifier="STABLE-A",
        name="Stable Twin A",
        dates=dates,
        returns=returns,
        initial_price=100.0,
    )

    second = _evidence(
        identifier="STABLE-B",
        name="Stable Twin B",
        dates=dates,
        returns=returns,
        initial_price=250.0,
    )

    similarity = compare_validated_assets(first, second)

    if not similarity.is_valid:
        raise RuntimeError("stable-twins comparison failed")

    return Demonstration(
        name="Stable Twins",
        question=(
            "Can instruments beginning at different price levels exhibit "
            "the same return, risk, distribution, and stability structure?"
        ),
        plain_english=(
            "The instruments have different sticker prices but move by "
            "the same percentages throughout the sample."
        ),
        mechanism="Identical return sequence with different initial prices",
        expected_result=(
            "Perfect surface similarity, matching risk, near-zero "
            "distribution distance, and no relationship reversal."
        ),
        instrument_a=first,
        instrument_b=second,
        similarity=similarity,
    )


def false_friends_demonstration() -> Demonstration:
    """Create two instruments whose relationship reverses halfway."""
    first_returns = np.array(
        [
            0.010,
            -0.020,
            0.030,
            0.020,
            -0.010,
            0.015,
            0.010,
            -0.020,
            0.030,
            0.020,
            -0.010,
            0.015,
        ],
        dtype="float64",
    )

    second_returns = np.concatenate(
        [
            first_returns[:6],
            -first_returns[6:],
        ]
    )

    dates = pd.date_range(
        "2026-01-02",
        periods=len(first_returns) + 1,
        freq="B",
    )

    first = _evidence(
        identifier="FALSE-A",
        name="False Friend A",
        dates=dates,
        returns=first_returns,
        initial_price=100.0,
    )

    second = _evidence(
        identifier="FALSE-B",
        name="False Friend B",
        dates=dates,
        returns=second_returns,
        initial_price=180.0,
    )

    similarity = compare_validated_assets(first, second)

    if not similarity.is_valid:
        raise RuntimeError("false-friends comparison failed")

    return Demonstration(
        name="False Friends",
        question=(
            "Can a full-period correlation hide a complete reversal in "
            "the relationship between two instruments?"
        ),
        plain_english=(
            "The instruments begin as perfect allies and then become "
            "perfect opposites. Aggregating both periods hides the switch."
        ),
        mechanism=(
            "Positive first-half relationship followed by an equal-sized "
            "negative second-half relationship"
        ),
        expected_result=(
            "Overall correlation near zero, first-half correlation near "
            "+1, second-half correlation near -1, and stability gap near 2."
        ),
        instrument_a=first,
        instrument_b=second,
        similarity=similarity,
    )
