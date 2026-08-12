"""Presentation-safe models for the synthetic public research preview."""

from __future__ import annotations

from dataclasses import dataclass

from market_memory_lab.demonstrations import (
    Demonstration,
    classify_evidence_adequacy,
    false_friends_demonstration,
    stable_twins_demonstration,
)


@dataclass(frozen=True)
class PreviewMetric:
    """One displayed measurement without changing its interpretation."""

    label: str
    value: float | int | None
    unavailable: bool


@dataclass(frozen=True)
class PreviewCase:
    """Identity-limited presentation model for one synthetic demonstration."""

    name: str
    question: str
    plain_english: str
    mechanism: str
    expected_result: str
    metrics: tuple[PreviewMetric, ...]
    adequacy_label: str
    adequacy_use: str
    adequacy_explanation: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    is_valid: bool
    real_market_evidence: bool
    memory_similarity: str
    regime_similarity: str
    lead_lag_direction: str
    causation_state: str
    trading_conclusion: str


@dataclass(frozen=True)
class PublicPreview:
    """Complete presentation model for the frozen synthetic preview."""

    title: str
    release_label: str
    central_question: str
    synthetic_notice: str
    cases: tuple[PreviewCase, ...]


def _metric(
    label: str,
    value: float | None,
) -> PreviewMetric:
    return PreviewMetric(
        label=label,
        value=value,
        unavailable=value is None,
    )


def _case_from_demonstration(
    demonstration: Demonstration,
) -> PreviewCase:
    similarity = demonstration.similarity
    adequacy = classify_evidence_adequacy(
        similarity.overlapping_returns
    )

    return PreviewCase(
        name=demonstration.name,
        question=demonstration.question,
        plain_english=demonstration.plain_english,
        mechanism=demonstration.mechanism,
        expected_result=demonstration.expected_result,
        metrics=(
            _metric(
                "Overlapping returns",
                similarity.overlapping_returns,
            ),
            _metric(
                "Pearson correlation",
                similarity.pearson_correlation,
            ),
            _metric(
                "Spearman correlation",
                similarity.spearman_correlation,
            ),
            _metric(
                "Symmetric volatility ratio",
                similarity.volatility_ratio,
            ),
            _metric(
                "Standardized Wasserstein distance",
                similarity.standardized_wasserstein_distance,
            ),
            _metric(
                "Drawdown difference",
                similarity.drawdown_difference,
            ),
            _metric(
                "First-half correlation",
                similarity.first_half_correlation,
            ),
            _metric(
                "Second-half correlation",
                similarity.second_half_correlation,
            ),
            _metric(
                "Correlation stability gap",
                similarity.correlation_stability_gap,
            ),
        ),
        adequacy_label=adequacy.label,
        adequacy_use=adequacy.decision_use,
        adequacy_explanation=adequacy.explanation,
        warnings=similarity.warnings,
        errors=similarity.errors,
        is_valid=similarity.is_valid,
        real_market_evidence=False,
        memory_similarity=similarity.memory_similarity,
        regime_similarity=similarity.regime_similarity,
        lead_lag_direction=similarity.lead_lag_direction,
        causation_state=similarity.causation_state,
        trading_conclusion=similarity.trading_conclusion,
    )


def build_public_preview() -> PublicPreview:
    """Build the deterministic synthetic-only public preview."""

    demonstrations = (
        stable_twins_demonstration(),
        false_friends_demonstration(),
    )

    return PublicPreview(
        title="Market Memory Lab",
        release_label="Synthetic Research Preview v0.1",
        central_question=(
            "What looks similar, what changed underneath, "
            "and where could the market be fooling us?"
        ),
        synthetic_notice=(
            "This preview uses frozen synthetic observations. "
            "Synthetic results are not market evidence, investment advice, "
            "a profitability result, or a claim of causation."
        ),
        cases=tuple(
            _case_from_demonstration(demonstration)
            for demonstration in demonstrations
        ),
    )
