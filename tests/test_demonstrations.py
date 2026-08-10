"""Tests for controlled demonstrations and adequacy policies."""

from __future__ import annotations

import pytest

from market_memory_lab.demonstrations import (
    classify_evidence_adequacy,
    false_friends_demonstration,
    stable_twins_demonstration,
)


def test_stable_twins_expected_structure() -> None:
    demo = stable_twins_demonstration()
    result = demo.similarity

    assert result.is_valid
    assert result.overlapping_returns == 12
    assert result.pearson_correlation == pytest.approx(1.0)
    assert result.spearman_correlation == pytest.approx(1.0)
    assert result.volatility_ratio == pytest.approx(1.0)
    assert result.standardized_wasserstein_distance == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert result.correlation_stability_gap == pytest.approx(0.0)


def test_false_friends_expected_reversal() -> None:
    demo = false_friends_demonstration()
    result = demo.similarity

    assert result.is_valid
    assert result.overlapping_returns == 12
    assert result.pearson_correlation == pytest.approx(0.0, abs=1e-12)
    assert result.first_half_correlation == pytest.approx(1.0)
    assert result.second_half_correlation == pytest.approx(-1.0)
    assert result.correlation_stability_gap == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("observations", "label", "decision_use"),
    [
        (0, "Critical", "Blocked"),
        (12, "Critical", "Blocked"),
        (29, "Critical", "Blocked"),
        (30, "Limited", "Research only"),
        (99, "Limited", "Research only"),
        (100, "Moderate", "Research only"),
        (251, "Moderate", "Research only"),
        (252, "Stronger descriptive base", "Research only"),
    ],
)
def test_adequacy_policy_boundaries(
    observations: int,
    label: str,
    decision_use: str,
) -> None:
    result = classify_evidence_adequacy(observations)

    assert result.label == label
    assert result.decision_use == decision_use


def test_adequacy_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        classify_evidence_adequacy(True)

    with pytest.raises(ValueError):
        classify_evidence_adequacy(-1)


def test_manifests_preserve_scientific_boundaries() -> None:
    demo = false_friends_demonstration()
    manifest = demo.manifest_json()

    assert '"real_market_evidence": false' in manifest
    assert '"memory_similarity": "not_tested"' in manifest
    assert '"causation": "not_established"' in manifest
    assert '"trading_conclusion": "abstain"' in manifest
