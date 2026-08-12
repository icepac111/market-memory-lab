"""Tests for the synthetic public research preview."""

from __future__ import annotations

import pytest

from market_memory_lab.public_preview import build_public_preview


def test_preview_contains_both_frozen_demonstrations() -> None:
    preview = build_public_preview()

    assert len(preview.cases) == 2
    assert all(case.name.strip() for case in preview.cases)


def test_preview_is_explicitly_synthetic_and_non_market() -> None:
    preview = build_public_preview()

    assert "synthetic" in preview.release_label.lower()
    assert "not market evidence" in preview.synthetic_notice.lower()
    assert all(
        not case.real_market_evidence
        for case in preview.cases
    )


def test_preview_preserves_scientific_abstention() -> None:
    preview = build_public_preview()

    for case in preview.cases:
        assert case.memory_similarity == "not_tested"
        assert case.regime_similarity == "not_tested"
        assert case.lead_lag_direction == "not_tested"
        assert case.causation_state == "not_established"
        assert case.trading_conclusion == "abstain"


def test_preview_preserves_existing_evidence_values() -> None:
    preview = build_public_preview()
    first_case = preview.cases[0]
    metrics = {
        metric.label: metric.value
        for metric in first_case.metrics
    }

    assert metrics["Overlapping returns"] == 12
    assert metrics["Pearson correlation"] == 1.0
    assert metrics["Spearman correlation"] == 1.0
    assert metrics["Symmetric volatility ratio"] == pytest.approx(1.0)


def test_preview_keeps_adequacy_separate_from_validity() -> None:
    preview = build_public_preview()

    for case in preview.cases:
        assert case.is_valid
        assert case.adequacy_label == "Critical"
        assert case.adequacy_use == "Blocked"


def test_preview_is_deterministic() -> None:
    assert build_public_preview() == build_public_preview()


def test_preview_does_not_expose_instrument_identity() -> None:
    preview = build_public_preview()

    for case in preview.cases:
        assert not hasattr(case, "asset_a")
        assert not hasattr(case, "asset_b")
        assert not hasattr(case, "ticker")
        assert not hasattr(case, "provider")
