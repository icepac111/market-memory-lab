"""Tests for the canonical public Learning Center."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "LEARNING_CENTER.md"
PAGE = ROOT / "dashboard" / "pages" / "5_Learning_Center.py"


def test_learning_center_files_exist() -> None:
    assert DOCUMENT.is_file()
    assert PAGE.is_file()


def test_learning_center_is_substantive() -> None:
    content = DOCUMENT.read_text(encoding="utf-8")

    required_sections = (
        "# Market Memory Lab Learning Center",
        "## Choose your path",
        "# 1. From prices to returns",
        "# 6. Pearson correlation",
        "# 10. Relationship stability gap",
        "# 11. Evidence adequacy",
        "# 13. Synthetic null laboratory",
        "# 15. The scientific decision ladder",
        "# 17. Current implementation boundary",
        "# 19. Final principle",
    )

    for section in required_sections:
        assert section in content


def test_learning_center_preserves_scientific_guardrails() -> None:
    raw_content = DOCUMENT.read_text(encoding="utf-8").lower()
    content = " ".join(raw_content.split())

    required_guardrails = (
        "correlation does not establish causation",
        "not a formal change-point test",
        "not a probability",
        "synthetic data are controlled mathematical examples",
        "not observations from real markets",
        "not currently implemented or validated",
        "abstain",
    )

    for statement in required_guardrails:
        assert statement in content


def test_learning_center_documents_three_explanation_levels() -> None:
    content = DOCUMENT.read_text(encoding="utf-8")

    assert "### 1. Completely new explanation" in content
    assert "### 2. Practical intuition" in content
    assert "### 3. Mathematical definition" in content


def test_learning_center_documents_required_concept_fields() -> None:
    content = DOCUMENT.read_text(encoding="utf-8")

    required_fields = (
        "### Symbol guide",
        "### Numerical example",
        "### What Market Memory Lab calculates",
        "### What automated tests verify",
        "### Assumptions",
        "### Failure modes",
        "### Allowed conclusion",
        "### Blocked conclusion",
        "### Code reference",
        "### Test reference",
    )

    for field in required_fields:
        assert field in content


def test_streamlit_page_reads_canonical_document() -> None:
    page = PAGE.read_text(encoding="utf-8")

    assert '"docs" / "LEARNING_CENTER.md"' in page
    assert "read_text(encoding=\"utf-8\")" in page
    assert "st.markdown(curriculum)" in page
    assert "canonical Markdown" in page


def test_streamlit_page_has_public_learning_paths() -> None:
    page = PAGE.read_text(encoding="utf-8")

    for label in ("New learner", "Student", "Researcher", "Full curriculum"):
        assert label in page


def test_streamlit_page_preserves_research_boundary() -> None:
    page = PAGE.read_text(encoding="utf-8").lower()

    assert "synthetic demonstrations" in page
    assert "not observations from real markets" in page
    assert "does not issue trading recommendations" in page


def test_learning_center_contains_no_html_entities() -> None:
    content = DOCUMENT.read_text(encoding="utf-8")

    forbidden = ("&amp;", "&nbsp;", "&gt;", "&lt;")
    for entity in forbidden:
        assert entity not in content


def test_learning_center_contains_no_character_branding() -> None:
    content = DOCUMENT.read_text(encoding="utf-8").lower()

    forbidden = ("bobby axelrod", "billions", "axe capital")
    for phrase in forbidden:
        assert phrase not in content
