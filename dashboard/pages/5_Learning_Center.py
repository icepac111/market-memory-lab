"""Public Learning Center rendered from the canonical Markdown curriculum."""

from pathlib import Path

import streamlit as st


def learning_center_path() -> Path:
    """Return the canonical Learning Center Markdown path."""
    return Path(__file__).resolve().parents[2] / "docs" / "LEARNING_CENTER.md"


def load_learning_center() -> str:
    """Load the canonical Learning Center or fail with a clear message."""
    path = learning_center_path()
    if not path.is_file():
        raise FileNotFoundError(
            "The canonical Learning Center was not found at "
            f"{path}. The dashboard does not maintain a duplicate copy."
        )

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"The canonical Learning Center is empty: {path}")

    return content


st.set_page_config(
    page_title="Learning Center | Market Memory Lab",
    page_icon="📘",
    layout="wide",
)

st.title("Learning Center")
st.caption(
    "Learn what the platform measures, what the evidence permits, "
    "and when the scientifically honest result is to abstain."
)

with st.sidebar:
    st.header("Learning paths")
    st.markdown(
        """
- **New learner:** begin with prices, returns, risk, and evidence
- **Student:** study formulas, symbols, and numerical examples
- **Researcher:** inspect assumptions, tests, and implementation boundaries
- **Reference:** use the complete mathematical curriculum below
"""
    )

    st.divider()
    st.info(
        "This page reads the same canonical Markdown document displayed "
        "on GitHub. Educational explanations are not duplicated."
    )

try:
    curriculum = load_learning_center()
except (OSError, ValueError) as exc:
    st.error("The Learning Center could not be loaded.")
    st.exception(exc)
    st.stop()

intro_tab, curriculum_tab, boundary_tab = st.tabs(
    ["Start here", "Full curriculum", "Scientific boundary"]
)

with intro_tab:
    st.subheader("A platform designed to challenge conclusions")
    st.markdown(
        """
Most systems optimize for giving an answer.

**Market Memory Lab optimizes for discovering when the answer is not yet
justified.**

Use the curriculum to move from a plain-language explanation to practical
intuition and then to the complete mathematical definition.
"""
    )

    learner_col, student_col, researcher_col = st.columns(3)

    with learner_col:
        st.markdown("### New learner")
        st.write(
            "Understand returns, volatility, drawdown, correlation, "
            "evidence quality, and abstention without beginning with notation."
        )

    with student_col:
        st.markdown("### Student")
        st.write(
            "Connect formulas to symbol guides, small numerical examples, "
            "assumptions, and failure modes."
        )

    with researcher_col:
        st.markdown("### Researcher")
        st.write(
            "Inspect implementation references, automated tests, null "
            "processes, blocked conclusions, and current limitations."
        )

    st.warning(
        "Synthetic demonstrations are controlled mathematical constructions. "
        "They are not observations from real markets."
    )

with curriculum_tab:
    st.markdown(curriculum)

with boundary_tab:
    st.subheader("What the Learning Center does not claim")
    st.markdown(
        """
The curriculum does not convert descriptive statistics into:

- causal evidence
- market-memory detection
- formal regime identification
- return forecasts
- verified profitability
- personalized investment advice
- legal or economic equivalence between instruments

A correct calculation can coexist with inadequate evidence.
"""
    )

    st.error(
        "Decision use remains blocked or research-only under the documented "
        "evidence policy. The interface does not issue trading recommendations."
    )

st.divider()
st.caption(
    "Canonical source: docs/LEARNING_CENTER.md · "
    "Research and educational use · No personalized investment advice"
)
