"""Streamlit entry point for the Market Memory Lab public preview."""

from __future__ import annotations

import html

import streamlit as st

from market_memory_lab.public_preview import (
    PreviewMetric,
    build_public_preview,
)


def _format_metric(metric: PreviewMetric) -> str:
    if metric.unavailable:
        return "Unavailable"

    if isinstance(metric.value, int):
        return str(metric.value)

    if isinstance(metric.value, float):
        return f"{metric.value:.4f}"

    return "Unavailable"


st.set_page_config(
    page_title="Market Memory Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    .mml-kicker {
        color: #65d9e8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .mml-hero {
        font-size: clamp(2.4rem, 7vw, 5.6rem);
        font-weight: 780;
        letter-spacing: -0.055em;
        line-height: 0.96;
        margin: 0.4rem 0 1.1rem;
    }
    .mml-question {
        color: #dce8ee;
        font-size: 1.35rem;
        line-height: 1.55;
        max-width: 900px;
    }
    .mml-boundary {
        background: rgba(20, 45, 58, 0.76);
        border: 1px solid rgba(101, 217, 232, 0.30);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: 1.4rem 0 2rem;
    }
    .mml-case {
        border-top: 1px solid rgba(220, 232, 238, 0.18);
        margin-top: 2.2rem;
        padding-top: 2.2rem;
    }
    .mml-conclusion {
        background: rgba(245, 184, 72, 0.09);
        border: 1px solid rgba(245, 184, 72, 0.36);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

preview = build_public_preview()

st.markdown(
    f'<div class="mml-kicker">{html.escape(preview.release_label)}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="mml-hero">{html.escape(preview.title)}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="mml-question">{html.escape(preview.central_question)}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="mml-boundary"><strong>Synthetic boundary:</strong> '
    f'{html.escape(preview.synthetic_notice)}</div>',
    unsafe_allow_html=True,
)

st.subheader("Five separate gates")
gate_columns = st.columns(5)
gate_labels = (
    "Data validity",
    "Calculation validity",
    "Evidence adequacy",
    "Scientific conclusion",
    "Investment decision",
)
gate_notes = (
    "Are the inputs valid?",
    "Was the quantity calculated correctly?",
    "Is the evidence sufficient?",
    "What does the evidence establish?",
    "Is any action justified?",
)

for column, label, note in zip(
    gate_columns,
    gate_labels,
    gate_notes,
    strict=True,
):
    with column:
        st.markdown(f"**{label}**")
        st.caption(note)

for index, case in enumerate(preview.cases, start=1):
    st.markdown('<div class="mml-case"></div>', unsafe_allow_html=True)
    st.caption(f"CONTROLLED SYNTHETIC CASE {index}")
    st.header(case.name)
    st.markdown(f"### {case.question}")
    st.write(case.plain_english)

    narrative_left, narrative_right = st.columns(2)
    with narrative_left:
        st.markdown("**Mechanism**")
        st.write(case.mechanism)
    with narrative_right:
        st.markdown("**Expected result**")
        st.write(case.expected_result)

    st.markdown("#### Measured evidence")
    metric_columns = st.columns(3)

    for metric_index, metric in enumerate(case.metrics):
        with metric_columns[metric_index % 3]:
            st.metric(
                metric.label,
                _format_metric(metric),
            )

    evidence_left, evidence_right = st.columns(2)

    with evidence_left:
        st.markdown("#### Evidence adequacy")
        st.write(f"**Label:** {case.adequacy_label}")
        st.write(f"**Permitted use:** {case.adequacy_use}")
        st.caption(case.adequacy_explanation)

    with evidence_right:
        st.markdown("#### Validation state")
        st.write(
            "**Calculation object:** "
            + ("Valid" if case.is_valid else "Blocked")
        )
        st.write(f"**Warnings:** {len(case.warnings)}")
        st.write(f"**Blocking errors:** {len(case.errors)}")

    if case.warnings:
        with st.expander("Warnings"):
            for warning in case.warnings:
                st.warning(warning)

    if case.errors:
        with st.expander("Blocking errors"):
            for error in case.errors:
                st.error(error)

    st.markdown(
        """
        <div class="mml-conclusion">
        <strong>Scientific conclusion boundary</strong><br>
        Memory similarity: not tested<br>
        Regime similarity: not tested<br>
        Lead-lag direction: not tested<br>
        Causation: not established<br>
        Trading conclusion: abstain
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.subheader("What this preview does not claim")
st.markdown(
    """
    - It does not establish market memory, regime change, or causation.
    - It does not demonstrate prediction, profitability, or a trading edge.
    - It does not convert valid calculations into investment decisions.
    - It does not present synthetic observations as historical market data.
    - It does not provide personalized investment advice.
    """
)

st.caption(
    "Market Memory Lab is open-source research infrastructure. "
    "This public preview uses frozen synthetic observations only."
)
