"""One-click explanations of Market Memory Lab's core research idea."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from market_memory_lab.demonstrations import (
    Demonstration,
    classify_evidence_adequacy,
    false_friends_demonstration,
    stable_twins_demonstration,
)

st.set_page_config(
    page_title="One-Click Demonstrations | Market Memory Lab",
    page_icon="M",
    layout="wide",
)

st.title("What Is Market Memory Lab Actually Doing?")

st.caption(
    "One-click demonstrations for researchers, investors, students, "
    "executives, and anyone who has never studied financial statistics."
)

st.info(
    "These demonstrations use controlled synthetic data. They explain "
    "how the engine works but provide no evidence about any real market."
)

selection = st.radio(
    "Choose a demonstration",
    [
        "Stable Twins",
        "False Friends",
    ],
    horizontal=True,
)

demo: Demonstration

if selection == "Stable Twins":
    demo = stable_twins_demonstration()
else:
    demo = false_friends_demonstration()

result = demo.similarity
adequacy = classify_evidence_adequacy(
    result.overlapping_returns
)

st.markdown(f"## {demo.name}")

st.markdown("### The question")

st.write(demo.question)

st.markdown("### Explain it like I am five")

st.success(demo.plain_english)

st.markdown("### What was deliberately built into the example")

st.write(demo.mechanism)

executive_one, executive_two, executive_three, executive_four = st.columns(4)

executive_one.metric(
    "Overlapping returns",
    str(result.overlapping_returns),
)

executive_two.metric(
    "Overall correlation",
    (
        f"{result.pearson_correlation:.4f}"
        if result.pearson_correlation is not None
        else "Unavailable"
    ),
)

executive_three.metric(
    "Relationship stability gap",
    (
        f"{result.correlation_stability_gap:.4f}"
        if result.correlation_stability_gap is not None
        else "Unavailable"
    ),
)

executive_four.metric(
    "Trading conclusion",
    "Abstain",
)

if adequacy.label == "Critical":
    st.error(
        f"Evidence adequacy: {adequacy.label}. "
        f"Decision use: {adequacy.decision_use}. "
        f"{adequacy.explanation}"
    )
else:
    st.warning(
        f"Evidence adequacy: {adequacy.label}. "
        f"Decision use: {adequacy.decision_use}. "
        f"{adequacy.explanation}"
    )

surface_tab, risk_tab, stability_tab, explanation_tab, audit_tab = st.tabs(
    [
        "What moved together?",
        "How did risk compare?",
        "Did the relationship change?",
        "Why does this matter?",
        "Audit trail",
    ]
)

with surface_tab:
    first, second = st.columns(2)

    first.metric(
        "Pearson return correlation",
        (
            f"{result.pearson_correlation:.4f}"
            if result.pearson_correlation is not None
            else "Unavailable"
        ),
    )

    second.metric(
        "Spearman rank correlation",
        (
            f"{result.spearman_correlation:.4f}"
            if result.spearman_correlation is not None
            else "Unavailable"
        ),
    )

    aligned = pd.concat(
        [
            demo.instrument_a.simple_return_series.rename(
                result.asset_a
            ),
            demo.instrument_b.simple_return_series.rename(
                result.asset_b
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    normalized_wealth = (1.0 + aligned).cumprod()

    st.line_chart(
        normalized_wealth,
        height=350,
    )

    st.markdown(
        """
        **Plain English**

        The chart begins both instruments at the same reference value.
        That lets the viewer compare percentage movement rather than
        sticker price.

        **Caution**

        Visual resemblance does not establish causation, persistence,
        or an investment opportunity.
        """
    )

with risk_tab:
    first, second, third = st.columns(3)

    first.metric(
        f"{result.asset_a} volatility",
        (
            f"{result.volatility_a:.6f}"
            if result.volatility_a is not None
            else "Unavailable"
        ),
    )

    second.metric(
        f"{result.asset_b} volatility",
        (
            f"{result.volatility_b:.6f}"
            if result.volatility_b is not None
            else "Unavailable"
        ),
    )

    third.metric(
        "Volatility ratio",
        (
            f"{result.volatility_ratio:.4f}"
            if result.volatility_ratio is not None
            else "Unavailable"
        ),
    )

    risk_table = pd.DataFrame(
        [
            {
                "Instrument": result.asset_a,
                "Maximum drawdown": (
                    f"{100.0 * result.maximum_drawdown_a:.2f}%"
                ),
            },
            {
                "Instrument": result.asset_b,
                "Maximum drawdown": (
                    f"{100.0 * result.maximum_drawdown_b:.2f}%"
                ),
            },
            {
                "Instrument": "Absolute difference",
                "Maximum drawdown": (
                    f"{100.0 * result.drawdown_difference:.2f}%"
                ),
            },
        ]
    )

    st.dataframe(
        risk_table,
        hide_index=True,
        width="stretch",
    )

with stability_tab:
    first, second, third = st.columns(3)

    first.metric(
        "First-half correlation",
        (
            f"{result.first_half_correlation:.4f}"
            if result.first_half_correlation is not None
            else "Unavailable"
        ),
    )

    second.metric(
        "Second-half correlation",
        (
            f"{result.second_half_correlation:.4f}"
            if result.second_half_correlation is not None
            else "Unavailable"
        ),
    )

    third.metric(
        "Absolute stability gap",
        (
            f"{result.correlation_stability_gap:.4f}"
            if result.correlation_stability_gap is not None
            else "Unavailable"
        ),
    )

    if demo.name == "False Friends":
        st.error(
            "The full-period correlation is near zero, but the first "
            "half is +1 and the second half is -1. Aggregation concealed "
            "a complete relationship reversal."
        )
    else:
        st.success(
            "Both halves preserve the same positive relationship in this "
            "controlled example."
        )

    st.caption(
        "The two-half comparison is a descriptive diagnostic, not a "
        "formal break test or proof of a market regime."
    )

with explanation_tab:
    st.markdown("### The simple version")

    if demo.name == "False Friends":
        st.markdown(
            """
            A normal screener could see overall correlation near zero and
            conclude that the instruments are unrelated.

            Market Memory Lab looks inside the sample and discovers that
            the instruments had two extreme relationships with opposite
            signs.

            **The relationship did not disappear. The relationship
            reversed.**
            """
        )
    else:
        st.markdown(
            """
            Different nominal prices do not prevent two instruments from
            behaving identically in percentage terms.

            Market Memory Lab compares aligned returns and separate risk
            dimensions instead of confusing a price tag with market
            behavior.
            """
        )

    st.markdown("### What the result supports")

    st.write(demo.expected_result)

    st.markdown("### What the result does not support")

    unsupported = pd.DataFrame(
        [
            {
                "Claim": "Real-market evidence",
                "State": "No",
            },
            {
                "Claim": "Market memory",
                "State": "Not tested",
            },
            {
                "Claim": "Regime identification",
                "State": "Not tested",
            },
            {
                "Claim": "Lead-lag direction",
                "State": "Not tested",
            },
            {
                "Claim": "Causation",
                "State": "Not established",
            },
            {
                "Claim": "Trading recommendation",
                "State": "Abstain",
            },
        ]
    )

    st.dataframe(
        unsupported,
        hide_index=True,
        width="stretch",
    )

with audit_tab:
    audit = pd.DataFrame(
        [
            {
                "Field": "Demonstration",
                "Value": demo.name,
            },
            {
                "Field": "Mechanism",
                "Value": demo.mechanism,
            },
            {
                "Field": "Instrument A hash",
                "Value": (
                    demo.instrument_a.validation.canonical_sha256
                ),
            },
            {
                "Field": "Instrument B hash",
                "Value": (
                    demo.instrument_b.validation.canonical_sha256
                ),
            },
            {
                "Field": "Return observations",
                "Value": str(result.overlapping_returns),
            },
            {
                "Field": "Evidence adequacy",
                "Value": adequacy.label,
            },
            {
                "Field": "Decision use",
                "Value": adequacy.decision_use,
            },
        ]
    )

    st.dataframe(
        audit,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download demonstration manifest",
        data=demo.manifest_json(),
        file_name=(
            demo.name.lower().replace(" ", "_")
            + "_manifest.json"
        ),
        mime="application/json",
    )

st.markdown("---")

st.markdown(
    """
    ### Market Memory Lab's operating rule

    A calculation may be mathematically valid while the evidence remains
    inadequate.

    The platform separates:

    1. Data validity
    2. Calculation validity
    3. Evidence adequacy
    4. Scientific conclusion
    5. Investment decision
    """
)
