"""Interactive scientific preview for Market Memory Lab."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from market_memory_lab.synthetic import iid_gaussian

st.set_page_config(
    page_title="Market Memory Lab",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --mml-bg: #07111f;
        --mml-panel: #0d1b2a;
        --mml-border: #22364d;
        --mml-blue: #58a6ff;
        --mml-cyan: #39d0d8;
        --mml-green: #3ddc97;
        --mml-amber: #f2c14e;
        --mml-red: #ff6b6b;
        --mml-text: #e6edf3;
        --mml-muted: #94a3b8;
    }

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(57, 208, 216, 0.10),
                transparent 32%
            ),
            radial-gradient(
                circle at top left,
                rgba(88, 166, 255, 0.10),
                transparent 28%
            ),
            var(--mml-bg);
        color: var(--mml-text);
    }

    [data-testid="stSidebar"] {
        background: #081522;
        border-right: 1px solid var(--mml-border);
    }

    .mml-hero {
        border: 1px solid var(--mml-border);
        border-radius: 18px;
        padding: 28px 30px;
        background:
            linear-gradient(
                135deg,
                rgba(88, 166, 255, 0.12),
                rgba(57, 208, 216, 0.05)
            ),
            var(--mml-panel);
        margin-bottom: 20px;
    }

    .mml-kicker {
        color: var(--mml-cyan);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .mml-title {
        color: var(--mml-text);
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 10px;
    }

    .mml-subtitle {
        color: var(--mml-muted);
        font-size: 1.05rem;
        line-height: 1.55;
        max-width: 950px;
    }

    .mml-question {
        color: white;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 18px;
    }

    .mml-card {
        border: 1px solid var(--mml-border);
        border-radius: 14px;
        padding: 18px;
        background: rgba(13, 27, 42, 0.88);
        min-height: 145px;
    }

    .mml-label {
        color: var(--mml-muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .mml-value {
        color: var(--mml-text);
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .mml-detail {
        color: var(--mml-muted);
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .mml-status-good {
        color: var(--mml-green);
        font-weight: 700;
    }

    .mml-status-pending {
        color: var(--mml-amber);
        font-weight: 700;
    }

    .mml-status-blocked {
        color: var(--mml-red);
        font-weight: 700;
    }

    .mml-evidence {
        border-left: 4px solid var(--mml-cyan);
        border-radius: 8px;
        padding: 14px 18px;
        background: rgba(57, 208, 216, 0.07);
        color: var(--mml-text);
        margin: 12px 0;
    }

    .mml-warning {
        border-left: 4px solid var(--mml-amber);
        border-radius: 8px;
        padding: 14px 18px;
        background: rgba(242, 193, 78, 0.07);
        color: var(--mml-text);
        margin: 12px 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(13, 27, 42, 0.88);
        border: 1px solid var(--mml-border);
        border-radius: 13px;
        padding: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--mml-muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--mml-text);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(13, 27, 42, 0.75);
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_lag_one_correlation(values: pd.Series) -> float:
    """Return lag-one sample correlation or NaN when undefined."""
    if len(values) < 3:
        return float("nan")

    current = values.iloc[1:].to_numpy()
    previous = values.iloc[:-1].to_numpy()

    if np.isclose(np.std(current), 0.0):
        return float("nan")

    if np.isclose(np.std(previous), 0.0):
        return float("nan")

    return float(np.corrcoef(previous, current)[0, 1])


def format_number(value: float, digits: int = 5) -> str:
    """Format a finite number while making undefined values explicit."""
    if not math.isfinite(value):
        return "Undefined"

    return f"{value:.{digits}f}"


def research_card(
    title: str,
    status: str,
    detail: str,
    status_class: str,
) -> str:
    """Create one research-status card."""
    return f"""
    <div class="mml-card">
        <div class="mml-label">{title}</div>
        <div class="mml-value {status_class}">{status}</div>
        <div class="mml-detail">{detail}</div>
    </div>
    """


with st.sidebar:
    st.markdown("## Market Memory Lab")
    st.caption("Scientific preview, version 0.1")

    st.markdown("---")
    st.markdown("### Synthetic process")

    sample_size = st.slider(
        "Number of observations",
        min_value=100,
        max_value=100_000,
        value=5_000,
        step=100,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
    )

    target_mean = st.number_input(
        "Population mean",
        value=0.0,
        step=0.01,
        format="%.4f",
    )

    target_sigma = st.number_input(
        "Population standard deviation",
        min_value=0.0001,
        value=1.0,
        step=0.05,
        format="%.4f",
    )

    st.markdown("---")
    st.markdown("### Scientific state")
    st.success("IID Gaussian generator verified")
    st.info("All displayed observations are synthetic")
    st.warning("No market inference is produced")

result = iid_gaussian(
    n=int(sample_size),
    seed=int(seed),
    mean=float(target_mean),
    sigma=float(target_sigma),
)

values = result.values
observed_mean = float(values.mean())
observed_variance = float(values.var(ddof=1))
target_variance = float(target_sigma) ** 2

mean_standard_error = float(target_sigma) / math.sqrt(sample_size)
variance_standard_error = (
    target_variance * math.sqrt(2.0 / (sample_size - 1))
)

mean_error_in_se = (
    (observed_mean - float(target_mean)) / mean_standard_error
)

variance_error_in_se = (
    (observed_variance - target_variance)
    / variance_standard_error
)

lag_one = safe_lag_one_correlation(values)

st.markdown(
    """
    <div class="mml-hero">
        <div class="mml-kicker">
            Open financial falsification infrastructure
        </div>
        <div class="mml-title">Market Memory Lab</div>
        <div class="mml-subtitle">
            Testing whether apparent financial memory, complexity,
            information flow, and regime change remain credible after
            competing estimators, null processes, structural breaks,
            and out-of-sample evaluation.
        </div>
        <div class="mml-question">
            Did the market move, or did the market's structure change?
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, null_tab, research_tab, methods_tab = st.tabs(
    [
        "Structural Radar",
        "NullLab Explorer",
        "Research Pipeline",
        "Scientific Guardrails",
    ]
)

with overview_tab:
    st.subheader("What changed structurally?")
    st.caption(
        "This preview reports project evidence states, not live markets."
    )

    first, second, third = st.columns(3)

    with first:
        st.markdown(
            research_card(
                "Verified mechanism",
                "IID Gaussian",
                (
                    "Independent Gaussian innovations with explicit "
                    "seed, mean, standard deviation, and immutable "
                    "ground-truth metadata."
                ),
                "mml-status-good",
            ),
            unsafe_allow_html=True,
        )

    with second:
        st.markdown(
            research_card(
                "Estimator disagreement",
                "Not available",
                (
                    "No memory estimator has been admitted into the "
                    "benchmark. The platform will not fabricate an "
                    "agreement or disagreement result."
                ),
                "mml-status-pending",
            ),
            unsafe_allow_html=True,
        )

    with third:
        st.markdown(
            research_card(
                "Market conclusion",
                "Abstain",
                (
                    "Synthetic Gaussian observations cannot establish "
                    "anything about an empirical financial market."
                ),
                "mml-status-blocked",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Current research ledger")

    ledger = pd.DataFrame(
        [
            {
                "Component": "Core financial metrics",
                "State": "Validated",
                "Evidence": "16 automated tests",
                "Next challenge": "Cross-library reference comparison",
            },
            {
                "Component": "IID Gaussian null",
                "State": "Validated",
                "Evidence": "12 automated tests",
                "Next challenge": "Monte Carlo calibration",
            },
            {
                "Component": "Heavy-tailed null",
                "State": "Pending",
                "Evidence": "None",
                "Next challenge": "Student-t implementation",
            },
            {
                "Component": "Memory estimators",
                "State": "Blocked",
                "Evidence": "None",
                "Next challenge": "Null processes first",
            },
            {
                "Component": "Thermodynamic analogy",
                "State": "Blocked",
                "Evidence": "None",
                "Next challenge": "Formal state mapping",
            },
        ]
    )

    st.dataframe(
        ledger,
        hide_index=True,
        use_container_width=True,
    )

with null_tab:
    st.subheader("Verified IID Gaussian null")

    st.markdown(
        """
        <div class="mml-evidence">
            <strong>Ground truth:</strong> observations are independent
            Gaussian innovations. Temporal dependence is absent by
            construction. This is a synthetic control, not market data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_one, metric_two, metric_three, metric_four = st.columns(4)

    metric_one.metric(
        "Observed mean",
        format_number(observed_mean),
        delta=(
            f"{mean_error_in_se:+.2f} mean standard errors"
        ),
    )

    metric_two.metric(
        "Observed variance",
        format_number(observed_variance),
        delta=(
            f"{variance_error_in_se:+.2f} variance SEs"
        ),
    )

    metric_three.metric(
        "Lag-1 sample correlation",
        format_number(lag_one),
    )

    metric_four.metric(
        "Seed",
        str(int(seed)),
        delta="Reproducible",
    )

    chart_data = pd.DataFrame(
        {
            "Observation": values.index,
            "Synthetic return": values.to_numpy(),
        }
    ).set_index("Observation")

    st.line_chart(
        chart_data,
        height=350,
        color="#39d0d8",
    )

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("#### Distribution preview")
        histogram_counts, histogram_edges = np.histogram(
            values.to_numpy(),
            bins=40,
        )

        histogram = pd.DataFrame(
            {
                "Bin midpoint": (
                    histogram_edges[:-1]
                    + histogram_edges[1:]
                )
                / 2.0,
                "Count": histogram_counts,
            }
        ).set_index("Bin midpoint")

        st.bar_chart(
            histogram,
            height=300,
            color="#58a6ff",
        )

    with right:
        st.markdown("#### Provenance")

        provenance = pd.DataFrame(
            {
                "Field": [
                    "Process",
                    "Mechanism",
                    "Expected dependence",
                    "Sample size",
                    "Seed",
                    "Target mean",
                    "Target standard deviation",
                    "Data status",
                ],
                "Value": [
                    result.process,
                    result.mechanism,
                    result.expected_dependence,
                    str(sample_size),
                    str(int(seed)),
                    str(float(target_mean)),
                    str(float(target_sigma)),
                    "Synthetic",
                ],
            }
        )

        st.dataframe(
            provenance,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="mml-warning">
            <strong>Interpretation limit:</strong> a nonzero sample
            autocorrelation can occur by sampling variation even when
            population autocorrelation is zero. No memory conclusion is
            drawn from this single path.
        </div>
        """,
        unsafe_allow_html=True,
    )

with research_tab:
    st.subheader("Research program")

    pipeline = pd.DataFrame(
        [
            {
                "Stage": 1,
                "Laboratory": "Gaussian null",
                "Purpose": "Basic no-memory calibration",
                "State": "Validated",
            },
            {
                "Stage": 2,
                "Laboratory": "Heavy-tail null",
                "Purpose": "Separate tail behavior from dependence",
                "State": "Next",
            },
            {
                "Stage": 3,
                "Laboratory": "Short-memory null",
                "Purpose": "Separate AR dependence from long memory",
                "State": "Pending",
            },
            {
                "Stage": 4,
                "Laboratory": "Variance persistence",
                "Purpose": "Distinguish GARCH effects from return memory",
                "State": "Pending",
            },
            {
                "Stage": 5,
                "Laboratory": "Structural breaks",
                "Purpose": "Test spurious long-memory detection",
                "State": "Pending",
            },
            {
                "Stage": 6,
                "Laboratory": "Estimator Arena",
                "Purpose": "Compare Hurst, DFA, GPH, and alternatives",
                "State": "Blocked by stages 2 through 5",
            },
            {
                "Stage": 7,
                "Laboratory": "Entropy Lab",
                "Purpose": "Test incremental complexity information",
                "State": "Pending",
            },
            {
                "Stage": 8,
                "Laboratory": "Thermodynamic Trial",
                "Purpose": "Test formal physics-inspired mappings",
                "State": "Blocked by earlier evidence",
            },
        ]
    )

    st.dataframe(
        pipeline,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Candidate dissertation umbrella")

    st.markdown(
        """
        <div class="mml-evidence">
            Can apparent financial memory and thermodynamic-style state
            transitions be distinguished from heavy tails, volatility
            clustering, short-range dependence, and structural breaks
            across traditional and digitally native markets?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Status: candidate research direction, not a selected dissertation "
        "topic and not an established finding."
    )

with methods_tab:
    st.subheader("Scientific guardrails")

    st.markdown(
        """
        **No estimator is truth.**

        **No analogy is law.**

        **No correlation is causation.**

        **No in-sample fit is discovery.**

        **No forecast matters without a baseline.**

        **No statistical effect guarantees economic value.**

        **No AI output becomes numerical ground truth.**

        **No failed experiment disappears.**

        **No number appears without provenance.**

        **No conclusion is protected from falsification.**
        """
    )

    st.markdown("---")

    st.markdown("### What this preview establishes")

    st.success(
        "The verified IID Gaussian generator can be explored "
        "interactively with reproducible seeds."
    )

    st.markdown("### What this preview does not establish")

    st.error(
        "It does not detect market memory, predict prices, estimate alpha, "
        "recommend trades, or establish thermodynamic market laws."
    )

    st.markdown("### Required path to an empirical claim")

    st.markdown(
        """
        1. Register the hypothesis.
        2. Preserve data provenance.
        3. Declare the estimator and assumptions.
        4. Challenge the result with null processes.
        5. Compare competing estimators.
        6. Test structural-break explanations.
        7. Validate chronologically out of sample.
        8. Quantify uncertainty.
        9. Record contradicting evidence.
        10. State what would falsify the conclusion.
        """
    )

st.markdown("---")
st.caption(
    "Market Memory Lab, scientific preview v0.1. "
    "All displayed observations are synthetic."
)
