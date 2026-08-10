"""Empirical financial-asset validation and descriptive research page."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from market_memory_lab.empirical import (
    AssetMetadata,
    analyze_empirical_prices,
)

st.set_page_config(
    page_title="Empirical Asset Lab | Market Memory Lab",
    page_icon="M",
    layout="wide",
)

st.title("Empirical Asset Lab")
st.caption(
    "Bring a stock, index, FX pair, commodity, cryptoasset, or other "
    "instrument into a strict validation gate before any structural claim."
)

st.info(
    "This page performs validated descriptive analysis only. "
    "Memory, alpha, thermodynamic state, and trading conclusions "
    "remain unavailable."
)

template = pd.DataFrame(
    {
        "Date": [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
        ],
        "Adjusted Close": [
            100.00,
            101.25,
            99.80,
            102.10,
        ],
    }
)

with st.sidebar:
    st.header("Asset evidence intake")

    uploaded_file = st.file_uploader(
        "Upload a CSV price series",
        type=["csv"],
        help=(
            "The file remains inside this Streamlit session. "
            "Do not upload licensed or confidential data to a public app."
        ),
    )

    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False),
        file_name="market_memory_lab_template.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Declared metadata")

    dataset_name = st.text_input(
        "Dataset name",
        value="Uploaded empirical series",
    )

    source = st.text_input(
        "Source",
        value="User supplied CSV",
    )

    asset_identifier = st.text_input(
        "Asset identifier",
        value="UNDECLARED",
    )

    asset_class = st.selectbox(
        "Asset class",
        [
            "Equity",
            "Index",
            "ETF",
            "FX",
            "Commodity",
            "Cryptoasset",
            "Tokenized asset",
            "Interest rate",
            "Other",
        ],
    )

    venue = st.text_input(
        "Venue or market",
        value="Undeclared",
    )

    currency = st.text_input(
        "Currency",
        value="USD",
    )

    timezone = st.text_input(
        "Timezone",
        value="Undeclared",
    )

    adjustment_status = st.selectbox(
        "Price adjustment status",
        [
            "Adjusted close",
            "Unadjusted close",
            "Unknown",
        ],
    )

    license_note = st.text_input(
        "License or access note",
        value="User is responsible for data rights",
    )

if uploaded_file is None:
    st.subheader("Start with a documented empirical series")

    first, second, third = st.columns(3)

    first.metric(
        "Data state",
        "No dataset",
    )
    second.metric(
        "Memory conclusion",
        "Not tested",
    )
    third.metric(
        "Trading conclusion",
        "Abstain",
    )

    st.markdown(
        """
        ### Required steps

        1. Download the CSV template from the sidebar.
        2. Replace the example dates and prices with an empirical series.
        3. Upload the CSV.
        4. Declare the source, asset, venue, currency, timezone, and
           adjustment status.
        5. Map the date and price columns.
        6. Review every data-health warning before interpreting results.

        ### Supported at this stage

        - Price validation
        - Simple returns
        - Logarithmic returns
        - Wealth index
        - Arithmetic annualized return
        - Annualized sample volatility
        - Maximum drawdown
        - Canonical dataset hash
        - Reproducibility manifest

        ### Explicitly unavailable

        - Hurst exponent
        - DFA
        - GPH
        - Entropy
        - Market temperature
        - Alpha
        - Buy, sell, or short recommendation
        """
    )

    st.stop()

try:
    uploaded_data = pd.read_csv(uploaded_file)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, OSError) as error:
    st.error(f"CSV parsing failed: {error}")
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

st.subheader("Column mapping")

mapping_one, mapping_two, mapping_three = st.columns(3)

with mapping_one:
    date_column = st.selectbox(
        "Date column",
        options=list(uploaded_data.columns),
    )

with mapping_two:
    price_column = st.selectbox(
        "Price column",
        options=list(uploaded_data.columns),
        index=(
            list(uploaded_data.columns).index("Adjusted Close")
            if "Adjusted Close" in uploaded_data.columns
            else min(1, len(uploaded_data.columns) - 1)
        ),
    )

with mapping_three:
    frequency_choice = st.selectbox(
        "Observation frequency",
        [
            "Undeclared",
            "Daily",
            "Weekly",
            "Monthly",
            "Quarterly",
            "Annual",
        ],
    )

frequency_map = {
    "Undeclared": None,
    "Daily": 252,
    "Weekly": 52,
    "Monthly": 12,
    "Quarterly": 4,
    "Annual": 1,
}

periods_per_year = frequency_map[frequency_choice]

metadata = AssetMetadata(
    dataset_name=dataset_name.strip() or "Undeclared",
    source=source.strip() or "Undeclared",
    asset_identifier=asset_identifier.strip() or "Undeclared",
    asset_class=asset_class,
    venue=venue.strip() or "Undeclared",
    currency=currency.strip() or "Undeclared",
    timezone=timezone.strip() or "Undeclared",
    frequency=frequency_choice,
    adjustment_status=adjustment_status,
    license_note=license_note.strip() or "Undeclared",
)

evidence = analyze_empirical_prices(
    uploaded_data,
    date_column=date_column,
    price_column=price_column,
    metadata=metadata,
    periods_per_year=periods_per_year,
)

st.subheader("Executive evidence summary")

summary_one, summary_two, summary_three, summary_four = st.columns(4)

summary_one.metric(
    "Validation state",
    "Passed" if evidence.is_valid else "Blocked",
)

summary_two.metric(
    "Validated observations",
    str(evidence.validation.valid_rows),
)

summary_three.metric(
    "Memory conclusion",
    "Not tested",
)

summary_four.metric(
    "Trading conclusion",
    "Abstain",
)

if evidence.errors:
    st.error("Analysis blocked because the dataset failed validation.")

    for error in evidence.errors:
        st.write(f"- {error}")

if evidence.warnings:
    for warning in evidence.warnings:
        st.warning(warning)

health = pd.DataFrame(
    [
        {
            "Check": "Input rows",
            "Result": evidence.validation.input_rows,
            "State": "Information",
        },
        {
            "Check": "Invalid dates",
            "Result": evidence.validation.invalid_dates,
            "State": (
                "Passed"
                if evidence.validation.invalid_dates == 0
                else "Blocked"
            ),
        },
        {
            "Check": "Invalid prices",
            "Result": evidence.validation.invalid_prices,
            "State": (
                "Passed"
                if evidence.validation.invalid_prices == 0
                else "Blocked"
            ),
        },
        {
            "Check": "Duplicate dates",
            "Result": evidence.validation.duplicate_dates,
            "State": (
                "Passed"
                if evidence.validation.duplicate_dates == 0
                else "Blocked"
            ),
        },
        {
            "Check": "Nonpositive prices",
            "Result": evidence.validation.nonpositive_prices,
            "State": (
                "Passed"
                if evidence.validation.nonpositive_prices == 0
                else "Blocked"
            ),
        },
        {
            "Check": "Originally chronological",
            "Result": evidence.validation.originally_sorted,
            "State": (
                "Passed"
                if evidence.validation.originally_sorted
                else "Corrected"
            ),
        },
    ]
)

health["Result"] = health["Result"].astype("string")

st.dataframe(
    health,
    hide_index=True,
    width="stretch",
)

if not evidence.is_valid:
    st.stop()

st.subheader("Verified descriptive metrics")

metric_one, metric_two, metric_three, metric_four = st.columns(4)

metric_one.metric(
    "Cumulative return",
    f"{100.0 * evidence.cumulative_return:.2f}%",
)

metric_two.metric(
    "Maximum drawdown",
    f"{100.0 * evidence.maximum_drawdown:.2f}%",
)

if evidence.annualized_volatility is None:
    metric_three.metric(
        "Annualized volatility",
        "Unavailable",
    )
else:
    metric_three.metric(
        "Annualized volatility",
        f"{100.0 * evidence.annualized_volatility:.2f}%",
    )

if evidence.arithmetic_annualized_return is None:
    metric_four.metric(
        "Arithmetic annualized return",
        "Unavailable",
    )
else:
    metric_four.metric(
        "Arithmetic annualized return",
        f"{100.0 * evidence.arithmetic_annualized_return:.2f}%",
    )

price_tab, return_tab, risk_tab, provenance_tab = st.tabs(
    [
        "Price",
        "Returns",
        "Risk",
        "Provenance",
    ]
)

with price_tab:
    price_frame = evidence.canonical_data.set_index("date")[["price"]]
    st.line_chart(
        price_frame,
        height=360,
        color="#39d0d8",
    )

    with st.expander("Explain this chart"):
        st.markdown(
            """
            **Plain English**

            The chart shows the validated price values in chronological
            order.

            **Research limitation**

            A rising or falling price chart does not establish memory,
            efficiency, predictability, causation, or investability.
            """
        )

with return_tab:
    return_frame = pd.DataFrame(
        {
            "Simple return": evidence.simple_return_series,
            "Log return": evidence.log_return_series,
        }
    )

    st.line_chart(
        return_frame,
        height=360,
    )

    with st.expander("Explain returns"):
        st.latex(r"r_t = \frac{P_t}{P_{t-1}} - 1")
        st.latex(r"g_t = \log\left(\frac{P_t}{P_{t-1}}\right)")

        st.markdown(
            """
            Simple and logarithmic returns are different mathematical
            objects and are not silently treated as interchangeable.
            """
        )

with risk_tab:
    wealth_frame = pd.DataFrame(
        {
            "Wealth index": evidence.wealth_series,
        }
    )

    st.line_chart(
        wealth_frame,
        height=300,
        color="#58a6ff",
    )

    st.markdown(
        f"""
        **Maximum drawdown:**  
        `{100.0 * evidence.maximum_drawdown:.4f}%`

        **Plain English:**  
        Maximum drawdown is the largest observed decline from a prior
        wealth peak within the selected sample.

        **Limitation:**  
        Historical drawdown does not define the largest possible future
        loss.
        """
    )

with provenance_tab:
    provenance = pd.DataFrame(
        [
            {
                "Field": "Dataset",
                "Value": metadata.dataset_name,
            },
            {
                "Field": "Source",
                "Value": metadata.source,
            },
            {
                "Field": "Asset identifier",
                "Value": metadata.asset_identifier,
            },
            {
                "Field": "Asset class",
                "Value": metadata.asset_class,
            },
            {
                "Field": "Venue",
                "Value": metadata.venue,
            },
            {
                "Field": "Currency",
                "Value": metadata.currency,
            },
            {
                "Field": "Timezone",
                "Value": metadata.timezone,
            },
            {
                "Field": "Frequency",
                "Value": metadata.frequency,
            },
            {
                "Field": "Adjustment",
                "Value": metadata.adjustment_status,
            },
            {
                "Field": "First observation",
                "Value": evidence.validation.first_observation,
            },
            {
                "Field": "Last observation",
                "Value": evidence.validation.last_observation,
            },
            {
                "Field": "Canonical SHA-256",
                "Value": evidence.validation.canonical_sha256,
            },
        ]
    )

    st.dataframe(
        provenance,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download reproducibility manifest",
        data=evidence.manifest_json(),
        file_name="market_memory_lab_manifest.json",
        mime="application/json",
    )

st.markdown("---")

st.info(
    "Next scientific gate: challenge empirical statistics against "
    "competing Gaussian, heavy-tail, short-memory, volatility-persistent, "
    "and structural-break mechanisms."
)

if not math.isfinite(evidence.cumulative_return):
    st.error("A nonfinite empirical metric was detected.")
