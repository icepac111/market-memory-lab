"""Interactive pairwise structural-similarity research laboratory."""

from __future__ import annotations

import json
import math

import pandas as pd
import streamlit as st

from market_memory_lab.empirical import (
    AssetMetadata,
    EmpiricalEvidence,
    analyze_empirical_prices,
)
from market_memory_lab.similarity import (
    SimilarityEvidence,
    compare_validated_assets,
)

st.set_page_config(
    page_title="Pairwise Similarity Lab | Market Memory Lab",
    page_icon="M",
    layout="wide",
)


FREQUENCIES = {
    "Undeclared": None,
    "Daily": 252,
    "Weekly": 52,
    "Monthly": 12,
    "Quarterly": 4,
    "Annual": 1,
}


def read_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV using narrowly scoped failure handling."""
    try:
        return pd.read_csv(uploaded_file)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeError,
        OSError,
    ) as error:
        st.error(f"CSV parsing failed: {error}")
        st.stop()


def metadata_form(
    *,
    prefix: str,
    default_identifier: str,
) -> AssetMetadata:
    """Collect declared provenance for one instrument."""
    st.markdown(f"#### Instrument {prefix}")

    identifier = st.text_input(
        f"{prefix}: asset identifier",
        value=default_identifier,
        key=f"{prefix}_identifier",
    )

    dataset_name = st.text_input(
        f"{prefix}: dataset name",
        value=f"Dataset {default_identifier}",
        key=f"{prefix}_dataset",
    )

    source = st.text_input(
        f"{prefix}: source",
        value="User-supplied CSV",
        key=f"{prefix}_source",
    )

    asset_class = st.selectbox(
        f"{prefix}: asset class",
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
        key=f"{prefix}_class",
    )

    venue = st.text_input(
        f"{prefix}: venue or market",
        value="Undeclared",
        key=f"{prefix}_venue",
    )

    currency = st.text_input(
        f"{prefix}: currency",
        value="USD",
        key=f"{prefix}_currency",
    )

    timezone = st.text_input(
        f"{prefix}: timezone",
        value="UTC",
        key=f"{prefix}_timezone",
    )

    adjustment = st.selectbox(
        f"{prefix}: price adjustment",
        [
            "Adjusted close",
            "Unadjusted close",
            "Unknown",
        ],
        key=f"{prefix}_adjustment",
    )

    frequency = st.selectbox(
        f"{prefix}: observation frequency",
        list(FREQUENCIES),
        index=1,
        key=f"{prefix}_frequency",
    )

    license_note = st.text_input(
        f"{prefix}: license or access note",
        value="User is responsible for data rights",
        key=f"{prefix}_license",
    )

    return AssetMetadata(
        dataset_name=dataset_name.strip() or "Undeclared",
        source=source.strip() or "Undeclared",
        asset_identifier=identifier.strip() or "Undeclared",
        asset_class=asset_class,
        venue=venue.strip() or "Undeclared",
        currency=currency.strip() or "Undeclared",
        timezone=timezone.strip() or "Undeclared",
        frequency=frequency,
        adjustment_status=adjustment,
        license_note=license_note.strip() or "Undeclared",
    )


def build_evidence(
    *,
    uploaded_file,
    metadata: AssetMetadata,
    prefix: str,
) -> EmpiricalEvidence:
    """Map and validate one uploaded empirical dataset."""
    data = read_csv(uploaded_file)

    if data.empty:
        st.error(f"Instrument {prefix} contains no rows.")
        st.stop()

    st.markdown(f"#### Instrument {prefix} column mapping")

    first, second = st.columns(2)

    with first:
        date_column = st.selectbox(
            f"{prefix}: date column",
            list(data.columns),
            key=f"{prefix}_date_column",
        )

    columns = list(data.columns)

    preferred_index = (
        columns.index("Adjusted Close")
        if "Adjusted Close" in columns
        else min(1, len(columns) - 1)
    )

    with second:
        price_column = st.selectbox(
            f"{prefix}: price column",
            columns,
            index=preferred_index,
            key=f"{prefix}_price_column",
        )

    periods_per_year = FREQUENCIES[metadata.frequency]

    return analyze_empirical_prices(
        data,
        date_column=date_column,
        price_column=price_column,
        metadata=metadata,
        periods_per_year=periods_per_year,
    )


def finite_text(
    value: float | None,
    *,
    digits: int = 4,
) -> str:
    """Render optional finite values without creating false precision."""
    if value is None or not math.isfinite(value):
        return "Unavailable"

    return f"{value:.{digits}f}"


def percentage_text(
    value: float | None,
    *,
    digits: int = 2,
) -> str:
    """Render an optional decimal as a percentage."""
    if value is None or not math.isfinite(value):
        return "Unavailable"

    return f"{100.0 * value:.{digits}f}%"


def evidence_manifest(
    first: EmpiricalEvidence,
    second: EmpiricalEvidence,
    similarity: SimilarityEvidence,
) -> str:
    """Create one downloadable JSON evidence package."""
    payload = {
        "instrument_a": first.manifest(),
        "instrument_b": second.manifest(),
        "pairwise_similarity": similarity.manifest(),
        "interpretation_contract": {
            "composite_similarity_score": "not_produced",
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


st.title("Pairwise Structural Similarity Lab")

st.caption(
    "Compare two financial instruments across separate evidence "
    "dimensions without collapsing disagreement into one magical score."
)

st.info(
    "Returns are calculated independently for each instrument and only "
    "then aligned on common return dates. This avoids generating returns "
    "across mismatched market observations."
)

with st.sidebar:
    st.header("Pairwise evidence intake")

    uploaded_a = st.file_uploader(
        "Upload Instrument A CSV",
        type=["csv"],
        key="upload_a",
    )

    uploaded_b = st.file_uploader(
        "Upload Instrument B CSV",
        type=["csv"],
        key="upload_b",
    )

    minimum_overlap = st.number_input(
        "Minimum overlapping returns",
        min_value=3,
        max_value=100_000,
        value=6,
        step=1,
    )

    st.markdown("---")

    metadata_a = metadata_form(
        prefix="A",
        default_identifier="INSTRUMENT-A",
    )

    st.markdown("---")

    metadata_b = metadata_form(
        prefix="B",
        default_identifier="INSTRUMENT-B",
    )

if uploaded_a is None or uploaded_b is None:
    st.subheader("The analyst desk is ready")

    first, second, third = st.columns(3)

    first.metric(
        "Instrument A",
        "Waiting for data",
    )

    second.metric(
        "Instrument B",
        "Waiting for data",
    )

    third.metric(
        "Trading conclusion",
        "Abstain",
    )

    st.markdown(
        """
        ### What this laboratory compares

        **Surface similarity**

        - Pearson return correlation
        - Spearman rank correlation

        **Risk similarity**

        - Sample volatility
        - Symmetric volatility ratio
        - Maximum drawdown
        - Drawdown difference

        **Distribution similarity**

        - Wasserstein distance after removing separate mean and scale

        **Relationship stability**

        - First-half correlation
        - Second-half correlation
        - Absolute stability gap

        ### What remains unavailable

        - Market-memory similarity
        - Regime similarity
        - Lead-lag direction
        - Causation
        - Buy, sell, or short recommendation
        """
    )

    st.stop()

st.subheader("Independent validation")

mapping_a, mapping_b = st.columns(2)

with mapping_a:
    evidence_a = build_evidence(
        uploaded_file=uploaded_a,
        metadata=metadata_a,
        prefix="A",
    )

with mapping_b:
    evidence_b = build_evidence(
        uploaded_file=uploaded_b,
        metadata=metadata_b,
        prefix="B",
    )

validation_table = pd.DataFrame(
    [
        {
            "Instrument": evidence_a.metadata.asset_identifier,
            "Validation": (
                "Passed" if evidence_a.is_valid else "Blocked"
            ),
            "Valid observations": str(
                evidence_a.validation.valid_rows
            ),
            "Dataset SHA-256": (
                evidence_a.validation.canonical_sha256
                or "Unavailable"
            ),
        },
        {
            "Instrument": evidence_b.metadata.asset_identifier,
            "Validation": (
                "Passed" if evidence_b.is_valid else "Blocked"
            ),
            "Valid observations": str(
                evidence_b.validation.valid_rows
            ),
            "Dataset SHA-256": (
                evidence_b.validation.canonical_sha256
                or "Unavailable"
            ),
        },
    ]
)

st.dataframe(
    validation_table,
    hide_index=True,
    width="stretch",
)

for label, evidence in (
    ("Instrument A", evidence_a),
    ("Instrument B", evidence_b),
):
    for error in evidence.errors:
        st.error(f"{label}: {error}")

    for warning in evidence.warnings:
        st.warning(f"{label}: {warning}")

similarity = compare_validated_assets(
    evidence_a,
    evidence_b,
    minimum_overlap=int(minimum_overlap),
)

if similarity.errors:
    st.error("Pairwise analysis is blocked.")

    for error in similarity.errors:
        st.write(f"- {error}")

    st.stop()

st.subheader("Executive evidence summary")

executive_one, executive_two, executive_three, executive_four = st.columns(4)

executive_one.metric(
    "Overlapping returns",
    str(similarity.overlapping_returns),
)

executive_two.metric(
    "Pearson correlation",
    finite_text(similarity.pearson_correlation),
)

executive_three.metric(
    "Stability gap",
    finite_text(similarity.correlation_stability_gap),
)

executive_four.metric(
    "Trading conclusion",
    "Abstain",
)

for warning in similarity.warnings:
    st.warning(warning)

surface_tab, risk_tab, distribution_tab, stability_tab, audit_tab = st.tabs(
    [
        "Surface Similarity",
        "Risk Similarity",
        "Distribution",
        "Relationship Stability",
        "Audit",
    ]
)

with surface_tab:
    surface_one, surface_two = st.columns(2)

    surface_one.metric(
        "Pearson return correlation",
        finite_text(similarity.pearson_correlation),
    )

    surface_two.metric(
        "Spearman rank correlation",
        finite_text(similarity.spearman_correlation),
    )

    aligned_returns = pd.concat(
        [
            evidence_a.simple_return_series.rename(
                evidence_a.metadata.asset_identifier
            ),
            evidence_b.simple_return_series.rename(
                evidence_b.metadata.asset_identifier
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    normalized_wealth = (1.0 + aligned_returns).cumprod()

    st.line_chart(
        normalized_wealth,
        height=360,
    )

    with st.expander("Executive explanation"):
        st.markdown(
            """
            **Pearson correlation** measures linear co-movement in aligned
            returns.

            **Spearman correlation** compares the rank ordering of returns
            and can capture monotonic relationships that are not perfectly
            linear.

            Neither statistic establishes causation, lead-lag direction,
            stability, or profitability.
            """
        )

with risk_tab:
    risk_one, risk_two, risk_three = st.columns(3)

    risk_one.metric(
        f"{similarity.asset_a} sample volatility",
        finite_text(similarity.volatility_a, digits=6),
    )

    risk_two.metric(
        f"{similarity.asset_b} sample volatility",
        finite_text(similarity.volatility_b, digits=6),
    )

    risk_three.metric(
        "Symmetric volatility ratio",
        finite_text(similarity.volatility_ratio),
    )

    drawdown_table = pd.DataFrame(
        [
            {
                "Instrument": similarity.asset_a,
                "Maximum drawdown": percentage_text(
                    similarity.maximum_drawdown_a
                ),
            },
            {
                "Instrument": similarity.asset_b,
                "Maximum drawdown": percentage_text(
                    similarity.maximum_drawdown_b
                ),
            },
            {
                "Instrument": "Absolute difference",
                "Maximum drawdown": percentage_text(
                    similarity.drawdown_difference
                ),
            },
        ]
    )

    st.dataframe(
        drawdown_table,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "A volatility ratio near 1 indicates similar sample volatility. "
        "It is not a probability and has no universal investment threshold."
    )

with distribution_tab:
    st.metric(
        "Standardized Wasserstein distance",
        finite_text(
            similarity.standardized_wasserstein_distance,
            digits=6,
        ),
    )

    st.markdown(
        """
        Each return distribution is standardized separately before the
        distance is calculated.

        A smaller value indicates closer distributional shape after
        removing each series' own mean and scale.

        The distance is not bounded above, is not a probability, and does
        not establish that the instruments share an economic mechanism.
        """
    )

with stability_tab:
    stability_one, stability_two, stability_three = st.columns(3)

    stability_one.metric(
        "First-half correlation",
        finite_text(similarity.first_half_correlation),
    )

    stability_two.metric(
        "Second-half correlation",
        finite_text(similarity.second_half_correlation),
    )

    stability_three.metric(
        "Absolute stability gap",
        finite_text(similarity.correlation_stability_gap),
    )

    st.markdown(
        """
        **Plain English**

        The overlapping return sample is split into two nonoverlapping
        halves. Correlation is calculated separately in each half.

        A large gap means the observed relationship changed materially
        across the selected sample.

        **Research limitation**

        A two-half comparison is a diagnostic, not a formal structural-break
        test and not proof of a market regime change.
        """
    )

with audit_tab:
    compatibility = pd.DataFrame(
        [
            {
                "Field": "Currency",
                "Instrument A": evidence_a.metadata.currency,
                "Instrument B": evidence_b.metadata.currency,
            },
            {
                "Field": "Timezone",
                "Instrument A": evidence_a.metadata.timezone,
                "Instrument B": evidence_b.metadata.timezone,
            },
            {
                "Field": "Frequency",
                "Instrument A": evidence_a.metadata.frequency,
                "Instrument B": evidence_b.metadata.frequency,
            },
            {
                "Field": "Adjustment status",
                "Instrument A": (
                    evidence_a.metadata.adjustment_status
                ),
                "Instrument B": (
                    evidence_b.metadata.adjustment_status
                ),
            },
            {
                "Field": "Overlap start",
                "Instrument A": similarity.overlap_start,
                "Instrument B": similarity.overlap_start,
            },
            {
                "Field": "Overlap end",
                "Instrument A": similarity.overlap_end,
                "Instrument B": similarity.overlap_end,
            },
        ]
    )

    st.dataframe(
        compatibility,
        hide_index=True,
        width="stretch",
    )

    manifest_json = evidence_manifest(
        evidence_a,
        evidence_b,
        similarity,
    )

    st.download_button(
        "Download pairwise evidence manifest",
        data=manifest_json,
        file_name="pairwise_similarity_manifest.json",
        mime="application/json",
    )

st.markdown("---")

status_table = pd.DataFrame(
    [
        {
            "Question": "Surface return similarity",
            "State": "Calculated",
        },
        {
            "Question": "Risk similarity",
            "State": "Calculated",
        },
        {
            "Question": "Distributional similarity",
            "State": "Calculated",
        },
        {
            "Question": "Relationship stability",
            "State": "Descriptive diagnostic",
        },
        {
            "Question": "Memory similarity",
            "State": "Not tested",
        },
        {
            "Question": "Regime similarity",
            "State": "Not tested",
        },
        {
            "Question": "Lead-lag direction",
            "State": "Not tested",
        },
        {
            "Question": "Causation",
            "State": "Not established",
        },
        {
            "Question": "Trading decision",
            "State": "Abstain",
        },
    ]
)

st.dataframe(
    status_table,
    hide_index=True,
    width="stretch",
)

st.info(
    "Next gate: compare an instrument against a validated universe, "
    "preserve multiple-testing controls, and identify false friends."
)
