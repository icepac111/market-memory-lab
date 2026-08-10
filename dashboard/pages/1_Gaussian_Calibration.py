"""Monte Carlo calibration page for the verified IID Gaussian null."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from market_memory_lab.calibration import (
    central_interval,
    empirical_percentile,
    gaussian_null_calibration,
)
from market_memory_lab.synthetic import iid_gaussian

st.set_page_config(
    page_title="Gaussian Calibration | Market Memory Lab",
    page_icon="M",
    layout="wide",
)

st.title("Gaussian Monte Carlo Calibration")
st.caption(
    "Compare one observed synthetic path against repeated samples "
    "generated under the same known no-memory mechanism."
)

with st.sidebar:
    st.header("Calibration controls")

    sample_size = st.slider(
        "Observations per path",
        min_value=100,
        max_value=20_000,
        value=5_000,
        step=100,
    )

    replications = st.slider(
        "Monte Carlo replications",
        min_value=100,
        max_value=5_000,
        value=1_000,
        step=100,
    )

    observed_seed = st.number_input(
        "Observed-path seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
    )

    calibration_seed = st.number_input(
        "Calibration seed",
        min_value=0,
        max_value=2**32 - 1,
        value=20260809,
        step=1,
    )

    population_mean = st.number_input(
        "Population mean",
        value=0.0,
        format="%.4f",
    )

    population_sigma = st.number_input(
        "Population standard deviation",
        min_value=0.0001,
        value=1.0,
        format="%.4f",
    )

observed = iid_gaussian(
    n=int(sample_size),
    seed=int(observed_seed),
    mean=float(population_mean),
    sigma=float(population_sigma),
)

calibration = gaussian_null_calibration(
    sample_size=int(sample_size),
    replications=int(replications),
    seed=int(calibration_seed),
    population_mean=float(population_mean),
    population_sigma=float(population_sigma),
    batch_size=min(100, int(replications)),
)

values = observed.values.to_numpy()

observed_mean = float(np.mean(values))
observed_variance = float(np.var(values, ddof=1))

mean_standard_error = (
    float(population_sigma) / math.sqrt(sample_size)
)
variance_standard_error = (
    float(population_sigma) ** 2
    * math.sqrt(2.0 / (sample_size - 1))
)

observed_mean_z = (
    observed_mean - float(population_mean)
) / mean_standard_error

observed_variance_z = (
    observed_variance - float(population_sigma) ** 2
) / variance_standard_error

previous = values[:-1]
current = values[1:]

observed_lag_one = float(
    np.corrcoef(previous, current)[0, 1]
)

mean_percentile = empirical_percentile(
    calibration.sample_mean_z,
    observed_mean_z,
)
variance_percentile = empirical_percentile(
    calibration.sample_variance_z,
    observed_variance_z,
)
lag_percentile = empirical_percentile(
    calibration.lag_one_correlation,
    observed_lag_one,
)

mean_interval = central_interval(
    calibration.sample_mean_z
)
variance_interval = central_interval(
    calibration.sample_variance_z
)
lag_interval = central_interval(
    calibration.lag_one_correlation
)

st.info(
    "Ground truth: every path is generated from independent Gaussian "
    "innovations. Any observed nonzero sample statistic may arise through "
    "finite-sample variation."
)

first, second, third = st.columns(3)

first.metric(
    "Mean deviation",
    f"{observed_mean_z:.3f} SE",
)
first.caption(
    f"Monte Carlo percentile: {100.0 * mean_percentile:.1f}%"
)
first.caption(
    "Central 95% null interval: "
    f"[{mean_interval[0]:.3f}, {mean_interval[1]:.3f}]"
)

second.metric(
    "Variance deviation",
    f"{observed_variance_z:.3f} scaled units",
)
second.caption(
    f"Monte Carlo percentile: {100.0 * variance_percentile:.1f}%"
)
second.caption(
    "Central 95% null interval: "
    f"[{variance_interval[0]:.3f}, {variance_interval[1]:.3f}]"
)

third.metric(
    "Lag-1 sample correlation",
    f"{observed_lag_one:.5f}",
)
third.caption(
    f"Monte Carlo percentile: {100.0 * lag_percentile:.1f}%"
)
third.caption(
    "Central 95% null interval: "
    f"[{lag_interval[0]:.5f}, {lag_interval[1]:.5f}]"
)

st.subheader("Observed path")

path_frame = pd.DataFrame(
    {
        "Observation": np.arange(sample_size),
        "Synthetic return": values,
    }
).set_index("Observation")

st.line_chart(
    path_frame,
    height=320,
    color="#39d0d8",
)

st.subheader("Null reference distributions")

selected_statistic = st.selectbox(
    "Statistic",
    options=[
        "Mean deviation",
        "Variance deviation",
        "Lag-1 sample correlation",
    ],
)

if selected_statistic == "Mean deviation":
    reference = calibration.sample_mean_z
    observed_value = observed_mean_z
elif selected_statistic == "Variance deviation":
    reference = calibration.sample_variance_z
    observed_value = observed_variance_z
else:
    reference = calibration.lag_one_correlation
    observed_value = observed_lag_one

counts, edges = np.histogram(
    reference,
    bins=40,
)

histogram = pd.DataFrame(
    {
        "Bin midpoint": np.round(
            (edges[:-1] + edges[1:]) / 2.0,
            4,
        ),
        "Monte Carlo count": counts,
    }
).set_index("Bin midpoint")

st.bar_chart(
    histogram,
    height=320,
    color="#58a6ff",
)

st.write(
    f"Observed statistic: **{observed_value:.6f}**"
)

st.warning(
    "A Monte Carlo percentile is descriptive. It is not automatically "
    "a preregistered significance decision, evidence of market memory, "
    "or a trading signal."
)

st.subheader("Computation contract")

contract = pd.DataFrame(
    [
        {
            "Property": "Mechanism",
            "Value": "IID Gaussian innovations",
        },
        {
            "Property": "Sample size",
            "Value": str(sample_size),
        },
        {
            "Property": "Replications",
            "Value": str(replications),
        },
        {
            "Property": "Observed seed",
            "Value": str(int(observed_seed)),
        },
        {
            "Property": "Calibration seed",
            "Value": str(int(calibration_seed)),
        },
        {
            "Property": "Time complexity",
            "Value": "O(replications x sample size)",
        },
        {
            "Property": "Peak simulation memory",
            "Value": "O(batch size x sample size + replications)",
        },
        {
            "Property": "Conclusion",
            "Value": "Synthetic calibration only",
        },
    ]
)

st.dataframe(
    contract,
    hide_index=True,
    width="stretch",
)
