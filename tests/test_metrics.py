"""Reference and property tests for core financial metrics."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from market_memory_lab.metrics import (
    annualized_sharpe,
    annualized_volatility,
    arithmetic_annualized_return,
    cumulative_return,
    drawdown_series,
    log_returns,
    maximum_drawdown,
    simple_returns,
    validate_prices,
    wealth_index,
)


def test_simple_returns_known_path() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    actual = simple_returns(prices).to_numpy()
    expected = np.array([0.10, -0.10])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_log_returns_telescope_to_total_log_ratio() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    actual = float(log_returns(prices).sum())
    expected = math.log(99.0 / 100.0)

    assert actual == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


def test_cumulative_return_compounds_instead_of_adding() -> None:
    returns = pd.Series([0.10, -0.10])

    assert cumulative_return(returns) == pytest.approx(-0.01)


def test_wealth_index_known_values() -> None:
    returns = pd.Series([0.10, -0.10])
    actual = wealth_index(
        returns,
        initial_wealth=100.0,
    ).to_numpy()
    expected = np.array([110.0, 99.0])

    np.testing.assert_allclose(actual, expected)


def test_arithmetic_annualized_return_matches_definition() -> None:
    returns = pd.Series([0.01, 0.02, -0.01])
    expected = 12 * returns.mean()

    actual = arithmetic_annualized_return(
        returns,
        periods_per_year=12,
    )

    assert actual == pytest.approx(expected)


def test_annualized_volatility_matches_sample_definition() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    expected = returns.std(ddof=1) * math.sqrt(12)

    actual = annualized_volatility(
        returns,
        periods_per_year=12,
    )

    assert actual == pytest.approx(expected)


def test_sharpe_matches_explicit_periodic_definition() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    risk_free = pd.Series([0.001, 0.001, 0.001, 0.001])
    excess = returns - risk_free

    expected = (
        math.sqrt(12)
        * excess.mean()
        / excess.std(ddof=1)
    )

    actual = annualized_sharpe(
        returns,
        risk_free,
        periods_per_year=12,
    )

    assert actual == pytest.approx(expected)


def test_drawdown_measures_first_period_loss_from_initial_wealth() -> None:
    returns = pd.Series([-0.20, 0.10])
    actual = drawdown_series(returns).to_numpy()
    expected = np.array([-0.20, -0.12])

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
    )


def test_maximum_drawdown_known_path() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    assert maximum_drawdown(returns) == pytest.approx(-0.20)


def test_total_loss_is_valid_simple_return() -> None:
    returns = pd.Series([0.10, -1.00, 0.50])

    assert cumulative_return(returns) == pytest.approx(-1.00)
    assert maximum_drawdown(returns) == pytest.approx(-1.00)


def test_prices_must_be_strictly_positive() -> None:
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        validate_prices([100.0, 0.0])


def test_returns_cannot_be_below_minus_one() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be less than -1",
    ):
        cumulative_return([0.10, -1.01])


def test_sharpe_rejects_zero_excess_volatility() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    risk_free = pd.Series([0.00, 0.00, 0.00])

    with pytest.raises(
        ValueError,
        match="undefined",
    ):
        annualized_sharpe(
            returns,
            risk_free,
            periods_per_year=252,
        )


def test_frequency_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        annualized_volatility(
            [0.01, -0.01],
            periods_per_year=0,
        )


@given(
    st.lists(
        st.floats(
            min_value=-0.95,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=50,
    )
)
def test_cumulative_return_equals_final_wealth_minus_one(
    values: list[float],
) -> None:
    returns = pd.Series(values, dtype="float64")

    actual = cumulative_return(returns)
    expected = float((1.0 + returns).prod() - 1.0)

    assert actual == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


@given(
    st.lists(
        st.floats(
            min_value=-0.95,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=50,
    )
)
def test_drawdown_is_never_positive(
    values: list[float],
) -> None:
    drawdowns = drawdown_series(values)

    assert bool((drawdowns <= 1e-12).all())
    assert bool((drawdowns >= -1.0 - 1e-12).all())
