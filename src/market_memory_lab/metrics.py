"""Core financial metrics with explicit assumptions and validation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _as_finite_series(
    values: Iterable[float] | pd.Series,
    *,
    name: str,
    minimum_length: int = 1,
) -> pd.Series:
    """
    Convert input to a finite float64 Series.

    Missing values are removed explicitly. Infinite values are rejected.
    """
    series = pd.Series(values, dtype="float64").dropna()

    if len(series) < minimum_length:
        raise ValueError(
            f"{name} requires at least {minimum_length} valid observations"
        )

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")

    return series


def validate_prices(
    prices: Iterable[float] | pd.Series,
) -> pd.Series:
    """
    Validate strictly positive prices.

    Positive prices are required for both simple and logarithmic returns.
    """
    series = _as_finite_series(
        prices,
        name="prices",
        minimum_length=2,
    )

    if (series <= 0.0).any():
        raise ValueError("prices must be strictly positive")

    return series


def validate_simple_returns(
    returns: Iterable[float] | pd.Series,
    *,
    minimum_length: int = 1,
) -> pd.Series:
    """
    Validate simple returns.

    A simple return less than -1 is impossible. A return of exactly -1
    represents complete loss and is allowed for cumulative wealth, but
    logarithmic transformations of such a return are undefined.
    """
    series = _as_finite_series(
        returns,
        name="returns",
        minimum_length=minimum_length,
    )

    if (series < -1.0).any():
        raise ValueError("simple returns cannot be less than -1")

    return series


def simple_returns(
    prices: Iterable[float] | pd.Series,
) -> pd.Series:
    """
    Compute simple returns:

        r_t = P_t / P_(t-1) - 1

    Missing prices are removed before calculation. Callers must therefore
    verify that dropping missing timestamps is appropriate for the data.
    """
    series = validate_prices(prices)
    result = series.pct_change(fill_method=None).dropna()
    result.name = "simple_return"
    return result


def log_returns(
    prices: Iterable[float] | pd.Series,
) -> pd.Series:
    """
    Compute logarithmic returns:

        g_t = ln(P_t / P_(t-1))
    """
    series = validate_prices(prices)
    result = np.log(series / series.shift(1)).dropna()
    result.name = "log_return"
    return result


def wealth_index(
    returns: Iterable[float] | pd.Series,
    *,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """
    Compute the wealth path from simple returns:

        W_t = W_0 * product_{i=1}^{t}(1 + r_i)
    """
    if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
        raise ValueError("initial_wealth must be finite and strictly positive")

    series = validate_simple_returns(returns)
    wealth = initial_wealth * (1.0 + series).cumprod()
    wealth.name = "wealth_index"
    return wealth


def cumulative_return(
    returns: Iterable[float] | pd.Series,
) -> float:
    """
    Compute compounded cumulative simple return:

        R_(1,T) = product_{t=1}^{T}(1 + r_t) - 1
    """
    wealth = wealth_index(returns, initial_wealth=1.0)
    return float(wealth.iloc[-1] - 1.0)


def arithmetic_annualized_return(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: int,
) -> float:
    """
    Compute arithmetic annualized mean:

        mu_ann = N * mean(r_t)

    This is not the compound annual growth rate.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")

    series = validate_simple_returns(returns)
    return float(periods_per_year * series.mean())


def annualized_volatility(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: int,
) -> float:
    """
    Compute sample annualized volatility:

        sigma_ann = std(r_t, ddof=1) * sqrt(N)
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")

    series = validate_simple_returns(
        returns,
        minimum_length=2,
    )

    return float(series.std(ddof=1) * math.sqrt(periods_per_year))


def annualized_sharpe(
    returns: Iterable[float] | pd.Series,
    risk_free_returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: int,
) -> float:
    """
    Compute annualized Sharpe ratio from aligned periodic simple returns:

        SR_ann = sqrt(N) * mean(r_t - rf_t) / std(r_t - rf_t, ddof=1)

    Asset and risk-free returns must be expressed at the same frequency.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")

    asset = pd.Series(returns, dtype="float64").rename("asset")
    risk_free = pd.Series(
        risk_free_returns,
        dtype="float64",
    ).rename("risk_free")

    aligned = pd.concat(
        [asset, risk_free],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        raise ValueError(
            "Sharpe ratio requires at least two aligned observations"
        )

    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Sharpe ratio inputs must be finite")

    if (aligned["asset"] < -1.0).any():
        raise ValueError("asset returns cannot be less than -1")

    if (aligned["risk_free"] < -1.0).any():
        raise ValueError("risk-free returns cannot be less than -1")

    excess = aligned["asset"] - aligned["risk_free"]
    denominator = float(excess.std(ddof=1))

    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        raise ValueError(
            "Sharpe ratio is undefined when excess-return volatility is zero"
        )

    return float(
        math.sqrt(periods_per_year)
        * float(excess.mean())
        / denominator
    )


def drawdown_series(
    returns: Iterable[float] | pd.Series,
) -> pd.Series:
    """
    Compute drawdown relative to the running wealth peak:

        D_t = W_t / max_{s <= t}(W_s) - 1

    An initial wealth observation is inserted so that a first-period loss
    is measured relative to starting wealth.
    """
    series = validate_simple_returns(returns)

    wealth = wealth_index(series, initial_wealth=1.0)
    initial_index = "__initial__"

    extended = pd.concat(
        [
            pd.Series([1.0], index=[initial_index], dtype="float64"),
            wealth,
        ]
    )

    running_peak = extended.cummax()
    drawdown = extended / running_peak - 1.0
    result = drawdown.iloc[1:].copy()
    result.index = wealth.index
    result.name = "drawdown"
    return result


def maximum_drawdown(
    returns: Iterable[float] | pd.Series,
) -> float:
    """Return the most negative drawdown observation."""
    return float(drawdown_series(returns).min())
