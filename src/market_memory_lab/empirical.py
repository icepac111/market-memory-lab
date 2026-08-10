"""Validation and evidence preparation for empirical financial series."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_memory_lab.metrics import (
    annualized_volatility,
    arithmetic_annualized_return,
    cumulative_return,
    log_returns,
    maximum_drawdown,
    simple_returns,
    wealth_index,
)


@dataclass(frozen=True)
class AssetMetadata:
    """Declared identity and provenance of one empirical asset series."""

    dataset_name: str
    source: str
    asset_identifier: str
    asset_class: str
    venue: str
    currency: str
    timezone: str
    frequency: str
    adjustment_status: str
    license_note: str


@dataclass(frozen=True)
class ValidationSummary:
    """Auditable validation information for an empirical price series."""

    input_rows: int
    valid_rows: int
    invalid_dates: int
    invalid_prices: int
    duplicate_dates: int
    nonpositive_prices: int
    originally_sorted: bool
    first_observation: str | None
    last_observation: str | None
    canonical_sha256: str | None


@dataclass(frozen=True)
class EmpiricalEvidence:
    """Validated descriptive evidence for one empirical price series."""

    metadata: AssetMetadata
    validation: ValidationSummary
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    canonical_data: pd.DataFrame
    simple_return_series: pd.Series
    log_return_series: pd.Series
    wealth_series: pd.Series
    arithmetic_annualized_return: float | None
    annualized_volatility: float | None
    cumulative_return: float | None
    maximum_drawdown: float | None
    periods_per_year: int | None

    @property
    def is_valid(self) -> bool:
        """Whether the dataset passed all blocking validation rules."""
        return not self.errors

    def manifest(self) -> dict[str, Any]:
        """Return a serializable evidence and reproducibility manifest."""
        return {
            "metadata": asdict(self.metadata),
            "validation": asdict(self.validation),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "analysis": {
                "periods_per_year": self.periods_per_year,
                "arithmetic_annualized_return": (
                    self.arithmetic_annualized_return
                ),
                "annualized_volatility": self.annualized_volatility,
                "cumulative_return": self.cumulative_return,
                "maximum_drawdown": self.maximum_drawdown,
                "return_definition": "simple and logarithmic returns",
                "conclusion_state": (
                    "descriptive_only"
                    if self.is_valid
                    else "invalid_data"
                ),
                "memory_conclusion": "not_tested",
                "trading_conclusion": "abstain",
            },
        }

    def manifest_json(self) -> str:
        """Return the evidence manifest as stable formatted JSON."""
        return json.dumps(
            self.manifest(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )


def _empty_series(name: str) -> pd.Series:
    """Create an empty float64 series with a stable name."""
    return pd.Series(dtype="float64", name=name)


def _empty_canonical() -> pd.DataFrame:
    """Create the canonical empty empirical-data frame."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "price": pd.Series(dtype="float64"),
        }
    )


def _canonical_hash(frame: pd.DataFrame) -> str:
    """Hash canonical CSV bytes using SHA-256."""
    payload = frame.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def analyze_empirical_prices(
    data: pd.DataFrame,
    *,
    date_column: str,
    price_column: str,
    metadata: AssetMetadata,
    periods_per_year: int | None,
) -> EmpiricalEvidence:
    """
    Validate and analyze one univariate empirical price series.

    Blocking requirements:

    - specified columns exist
    - every date parses successfully
    - every price is numeric and finite
    - every price is strictly positive
    - dates are unique
    - at least three observations remain

    Data are sorted chronologically after recording whether the input was
    originally sorted. Missing financial-calendar dates are not imputed.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if periods_per_year is not None:
        if (
            not isinstance(periods_per_year, int)
            or isinstance(periods_per_year, bool)
        ):
            raise TypeError("periods_per_year must be an integer or None")

        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [
        column
        for column in (date_column, price_column)
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(str(column) for column in missing_columns)
        )

        validation = ValidationSummary(
            input_rows=len(data),
            valid_rows=0,
            invalid_dates=0,
            invalid_prices=0,
            duplicate_dates=0,
            nonpositive_prices=0,
            originally_sorted=False,
            first_observation=None,
            last_observation=None,
            canonical_sha256=None,
        )

        return EmpiricalEvidence(
            metadata=metadata,
            validation=validation,
            errors=tuple(errors),
            warnings=tuple(warnings),
            canonical_data=_empty_canonical(),
            simple_return_series=_empty_series("simple_return"),
            log_return_series=_empty_series("log_return"),
            wealth_series=_empty_series("wealth_index"),
            arithmetic_annualized_return=None,
            annualized_volatility=None,
            cumulative_return=None,
            maximum_drawdown=None,
            periods_per_year=periods_per_year,
        )

    selected = data[[date_column, price_column]].copy()

    parsed_dates = pd.to_datetime(
        selected[date_column],
        errors="coerce",
    )

    numeric_prices = pd.to_numeric(
        selected[price_column],
        errors="coerce",
    )

    invalid_dates = int(parsed_dates.isna().sum())

    numeric_values = numeric_prices.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )
    invalid_price_mask = ~np.isfinite(numeric_values)
    invalid_prices = int(np.count_nonzero(invalid_price_mask))

    finite_price_mask = np.isfinite(numeric_values)
    nonpositive_prices = int(
        np.count_nonzero(
            finite_price_mask & (numeric_values <= 0.0)
        )
    )

    valid_date_values = parsed_dates.dropna()
    duplicate_dates = int(valid_date_values.duplicated().sum())

    originally_sorted = bool(
        parsed_dates.dropna().is_monotonic_increasing
    )

    if invalid_dates:
        errors.append(
            f"{invalid_dates} observation(s) contain invalid dates"
        )

    if invalid_prices:
        errors.append(
            f"{invalid_prices} observation(s) contain missing, "
            "nonnumeric, or nonfinite prices"
        )

    if nonpositive_prices:
        errors.append(
            f"{nonpositive_prices} observation(s) contain "
            "nonpositive prices"
        )

    if duplicate_dates:
        errors.append(
            f"{duplicate_dates} duplicate date observation(s) detected"
        )

    if not originally_sorted:
        warnings.append(
            "Input observations were not chronologically sorted; "
            "the canonical series was sorted before analysis"
        )

    canonical = pd.DataFrame(
        {
            "date": parsed_dates,
            "price": numeric_prices.astype("float64"),
        }
    )

    if not errors:
        canonical = canonical.sort_values(
            "date",
            kind="stable",
        ).reset_index(drop=True)

        if len(canonical) < 3:
            errors.append(
                "At least three validated price observations are required"
            )

    if errors:
        first_observation = None
        last_observation = None
        canonical_sha256 = None
        valid_rows = 0
        safe_canonical = _empty_canonical()
    else:
        first_observation = canonical["date"].iloc[0].isoformat()
        last_observation = canonical["date"].iloc[-1].isoformat()
        canonical_sha256 = _canonical_hash(canonical)
        valid_rows = len(canonical)
        safe_canonical = canonical

    validation = ValidationSummary(
        input_rows=len(data),
        valid_rows=valid_rows,
        invalid_dates=invalid_dates,
        invalid_prices=invalid_prices,
        duplicate_dates=duplicate_dates,
        nonpositive_prices=nonpositive_prices,
        originally_sorted=originally_sorted,
        first_observation=first_observation,
        last_observation=last_observation,
        canonical_sha256=canonical_sha256,
    )

    if errors:
        return EmpiricalEvidence(
            metadata=metadata,
            validation=validation,
            errors=tuple(errors),
            warnings=tuple(warnings),
            canonical_data=safe_canonical,
            simple_return_series=_empty_series("simple_return"),
            log_return_series=_empty_series("log_return"),
            wealth_series=_empty_series("wealth_index"),
            arithmetic_annualized_return=None,
            annualized_volatility=None,
            cumulative_return=None,
            maximum_drawdown=None,
            periods_per_year=periods_per_year,
        )

    prices = safe_canonical.set_index("date")["price"]

    simple = simple_returns(prices)
    logarithmic = log_returns(prices)
    wealth = wealth_index(simple)

    annualized_return_value: float | None = None
    annualized_volatility_value: float | None = None

    if periods_per_year is None:
        warnings.append(
            "Annualized metrics are unavailable because "
            "periods_per_year was not declared"
        )
    else:
        annualized_return_value = arithmetic_annualized_return(
            simple,
            periods_per_year=periods_per_year,
        )
        annualized_volatility_value = annualized_volatility(
            simple,
            periods_per_year=periods_per_year,
        )

    return EmpiricalEvidence(
        metadata=metadata,
        validation=validation,
        errors=tuple(errors),
        warnings=tuple(warnings),
        canonical_data=safe_canonical,
        simple_return_series=simple,
        log_return_series=logarithmic,
        wealth_series=wealth,
        arithmetic_annualized_return=annualized_return_value,
        annualized_volatility=annualized_volatility_value,
        cumulative_return=cumulative_return(simple),
        maximum_drawdown=maximum_drawdown(simple),
        periods_per_year=periods_per_year,
    )
