#!/usr/bin/env python3
"""
Market Memory Lab: U.S. 10Y Nominal Treasury vs 10Y TIPS downloader
====================================================================

Purpose
-------
Create a non-destructive, provenance-first raw-data checkpoint for:

Official economic layer
    DGS10  : 10-Year Treasury constant-maturity nominal yield
    DFII10 : 10-Year Treasury inflation-indexed constant-maturity real yield

Optional investable-market context
    IEF    : iShares 7-10 Year Treasury Bond ETF
    TIP    : iShares TIPS Bond ETF

This script DOWNLOADS AND AUDITS DATA ONLY. It does not generate findings,
scientific charts, molecular visuals, investment recommendations, commits,
or pushes.

Safety and evidence rules
-------------------------
* Never overwrites an existing run directory.
* Never fabricates, interpolates, or forward-fills observations.
* Preserves source-native raw downloads where possible.
* Records retrieval timestamps, parameters, environment information, hashes,
  coverage, missingness, and validation outcomes.
* Aligns DGS10 and DFII10 only on common valid observation dates.
* Aligns IEF and TIP only on common valid trading dates.
* Keeps official yield evidence separate from ETF market-proxy evidence.
* Does not call the DGS10-DFII10 spread exact expected inflation.
* Does not treat yield as total return.
* Does not treat IEF or TIP as exact constant-maturity instruments.

Example
-------
python download_us10y_nominal_vs_tips_v0_1.py \
    --project-root /absolute/path/to/Market-Memory-Lab

Optional flags
--------------
--skip-etfs       Download only official FRED series.
--start 2003-01-02
--end 2026-09-21 End is inclusive in the contract; yfinance is queried with
                 the following calendar day because its end boundary is
                 normally exclusive.
--case-relpath cases/us10y_nominal_vs_tips_v0_1
--run-id 20260921T194106Z
--allow-partial   Preserve a run when one source layer fails. The manifest
                  will mark the run incomplete. Default behavior fails.

Dependencies
------------
Required: pandas, requests
Optional ETF layer: yfinance

No package installation is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import socket
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required dependency 'pandas'. Use the existing Market Memory "
        "Lab environment; this script will not install packages."
    ) from exc

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing required dependency 'requests'. Use the existing Market Memory "
        "Lab environment; this script will not install packages."
    ) from exc

SCRIPT_VERSION = "0.1.0"
CASE_VERSION = "us10y_nominal_vs_tips_v0_1"
DEFAULT_START = date(2003, 1, 2)
DEFAULT_CASE_RELPATH = Path("cases") / CASE_VERSION
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
USER_AGENT = (
    "MarketMemoryLab/0.1 evidence-audited-research "
    "(educational research; contact configured locally)"
)

OFFICIAL_SERIES: dict[str, dict[str, str]] = {
    "DGS10": {
        "series_id": "DGS10",
        "series_name": (
            "Market Yield on U.S. Treasury Securities at 10-Year Constant "
            "Maturity, Quoted on an Investment Basis"
        ),
        "instrument": "U.S. 10-Year nominal Treasury constant-maturity yield",
        "country": "United States",
        "currency": "USD",
        "maturity": "10-year constant maturity",
        "yield_definition": "Nominal market yield, quoted on an investment basis",
        "nominal_or_real": "nominal",
        "frequency_requested": "daily",
        "units": "percent",
        "source_organization": "Board of Governors of the Federal Reserve System (US)",
        "source_release": "H.15 Selected Interest Rates",
        "access_provider": "Federal Reserve Bank of St. Louis FRED",
        "series_page": "https://fred.stlouisfed.org/series/DGS10",
        "known_limitations": (
            "Constant-maturity yield series; not a security price, wealth index, "
            "or realized total-return series."
        ),
    },
    "DFII10": {
        "series_id": "DFII10",
        "series_name": (
            "Market Yield on U.S. Treasury Securities at 10-Year Constant "
            "Maturity, Quoted on an Investment Basis, Inflation-Indexed"
        ),
        "instrument": "U.S. 10-Year inflation-indexed Treasury constant-maturity yield",
        "country": "United States",
        "currency": "USD",
        "maturity": "10-year constant maturity",
        "yield_definition": "Inflation-indexed market yield, quoted on an investment basis",
        "nominal_or_real": "real",
        "frequency_requested": "daily",
        "units": "percent",
        "source_organization": "Board of Governors of the Federal Reserve System (US)",
        "source_release": "H.15 Selected Interest Rates",
        "access_provider": "Federal Reserve Bank of St. Louis FRED",
        "series_page": "https://fred.stlouisfed.org/series/DFII10",
        "known_limitations": (
            "Constant-maturity inflation-indexed yield series; not a security price, "
            "wealth index, or realized total-return series."
        ),
    },
}

ETF_SPECS: dict[str, dict[str, str]] = {
    "IEF": {
        "symbol": "IEF",
        "instrument": "iShares 7-10 Year Treasury Bond ETF",
        "role": "Broad investable nominal-Treasury market proxy",
        "currency": "USD",
        "provider": "Yahoo Finance accessed through yfinance",
        "known_limitations": (
            "Not a single 10-year constant-maturity Treasury. Portfolio composition, "
            "duration, fees, distributions, rebalancing, and market pricing matter."
        ),
    },
    "TIP": {
        "symbol": "TIP",
        "instrument": "iShares TIPS Bond ETF",
        "role": "Broad investable TIPS market proxy",
        "currency": "USD",
        "provider": "Yahoo Finance accessed through yfinance",
        "known_limitations": (
            "Not a single 10-year constant-maturity TIPS security. Portfolio composition, "
            "duration, fees, distributions, rebalancing, and market pricing matter."
        ),
    },
}


class EvidenceError(RuntimeError):
    """Raised when a source or validation cannot satisfy the data contract."""


@dataclass(frozen=True)
class DownloadRecord:
    layer: str
    identifier: str
    request_url: str | None
    request_parameters: dict[str, Any]
    retrieved_at_utc: str
    raw_file: str | None
    raw_sha256: str | None
    rows_raw: int | None
    rows_valid: int | None
    earliest_valid_date: str | None
    latest_valid_date: str | None
    missing_value_count: int | None
    status: str
    error_type: str | None = None
    error_message: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date {value!r}; expected YYYY-MM-DD") from exc


def safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "run-id may contain only letters, digits, dot, underscore, and hyphen"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and audit official U.S. 10Y nominal/real yields and optional ETF context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Existing Market Memory Lab repository root. Must already exist.",
    )
    parser.add_argument(
        "--case-relpath",
        type=Path,
        default=DEFAULT_CASE_RELPATH,
        help="Case path relative to the project root.",
    )
    parser.add_argument("--start", type=parse_iso_date, default=DEFAULT_START)
    parser.add_argument(
        "--end",
        type=parse_iso_date,
        default=date.today(),
        help="Inclusive contract end date.",
    )
    parser.add_argument(
        "--run-id",
        type=safe_run_id,
        default=None,
        help="Immutable run identifier. Defaults to current UTC timestamp.",
    )
    parser.add_argument("--skip-etfs", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Preserve and report incomplete source layers instead of failing the run.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def ensure_within(parent: Path, child: Path) -> None:
    parent_r = parent.resolve()
    child_r = child.resolve()
    if child_r != parent_r and parent_r not in child_r.parents:
        raise EvidenceError(f"Unsafe path outside project root: {child_r}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def relative_to_run(path: Path | None, run_root: Path) -> str | None:
    return None if path is None else path.relative_to(run_root).as_posix()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def response_to_file(response: requests.Response, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=128 * 1024):
            if chunk:
                handle.write(chunk)
    os.replace(temporary, path)


def normalize_official_csv(raw_path: Path, series_id: str) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    if frame.empty:
        raise EvidenceError(f"{series_id}: source returned an empty CSV")

    date_candidates = [c for c in frame.columns if c.upper() in {"DATE", "OBSERVATION_DATE"}]
    if not date_candidates:
        raise EvidenceError(f"{series_id}: no date column; columns={list(frame.columns)!r}")
    date_column = date_candidates[0]

    value_candidates = [c for c in frame.columns if c.upper() == series_id.upper()]
    if not value_candidates:
        non_date = [c for c in frame.columns if c != date_column]
        if len(non_date) != 1:
            raise EvidenceError(
                f"{series_id}: cannot identify value column; columns={list(frame.columns)!r}"
            )
        value_column = non_date[0]
    else:
        value_column = value_candidates[0]

    normalized = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(frame[date_column], errors="coerce"),
            series_id: pd.to_numeric(
                frame[value_column].replace({".": None, "": None}), errors="coerce"
            ),
        }
    )
    if normalized["observation_date"].isna().any():
        bad_count = int(normalized["observation_date"].isna().sum())
        raise EvidenceError(f"{series_id}: {bad_count} unparsable observation dates")
    if normalized["observation_date"].duplicated().any():
        dupes = normalized.loc[
            normalized["observation_date"].duplicated(keep=False), "observation_date"
        ].dt.strftime("%Y-%m-%d").tolist()
        raise EvidenceError(f"{series_id}: duplicate dates detected: {dupes[:10]}")

    normalized = normalized.sort_values("observation_date", kind="mergesort").reset_index(drop=True)
    normalized["observation_date"] = normalized["observation_date"].dt.strftime("%Y-%m-%d")
    return normalized


def validate_official_frame(
    frame: pd.DataFrame,
    series_id: str,
    start: date,
    end: date,
) -> dict[str, Any]:
    values = frame[series_id]
    valid = frame.loc[values.notna()].copy()
    if valid.empty:
        raise EvidenceError(f"{series_id}: no numeric observations")

    parsed_dates = pd.to_datetime(valid["observation_date"], errors="raise").dt.date
    outside = (parsed_dates < start) | (parsed_dates > end)
    if bool(outside.any()):
        examples = valid.loc[outside, "observation_date"].head(10).tolist()
        raise EvidenceError(f"{series_id}: source returned dates outside requested range: {examples}")

    nonfinite = ~pd.Series(pd.array(values.dropna(), dtype="Float64")).map(
        lambda x: pd.notna(x) and float("-inf") < float(x) < float("inf")
    )
    if bool(nonfinite.any()):
        raise EvidenceError(f"{series_id}: non-finite numeric values detected")

    return {
        "rows_raw": int(len(frame)),
        "rows_valid": int(values.notna().sum()),
        "missing_value_count": int(values.isna().sum()),
        "earliest_valid_date": str(valid["observation_date"].iloc[0]),
        "latest_valid_date": str(valid["observation_date"].iloc[-1]),
        "is_monotonic_date": bool(frame["observation_date"].is_monotonic_increasing),
        "duplicate_date_count": int(frame["observation_date"].duplicated().sum()),
    }


def download_fred_series(
    session: requests.Session,
    series_id: str,
    start: date,
    end: date,
    timeout_seconds: int,
    raw_dir: Path,
    processed_dir: Path,
    run_root: Path,
) -> tuple[DownloadRecord, pd.DataFrame, dict[str, Any]]:
    params = {
        "id": series_id,
        "cosd": start.isoformat(),
        "coed": end.isoformat(),
    }
    retrieved_at = iso_z(utc_now())
    raw_path = raw_dir / "official" / f"fred_{series_id}_{start}_{end}.csv"
    logging.info("Downloading official series %s", series_id)
    response = session.get(
        FRED_CSV_URL,
        params=params,
        timeout=timeout_seconds,
        stream=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text" not in content_type.lower() and "csv" not in content_type.lower():
        logging.warning("%s returned unusual Content-Type: %s", series_id, content_type)
    response_to_file(response, raw_path)
    if raw_path.stat().st_size == 0:
        raise EvidenceError(f"{series_id}: downloaded file is empty")

    normalized = normalize_official_csv(raw_path, series_id)
    audit = validate_official_frame(normalized, series_id, start, end)
    normalized_path = processed_dir / "official" / f"{series_id}_normalized.csv"
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(normalized_path, index=False, lineterminator="\n", float_format="%.10g")

    record = DownloadRecord(
        layer="official_yield",
        identifier=series_id,
        request_url=response.url,
        request_parameters=params,
        retrieved_at_utc=retrieved_at,
        raw_file=relative_to_run(raw_path, run_root),
        raw_sha256=sha256_file(raw_path),
        rows_raw=audit["rows_raw"],
        rows_valid=audit["rows_valid"],
        earliest_valid_date=audit["earliest_valid_date"],
        latest_valid_date=audit["latest_valid_date"],
        missing_value_count=audit["missing_value_count"],
        status="success",
    )
    audit["normalized_file"] = relative_to_run(normalized_path, run_root)
    audit["normalized_sha256"] = sha256_file(normalized_path)
    audit["response_headers_selected"] = {
        key: response.headers.get(key)
        for key in ("Content-Type", "Content-Length", "ETag", "Last-Modified")
        if response.headers.get(key) is not None
    }
    return record, normalized, audit


def normalize_yfinance_columns(frame: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        raise EvidenceError("yfinance returned an empty dataset")

    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        # yfinance versions may return (Price, Ticker) or (Ticker, Price).
        level0 = set(map(str, output.columns.get_level_values(0)))
        level1 = set(map(str, output.columns.get_level_values(1)))
        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if level0 & price_fields:
            output.columns = [f"{ticker}__{field}" for field, ticker in output.columns]
        elif level1 & price_fields:
            output.columns = [f"{ticker}__{field}" for ticker, field in output.columns]
        else:
            output.columns = ["__".join(map(str, item)) for item in output.columns]
    else:
        # A single symbol can produce non-MultiIndex columns.
        if len(symbols) != 1:
            raise EvidenceError(
                "Unexpected single-level yfinance columns for multiple symbols: "
                f"{list(map(str, output.columns))!r}"
            )
        output.columns = [f"{symbols[0]}__{column}" for column in output.columns]

    label = output.index.name or "Date"
    output = output.reset_index().rename(columns={label: "observation_date"})
    if "observation_date" not in output.columns:
        output = output.rename(columns={output.columns[0]: "observation_date"})

    parsed = pd.to_datetime(output["observation_date"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise EvidenceError("ETF data contain unparsable observation dates")
    output["observation_date"] = parsed.dt.strftime("%Y-%m-%d")
    if output["observation_date"].duplicated().any():
        raise EvidenceError("ETF data contain duplicate observation dates")
    return output.sort_values("observation_date", kind="mergesort").reset_index(drop=True)


def find_etf_price_column(frame: pd.DataFrame, symbol: str) -> tuple[str, str]:
    candidates = [
        (f"{symbol}__Adj Close", "adjusted_close"),
        (f"{symbol}__Close", "close_auto_adjust_interpretation_required"),
    ]
    for column, interpretation in candidates:
        if column in frame.columns:
            return column, interpretation
    available = [c for c in frame.columns if c.startswith(f"{symbol}__")]
    raise EvidenceError(f"{symbol}: no adjusted-close or close column; available={available!r}")


def download_etfs(
    symbols: Sequence[str],
    start: date,
    end: date,
    raw_dir: Path,
    processed_dir: Path,
    run_root: Path,
) -> tuple[list[DownloadRecord], pd.DataFrame, dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise EvidenceError(
            "ETF layer requested but yfinance is not installed in the active environment"
        ) from exc

    request_end_exclusive = end + timedelta(days=1)
    retrieved_at = iso_z(utc_now())
    params = {
        "tickers": list(symbols),
        "start": start.isoformat(),
        "end_exclusive": request_end_exclusive.isoformat(),
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "repair": False,
        "keepna": True,
        "threads": False,
        "progress": False,
    }
    logging.info("Downloading ETF context: %s", ", ".join(symbols))
    downloaded = yf.download(
        tickers=list(symbols),
        start=params["start"],
        end=params["end_exclusive"],
        interval=params["interval"],
        auto_adjust=params["auto_adjust"],
        actions=params["actions"],
        repair=params["repair"],
        keepna=params["keepna"],
        threads=params["threads"],
        progress=params["progress"],
        group_by="column",
        multi_level_index=True,
    )
    normalized = normalize_yfinance_columns(downloaded, symbols)
    parsed_dates = pd.to_datetime(normalized["observation_date"], errors="raise").dt.date
    outside = (parsed_dates < start) | (parsed_dates > end)
    if bool(outside.any()):
        examples = normalized.loc[outside, "observation_date"].head(10).tolist()
        raise EvidenceError(f"ETF download contains dates outside contract: {examples}")

    raw_path = raw_dir / "market_proxy" / f"yfinance_{'_'.join(symbols)}_{start}_{end}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(raw_path, index=False, lineterminator="\n", float_format="%.10g")

    records: list[DownloadRecord] = []
    audits: dict[str, Any] = {}
    selected = pd.DataFrame({"observation_date": normalized["observation_date"]})

    for symbol in symbols:
        price_column, interpretation = find_etf_price_column(normalized, symbol)
        numeric = pd.to_numeric(normalized[price_column], errors="coerce")
        valid_mask = numeric.notna()
        if not bool(valid_mask.any()):
            raise EvidenceError(f"{symbol}: no valid price observations")
        if bool((numeric[valid_mask] <= 0).any()):
            raise EvidenceError(f"{symbol}: non-positive price observations detected")

        selected[f"{symbol}_market_price"] = numeric
        valid_dates = normalized.loc[valid_mask, "observation_date"]
        records.append(
            DownloadRecord(
                layer="market_proxy",
                identifier=symbol,
                request_url=None,
                request_parameters=params,
                retrieved_at_utc=retrieved_at,
                raw_file=relative_to_run(raw_path, run_root),
                raw_sha256=sha256_file(raw_path),
                rows_raw=int(len(normalized)),
                rows_valid=int(valid_mask.sum()),
                earliest_valid_date=str(valid_dates.iloc[0]),
                latest_valid_date=str(valid_dates.iloc[-1]),
                missing_value_count=int(numeric.isna().sum()),
                status="success",
            )
        )
        audits[symbol] = {
            "selected_source_column": price_column,
            "price_interpretation": interpretation,
            "rows_valid": int(valid_mask.sum()),
            "missing_value_count": int(numeric.isna().sum()),
            "earliest_valid_date": str(valid_dates.iloc[0]),
            "latest_valid_date": str(valid_dates.iloc[-1]),
            "warning": (
                "ETF market data are a secondary illustration. Validate adjusted-price "
                "and distribution treatment before calling derived changes total returns."
            ),
        }

    processed_path = processed_dir / "market_proxy" / "IEF_TIP_selected_prices.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(processed_path, index=False, lineterminator="\n", float_format="%.10g")
    audits["combined"] = {
        "processed_file": relative_to_run(processed_path, run_root),
        "processed_sha256": sha256_file(processed_path),
        "raw_normalized_file": relative_to_run(raw_path, run_root),
        "raw_sha256": sha256_file(raw_path),
        "yfinance_version": package_version("yfinance"),
        "legal_usage_note": (
            "Review Yahoo Finance terms and the intended public-use rights before "
            "redistributing source data. Publish derived analysis only when permitted."
        ),
    }
    return records, selected, audits


def construct_official_common_panel(
    frames: Mapping[str, pd.DataFrame],
    processed_dir: Path,
    run_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dgs = frames["DGS10"].copy()
    dfii = frames["DFII10"].copy()
    merged = dgs.merge(dfii, on="observation_date", how="outer", validate="one_to_one")
    merged = merged.sort_values("observation_date", kind="mergesort").reset_index(drop=True)
    merged["is_common_valid"] = merged[["DGS10", "DFII10"]].notna().all(axis=1)
    merged["inflation_compensation_pct_points"] = (
        merged["DGS10"] - merged["DFII10"]
    ).where(merged["is_common_valid"])

    common = merged.loc[
        merged["is_common_valid"],
        ["observation_date", "DGS10", "DFII10", "inflation_compensation_pct_points"],
    ].reset_index(drop=True)
    if common.empty:
        raise EvidenceError("DGS10 and DFII10 have no common valid observation dates")

    panel_path = processed_dir / "official" / "nominal_real_outer_panel.csv"
    common_path = processed_dir / "official" / "nominal_real_common_valid.csv"
    merged.to_csv(panel_path, index=False, lineterminator="\n", float_format="%.10g")
    common.to_csv(common_path, index=False, lineterminator="\n", float_format="%.10g")

    audit = {
        "outer_union_rows": int(len(merged)),
        "common_valid_rows": int(len(common)),
        "dgs10_only_rows": int((merged["DGS10"].notna() & merged["DFII10"].isna()).sum()),
        "dfii10_only_rows": int((merged["DGS10"].isna() & merged["DFII10"].notna()).sum()),
        "both_missing_rows": int((merged[["DGS10", "DFII10"]].isna().all(axis=1)).sum()),
        "earliest_common_date": str(common["observation_date"].iloc[0]),
        "latest_common_date": str(common["observation_date"].iloc[-1]),
        "outer_panel_file": relative_to_run(panel_path, run_root),
        "outer_panel_sha256": sha256_file(panel_path),
        "common_valid_file": relative_to_run(common_path, run_root),
        "common_valid_sha256": sha256_file(common_path),
        "interpretation_boundary": (
            "DGS10 minus DFII10 is labeled inflation compensation, not exact expected inflation. "
            "The spread may include expected inflation, inflation risk premium, relative "
            "liquidity, market technicals, and measurement effects."
        ),
    }
    return common, audit


def construct_etf_common_panel(
    selected: pd.DataFrame,
    processed_dir: Path,
    run_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["observation_date", "IEF_market_price", "TIP_market_price"]
    missing_columns = [c for c in columns if c not in selected.columns]
    if missing_columns:
        raise EvidenceError(f"ETF selected dataset missing columns: {missing_columns}")
    common = selected.loc[selected[columns[1:]].notna().all(axis=1), columns].copy()
    common = common.sort_values("observation_date", kind="mergesort").reset_index(drop=True)
    if common.empty:
        raise EvidenceError("IEF and TIP have no common valid market dates")
    path = processed_dir / "market_proxy" / "IEF_TIP_common_valid_prices.csv"
    common.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")
    audit = {
        "common_valid_rows": int(len(common)),
        "earliest_common_date": str(common["observation_date"].iloc[0]),
        "latest_common_date": str(common["observation_date"].iloc[-1]),
        "common_valid_file": relative_to_run(path, run_root),
        "common_valid_sha256": sha256_file(path),
        "interpretation_boundary": (
            "IEF and TIP are investable-market proxies, not exact DGS10 and DFII10 instruments. "
            "No total-return claim is authorized until adjusted-price and distribution handling "
            "are explicitly validated."
        ),
    }
    return common, audit


def collect_hashes(run_root: Path, excluded_names: Iterable[str] = ()) -> list[tuple[str, str]]:
    excluded = set(excluded_names)
    rows: list[tuple[str, str]] = []
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded or path.name.endswith(".part") or path.name.endswith(".tmp"):
            continue
        rows.append((sha256_file(path), relative))
    return rows


def write_sha256sums(run_root: Path) -> Path:
    output = run_root / "SHA256SUMS"
    rows = collect_hashes(run_root, excluded_names={"SHA256SUMS"})
    write_text(output, "".join(f"{digest}  {relative}\n" for digest, relative in rows))
    return output


def environmental_manifest() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "pandas_version": package_version("pandas"),
        "requests_version": package_version("requests"),
        "yfinance_version": package_version("yfinance"),
        "timezone_environment": os.environ.get("TZ"),
    }


def make_claim_boundaries() -> list[dict[str, str]]:
    return [
        {
            "status": "allowed_if_verified",
            "claim": "Describe observed DGS10, DFII10, and their common-date difference.",
        },
        {
            "status": "allowed_if_verified",
            "claim": "Describe IEF and TIP market-price paths as imperfect market proxies.",
        },
        {
            "status": "prohibited_from_download_only",
            "claim": "The spread is exact expected inflation.",
        },
        {
            "status": "prohibited_from_download_only",
            "claim": "Yield is historical or future total return.",
        },
        {
            "status": "prohibited_from_download_only",
            "claim": "IEF or TIP exactly replicates a constant-maturity 10-year instrument.",
        },
        {
            "status": "prohibited",
            "claim": "One instrument is the best, safest, or a buy/sell recommendation.",
        },
        {
            "status": "prohibited_from_observational_comparison",
            "claim": "Observed movements prove monetary-policy, inflation, growth, or fiscal causation.",
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.Formatter.converter = __import__("time").gmtime

    if args.start > args.end:
        raise SystemExit("--start must be on or before --end")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.case_relpath.is_absolute():
        raise SystemExit("--case-relpath must be relative to --project-root")

    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise SystemExit(f"Project root does not exist or is not a directory: {project_root}")

    run_id = args.run_id or utc_now().strftime("%Y%m%dT%H%M%SZ")
    case_root = project_root / args.case_relpath
    run_root = case_root / "data" / "runs" / run_id
    ensure_within(project_root, run_root)
    if run_root.exists():
        raise SystemExit(
            f"Refusing to overwrite existing run: {run_root}\n"
            "Choose a new --run-id. Existing evidence was not modified."
        )

    dirs = {
        "raw": run_root / "raw",
        "processed": run_root / "processed",
        "manifests": run_root / "manifests",
        "validation": run_root / "validation",
        "logs": run_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=False)

    started_at = iso_z(utc_now())
    records: list[DownloadRecord] = []
    validations: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    official_frames: dict[str, pd.DataFrame] = {}
    overall_status = "in_progress"

    contract = {
        "case_version": CASE_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": started_at,
        "project_root": str(project_root),
        "run_root": str(run_root),
        "observation_window": {
            "start_inclusive": args.start.isoformat(),
            "end_inclusive": args.end.isoformat(),
        },
        "official_series": OFFICIAL_SERIES,
        "optional_market_proxies": {} if args.skip_etfs else ETF_SPECS,
        "transformations": {
            "official_alignment": "Outer union retained; analytic spread only on common valid dates.",
            "inflation_compensation": "DGS10 minus DFII10, percentage points.",
            "missing_value_rule": "Preserve source missingness; no interpolation or filling.",
            "etf_alignment": "Pairwise common valid market dates; no filling.",
        },
        "revision_policy": (
            "This run records retrieval time and immutable raw hashes. FRED/ALFRED revision "
            "behavior must be audited separately before any real-time-information claim."
        ),
        "license_and_reuse": (
            "Review each provider's current terms before redistributing raw data or public artifacts."
        ),
        "claim_boundaries": make_claim_boundaries(),
    }
    write_json(dirs["manifests"] / "data_contract.json", contract)
    write_json(dirs["manifests"] / "environment.json", environmental_manifest())

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/csv,*/*;q=0.8"})

    try:
        for series_id in OFFICIAL_SERIES:
            try:
                record, frame, audit = download_fred_series(
                    session=session,
                    series_id=series_id,
                    start=args.start,
                    end=args.end,
                    timeout_seconds=args.timeout_seconds,
                    raw_dir=dirs["raw"],
                    processed_dir=dirs["processed"],
                    run_root=run_root,
                )
                records.append(record)
                official_frames[series_id] = frame
                validations[f"official_{series_id}"] = audit
            except Exception as exc:
                logging.exception("Official series %s failed", series_id)
                failure = DownloadRecord(
                    layer="official_yield",
                    identifier=series_id,
                    request_url=FRED_CSV_URL,
                    request_parameters={
                        "id": series_id,
                        "cosd": args.start.isoformat(),
                        "coed": args.end.isoformat(),
                    },
                    retrieved_at_utc=iso_z(utc_now()),
                    raw_file=None,
                    raw_sha256=None,
                    rows_raw=None,
                    rows_valid=None,
                    earliest_valid_date=None,
                    latest_valid_date=None,
                    missing_value_count=None,
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                records.append(failure)
                errors.append({
                    "layer": "official_yield",
                    "identifier": series_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if not args.allow_partial:
                    raise

        if set(OFFICIAL_SERIES) <= set(official_frames):
            _, common_audit = construct_official_common_panel(
                official_frames, dirs["processed"], run_root
            )
            validations["official_common_panel"] = common_audit
        elif not args.allow_partial:
            raise EvidenceError("Official common panel unavailable")

        if not args.skip_etfs:
            try:
                etf_records, etf_selected, etf_audit = download_etfs(
                    symbols=list(ETF_SPECS),
                    start=args.start,
                    end=args.end,
                    raw_dir=dirs["raw"],
                    processed_dir=dirs["processed"],
                    run_root=run_root,
                )
                records.extend(etf_records)
                validations["market_proxy_download"] = etf_audit
                _, pair_audit = construct_etf_common_panel(
                    etf_selected, dirs["processed"], run_root
                )
                validations["market_proxy_common_panel"] = pair_audit
            except Exception as exc:
                logging.exception("ETF market-proxy layer failed")
                errors.append({
                    "layer": "market_proxy",
                    "identifier": "IEF,TIP",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                for symbol in ETF_SPECS:
                    records.append(
                        DownloadRecord(
                            layer="market_proxy",
                            identifier=symbol,
                            request_url=None,
                            request_parameters={
                                "tickers": list(ETF_SPECS),
                                "start": args.start.isoformat(),
                                "end_inclusive": args.end.isoformat(),
                            },
                            retrieved_at_utc=iso_z(utc_now()),
                            raw_file=None,
                            raw_sha256=None,
                            rows_raw=None,
                            rows_valid=None,
                            earliest_valid_date=None,
                            latest_valid_date=None,
                            missing_value_count=None,
                            status="failed",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    )
                if not args.allow_partial:
                    raise

        failed = [r for r in records if r.status != "success"]
        required_success = set(OFFICIAL_SERIES) <= {
            r.identifier for r in records if r.layer == "official_yield" and r.status == "success"
        }
        etf_success = args.skip_etfs or set(ETF_SPECS) <= {
            r.identifier for r in records if r.layer == "market_proxy" and r.status == "success"
        }
        if required_success and etf_success and not failed:
            overall_status = "complete"
        elif required_success and args.allow_partial:
            overall_status = "partial"
        else:
            overall_status = "failed"

    except Exception:
        overall_status = "failed"
        if not errors:
            errors.append({
                "layer": "run",
                "identifier": CASE_VERSION,
                "error_type": sys.exc_info()[0].__name__ if sys.exc_info()[0] else "UnknownError",
                "error_message": str(sys.exc_info()[1]),
                "traceback": traceback.format_exc(),
            })
    finally:
        session.close()
        completed_at = iso_z(utc_now())
        write_json(
            dirs["manifests"] / "download_records.json",
            [asdict(record) for record in records],
        )
        write_json(dirs["validation"] / "validation_report.json", validations)
        write_json(dirs["validation"] / "errors.json", errors)
        run_manifest = {
            "case_version": CASE_VERSION,
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "status": overall_status,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "official_layer_complete": set(OFFICIAL_SERIES) <= {
                r.identifier
                for r in records
                if r.layer == "official_yield" and r.status == "success"
            },
            "market_proxy_requested": not args.skip_etfs,
            "market_proxy_complete": args.skip_etfs or set(ETF_SPECS) <= {
                r.identifier
                for r in records
                if r.layer == "market_proxy" and r.status == "success"
            },
            "error_count": len(errors),
            "non_claim_statement": (
                "This checkpoint contains downloaded observations and data-quality metadata only. "
                "It establishes no investment superiority, forecast, causal explanation, alpha, "
                "trading edge, or individualized advice."
            ),
        }
        write_json(dirs["manifests"] / "run_manifest.json", run_manifest)
        write_sha256sums(run_root)

    print(json.dumps({
        "status": overall_status,
        "run_root": str(run_root),
        "records": len(records),
        "errors": len(errors),
        "sha256sums": str(run_root / "SHA256SUMS"),
    }, indent=2))

    if overall_status == "complete":
        return 0
    if overall_status == "partial" and args.allow_partial:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
