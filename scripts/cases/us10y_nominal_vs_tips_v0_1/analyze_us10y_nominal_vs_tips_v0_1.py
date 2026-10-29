#!/usr/bin/env python3
"""One-shot Market Memory Lab analysis and LinkedIn evidence pack.

Reads locally downloaded FRED DGS10.csv and DFII10.csv, verifies approved
hashes, creates an immutable run, computes descriptive and falsification
metrics, generates publication-ready scientific figures, and writes a bounded
LinkedIn draft. No network access, package installation, git action, or
investment recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "0.1.0"
EXPECTED = {
    "DGS10.csv": "3e0742b6d6b6e3d741d70e0113e26dcf8779b41785db56bc4e5f0773396f4893",
    "DFII10.csv": "e818d3a37c71b4ab14ddcb11740db38f07722debb2859c4e952e4fb86d00c09e",
}
START = pd.Timestamp("2003-01-02")
END = pd.Timestamp("2026-09-21")
NAVY = "#081B33"
BLUE = "#4D8AC9"
GOLD = "#C6A15B"
WHITE = "#F6F8FB"
STEEL = "#7C94AB"
RED = "#C96B6B"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_series(path: Path, col: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["observation_date", col]:
        raise ValueError(
            f"{path.name}: expected columns observation_date,{col}; got {list(df.columns)}"
        )
    raw_rows = len(df)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    if df["observation_date"].isna().any():
        raise ValueError(f"{path.name}: unparsable dates")
    if df["observation_date"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate dates")
    df[col] = pd.to_numeric(df[col].replace({".": np.nan, "": np.nan}), errors="coerce")
    df = df.sort_values("observation_date", kind="mergesort").reset_index(drop=True)
    audit = {
        "raw_rows": raw_rows,
        "numeric_rows": int(df[col].notna().sum()),
        "missing_rows": int(df[col].isna().sum()),
        "earliest_date": df["observation_date"].min().date().isoformat(),
        "latest_date": df["observation_date"].max().date().isoformat(),
        "earliest_valid": df.loc[df[col].notna(), "observation_date"]
        .min()
        .date()
        .isoformat(),
        "latest_valid": df.loc[df[col].notna(), "observation_date"]
        .max()
        .date()
        .isoformat(),
        "monotonic": bool(df["observation_date"].is_monotonic_increasing),
    }
    return df, audit


def percentile_rank(values: pd.Series, x: float) -> float:
    clean = values.dropna().to_numpy()
    return float(100.0 * np.mean(clean <= x))


def robust_summary(s: pd.Series) -> dict:
    x = s.dropna()
    if x.empty:
        return {}
    med = float(x.median())
    mad = float((x - med).abs().median())
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": med,
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
        "mad": mad,
    }


def nearest_prior(df: pd.DataFrame, target: pd.Timestamp) -> pd.Series:
    eligible = df.loc[df.index <= target]
    if eligible.empty:
        raise ValueError(f"No observation on or before {target.date()}")
    return eligible.iloc[-1]


def horizon_metrics(df: pd.DataFrame) -> list[dict]:
    latest_date = df.index.max()
    latest = df.loc[latest_date]
    horizons = {
        "1_month": pd.DateOffset(months=1),
        "3_months": pd.DateOffset(months=3),
        "1_year": pd.DateOffset(years=1),
        "5_years": pd.DateOffset(years=5),
    }
    out = []
    for label, offset in horizons.items():
        target = latest_date - offset
        prior = nearest_prior(df, target)
        prior_date = prior.name
        row = {
            "horizon": label,
            "latest_date": latest_date.date().isoformat(),
            "target_date": target.date().isoformat(),
            "comparison_date": prior_date.date().isoformat(),
        }
        for c in ["DGS10", "DFII10", "inflation_compensation"]:
            row[f"{c}_latest_pct"] = float(latest[c])
            row[f"{c}_change_bps"] = float((latest[c] - prior[c]) * 100)
        out.append(row)
    return out


def correlations(df: pd.DataFrame, label: str) -> dict:
    level = df[["DGS10", "DFII10"]].dropna()
    changes = df[["DGS10_change_bps", "DFII10_change_bps"]].dropna()
    weekly = df[["DGS10", "DFII10"]].resample("W-FRI").last().diff().dropna() * 100

    def pair(x, a, b):
        return {
            "n": len(x),
            "pearson": float(x[a].corr(x[b], method="pearson")),
            "spearman": float(x[a].corr(x[b], method="spearman")),
        }

    return {
        "window": label,
        "levels": pair(level, "DGS10", "DFII10"),
        "daily_changes_bps": pair(changes, "DGS10_change_bps", "DFII10_change_bps"),
        "weekly_changes_bps": pair(weekly, "DGS10", "DFII10"),
    }


def window_table(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.index.max()
    windows = [
        ("full_common", df.index.min()),
        ("10_years", latest - pd.DateOffset(years=10)),
        ("5_years", latest - pd.DateOffset(years=5)),
        ("3_years", latest - pd.DateOffset(years=3)),
        ("1_year", latest - pd.DateOffset(years=1)),
    ]
    rows = []
    for label, start in windows:
        x = df.loc[df.index >= start].copy()
        c = correlations(x, label)
        s = x["inflation_compensation"]
        rows.append(
            {
                "window": label,
                "actual_start": x.index.min().date().isoformat(),
                "actual_end": x.index.max().date().isoformat(),
                "n": len(x),
                "level_pearson": c["levels"]["pearson"],
                "daily_change_pearson": c["daily_changes_bps"]["pearson"],
                "weekly_change_pearson": c["weekly_changes_bps"]["pearson"],
                "spread_mean_pct": s.mean(),
                "spread_median_pct": s.median(),
                "spread_std_pct": s.std(),
                "spread_min_pct": s.min(),
                "spread_max_pct": s.max(),
                "latest_spread_percentile": percentile_rank(s, s.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def outlier_check(df: pd.DataFrame) -> dict:
    x = df[["DGS10_change_bps", "DFII10_change_bps"]].dropna().copy()
    base = float(x.iloc[:, 0].corr(x.iloc[:, 1]))
    cutoff_a = x.iloc[:, 0].abs().quantile(0.99)
    cutoff_b = x.iloc[:, 1].abs().quantile(0.99)
    trimmed = x.loc[(x.iloc[:, 0].abs() <= cutoff_a) & (x.iloc[:, 1].abs() <= cutoff_b)]
    return {
        "base_daily_change_pearson": base,
        "trimmed_daily_change_pearson": float(
            trimmed.iloc[:, 0].corr(trimmed.iloc[:, 1])
        ),
        "base_n": len(x),
        "trimmed_n": len(trimmed),
        "DGS10_abs_change_99pct_bps": float(cutoff_a),
        "DFII10_abs_change_99pct_bps": float(cutoff_b),
    }


def style_axis(ax, title: str, ylabel: str = ""):
    ax.set_facecolor(NAVY)
    ax.set_title(title, color=WHITE, fontsize=14, weight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=WHITE)
    ax.tick_params(colors=WHITE, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#39516B")
    ax.grid(True, color="#29435E", alpha=0.55, linewidth=0.7)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)


def make_figures(df: pd.DataFrame, out: Path, metrics: dict):
    dates = df.index
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10), sharex=True, facecolor=NAVY)
    axes[0].plot(dates, df.DGS10, color=GOLD, lw=1.35)
    style_axis(axes[0], "U.S. 10-Year Nominal Treasury Yield", "Percent")
    axes[1].plot(dates, df.DFII10, color=BLUE, lw=1.35)
    axes[1].axhline(0, color=WHITE, lw=0.8, alpha=0.6)
    style_axis(axes[1], "U.S. 10-Year Real TIPS Yield", "Percent")
    axes[2].plot(dates, df.inflation_compensation, color=WHITE, lw=1.35)
    style_axis(
        axes[2], "Nominal Minus Real: Inflation Compensation", "Percentage points"
    )
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    latest = df.iloc[-1]
    fig.suptitle(
        "TWO BONDS, TWO INFLATION WORLDS",
        color=WHITE,
        fontsize=22,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        f"Common valid observations | {df.index.min().date()} to {df.index.max().date()} | Latest spread {latest.inflation_compensation:.2f}%",
        ha="center",
        color=STEEL,
        fontsize=10,
    )
    fig.text(
        0.01,
        0.005,
        "Source: Federal Reserve H.15 series via locally preserved FRED CSVs. Spread is inflation compensation, not exact expected inflation.",
        color=STEEL,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.945])
    save_fig(fig, out / "01_rate_structure.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=NAVY)
    axes[0].scatter(df.DFII10, df.DGS10, s=8, alpha=0.25, color=GOLD, edgecolors="none")
    style_axis(axes[0], "Levels", "Nominal yield (%)")
    axes[0].set_xlabel("Real yield (%)", color=WHITE)
    axes[0].text(
        0.03,
        0.95,
        f"Pearson r = {metrics['correlations']['levels']['pearson']:.2f}\nSpearman ρ = {metrics['correlations']['levels']['spearman']:.2f}",
        transform=axes[0].transAxes,
        va="top",
        color=WHITE,
    )
    c = df[["DFII10_change_bps", "DGS10_change_bps"]].dropna()
    axes[1].scatter(
        c.DFII10_change_bps,
        c.DGS10_change_bps,
        s=8,
        alpha=0.25,
        color=BLUE,
        edgecolors="none",
    )
    style_axis(axes[1], "Daily Changes", "Nominal change (bps)")
    axes[1].set_xlabel("Real change (bps)", color=WHITE)
    axes[1].text(
        0.03,
        0.95,
        f"Pearson r = {metrics['correlations']['daily_changes_bps']['pearson']:.2f}\nSpearman ρ = {metrics['correlations']['daily_changes_bps']['spearman']:.2f}",
        transform=axes[1].transAxes,
        va="top",
        color=WHITE,
    )
    fig.suptitle(
        "LEVELS CAN LOOK CONNECTED. CHANGES ARE THE HARDER TEST.",
        color=WHITE,
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Descriptive correlations only. No causal or forecasting interpretation.",
        color=STEEL,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "02_levels_vs_changes.png")

    roll = (
        df[["DGS10_change_bps", "DFII10_change_bps"]]
        .rolling(252, min_periods=126)
        .corr()
        .unstack()
        .iloc[:, 1]
    )
    fig, ax = plt.subplots(figsize=(13.5, 5.5), facecolor=NAVY)
    ax.plot(roll.index, roll, color=GOLD, lw=1.3)
    ax.axhline(0, color=WHITE, lw=0.8, alpha=0.6)
    style_axis(
        ax, "252-Observation Rolling Correlation of Daily Yield Changes", "Correlation"
    )
    ax.set_ylim(-1, 1)
    fig.text(
        0.01,
        0.01,
        "Rolling window requires at least 126 valid pairs. Instability weakens any claim of a permanent relationship.",
        color=STEEL,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, out / "03_rolling_change_correlation.png")

    vol = (
        df[["DGS10_change_bps", "DFII10_change_bps", "spread_change_bps"]]
        .rolling(63, min_periods=32)
        .std()
    )
    fig, ax = plt.subplots(figsize=(13.5, 5.5), facecolor=NAVY)
    ax.plot(vol.index, vol.DGS10_change_bps, label="Nominal", color=GOLD, lw=1.1)
    ax.plot(vol.index, vol.DFII10_change_bps, label="Real", color=BLUE, lw=1.1)
    ax.plot(
        vol.index,
        vol.spread_change_bps,
        label="Inflation compensation",
        color=WHITE,
        lw=1.1,
    )
    style_axis(
        ax,
        "63-Observation Volatility of Daily Basis-Point Changes",
        "Standard deviation (bps)",
    )
    leg = ax.legend(frameon=False, ncol=3, loc="upper left")
    for t in leg.get_texts():
        t.set_color(WHITE)
    fig.text(
        0.01,
        0.01,
        "Yield-change volatility is not bond total-return volatility.",
        color=STEEL,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, out / "04_change_volatility.png")

    fig, ax = plt.subplots(figsize=(13.5, 7.5), facecolor=NAVY)
    ax.set_facecolor(NAVY)
    ax.axis("off")
    latest = df.iloc[-1]
    ax.scatter(
        [0.25, 0.75],
        [0.56, 0.56],
        s=[9000, 9000],
        c=[GOLD, BLUE],
        alpha=0.95,
        transform=ax.transAxes,
    )
    ax.plot(
        [0.34, 0.66],
        [0.56, 0.56],
        color=WHITE,
        lw=max(3, latest.inflation_compensation * 2),
        transform=ax.transAxes,
    )
    ax.text(
        0.25,
        0.56,
        f"NOMINAL 10Y\n{latest.DGS10:.2f}%",
        ha="center",
        va="center",
        color=NAVY,
        weight="bold",
        fontsize=17,
        transform=ax.transAxes,
    )
    ax.text(
        0.75,
        0.56,
        f"REAL 10Y\n{latest.DFII10:.2f}%",
        ha="center",
        va="center",
        color=WHITE,
        weight="bold",
        fontsize=17,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.64,
        f"INFLATION COMPENSATION\n{latest.inflation_compensation:.2f}%",
        ha="center",
        va="bottom",
        color=WHITE,
        weight="bold",
        fontsize=15,
        transform=ax.transAxes,
    )
    forces = [
        "Inflation information",
        "Monetary policy",
        "Growth expectations",
        "Treasury supply",
        "Risk aversion",
        "Relative liquidity",
    ]
    xs = [0.12, 0.28, 0.43, 0.58, 0.73, 0.88]
    for x, label in zip(xs, forces):
        ax.annotate(
            label,
            xy=(0.5, 0.7),
            xytext=(x, 0.9),
            ha="center",
            color=STEEL,
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#39516B", "alpha": 0.75},
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )
    ax.text(
        0.5,
        0.18,
        "Observed structure, not a causal model or investment recommendation",
        ha="center",
        color=STEEL,
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.97,
        "THE NOMINAL-REAL TREASURY MOLECULE",
        ha="center",
        va="top",
        color=WHITE,
        weight="bold",
        fontsize=22,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.08,
        f"Common date: {df.index.max().date()} | Source: Federal Reserve H.15 via preserved FRED CSVs",
        ha="center",
        color=STEEL,
        fontsize=8,
        transform=ax.transAxes,
    )
    save_fig(fig, out / "05_molecular_structure.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--run-id", default=None)
    a = p.parse_args()
    root = a.project_root.expanduser().resolve()
    inp = a.input_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing project root: {root}")
    run_id = a.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = root / "cases/us10y_nominal_vs_tips_v0_1/analysis/runs" / run_id
    if run.exists():
        raise SystemExit(f"Refusing overwrite: {run}")
    for d in ["raw", "processed", "metrics", "figures", "publication", "validation"]:
        (run / d).mkdir(parents=True)

    checks = {}
    for name, expected in EXPECTED.items():
        src = inp / name
        if not src.is_file():
            raise SystemExit(f"Missing source: {src}")
        actual = sha256(src)
        checks[name] = {
            "expected": expected,
            "actual": actual,
            "pass": actual == expected,
        }
        if actual != expected:
            raise SystemExit(f"Hash mismatch for {name}: {actual}")
        shutil.copy2(src, run / "raw" / name)

    dgs, audit_dgs = load_series(run / "raw/DGS10.csv", "DGS10")
    real, audit_real = load_series(run / "raw/DFII10.csv", "DFII10")
    outer = dgs.merge(
        real, on="observation_date", how="outer", validate="one_to_one"
    ).sort_values("observation_date")
    outer = outer.loc[
        (outer.observation_date >= START) & (outer.observation_date <= END)
    ].copy()
    outer["common_valid"] = outer[["DGS10", "DFII10"]].notna().all(axis=1)
    common = outer.loc[
        outer.common_valid, ["observation_date", "DGS10", "DFII10"]
    ].set_index("observation_date")
    if common.empty:
        raise SystemExit("No common valid observations")
    common["inflation_compensation"] = common.DGS10 - common.DFII10
    common["DGS10_change_bps"] = common.DGS10.diff() * 100
    common["DFII10_change_bps"] = common.DFII10.diff() * 100
    common["spread_change_bps"] = common.inflation_compensation.diff() * 100
    common["rolling_63_nominal_vol_bps"] = common.DGS10_change_bps.rolling(
        63, min_periods=32
    ).std()
    common["rolling_63_real_vol_bps"] = common.DFII10_change_bps.rolling(
        63, min_periods=32
    ).std()
    common["rolling_63_spread_vol_bps"] = common.spread_change_bps.rolling(
        63, min_periods=32
    ).std()
    common["rolling_252_change_corr"] = common.DGS10_change_bps.rolling(
        252, min_periods=126
    ).corr(common.DFII10_change_bps)

    outer.to_csv(run / "processed/outer_union.csv", index=False, float_format="%.10g")
    common.to_csv(run / "processed/common_valid_analysis.csv", float_format="%.10g")
    latest = common.iloc[-1]
    corr = correlations(common, "full_common")
    windows = window_table(common)
    windows.to_csv(
        run / "metrics/window_sensitivity.csv", index=False, float_format="%.10g"
    )
    horizons = horizon_metrics(common)
    pd.DataFrame(horizons).to_csv(
        run / "metrics/horizon_changes.csv", index=False, float_format="%.10g"
    )
    metrics = {
        "version": VERSION,
        "common_start": common.index.min().date().isoformat(),
        "common_end": common.index.max().date().isoformat(),
        "common_observations": len(common),
        "latest": {
            "date": common.index.max().date().isoformat(),
            "DGS10_pct": float(latest.DGS10),
            "DFII10_pct": float(latest.DFII10),
            "inflation_compensation_pct": float(latest.inflation_compensation),
            "inflation_compensation_bps": float(latest.inflation_compensation * 100),
            "spread_full_history_percentile": percentile_rank(
                common.inflation_compensation, float(latest.inflation_compensation)
            ),
        },
        "correlations": corr,
        "summaries": {
            c: robust_summary(common[c])
            for c in [
                "DGS10",
                "DFII10",
                "inflation_compensation",
                "DGS10_change_bps",
                "DFII10_change_bps",
                "spread_change_bps",
            ]
        },
        "outlier_sensitivity": outlier_check(common),
        "horizon_changes": horizons,
    }
    write_json(run / "metrics/metrics.json", metrics)
    write_json(run / "validation/hash_verification.json", checks)
    write_json(
        run / "validation/data_audit.json",
        {
            "DGS10": audit_dgs,
            "DFII10": audit_real,
            "outer_rows": len(outer),
            "common_rows": len(common),
            "outer_missing_DGS10": int(outer.DGS10.isna().sum()),
            "outer_missing_DFII10": int(outer.DFII10.isna().sum()),
        },
    )
    make_figures(common, run / "figures", metrics)

    level_r = corr["levels"]["pearson"]
    change_r = corr["daily_changes_bps"]["pearson"]
    direction = "stronger" if abs(level_r) > abs(change_r) else "weaker"
    insights = f"""# Evidence summary\n\n## Verified data boundary\n\nThe analysis uses {len(common):,} common valid daily observations from {common.index.min().date()} through {common.index.max().date()}. Missing values were not filled.\n\n## Latest common observation\n\nOn {common.index.max().date()}, the nominal 10-year yield was {latest.DGS10:.2f}%, the real 10-year TIPS yield was {latest.DFII10:.2f}%, and their difference was {latest.inflation_compensation:.2f} percentage points, or {latest.inflation_compensation * 100:.0f} basis points. The spread sat at the {metrics["latest"]["spread_full_history_percentile"]:.1f}th percentile of this specific common-history sample.\n\n## Main falsification result\n\nThe Pearson correlation in levels was {level_r:.3f}. The Pearson correlation of daily basis-point changes was {change_r:.3f}. The relationship in levels was {direction} in absolute magnitude than the relationship in daily changes. This distinction limits any claim that a visually persistent level relationship automatically describes day-to-day repricing.\n\n## Interpretation boundary\n\nThe nominal-real difference is inflation compensation, not exact expected inflation. It may also contain inflation risk premium, liquidity differences, market technicals, and measurement effects. Yield movements are not bond total returns. These results do not establish causation, forecasting ability, investment superiority, alpha, or a buy/sell recommendation.\n"""
    atomic_text(run / "publication/verified_insights.md", insights)

    caption = f"""I wanted to understand what the U.S. 10-year Treasury and 10-year TIPS are actually telling us when they move apart.\n\nI analyzed {len(common):,} common valid daily observations from {common.index.min().date()} through {common.index.max().date()} using the Federal Reserve's nominal and inflation-indexed 10-year constant-maturity yield series.\n\nOn the latest common date in this dataset, the nominal yield was {latest.DGS10:.2f}% and the real yield was {latest.DFII10:.2f}%. Their difference was {latest.inflation_compensation:.2f} percentage points.\n\nThat difference is best described as inflation compensation, not exact expected inflation. It can also reflect inflation risk premium, relative liquidity, and market technicals.\n\nThe harder test was not whether the two yield levels looked connected. It was whether their daily changes moved together. The full-sample Pearson correlation was {level_r:.2f} in levels and {change_r:.2f} in daily basis-point changes. That difference matters because persistent series can appear tightly connected even when short-run repricing is less stable.\n\nMy molecular visual treats the nominal yield and real yield as two nodes joined by inflation compensation. It is an intuition layer, not a causal model.\n\nWhat this evidence does not prove:\n• future inflation\n• future Treasury or TIPS returns\n• which instrument is superior\n• a trading signal\n• a buy or sell recommendation\n\nThis is an educational market-structure study, not individualized investment advice.\n\n#FixedIncome #Treasury #TIPS #FinancialDataScience #FinTech #QuantitativeResearch #AssetPricing\n"""
    atomic_text(run / "publication/linkedin_caption_draft.txt", caption)
    claim_ledger = {
        "verified": [
            "Source hashes matched approved values",
            "Common-date arithmetic and descriptive metrics were computed from preserved local CSVs",
        ],
        "interpretation_limited": [
            "DGS10 minus DFII10 is inflation compensation, not pure expected inflation",
            "Correlations are descriptive",
        ],
        "not_established": [
            "Causation",
            "Forecasting ability",
            "Bond total return",
            "Investment superiority",
            "Alpha or trading edge",
        ],
        "prohibited": [
            "Buy/sell recommendation",
            "Guaranteed return",
            "Exact expected inflation claim",
        ],
    }
    write_json(run / "publication/claim_ledger.json", claim_ledger)
    manifest = {
        "status": "complete",
        "version": VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": checks,
        "outputs": sorted(
            str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()
        ),
        "non_claim": "No investment recommendation, alpha, causal identification, or forecast is established.",
    }
    write_json(run / "run_manifest.json", manifest)
    sums = []
    for f in sorted(
        p for p in run.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        sums.append(f"{sha256(f)}  {f.relative_to(run).as_posix()}\n")
    atomic_text(run / "SHA256SUMS", "".join(sums))
    print(
        json.dumps(
            {
                "status": "complete",
                "run_root": str(run),
                "latest_date": str(common.index.max().date()),
                "latest_nominal_pct": round(float(latest.DGS10), 4),
                "latest_real_pct": round(float(latest.DFII10), 4),
                "latest_inflation_compensation_pct": round(
                    float(latest.inflation_compensation), 4
                ),
                "common_observations": len(common),
                "level_pearson": round(level_r, 4),
                "daily_change_pearson": round(change_r, 4),
                "figures": str(run / "figures"),
                "caption": str(run / "publication/linkedin_caption_draft.txt"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
