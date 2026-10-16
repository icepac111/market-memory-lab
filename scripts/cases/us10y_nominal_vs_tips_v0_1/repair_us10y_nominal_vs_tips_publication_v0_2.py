#!/usr/bin/env python3
"""Market Memory Lab publication chart repair, v0.2.0.

Repairs the five-chart LinkedIn package from an existing completed analysis run.
The script does not download data, alter prior artifacts, stage git changes, or
make investment recommendations. It reads the preserved common-valid analysis,
recomputes every plotted diagnostic directly, validates critical identities,
and writes a new immutable publication-repair run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch

VERSION = "0.2.0"
NAVY = "#071A2F"
NAVY_2 = "#0B223D"
GRID = "#27445F"
WHITE = "#F5F7FA"
MUTED = "#91A7BA"
GOLD = "#D4A94F"
BLUE = "#4F91D0"
CYAN = "#76C7C0"
RED = "#E07872"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def pct_rank(s: pd.Series, x: float) -> float:
    clean = s.dropna().to_numpy()
    return float(100 * np.mean(clean <= x))


def style_ax(ax, title=None, ylabel=None):
    ax.set_facecolor(NAVY)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.55)
    ax.tick_params(colors=WHITE, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#42617D")
    if title:
        ax.set_title(title, loc="left", color=WHITE, fontsize=16, weight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=WHITE, fontsize=11)


def save(fig, path: Path):
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)


def load_and_validate(source_run: Path) -> tuple[pd.DataFrame, dict]:
    source = source_run / "processed" / "common_valid_analysis.csv"
    if not source.is_file():
        raise SystemExit(f"Missing required analysis file: {source}")
    df = pd.read_csv(source, parse_dates=["observation_date"])
    required = {"observation_date", "DGS10", "DFII10", "inflation_compensation"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values("observation_date", kind="mergesort").set_index(
        "observation_date"
    )
    if df.index.duplicated().any():
        raise SystemExit("Duplicate observation dates found")
    if not df.index.is_monotonic_increasing:
        raise SystemExit("Observation dates are not monotonic")
    if df[["DGS10", "DFII10", "inflation_compensation"]].isna().any().any():
        raise SystemExit("Common-valid panel contains missing core values")
    identity_error = (df.DGS10 - df.DFII10 - df.inflation_compensation).abs().max()
    if identity_error > 1e-10:
        raise SystemExit(f"Spread identity failed: max error {identity_error}")

    # Recompute plot inputs. Do not trust derived columns from the earlier run.
    df["nominal_change_bps"] = df.DGS10.diff() * 100
    df["real_change_bps"] = df.DFII10.diff() * 100
    df["spread_change_bps"] = df.inflation_compensation.diff() * 100
    df["rolling_252_cross_corr"] = df.nominal_change_bps.rolling(
        252, min_periods=126
    ).corr(df.real_change_bps)
    df["rolling_63_nominal_vol"] = df.nominal_change_bps.rolling(
        63, min_periods=32
    ).std()
    df["rolling_63_real_vol"] = df.real_change_bps.rolling(63, min_periods=32).std()
    df["rolling_63_spread_vol"] = df.spread_change_bps.rolling(63, min_periods=32).std()

    valid_roll = df.rolling_252_cross_corr.dropna()
    if valid_roll.empty:
        raise SystemExit("Rolling cross-correlation has no valid observations")
    if (valid_roll.abs() > 1 + 1e-12).any():
        raise SystemExit("Rolling cross-correlation escaped [-1, 1]")
    if valid_roll.nunique() <= 2:
        raise SystemExit(
            "Rolling cross-correlation is effectively constant; refusing publication"
        )

    changes = df[["nominal_change_bps", "real_change_bps"]].dropna()
    metrics = {
        "source_file": str(source),
        "source_sha256": sha256(source),
        "common_start": df.index.min().date().isoformat(),
        "common_end": df.index.max().date().isoformat(),
        "common_observations": int(len(df)),
        "change_pairs": int(len(changes)),
        "latest_nominal_pct": float(df.DGS10.iloc[-1]),
        "latest_real_pct": float(df.DFII10.iloc[-1]),
        "latest_inflation_compensation_pct": float(df.inflation_compensation.iloc[-1]),
        "latest_spread_percentile": pct_rank(
            df.inflation_compensation, float(df.inflation_compensation.iloc[-1])
        ),
        "level_pearson": float(df.DGS10.corr(df.DFII10, method="pearson")),
        "level_spearman": float(df.DGS10.corr(df.DFII10, method="spearman")),
        "change_pearson": float(
            changes.nominal_change_bps.corr(changes.real_change_bps, method="pearson")
        ),
        "change_spearman": float(
            changes.nominal_change_bps.corr(changes.real_change_bps, method="spearman")
        ),
        "rolling_cross_corr_min": float(valid_roll.min()),
        "rolling_cross_corr_max": float(valid_roll.max()),
        "rolling_cross_corr_latest": float(valid_roll.iloc[-1]),
        "rolling_cross_corr_valid_n": int(valid_roll.size),
        "spread_identity_max_abs_error": float(identity_error),
    }
    return df, metrics


def chart_1(df: pd.DataFrame, out: Path):
    latest = df.iloc[-1]
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.2), sharex=True, facecolor=NAVY)
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.08, right=0.98, hspace=0.22)
    fig.suptitle(
        "THE U.S. 10-YEAR RATE STRUCTURE",
        color=WHITE,
        fontsize=24,
        weight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        f"Nominal yield, real TIPS yield, and inflation compensation | {df.index.min().date()} to {df.index.max().date()}",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    axes[0].plot(df.index, df.DGS10, color=GOLD, lw=1.35)
    style_ax(axes[0], "Nominal 10-Year Treasury Constant-Maturity Yield", "Percent")
    axes[1].plot(df.index, df.DFII10, color=BLUE, lw=1.35)
    axes[1].axhline(0, color=WHITE, lw=0.8, alpha=0.55)
    style_ax(
        axes[1], "Real 10-Year Inflation-Indexed Constant-Maturity Yield", "Percent"
    )
    axes[2].plot(df.index, df.inflation_compensation, color=WHITE, lw=1.4)
    style_ax(axes[2], "Inflation Compensation: Nominal Minus Real", "Percentage points")
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax, value, color in [
        (axes[0], latest.DGS10, GOLD),
        (axes[1], latest.DFII10, BLUE),
        (axes[2], latest.inflation_compensation, WHITE),
    ]:
        ax.scatter(df.index[-1], value, s=32, color=color, zorder=5)
        ax.annotate(
            f"{value:.2f}%",
            (df.index[-1], value),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=10,
            weight="bold",
        )

    fig.text(
        0.01,
        0.018,
        "Source: Federal Reserve H.15 series via preserved FRED CSVs. Inflation compensation is not exact expected inflation.",
        color=MUTED,
        fontsize=8.5,
    )
    save(fig, out / "01_rate_structure_repaired.png")


def chart_2(df: pd.DataFrame, metrics: dict, out: Path):
    changes = df[["nominal_change_bps", "real_change_bps"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=NAVY)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.98, wspace=0.16)
    fig.suptitle(
        "LEVELS LOOK CONNECTED. SUCCESSIVE CHANGES ARE THE HARDER TEST.",
        color=WHITE,
        fontsize=20,
        weight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.885,
        f"{len(df):,} common yield observations and {len(changes):,} successive-observation change pairs",
        ha="center",
        color=MUTED,
        fontsize=10.5,
    )

    axes[0].scatter(
        df.DFII10,
        df.DGS10,
        s=9,
        alpha=0.24,
        color=GOLD,
        edgecolors="none",
        rasterized=True,
    )
    style_ax(axes[0], "Yield Levels", "Nominal yield (%)")
    axes[0].set_xlabel("Real yield (%)", color=WHITE, fontsize=11)
    axes[0].text(
        0.035,
        0.955,
        f"Pearson r = {metrics['level_pearson']:.2f}\nSpearman ρ = {metrics['level_spearman']:.2f}",
        transform=axes[0].transAxes,
        va="top",
        color=WHITE,
        fontsize=12,
        bbox=dict(
            facecolor=NAVY_2, edgecolor="none", alpha=0.78, boxstyle="round,pad=.45"
        ),
    )

    axes[1].scatter(
        changes.real_change_bps,
        changes.nominal_change_bps,
        s=9,
        alpha=0.25,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
    )
    style_ax(
        axes[1],
        "Changes Between Successive Common Observations",
        "Nominal change (bps)",
    )
    axes[1].set_xlabel("Real change (bps)", color=WHITE, fontsize=11)
    axes[1].text(
        0.035,
        0.955,
        f"Pearson r = {metrics['change_pearson']:.2f}\nSpearman ρ = {metrics['change_spearman']:.2f}",
        transform=axes[1].transAxes,
        va="top",
        color=WHITE,
        fontsize=12,
        bbox=dict(
            facecolor=NAVY_2, edgecolor="none", alpha=0.78, boxstyle="round,pad=.45"
        ),
    )
    fig.text(
        0.01,
        0.035,
        "Descriptive correlations only. Common information can move both series; no causal or forecasting claim is implied.",
        color=MUTED,
        fontsize=8.5,
    )
    save(fig, out / "02_levels_vs_changes_repaired.png")


def chart_3(df: pd.DataFrame, metrics: dict, out: Path):
    s = df.rolling_252_cross_corr.dropna()
    fig, ax = plt.subplots(figsize=(13.5, 6.3), facecolor=NAVY)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.98)
    fig.suptitle(
        "THE NOMINAL-REAL LINK IS STRONG, BUT NOT CONSTANT",
        color=WHITE,
        fontsize=22,
        weight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.885,
        "252-observation rolling cross-correlation of successive yield changes",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    ax.plot(s.index, s, color=GOLD, lw=1.6)
    ax.fill_between(s.index, 0, s, color=GOLD, alpha=0.09)
    ax.axhline(0, color=WHITE, lw=0.8, alpha=0.6)
    style_ax(ax, None, "Correlation")
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.scatter(s.index[-1], s.iloc[-1], s=42, color=WHITE, zorder=5)
    ax.annotate(
        f"Latest {s.iloc[-1]:.2f}",
        (s.index[-1], s.iloc[-1]),
        xytext=(-12, 12),
        textcoords="offset points",
        ha="right",
        color=WHITE,
        weight="bold",
    )
    ax.text(
        0.015,
        0.04,
        f"Observed range: {s.min():.2f} to {s.max():.2f}",
        transform=ax.transAxes,
        color=WHITE,
        fontsize=11,
        bbox=dict(
            facecolor=NAVY_2, edgecolor="none", alpha=0.8, boxstyle="round,pad=.4"
        ),
    )
    fig.text(
        0.01,
        0.035,
        "Each value uses up to 252 successive common-observation changes and requires at least 126 valid pairs.",
        color=MUTED,
        fontsize=8.5,
    )
    save(fig, out / "03_rolling_cross_correlation_repaired.png")


def chart_4(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(13.5, 6.3), facecolor=NAVY)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.98)
    fig.suptitle(
        "REPRICING INTENSITY CHANGES THROUGH TIME",
        color=WHITE,
        fontsize=22,
        weight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.885,
        "63-observation rolling volatility of successive yield changes",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    ax.plot(
        df.index, df.rolling_63_nominal_vol, color=GOLD, lw=1.25, label="Nominal yield"
    )
    ax.plot(df.index, df.rolling_63_real_vol, color=BLUE, lw=1.25, label="Real yield")
    ax.plot(
        df.index,
        df.rolling_63_spread_vol,
        color=WHITE,
        lw=1.25,
        label="Inflation compensation",
    )
    style_ax(ax, None, "Standard deviation (bps)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    leg = ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
    for t in leg.get_texts():
        t.set_color(WHITE)
    fig.text(
        0.01,
        0.035,
        "This measures yield-change volatility, not bond total-return volatility. Missing observations were not filled.",
        color=MUTED,
        fontsize=8.5,
    )
    save(fig, out / "04_change_volatility_repaired.png")


def chart_5(df: pd.DataFrame, metrics: dict, out: Path):
    latest = df.iloc[-1]
    fig, ax = plt.subplots(figsize=(13.5, 8), facecolor=NAVY)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.94,
        "THE NOMINAL-REAL TREASURY STRUCTURE",
        ha="center",
        va="top",
        color=WHITE,
        fontsize=24,
        weight="bold",
    )
    ax.text(
        0.5,
        0.885,
        "A visual decomposition of one maturity into nominal yield, real yield, and their difference",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    left = (0.24, 0.48)
    right = (0.76, 0.48)
    radius = 0.115
    ax.add_patch(
        Circle(
            left,
            radius,
            transform=ax.transAxes,
            facecolor=GOLD,
            edgecolor="#E9C979",
            lw=2,
        )
    )
    ax.add_patch(
        Circle(
            right,
            radius,
            transform=ax.transAxes,
            facecolor=BLUE,
            edgecolor="#78B1E8",
            lw=2,
        )
    )
    ax.plot(
        [left[0] + radius, right[0] - radius],
        [left[1], right[1]],
        transform=ax.transAxes,
        color=WHITE,
        lw=6,
        solid_capstyle="round",
    )
    ax.text(
        left[0],
        left[1],
        f"NOMINAL 10Y\n{latest.DGS10:.2f}%",
        ha="center",
        va="center",
        color=NAVY,
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        right[0],
        right[1],
        f"REAL 10Y\n{latest.DFII10:.2f}%",
        ha="center",
        va="center",
        color=WHITE,
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.59,
        f"INFLATION COMPENSATION\n{latest.inflation_compensation:.2f}%",
        ha="center",
        va="center",
        color=WHITE,
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
        bbox=dict(facecolor=NAVY_2, edgecolor="#42617D", boxstyle="round,pad=.55"),
    )

    factors = [
        "Inflation information",
        "Monetary policy",
        "Growth expectations",
        "Treasury supply",
        "Risk aversion",
        "Relative liquidity",
    ]
    xs = [0.12, 0.27, 0.42, 0.58, 0.73, 0.88]
    for x, label in zip(xs, factors):
        ax.text(
            x,
            0.76,
            label,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9.5,
            transform=ax.transAxes,
        )
        arrow = FancyArrowPatch(
            (x, 0.735),
            (0.5, 0.64),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.2,
            linestyle=(0, (3, 4)),
            color="#4A6985",
            alpha=0.85,
        )
        ax.add_patch(arrow)
    ax.text(
        0.5,
        0.70,
        "Candidate influences, not identified causal effects",
        ha="center",
        color=CYAN,
        fontsize=9.5,
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.245,
        f"Common observation: {df.index.max().date()}  |  Spread percentile in this sample: {metrics['latest_spread_percentile']:.1f}",
        ha="center",
        color=WHITE,
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.18,
        "Observed structure, not a forecasting model or investment recommendation",
        ha="center",
        color=MUTED,
        fontsize=10.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.075,
        "Source: Federal Reserve H.15 series via preserved FRED CSVs",
        ha="center",
        color=MUTED,
        fontsize=8.5,
        transform=ax.transAxes,
    )
    save(fig, out / "05_treasury_structure_repaired.png")


def main():
    p = argparse.ArgumentParser(
        description="Repair and validate the U.S. nominal-vs-real Treasury publication charts."
    )
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()
    root = args.project_root.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing project root: {root}")
    if not source_run.is_dir():
        raise SystemExit(f"Missing source run: {source_run}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_run = (
        root / "cases/us10y_nominal_vs_tips_v0_1/publication_repairs/runs" / run_id
    )
    if out_run.exists():
        raise SystemExit(f"Refusing overwrite: {out_run}")
    figures = out_run / "figures"
    validation = out_run / "validation"
    publication = out_run / "publication"
    for d in [figures, validation, publication]:
        d.mkdir(parents=True, exist_ok=False)

    df, metrics = load_and_validate(source_run)
    chart_1(df, figures)
    chart_2(df, metrics, figures)
    chart_3(df, metrics, figures)
    chart_4(df, figures)
    chart_5(df, metrics, figures)

    # Validate image dimensions and file sizes.
    from PIL import Image

    image_audit = {}
    for path in sorted(figures.glob("*.png")):
        with Image.open(path) as im:
            image_audit[path.name] = {
                "width": im.width,
                "height": im.height,
                "mode": im.mode,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            if im.width < 1800 or im.height < 900:
                raise SystemExit(f"Publication image too small: {path} {im.size}")
    if len(image_audit) != 5:
        raise SystemExit(f"Expected 5 figures, found {len(image_audit)}")

    write_json(validation / "recomputed_metrics.json", metrics)
    write_json(validation / "image_audit.json", image_audit)
    qa = {
        "rolling_cross_correlation_recomputed_directly": True,
        "rolling_cross_correlation_not_constant": True,
        "spread_identity_passed": metrics["spread_identity_max_abs_error"] <= 1e-10,
        "successive_observation_wording_used": True,
        "molecular_node_labels_unclipped_by_design": True,
        "candidate_influences_labeled_noncausal": True,
        "investment_advice_present": False,
        "status": "publication_candidate",
    }
    write_json(validation / "publication_qa.json", qa)

    caption = f"""I wanted to understand what the U.S. 10-year nominal Treasury yield and 10-year real TIPS yield reveal when they move together or apart.

I analyzed {metrics["common_observations"]:,} common valid observations from {metrics["common_start"]} through {metrics["common_end"]} using Federal Reserve H.15 constant-maturity yield series. Missing observations were not filled.

On the latest common date, the nominal yield was {metrics["latest_nominal_pct"]:.2f}% and the real yield was {metrics["latest_real_pct"]:.2f}%. Their difference was {metrics["latest_inflation_compensation_pct"]:.2f} percentage points, or {metrics["latest_inflation_compensation_pct"] * 100:.0f} basis points.

That difference is inflation compensation, not exact expected inflation. It can also reflect inflation risk premium, relative liquidity, market technicals, and measurement effects.

The harder test was whether changes between successive common observations moved together. The full-sample Pearson correlation was {metrics["level_pearson"]:.2f} in levels and {metrics["change_pearson"]:.2f} in successive changes. The rolling cross-correlation also moved through a range rather than remaining constant, which is why a single full-sample number should not be treated as a permanent market law.

The molecular visual is an intuition layer for the measured structure. The dotted arrows show candidate influences, not identified causal effects.

What this evidence does not establish:
• future inflation
• future Treasury or TIPS returns
• causation
• investment superiority
• a trading signal
• a buy or sell recommendation

This is an educational market-structure study, not individualized investment advice.

#FixedIncome #Treasury #TIPS #FinancialDataScience #FinTech #QuantitativeResearch #AssetPricing
"""
    atomic_text(publication / "linkedin_caption_candidate.txt", caption)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "source_analysis_sha256": metrics["source_sha256"],
        "python": sys.version,
        "platform": platform.platform(),
        "claim_boundary": "Descriptive market-structure analysis only. No causal, predictive, alpha, or investment recommendation claim.",
    }
    write_json(out_run / "run_manifest.json", manifest)
    sums = []
    for path in sorted(
        p for p in out_run.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        sums.append(f"{sha256(path)}  {path.relative_to(out_run).as_posix()}\n")
    atomic_text(out_run / "SHA256SUMS", "".join(sums))

    print(
        json.dumps(
            {
                "status": "complete",
                "publication_run": str(out_run),
                "source_run": str(source_run),
                "figures": str(figures),
                "caption": str(publication / "linkedin_caption_candidate.txt"),
                "rolling_corr_min": round(metrics["rolling_cross_corr_min"], 4),
                "rolling_corr_max": round(metrics["rolling_cross_corr_max"], 4),
                "rolling_corr_latest": round(metrics["rolling_cross_corr_latest"], 4),
                "image_count": len(image_audit),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
