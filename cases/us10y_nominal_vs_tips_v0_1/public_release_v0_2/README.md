# U.S. 10-Year Nominal Treasury Yield versus Real TIPS Yield

## Research question

What can be learned from how the U.S. 10-year nominal Treasury yield and
10-year real TIPS yield move together or apart?

## Data

The analysis uses 5,932 common valid observations from 2003-01-02 through
2026-09-17. Missing observations were not filled.

The analysis compares the nominal 10-year Treasury constant-maturity yield
with the 10-year real constant-maturity TIPS yield.

## Central identity

Nominal yield minus real yield equals market-implied inflation compensation
under this measurement construction.

Inflation compensation is not identical to expected inflation. It may also
reflect inflation-risk compensation, relative liquidity, market technicals,
and measurement effects.

## Verified descriptive results

- Latest nominal yield: 4.94%
- Latest real yield: 2.61%
- Latest difference: 2.33 percentage points
- Pearson correlation in levels: 0.942
- Pearson correlation in successive changes: 0.812
- Rolling change-correlation range: 0.615 to 0.949
- Common observations: 5,932

The rolling relationship changes through time. A full-sample correlation
should not be treated as a permanent market law.

## Evidence design

The release includes:

- five repaired figures;
- recomputed metrics;
- publication-quality checks;
- image checks;
- a run manifest;
- file checksums;
- reproducible source scripts stored under `scripts/cases/`.

## Claim boundaries

This is a descriptive market-structure study.

It does not establish:

- future inflation;
- future Treasury or TIPS returns;
- causal effects;
- investment superiority;
- a trading signal;
- a buy or sell recommendation;
- validated alpha.

The molecular figure is an intuition layer for measured relationships.
Candidate influences are not identified causal effects.

## Reproducibility

Source scripts:

- `scripts/cases/us10y_nominal_vs_tips_v0_1/download_us10y_nominal_vs_tips_v0_1.py`
- `scripts/cases/us10y_nominal_vs_tips_v0_1/analyze_us10y_nominal_vs_tips_v0_1.py`
- `scripts/cases/us10y_nominal_vs_tips_v0_1/repair_us10y_nominal_vs_tips_publication_v0_2.py`

This repository is for educational and scientific research. It does not
provide individualized investment advice.
