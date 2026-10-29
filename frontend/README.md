# Market Memory Lab — Evidence Workbench

An experimental React interface for inspecting financial-research outputs.

## What works

- Local JSON case loading without an upload request.
- INTC/MU hypothetical growth-of-$100 chart.
- Observation-date inspection and asset selection.
- Correlation and maximum-drawdown cards.
- Visible interpretation boundaries.
- Interactive decorative orbit.
- Demonstration case editor and preparation checklist.

The browser viewer displays precomputed outputs. It does not run the
Python research engine or retrieve live prices.

## Existing Python integration

The local historical case was produced using:

- YahooHistoricalProvider.fetch_prices
- analyze_empirical_prices
- compare_validated_assets

The original analytical functions were not modified for this interface.
A display baseline of 1.0 was added on the first price date when exporting
the existing return-based wealth series.

## Local development

From this directory:

```bash
npm ci
npm run dev
```

Use a Node version compatible with the locked Vite and ESLint dependencies.
The local build environment used Node v22.23.2.

## Checks

```bash
npm run lint
npm run build
```

## Case record: INTC versus MU

The following are author-reported outputs of a local descriptive run,
not independently reproduced results or a trading recommendation.

- Price coverage: 2025-01-02 through 2026-10-28.
- Price observations per asset: 457.
- Overlapping return observations: 456.
- Pearson correlation: 0.5424752579241587.
- Spearman correlation: 0.5339107152552416.
- INTC maximum drawdown: -0.41904359418853876.
- MU maximum drawdown: -0.4075308747713571.
- First-half correlation: 0.45183394758030976.
- Second-half correlation: 0.5992317274082772.

The half-sample difference is descriptive, not a formal break test.
No causality, regime, memory, alpha, or Turing-performance claim is made.

## Data boundary

Downloaded provider observations and private case JSON are not included.
The source code does not grant rights to third-party data.
Users must independently check applicable source terms.

This viewer accepts the private-equity-pair-case-v0.2 format for INTC/MU.
It is not a general-purpose arbitrary-case viewer yet.

## Limitations

- Manual file-based integration, not an automatic Python API.
- No live market feed.
- No persistent draft storage.
- No covariance-driven 3D visualization yet.
- The orbital illustration is decorative.
- Checklist completion is self-reported, not scientific validation.
- A successful build does not establish empirical validity.

## Attribution

Created by Pranshu Mihirbhai Jayswal as an independent learning and
research-engineering project. No institutional endorsement is implied.
