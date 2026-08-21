# TSLA Empirical Asset Lab Report v0.1

## Market Memory Lab

**Creator:** Pranshu Mihirbhai Jayswal
**Report date:** August 20, 2026
**Research status:** Descriptive empirical investigation
**Investment conclusion:** Abstain

## Executive summary

A daily adjusted-close series for Tesla, Inc. was processed through the
Market Memory Lab Empirical Asset Lab.

The uploaded series passed the displayed structural validation checks with
1,667 observations covering January 2, 2020 through August 20, 2026.

Across this historical interval, the application calculated a cumulative
return of 1,103.21%, annualized sample volatility of 64.91%, arithmetic
annualized return of 58.63%, and maximum drawdown of -73.63%.

The endpoint and the path tell different stories. Historical normalized
growth was extraordinary, but the path included an approximately 73.63%
decline from a previous peak.

This report does not establish memory, forecast skill, causation, alpha,
economic usefulness, profitability, suitability, or a trading conclusion.

## Dataset declaration

| Field | Declaration |
|---|---|
| Dataset name | TSLA Adjusted Close 2020-2026 |
| Asset identifier | TSLA |
| Asset class | Equity |
| Venue or market | NASDAQ |
| Currency | USD |
| Observation frequency | Daily |
| Price field | Adjusted Close |
| Timezone declaration | America/New_York |
| Source path | Yahoo Finance historical response accessed through yfinance |
| Observation count | 1,667 |
| First observation date | 2020-01-02 |
| Last observation date | 2026-08-20 |
| Raw local CSV SHA-256 | `95e08c0f0510a675749b1cbb4ddc81b36feb16fc9001d4927c69f4960dc8a4e5` |
| Data-distribution status | Raw provider observations are not included in this repository |
| Usage boundary | Private descriptive research; provider-data rights require independent review |

The SHA-256 above identifies the local downloaded CSV used during the
displayed analysis. It is not claimed to be the application's canonical
dataset hash unless independently shown to match that separately generated
artifact.

## Validation evidence

| Validation check | Result |
|---|---:|
| Validation state | Passed |
| Input rows | 1,667 |
| Invalid dates | 0 |
| Invalid prices | 0 |
| Duplicate dates | 0 |
| Nonpositive prices | 0 |
| Originally chronological | True |

These checks establish conformity with the displayed input-validation
requirements. They do not independently establish provider accuracy,
licensing rights, economic validity, or forecasting usefulness.

## Verified descriptive metrics

| Metric | Displayed result |
|---|---:|
| Cumulative return | 1,103.21% |
| Maximum drawdown | -73.63% |
| Annualized sample volatility | 64.91% |
| Arithmetic annualized return | 58.63% |
| Normalized historical ending value from $100 | Approximately $1,203.21 |

The normalized ending value is calculated as:

`$100 × (1 + 11.0321) = $1,203.21`

The value is a historical normalization over the declared interval. It is
not a forecast, recommendation, investor-specific realized return, or
expected future value.

## Interpretation

### Endpoint performance

The cumulative-return calculation describes very large historical growth
over the selected start and end dates.

The result is highly dependent on the declared interval and does not imply
that another entry date, exit date, observation convention, or future period
would produce a similar outcome.

### Path risk

The maximum drawdown of -73.63% indicates that the historical adjusted-price
path experienced a decline of approximately 73.63% from a previous peak.

This demonstrates why endpoint performance alone is inadequate evidence
about the experience or risk of the historical path.

### Volatility

Annualized sample volatility of 64.91% reflects substantial historical
dispersion in daily returns under the application's declared daily-frequency
annualization convention.

It is not a forecast of future volatility or a probability of loss.

### Arithmetic annualized return

The arithmetic annualized return of 58.63% is not equivalent to compounded
annual growth.

Arithmetic annualization scales an average periodic return and does not
reproduce the compounded wealth path, especially in the presence of large
volatility.

## Relationship to the earlier Toyota-Tesla investigation

An earlier private Toyota Motor ADR versus Tesla investigation covered
January 2020 through August 12, 2026 and reported:

- TSLA normalized historical ending value from $100: $1,141.79
- TM normalized historical ending value from $100: $155.15
- TSLA maximum drawdown: approximately -73.6%
- TM maximum drawdown: approximately -36.8%
- Overlapping daily returns: 1,660
- Pearson correlation: 0.2827
- Spearman correlation: 0.2776
- Symmetric volatility ratio: 0.4140
- Trading conclusion: abstain

The updated standalone TSLA analysis extends through August 20, 2026 and
reports a maximum drawdown of -73.63%.

The close agreement between the earlier and updated TSLA drawdown values is
a descriptive consistency observation. It is not labeled a formal
replication because the endpoints, analysis roles, overlap construction, and
potential provider revisions differ.

The prior private provider observations and private artifacts are not
included in this report or repository.

## Scientific conclusion boundary

### Supported within this report

- The uploaded series passed the displayed structural validation checks.
- The application produced the declared descriptive historical metrics.
- Historical normalized growth and historical path risk were both large.
- Endpoint performance alone concealed substantial drawdown and volatility.
- The result is suitable for documented descriptive research within the
  stated data and methodological boundaries.

### Not established

This report does not establish:

- persistent temporal memory
- forecast skill
- future price direction
- future return
- alpha
- causation
- statistical significance
- false-positive control
- economic usefulness
- profitability
- tradability
- investor suitability
- a buy recommendation
- a sell recommendation
- a short recommendation
- prediction-market credibility
- novelty
- patentability

## Application conclusions

| Conclusion | State |
|---|---|
| Memory conclusion | Not tested |
| Trading conclusion | Abstain |

## One-sentence finding

Across the validated daily TSLA adjusted-close series, historical normalized
growth was extraordinary, but it coexisted with approximately 64.91%
annualized sample volatility and a 73.63% maximum drawdown, demonstrating why
endpoint performance alone is inadequate evidence about the financial path,
future predictability, or investment usefulness.

## Reproducibility and provenance notes

- The raw downloaded CSV remains outside the repository.
- Raw provider observations must not be committed or redistributed through
  this report.
- The local raw CSV hash is recorded so the exact input can be identified
  without publishing the underlying observations.
- Future reruns should record their own endpoint, observation count, source
  declaration, frequency convention, price adjustment convention, and hash.
- Any difference in source observations, adjustment handling, timestamps,
  start date, end date, or canonicalization creates a distinct evidence run.
