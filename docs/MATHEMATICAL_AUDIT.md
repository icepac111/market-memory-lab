# Market Memory Lab Mathematical Audit

## Status

This document records the mathematical definitions currently implemented,
their assumptions, independent audit tests, and unresolved limitations.

Passing tests do not prove universal correctness. They establish that the
implementation satisfies the documented contracts and reference fixtures
included in the repository.

## Audit principles

1. Production functions must be checked against independent calculations.
2. Custom diagnostics must be labeled as custom.
3. Descriptive values must not be presented as probabilities.
4. Statistical association must not be presented as causation.
5. Large samples do not automatically remove bias.
6. Small samples must not be presented as decision-grade evidence.
7. Unsupported analysis must remain explicitly unavailable.
8. Floating-point values require declared tolerances.
9. Data transformations must preserve provenance.
10. Every conclusion must state what could invalidate it.

## Implemented definitions

### Simple return

R_t = P_t / P_(t-1) - 1

Requires strictly positive finite prices.

### Logarithmic return

G_t = log(P_t / P_(t-1))

Requires strictly positive finite prices.

### Wealth index

W_t = W_0 product(i=1 to t) (1 + R_i)

### Cumulative return

CR_T = product(t=1 to T) (1 + R_t) - 1

### Arithmetic annualized mean return

AR = N times mean(R_t)

This is not CAGR and not a forecast.

### Annualized sample volatility

AV = sample_std(R_t) times sqrt(N)

This is a square-root-of-time descriptive convention. Serial dependence,
conditional heteroskedasticity, and structural change can weaken the
interpretation.

### Annualized arithmetic Sharpe ratio

SR = sqrt(N) times mean(R_t - RF_t) / sample_std(R_t - RF_t)

Asset and risk-free returns must have the same periodic frequency.

### Drawdown

D_t = W_t / max(W_s for s less than or equal to t) - 1

### Maximum drawdown

MDD = min(D_t)

### Gaussian mean calibration

Z_mean = (sample_mean - population_mean)
         / (population_sigma / sqrt(n))

### Gaussian variance calibration

Z_variance = (sample_variance - population_variance)
             / (population_variance * sqrt(2 / (n - 1)))

This scaling uses the Gaussian variance of the unbiased sample variance.
Finite-sample normality is not assumed.

### Pearson correlation

Linear sample association between aligned returns.

### Spearman correlation

Rank-based monotonic sample association between aligned returns.

Neither correlation establishes causation.

### Symmetric volatility ratio

SVR = min(volatility_A, volatility_B)
      / max(volatility_A, volatility_B)

This is a Market Memory Lab descriptive diagnostic. It is bounded between
zero and one but is not a probability and has no universal threshold.

### Standardized Wasserstein distance

Each return series is standardized separately using its own sample mean
and sample standard deviation. Wasserstein-1 distance is then calculated.

This isolates remaining empirical-distribution shape differences after
removing separate location and scale. It is not total distribution
distance, not a probability, and has no universal threshold.

### Relationship stability gap

G = absolute_value(correlation_first_half - correlation_second_half)

This is a descriptive split-sample diagnostic. It is not a formal
change-point test, regime probability, or structural-break estimator.

### Active return

A_t = instrument_return_t - reference_return_t

### Mean active return

MAR = mean(A_t)

### Per-period ex-post tracking error

TE = sample_std(A_t)

Tracking error is currently not annualized.

### Tracking beta

Beta = sample_covariance(instrument_returns, reference_returns)
       / sample_variance(reference_returns)

This equals the ordinary least-squares slope when an intercept is included.

### Cumulative normalized divergence

CD_t = product(i=1 to t)(1 + instrument_return_i)
       - product(i=1 to t)(1 + reference_return_i)

This is not premium, discount, NAV deviation, or arbitrage profit.

## Independently audited invariances

- Positive scaling of every price does not change returns.
- Different initial prices do not change equal return paths.
- Pearson correlation is invariant to positive affine transformations.
- Spearman correlation is invariant to strictly increasing transformations.
- Wasserstein distance is nonnegative, symmetric, and zero for identical
  empirical distributions.
- Wealth equals manual sequential compounding.
- Cumulative return equals final-price divided by initial-price minus one.
- Drawdown matches an independent running-peak implementation.
- Tracking error matches manual active-return sample deviation.
- Tracking beta matches an independent least-squares slope.
- Cumulative divergence matches independently compounded wealth paths.

## Known unresolved limitations

### Market calendars and gaps

Missing timestamps are not imputed. Returns may span different calendar
gaps. A future gap diagnostic must distinguish ordinary closures from
missing market observations.

### Nonsynchronous closes

Daily observations from different timezones may represent different
information sets. Warning labels exist, but synchronized-close analysis
is not implemented.

### Currency conversion

Different declared currencies produce warnings. FX-neutral comparison is
not implemented.

### Corporate actions

Adjustment status is declared, but split, dividend, merger, and symbol
history verification is not automated.

### Serial dependence and heteroskedasticity

Basic annualization and Sharpe calculations do not currently apply HAC,
GARCH, block-bootstrap, or other dependence-aware inference.

### Monte Carlo uncertainty

Empirical percentiles are shown descriptively. Confidence intervals for
Monte Carlo percentiles are not yet implemented.

### Multiple comparisons

One-to-many analogue search is not yet implemented. False-discovery and
selection-bias controls are required before candidate ranking can support
formal inference.

### Structural breaks

Split-half stability is descriptive. Formal change-point methods are not
implemented.

### Long memory

Hurst, DFA, GPH, local Whittle, Lo-modified R/S, ARFIMA, and wavelet
estimators are not implemented.

### Tokenization

Tracking analysis is return based. Economic equivalence, legal claims,
redemption rights, conversion ratios, settlement, custody, smart-contract
risk, reserve quality, and synchronized premium or discount remain
unverified.

### Stablecoins

Peg deviation, depeg duration, recovery time, reserve evidence, redemption
mechanics, and venue-weighted prices are not implemented.

### Options

Strike, expiry, option type, exercise style, implied volatility, Greeks,
surface construction, arbitrage constraints, and corporate actions are not
implemented.

### Futures

Contract month, expiry, roll convention, continuous-series construction,
basis, carry, and term structure are not implemented.

### Machine learning

No optimization, model selection, feature selection, neural network,
cross-validation, or hyperparameter search currently supports a financial
claim in Market Memory Lab.

Before machine-learning admission, the project requires:

- chronological splits
- purging and embargo where labels overlap
- nested model selection
- leakage tests
- simple baselines
- repeated out-of-sample evaluation
- transaction-cost assumptions where applicable
- calibration checks
- stability analysis
- multiple-testing awareness
- complete experiment manifests

### Thermodynamic mappings

No market temperature, energy, free energy, equilibrium, phase transition,
or entropy law is currently established.

Any future mapping must define:

- state variables
- units
- invariance properties
- boundary conditions
- null behavior
- empirical interpretation
- incremental information
- falsification conditions

## Review policy

A discovered defect is recorded rather than hidden.

Requirements for a correction:

1. Reproducible failing example
2. Mathematical explanation
3. Regression test
4. Corrected implementation
5. Changelog entry
6. Recalculation of affected evidence
7. Versioned release note when externally visible

## Statistical safeguards added after the initial audit

### Bonferroni adjustment

For m tests:

p_adjusted_i = min(m times p_i, 1)

This controls family-wise error through the Bonferroni inequality and can
be conservative.

### Benjamini-Hochberg adjustment

For sorted p-values p_(1) through p_(m):

p_adjusted_(i)
= min(
    1,
    minimum over j greater than or equal to i
    of m times p_(j) divided by j
  )

Adjusted values are returned in original test order.

Benjamini-Hochberg controls false discovery rate under its applicable
assumptions. It does not establish economic importance or causation.

### Corrected Monte Carlo p-value

For B null simulations and r statistics at least as extreme as observed:

p_corrected = (r + 1) / (B + 1)

The system does not report a zero Monte Carlo p-value merely because no
finite simulation exceeded the observation.

### Binomial simulation uncertainty

The raw null exceedance probability receives an exact Clopper-Pearson
interval.

This interval quantifies finite simulation-count uncertainty. It is not a
confidence interval for investment return or model profitability.

### Newey-West mean standard error

Using Bartlett weights and maximum lag q:

gamma_j
= (1 / n) sum from t=j+1 to n of
  (x_t - mean_x)(x_(t-j) - mean_x)

long_run_variance
= gamma_0
  + 2 sum from j=1 to q of
    (1 - j / (q + 1)) gamma_j

standard_error_of_mean
= square_root(long_run_variance / n)

Required assumptions and limitations:

- observations are ordered
- observations represent equally spaced periods
- the lag choice is explicit
- this is asymptotic inference
- HAC does not solve endogeneity, omitted variables, structural breaks,
  data snooping, or poor model specification
