# Market Memory Lab Learning Center

## Learn what the platform measures, what the evidence permits, and when to abstain

Most systems optimize for giving an answer.

**Market Memory Lab optimizes for discovering when the answer is not yet justified.**

This Learning Center provides one canonical educational reference for the
public GitHub repository and the Streamlit application. The dashboard reads
this document directly, so the mathematical explanations do not need to be
maintained in two separate places.

Market Memory Lab is a research platform. It does not provide personalized
investment advice, guarantee financial outcomes, or convert a descriptive
relationship into a trading decision.

---

## Choose your path

### New learner

Start here if financial statistics are new to you.

Recommended order:

1. Prices and returns
2. Wealth and cumulative return
3. Volatility
4. Drawdown
5. Correlation
6. Relationship stability
7. Evidence adequacy
8. Why the platform may abstain

### Student

Use the formulas, symbol guides, numerical examples, assumptions, and failure
modes to understand how each calculation works.

Recommended order:

1. Simple and log returns
2. Annualized arithmetic return
3. Annualized sample volatility
4. Pearson and Spearman association
5. Standardized Wasserstein distance
6. Tracking integrity
7. Synthetic null processes
8. Statistical safeguards

### Researcher

Inspect the exact definitions, implementation references, automated tests,
custom diagnostics, limitations, null processes, and blocked conclusions.

Recommended references:

- `docs/MATHEMATICAL_AUDIT.md`
- `src/market_memory_lab/metrics.py`
- `src/market_memory_lab/empirical.py`
- `src/market_memory_lab/similarity.py`
- `src/market_memory_lab/tracking.py`
- `src/market_memory_lab/inference.py`
- `src/market_memory_lab/synthetic.py`
- `tests/`

---

# 1. From prices to returns

## Simple return

### 1. Completely new explanation

A simple return tells us how much a price increased or decreased relative to
its previous value.

If a price rises from 100 dollars to 105 dollars, the simple return is 5%.

### 2. Practical intuition

Prices themselves are difficult to compare across instruments because one
instrument may trade near 20 while another trades near 2,000. Returns express
changes relative to each instrument's own prior price.

### 3. Mathematical definition

\[
r_t = \frac{P_t}{P_{t-1}} - 1
\]

### Symbol guide

- \(r_t\): simple return for period \(t\)
- \(P_t\): price at the end of period \(t\)
- \(P_{t-1}\): price at the end of the previous period

### Numerical example

Suppose:

\[
P_{t-1} = 100,\qquad P_t = 105
\]

Then:

\[
r_t = \frac{105}{100} - 1 = 0.05
\]

The simple return is 0.05, or 5%.

### What Market Memory Lab calculates

The platform calculates each instrument's returns independently from that
instrument's validated chronological price series.

For pairwise analysis, the resulting returns are aligned on common return
dates. Raw prices are not aligned first to create returns across mismatched
calendars.

### What automated tests verify

Tests verify manual return calculations, chronological input behavior,
price-scale invariance, validation failures, and return alignment behavior.

### Assumptions

- Prices refer to consecutive observations within the declared series.
- Price values are finite and strictly positive.
- Adjustment status is declared.
- Observation dates have been parsed and validated.

### Failure modes

- Missing or duplicated dates
- Nonpositive prices
- Incorrect adjustment treatment
- Corporate actions represented inconsistently
- Comparing observations with incompatible frequencies
- Treating a data-provider error as a market move

### Allowed conclusion

The price changed by the calculated proportion between the two observations.

### Blocked conclusion

A positive return does not establish predictability, skill, causation,
profitability after costs, or future performance.

### Code reference

`src/market_memory_lab/empirical.py`

### Test reference

`tests/test_empirical.py` and `tests/test_mathematical_audit.py`

---

## Log return

### 1. Completely new explanation

A log return is another way to describe a price change. Log returns are useful
because consecutive log returns can be added across time.

### 2. Practical intuition

Simple returns compound by multiplication. Log returns aggregate by addition.
This can simplify mathematical analysis, but a log return is not identical to
a simple return.

### 3. Mathematical definition

\[
g_t = \log\left(\frac{P_t}{P_{t-1}}\right)
\]

### Symbol guide

- \(g_t\): log return for period \(t\)
- \(\log\): natural logarithm
- \(P_t\): current price
- \(P_{t-1}\): previous price

### Numerical example

If a price rises from 100 to 105:

\[
g_t = \log(105/100) \approx 0.04879
\]

The corresponding simple return is 0.05. They are close for this small change,
but they are not the same number.

### What Market Memory Lab calculates

The platform calculates log price ratios from validated, strictly positive
prices.

### What automated tests verify

Tests verify manual log-return calculations and expected invariance to a
positive rescaling of the entire price path.

### Assumptions and failure modes

Log returns require strictly positive prices. They are invalid for a price of
zero or below. Interpretation may also fail when the selected data field is
not an economically meaningful price.

### Allowed conclusion

The logarithmic change between two positive consecutive prices was measured.

### Blocked conclusion

A log-return series is not automatically Gaussian, stationary, independent,
or suitable for a specific forecasting model.

### Code reference

`src/market_memory_lab/empirical.py`

### Test reference

`tests/test_empirical.py` and `tests/test_mathematical_audit.py`

---

# 2. Wealth and cumulative return

## Wealth path

### 1. Completely new explanation

A wealth path shows what happens to an initial amount when each simple return
is applied in sequence.

### 2. Practical intuition

A gain followed by a loss cannot generally be summarized by adding the two
percentage returns. Compounding preserves the order-by-order multiplication.

### 3. Mathematical definition

\[
W_t = W_0 \prod_{i=1}^{t}(1+r_i)
\]

### Symbol guide

- \(W_t\): wealth after period \(t\)
- \(W_0\): initial wealth
- \(r_i\): simple return during period \(i\)
- \(\prod\): multiply all listed terms

### Numerical example

Let initial wealth be 100 and returns be 10% followed by minus 5%.

\[
W_2 = 100(1.10)(0.95) = 104.50
\]

### What Market Memory Lab calculates

The platform constructs a normalized compounded wealth path from simple
returns.

### What automated tests verify

Tests independently verify sequential compounding and initial-price
independence.

### Assumptions and failure modes

- Returns must be ordered correctly.
- Simple returns below minus 100% are incompatible with an ordinary long-only
  wealth interpretation.
- Fees, taxes, spreads, financing, and execution are not automatically
  included.
- A normalized wealth path is not a realized portfolio record.

### Allowed conclusion

The path describes mathematical compounding of the supplied returns.

### Blocked conclusion

It does not establish that the path was investable or achievable.

### Code reference

`src/market_memory_lab/metrics.py` and
`src/market_memory_lab/empirical.py`

### Test reference

`tests/test_metrics.py` and `tests/test_mathematical_audit.py`

---

## Cumulative return

### Mathematical definition

\[
R_{\mathrm{cum}} =
\prod_{t=1}^{T}(1+r_t)-1
\]

Using the previous example:

\[
R_{\mathrm{cum}} = (1.10)(0.95)-1 = 0.045
\]

The cumulative return is 4.5%.

### Allowed conclusion

The supplied return sequence compounded to the calculated sample return.

### Blocked conclusion

Historical cumulative return does not establish expected future return.

---

# 3. Annualized arithmetic return

### 1. Completely new explanation

This calculation scales the average periodic return by the declared number of
periods in one year.

### 2. Practical intuition

If daily returns have an arithmetic average, multiplying by the assumed number
of trading periods gives an annualized arithmetic description.

It is not the same as compounded annual growth.

### 3. Mathematical definition

\[
R_{\mathrm{annual}} = N\bar{r}
\]

### Symbol guide

- \(R_{\mathrm{annual}}\): arithmetic annualized return
- \(N\): declared periods per year
- \(\bar{r}\): arithmetic mean periodic return

### Numerical example

If the mean daily return is 0.0004 and \(N=252\):

\[
R_{\mathrm{annual}} = 252(0.0004)=0.1008
\]

The arithmetic annualized value is 10.08%.

### What Market Memory Lab calculates

The platform reports this as an arithmetic annualized descriptive statistic.

### What automated tests verify

Tests verify the calculation against an independent manual formula.

### Assumptions and failure modes

- The frequency declaration and \(N\) must be meaningful.
- Irregular observations undermine routine annualization.
- Serial dependence, changing distributions, and structural breaks complicate
  interpretation.
- The value is sensitive to the sample period.

### Allowed conclusion

The sample mean was scaled using the declared annualization factor.

### Blocked conclusion

This is not CAGR, expected return, forecast return, or guaranteed return.

### Code reference

`src/market_memory_lab/metrics.py`

### Test reference

`tests/test_metrics.py` and `tests/test_mathematical_audit.py`

---

# 4. Annualized sample volatility

### 1. Completely new explanation

Volatility describes how dispersed the observed returns were around their
average.

### 2. Practical intuition

A series with frequent large movements generally has higher sample volatility
than a series with small movements.

Volatility does not distinguish favorable changes from unfavorable changes.

### 3. Mathematical definition

\[
\sigma_{\mathrm{annual}} =
s_r\sqrt{N}
\]

where:

\[
s_r =
\sqrt{
\frac{1}{T-1}
\sum_{t=1}^{T}(r_t-\bar{r})^2
}
\]

### Symbol guide

- \(\sigma_{\mathrm{annual}}\): annualized sample volatility
- \(s_r\): sample standard deviation of periodic returns
- \(N\): periods per year
- \(T\): number of returns
- \(r_t\): return at time \(t\)
- \(\bar{r}\): sample mean return

### Numerical example

If daily sample volatility is 1% and \(N=252\):

\[
\sigma_{\mathrm{annual}}
=0.01\sqrt{252}
\approx 0.1587
\]

The annualized descriptive value is approximately 15.87%.

### What Market Memory Lab calculates

The platform uses sample standard deviation with one estimated degree of
freedom and applies the declared square-root-of-time convention.

### What automated tests verify

Tests compare the implementation with an independent manual sample-volatility
calculation.

### Assumptions

Routine square-root-of-time interpretation is most natural under stable,
equally spaced observations with sufficiently controlled dependence.

### Failure modes

- Serial dependence
- Volatility clustering
- Structural change
- Irregular time spacing
- Incorrect annualization factor
- Too few observations
- Treating volatility as a complete definition of risk

### Allowed conclusion

The calculation describes sample return dispersion under the declared
annualization convention.

### Blocked conclusion

It is not a forecast, loss probability, complete risk measure, or proof of a
stable data-generating process.

### Code reference

`src/market_memory_lab/metrics.py`

### Test reference

`tests/test_metrics.py` and `tests/test_mathematical_audit.py`

---

# 5. Drawdown

### 1. Completely new explanation

Drawdown measures how far wealth is below its previous highest level.

### 2. Practical intuition

An investment can have positive long-run growth while experiencing severe
declines along the way. Drawdown exposes those declines.

### 3. Mathematical definition

\[
D_t = \frac{W_t}{\max_{1\leq i\leq t} W_i}-1
\]

Maximum drawdown is:

\[
D_{\max} = \min_t D_t
\]

### Symbol guide

- \(D_t\): drawdown at time \(t\)
- \(W_t\): wealth at time \(t\)
- running maximum: highest wealth observed up to time \(t\)
- \(D_{\max}\): most negative sample drawdown

### Numerical example

Suppose wealth reaches 120 and later falls to 90:

\[
D_t = \frac{90}{120}-1=-0.25
\]

The drawdown is minus 25%.

### What Market Memory Lab calculates

The platform builds a compounded wealth path, calculates its running peak,
and records the most negative drawdown.

### What automated tests verify

Tests verify manual drawdown paths, maximum drawdown, and expected bounds.

### Assumptions and failure modes

Drawdown depends on the selected sample, frequency, price definition, and
compounding path. It does not include an unobserved decline before the sample
begins.

### Allowed conclusion

The sample wealth path fell by the calculated proportion from an earlier
sample peak.

### Blocked conclusion

Historical maximum drawdown is not the worst possible future loss.

### Code reference

`src/market_memory_lab/metrics.py`

### Test reference

`tests/test_metrics.py` and `tests/test_mathematical_audit.py`

---

# 6. Pearson correlation

### 1. Completely new explanation

Pearson correlation summarizes how strongly two variables moved together in a
linear way within the selected sample.

### 2. Practical intuition

- Near \(+1\): strong positive linear co-movement
- Near \(0\): weak overall linear association
- Near \(-1\): strong negative linear co-movement

A value near zero can hide changing relationships.

### 3. Mathematical definition

\[
\rho_{A,B} =
\frac{
\sum_{t=1}^{T}(A_t-\bar{A})(B_t-\bar{B})
}{
\sqrt{\sum_{t=1}^{T}(A_t-\bar{A})^2}
\sqrt{\sum_{t=1}^{T}(B_t-\bar{B})^2}
}
\]

### Symbol guide

- \(A_t, B_t\): aligned observations
- \(\bar{A}, \bar{B}\): sample means
- \(T\): aligned observation count
- \(\rho_{A,B}\): sample Pearson correlation

### Numerical example

If:

\[
A=(1,2,3),\qquad B=(2,4,6)
\]

then the sample Pearson correlation is \(+1\), because \(B\) is a positive
linear transformation of \(A\).

### What Market Memory Lab calculates

The platform calculates correlation after independently calculated return
series have been aligned on common return dates.

### What automated tests verify

Tests verify affine invariance and controlled cases with known relationships.

### Assumptions and failure modes

- Observations must be meaningfully aligned.
- Constant series do not provide a defined ordinary correlation.
- Outliers can strongly affect Pearson correlation.
- A changing relationship can be hidden by one full-sample value.
- Common exposure can create correlation without direct causation.
- Data snooping can produce attractive correlations by chance.

### Allowed conclusion

The aligned sample has the reported degree of linear association.

### Blocked conclusion

Correlation does not establish causation, direction, stability,
predictability, legal equivalence, or profitability.

### Code reference

`src/market_memory_lab/similarity.py`

### Test reference

`tests/test_similarity.py` and `tests/test_mathematical_audit.py`

---

# 7. Spearman rank correlation

### 1. Completely new explanation

Spearman correlation compares the ordering of observations instead of their
raw numerical distances.

### 2. Practical intuition

It can detect a consistent increasing or decreasing relationship even when
that relationship is not perfectly linear.

### 3. Mathematical definition

Spearman correlation is the Pearson correlation of the ranked observations:

\[
\rho_S = \operatorname{Corr}(\operatorname{rank}(A),
\operatorname{rank}(B))
\]

### Numerical example

If:

\[
A=(1,2,3),\qquad B=(1,4,9)
\]

the relationship is nonlinear, but both sequences have the same rank order,
so Spearman correlation is \(+1\).

### What automated tests verify

Tests verify invariance under strictly monotonic transformations in controlled
examples.

### Assumptions and failure modes

Ties require rank handling. Spearman correlation still does not establish
causation, stability, or a common mechanism.

### Allowed conclusion

The aligned sample has the reported monotonic rank association.

### Blocked conclusion

High rank association is not proof that the two instruments are economically
equivalent.

### Code reference

`src/market_memory_lab/similarity.py`

### Test reference

`tests/test_similarity.py` and `tests/test_mathematical_audit.py`

---

# 8. Symmetric volatility ratio

### 1. Completely new explanation

This diagnostic compares the two volatility estimates without treating one
instrument as automatically more important.

### 2. Practical intuition

A value near 1 means the estimated sample volatilities are similar. A value
near 0 means they are very different.

### 3. Mathematical definition

\[
V_{\mathrm{ratio}} =
\frac{\min(\sigma_A,\sigma_B)}
{\max(\sigma_A,\sigma_B)}
\]

### Symbol guide

- \(\sigma_A\): sample volatility of instrument A
- \(\sigma_B\): sample volatility of instrument B
- \(V_{\mathrm{ratio}}\): symmetric volatility ratio

### Numerical example

If volatility A is 10% and volatility B is 20%:

\[
V_{\mathrm{ratio}} = \frac{0.10}{0.20}=0.50
\]

### What Market Memory Lab calculates

The platform calculates this custom descriptive diagnostic from the two
sample-volatility estimates.

### Assumptions and failure modes

If both volatility estimates are zero, ordinary ratio interpretation fails.
The estimate also inherits every limitation of the underlying sample
volatilities.

### Allowed conclusion

The two sample-volatility estimates have the reported proportional similarity.

### Blocked conclusion

The value is not a probability, universal similarity score, economic
equivalence measure, or investment rule.

### Code reference

`src/market_memory_lab/similarity.py`

### Test reference

`tests/test_similarity.py`

---

# 9. Standardized Wasserstein distance

### 1. Completely new explanation

This diagnostic compares the overall shapes of two return distributions after
each series has been centered and scaled separately.

### 2. Practical intuition

Centering removes each sample mean. Scaling removes each sample standard
deviation. The remaining distance focuses on differences in distributional
shape.

### 3. Mathematical definition

First standardize each series:

\[
Z_A = \frac{R_A-\bar{R}_A}{s_A},
\qquad
Z_B = \frac{R_B-\bar{R}_B}{s_B}
\]

Then calculate the Wasserstein-1 distance between the two empirical
standardized distributions:

\[
W_1(F_A,F_B)
=
\int_0^1
\left|
F_A^{-1}(u)-F_B^{-1}(u)
\right|du
\]

### Symbol guide

- \(R_A, R_B\): aligned return samples
- \(\bar{R}_A, \bar{R}_B\): sample means
- \(s_A, s_B\): sample standard deviations
- \(F_A^{-1}, F_B^{-1}\): empirical quantile functions
- \(u\): quantile level between 0 and 1
- \(W_1\): Wasserstein-1 distance

### Numerical example

If two standardized samples are identical, their Wasserstein distance is
zero.

### What Market Memory Lab calculates

Each return series is standardized separately before the empirical
Wasserstein-1 distance is calculated.

### What automated tests verify

Tests verify identity, symmetry, and controlled transformation behavior.

### Assumptions and failure modes

- Each sample requires nonzero finite sample standard deviation.
- Small samples represent distributional shape poorly.
- Standardization deliberately removes location and scale differences.
- Similar empirical distributions can arise from different mechanisms.

### Allowed conclusion

The standardized empirical distributions have the reported sample distance.

### Blocked conclusion

The distance is not a probability, is not bounded above, is not a universal
similarity score, and does not establish a common economic mechanism.

### Code reference

`src/market_memory_lab/similarity.py`

### Test reference

`tests/test_similarity.py` and `tests/test_mathematical_audit.py`

---

# 10. Relationship stability gap

### 1. Completely new explanation

This diagnostic asks whether the correlation looks different in the first and
second halves of the selected sample.

### 2. Practical intuition

A single full-sample correlation can average together relationships that
changed dramatically.

### 3. Mathematical definition

\[
G =
\left|
\rho_{\mathrm{first}}
-
\rho_{\mathrm{second}}
\right|
\]

### Symbol guide

- \(G\): relationship stability gap
- \(\rho_{\mathrm{first}}\): first-half sample Pearson correlation
- \(\rho_{\mathrm{second}}\): second-half sample Pearson correlation

Its possible range is from 0 to 2.

### Numerical example: False Friends

Suppose:

\[
\rho_{\mathrm{first}}=+1
\]

and:

\[
\rho_{\mathrm{second}}=-1
\]

Then:

\[
G=|1-(-1)|=2
\]

The relationship did not disappear. The relationship reversed.

### What Market Memory Lab calculates

The aligned return sample is split into predefined halves, correlation is
calculated within each half, and the absolute difference is reported.

### What automated tests verify

Tests verify controlled stable and reversing relationships.

### Assumptions and failure modes

- The split point is predetermined by the diagnostic.
- Each half needs enough nonconstant observations.
- Results can depend strongly on the selected date range.
- A two-part split can miss more complicated changes.
- Repeatedly searching for the most dramatic split creates data-snooping risk.

### Allowed conclusion

The two predefined sample halves have correlations separated by the reported
absolute gap.

### Blocked conclusion

This is not a formal change-point test, regime probability, causal result,
forecast, or proof of a real market regime.

### Code reference

`src/market_memory_lab/similarity.py`

### Test reference

`tests/test_similarity.py` and `tests/test_demonstrations.py`

---

# 11. Evidence adequacy

Evidence adequacy is separate from calculation validity.

A formula may be implemented correctly while the available observations remain
too limited for a strong interpretation.

Market Memory Lab currently uses these interface safeguards:

| Aligned returns | Evidence label | Decision use |
|---:|---|---|
| Fewer than 30 | Critical | Blocked |
| 30 to 99 | Limited | Research only |
| 100 to 251 | Moderate | Research only |
| 252 or more | Stronger descriptive base | Research only |

These categories are Market Memory Lab interface safeguards. They are not
universal statistical laws.

A larger sample does not automatically fix:

- poor data quality
- selection bias
- structural change
- dependence
- multiple testing
- model misspecification
- economically meaningless comparisons
- lack of out-of-sample validation

### Allowed conclusion

The interface assigned the declared evidence category using its documented
observation-count policy.

### Blocked conclusion

The label is not a posterior probability that a hypothesis is true and is not
permission to trade.

### Code reference

`src/market_memory_lab/demonstrations.py`

### Test reference

`tests/test_demonstrations.py`

---

# 12. Tracking integrity

Tracking analysis compares an instrument with a declared or selected reference.

A declared reference relationship is a hypothesis to test, not proof of legal
or economic equivalence.

## Active return

\[
a_t = r_{I,t}-r_{R,t}
\]

- \(a_t\): active return
- \(r_{I,t}\): instrument return
- \(r_{R,t}\): reference return

## Mean active return

\[
\bar{a}=\frac{1}{T}\sum_{t=1}^{T}a_t
\]

## Per-period ex-post tracking error

\[
TE=s_a
\]

where \(s_a\) is the sample standard deviation of active returns.

## Tracking beta

\[
\beta =
\frac{\operatorname{Cov}(r_I,r_R)}
{\operatorname{Var}(r_R)}
\]

## Cumulative normalized divergence

\[
\Delta_T =
W_{I,T}-W_{R,T}
\]

where both wealth paths begin from the same normalized initial value.

### Numerical example

If an instrument returns 1.2% and its reference returns 1.0% during a period:

\[
a_t=0.012-0.010=0.002
\]

The active return is 0.2 percentage points.

### What automated tests verify

Tests independently verify tracking error, beta against an ordinary least
squares slope with an intercept, and cumulative divergence against manually
compounded normalized wealth paths.

### Assumptions and failure modes

- Return dates and frequencies must be compatible.
- Tracking depends on the declared adjustment treatment.
- Currency differences can dominate the comparison.
- A benchmark may be unsuitable.
- Cumulative divergence depends on the full return path.
- Market prices may not be the same as net asset value or redemption value.

### Allowed conclusion

The supplied instrument and reference had the reported sample tracking
diagnostics.

### Blocked conclusion

Tracking diagnostics are not automatically premium or discount, NAV deviation,
arbitrage profit, redemption evidence, or legal equivalence.

### Code reference

`src/market_memory_lab/tracking.py`

### Test reference

`tests/test_tracking.py` and `tests/test_mathematical_audit.py`

---

# 13. Synthetic null laboratory

Synthetic data are controlled mathematical examples, not observations from real
markets.

## Gaussian IID null

### Completely new explanation

This process creates independent random observations from the same Gaussian
distribution.

### Mathematical definition

\[
X_t = \mu+\sigma Z_t
\]

where:

\[
Z_t \overset{\mathrm{IID}}{\sim}N(0,1)
\]

### Symbols

- \(X_t\): generated observation
- \(\mu\): selected mean
- \(\sigma\): selected standard deviation
- \(Z_t\): independent standard Gaussian draw
- IID: independent and identically distributed

### Truth built into the process

- No temporal dependence
- Gaussian tails
- Constant generating parameters

### Allowed conclusion

The generator provides a controlled light-tailed independent null process.

### Blocked conclusion

A Gaussian simulation is not evidence that real markets are Gaussian.

---

## Student-t IID null

### Completely new explanation

This process preserves independence but produces heavier tails than a Gaussian
distribution.

### Mathematical definition

\[
X_t =
\mu+
\sigma
\sqrt{\frac{\nu-2}{\nu}}
T_t
\]

where \(T_t\) is IID Student-t with \(\nu>2\) degrees of freedom.

### Symbols

- \(\nu\): degrees of freedom
- \(T_t\): Student-t random draw
- remaining symbols have the same roles as above

For \(2<\nu\leq4\), variance exists but the fourth moment is infinite.

For \(\nu>4\), theoretical excess kurtosis is:

\[
\frac{6}{\nu-4}
\]

### Truth built into the process

- Heavy tails
- No temporal dependence

### Allowed conclusion

The generator can test whether a statistic responds to heavy tails even when
temporal dependence is absent.

### Blocked conclusion

A result reproduced by Student-t IID data is not evidence of market memory.

---

## Stationary AR(1) short-memory null

### Completely new explanation

An AR(1) process lets the current observation depend partly on the immediately
previous observation.

### Practical intuition

Dependence is real, but it decays geometrically with increasing lag. This is a
short-range dependence mechanism, not long-range dependence.

### Mathematical definition

\[
X_t-\mu =
\phi(X_{t-1}-\mu)+\epsilon_t
\]

with:

\[
|\phi|<1
\]

The stationary variance is:

\[
\operatorname{Var}(X_t)
=
\frac{\sigma_\epsilon^2}{1-\phi^2}
\]

The theoretical autocorrelation is:

\[
\rho(k)=\phi^k
\]

### Symbols

- \(\mu\): stationary mean
- \(\phi\): autoregressive coefficient
- \(\epsilon_t\): independent innovation
- \(\sigma_\epsilon^2\): innovation variance
- \(k\): lag
- \(\rho(k)\): autocorrelation at lag \(k\)

### Numerical example

If \(\phi=0.5\):

\[
\rho(1)=0.5,\qquad
\rho(2)=0.25,\qquad
\rho(3)=0.125
\]

The dependence decays geometrically.

### What Market Memory Lab calculates

The first observation is generated from the stationary distribution, followed
by the AR(1) recurrence using an explicit random seed.

### What automated tests verify

Tests verify parameter validation, reproducibility, stationary initialization,
sample behavior, and the expected short-memory interpretation.

### Assumptions and failure modes

- Stationarity requires \(|\phi|<1\).
- Finite samples may differ from theoretical moments.
- Near-unit-root behavior can look highly persistent.
- Structural breaks may be mistaken for autoregressive persistence.
- Passing or failing one AR(1) challenge cannot identify a unique mechanism.

### Allowed conclusion

This is a controlled stationary short-range dependence null.

### Blocked conclusion

AR(1) dependence is not long-range dependence, a market law, or evidence from
real financial data.

### Code reference

`src/market_memory_lab/synthetic.py`

### Test reference

`tests/test_synthetic.py`

---

# 14. Statistical safeguards

## Bonferroni adjustment

\[
p_i^{\mathrm{adjusted}}=\min(mp_i,1)
\]

- \(p_i\): original p-value
- \(m\): number of tested hypotheses

This controls a familywise error criterion under its stated testing setup, but
it does not repair invalid tests, biased data, or data leakage.

## Benjamini-Hochberg adjustment

For ordered p-values:

\[
p_{(1)}\leq p_{(2)}\leq\cdots\leq p_{(m)}
\]

the adjusted ordered values use reverse cumulative minima of:

\[
\frac{m}{i}p_{(i)}
\]

with values capped at 1.

This procedure requires a clearly defined testing family and an appropriate
interpretation of false-discovery control.

## Monte Carlo corrected p-value

If \(r\) simulated statistics are at least as extreme as the observed
statistic among \(B\) simulations:

\[
\hat{p}=\frac{r+1}{B+1}
\]

A finite simulation should not report a p-value of zero merely because no
simulation exceeded the observation.

## Monte Carlo uncertainty

Market Memory Lab implements an exact Clopper-Pearson interval for the raw
exceedance probability.

Simulation uncertainty does not include every other source of uncertainty,
such as model misspecification or data selection.

## Newey-West mean standard error

The platform implements a Bartlett-weighted heteroskedasticity and
autocorrelation consistent estimate of long-run variance for the sample mean.

It requires:

- ordered observations
- consecutive equally spaced periods
- an explicit lag choice

It does not solve:

- endogeneity
- omitted variables
- structural breaks
- data snooping
- poor model specification
- an inappropriate scientific question

### Code reference

`src/market_memory_lab/inference.py`

### Test reference

`tests/test_inference.py`

---

# 15. The scientific decision ladder

Market Memory Lab separates five layers.

## 1. Data validity

Are dates, values, identities, adjustments, licenses, and provenance adequate?

## 2. Calculation validity

Was the declared formula implemented and tested correctly?

## 3. Evidence adequacy

Is the sample sufficient for the intended interpretation?

## 4. Scientific conclusion

Which conclusions survive the implemented assumptions, nulls, sensitivity
checks, and structural challenges?

## 5. Investment decision

Has the result survived unseen data, costs, constraints, economic evaluation,
and decision-specific validation?

A calculation can be valid while the evidence remains inadequate.

Evidence can be statistically unusual while still being economically
irrelevant.

An interesting research result can still require abstention from an investment
decision.

---

# 16. Ten questions every future result must answer

1. What was measured?
2. Why was this method selected?
3. What assumptions are required?
4. Which nulls reproduce the finding?
5. Which estimators disagree?
6. Has it survived unseen data?
7. Is it statistically unusual?
8. Is it economically meaningful?
9. What would overturn the conclusion?
10. Can another researcher reproduce it?

If the platform cannot answer a necessary question, it should expose the gap
rather than hide it.

---

# 17. Current implementation boundary

The current repository includes validated descriptive calculations, pairwise
similarity diagnostics, tracking diagnostics, statistical safeguards,
instrument identity infrastructure, reproducibility support, and three
synthetic null processes.

The following are not currently implemented or validated:

- GARCH null
- mean-break null
- variance-break null
- formal change-point inference
- Hurst exponent
- classical or modified rescaled-range analysis
- detrended fluctuation analysis
- GPH
- local Whittle
- ARFIMA
- wavelet long-memory estimators
- entropy or transfer entropy
- Granger causality
- regime probabilities
- one-to-many Analogue Search
- machine-learning forecasting
- purged cross-validation
- portfolio optimization
- options pricing or Greeks
- futures roll and basis analysis
- stablecoin peg integrity
- token legal or economic equivalence
- thermodynamic market laws
- trading recommendations
- public live market-data connectors

These boundaries are part of the scientific product, not missing marketing
language.

---

# 18. Reproducibility and provenance

A reproducible investigation should record, where applicable:

- dataset name
- source
- instrument identity
- asset class
- venue
- currency
- timezone
- frequency
- price adjustment declaration
- license or access note
- first and last observation
- observation count
- canonical dataset hash
- software version
- calculation settings
- random seed
- null-process parameters
- evidence policy
- allowed conclusion
- blocked conclusion

A reproducibility manifest supports inspection. It does not prove that the
data source, scientific model, or interpretation is correct.

---

# 19. Final principle

The purpose of Market Memory Lab is not to make every pattern actionable.

Its purpose is to distinguish among:

- a valid calculation
- an interesting descriptive result
- adequate scientific evidence
- a surviving mechanism
- an economically meaningful finding
- a justified decision

When those layers do not align, the correct output is:

## Abstain
