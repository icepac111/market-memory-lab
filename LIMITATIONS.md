# Limitations

## General limitations

- Financial data may contain errors, revisions, missing observations,
  survivorship effects, corporate actions, and timestamp inconsistencies.
- Statistical dependence does not establish economic causation.
- A memory estimate does not automatically imply predictability.
- Predictability does not automatically imply executable profit.
- Paper-trading performance does not establish live-market performance.
- Results may be sensitive to estimator, window, frequency, asset,
  venue, regime definition, and null model.
- Multiple testing can generate apparently significant findings.
- Physics-inspired terminology can be misleading when mappings are not
  formally specified.
- Machine-learning performance may decay under structural change.

## Resource limitations

The initial development environment has limited memory relative to
large-scale institutional systems.

The project will therefore prefer:

- curated universes
- Parquet storage
- DuckDB queries
- streaming or chunked processing
- controlled parallelism
- lightweight reproducible baselines

## Current scientific limitation

No thermodynamic market law is currently claimed or established by this
repository.
