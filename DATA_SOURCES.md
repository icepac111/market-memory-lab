# Data Sources

## Source policy

Every connector must preserve:

- provider name
- source identifier
- asset identifier
- asset class
- venue
- currency
- timezone
- observation timestamp
- retrieval timestamp
- field name
- value
- unit
- adjustment status
- revision status
- license or redistribution note

## Source hierarchy

Preferred order:

1. Primary official source
2. Exchange or original venue
3. Issuer or administrator
4. Documented research dataset
5. Commercial or community aggregator

Lower-priority sources must not silently overwrite higher-priority
sources.

## Free-data phase

The initial project will use public, free, or appropriately licensed
sources and frozen research datasets.

## Licensed-data phase

Future institutional or paid datasets must remain separated from the
public repository unless their licenses explicitly permit redistribution.

The scientific engine must remain provider-independent so that a data
connector can be replaced without changing the mathematical definition
of an estimator.

## Data failure policy

When retrieval fails, the system must report failure.

It must not present an old observation as current without an explicit
stale-data warning.
