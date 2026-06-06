# Phase 7 — Inventory Simulation (service-vs-inventory frontier)

> **M5 has no real lead times/inventory/costs — these are ASSUMPTIONS.** This validates the reorder MACHINERY and the frontier, NOT quotable service levels. Every row is stamped with its lead-time regime.

## Frontier @ base regime (lead mean=3d): service vs inventory

Read across q for each method: higher fill should cost more DOH. Compare methods at matched fill.

| method         |    q |   fill_rate |   service_days |   avg_inv_units |   DOH |   lost_value |    n |
|:---------------|-----:|------------:|---------------:|----------------:|------:|-------------:|-----:|
| lgbm           | 0.5  |       0.81  |          0.818 |           5.112 | 1.511 |    297264    | 5355 |
| lgbm           | 0.9  |       0.974 |          0.978 |          15.516 | 4.661 |     40736.4  | 5355 |
| lgbm           | 0.95 |       0.984 |          0.987 |          18.807 | 5.661 |     25614.5  | 5355 |
| lgbm           | 0.99 |       0.993 |          0.995 |          25.065 | 7.579 |     11891.3  | 5355 |
| moving_average | 0.5  |       0.878 |          0.898 |           7.389 | 2.22  |    185896    | 5355 |
| moving_average | 0.9  |       0.98  |          0.984 |          17.07  | 5.186 |     29894.8  | 5355 |
| moving_average | 0.95 |       0.988 |          0.99  |          19.971 | 6.082 |     18861.6  | 5355 |
| moving_average | 0.99 |       0.995 |          0.996 |          25.545 | 7.795 |      6808.28 | 5355 |
| seasonal_naive | 0.5  |       0.877 |          0.899 |           8.465 | 2.584 |    185130    | 5355 |
| seasonal_naive | 0.9  |       0.988 |          0.989 |          20.067 | 6.18  |     18053.2  | 5355 |
| seasonal_naive | 0.95 |       0.994 |          0.995 |          23.58  | 7.262 |      9901.2  | 5355 |
| seasonal_naive | 0.99 |       0.998 |          0.998 |          30.268 | 9.349 |      2325.87 | 5355 |

## The number that matters: DOH to reach a fill target (base regime)

| method         |    q |   fill_rate |   DOH |
|:---------------|-----:|------------:|------:|
| lgbm           | 0.5  |       0.81  |  1.51 |
| lgbm           | 0.9  |       0.974 |  4.66 |
| lgbm           | 0.95 |       0.984 |  5.66 |
| lgbm           | 0.99 |       0.993 |  7.58 |
| moving_average | 0.5  |       0.878 |  2.22 |
| moving_average | 0.9  |       0.98  |  5.19 |
| moving_average | 0.95 |       0.988 |  6.08 |
| moving_average | 0.99 |       0.995 |  7.79 |
| seasonal_naive | 0.5  |       0.877 |  2.58 |
| seasonal_naive | 0.9  |       0.988 |  6.18 |
| seasonal_naive | 0.95 |       0.994 |  7.26 |
| seasonal_naive | 0.99 |       0.998 |  9.35 |

## Intermittent verdict: per-class fill & DOH @ base, q=0.95 (value-weighted)

| intermittency   | method         |   fill_rate |   service_days |   avg_inv_units |    DOH |   lost_value |    n |
|:----------------|:---------------|------------:|---------------:|----------------:|-------:|-------------:|-----:|
| erratic         | lgbm           |       0.976 |          0.985 |          32.397 |  5.37  |      7571.14 |  855 |
| erratic         | moving_average |       0.977 |          0.982 |          33.151 |  5.548 |      7967.34 |  855 |
| erratic         | seasonal_naive |       0.989 |          0.991 |          39.139 |  6.623 |      3245.57 |  855 |
| intermittent    | lgbm           |       0.962 |          0.985 |           5.59  |  6.768 |      9147.13 | 1500 |
| intermittent    | moving_average |       0.98  |          0.991 |           6.592 |  8.061 |      4699.64 | 1500 |
| intermittent    | seasonal_naive |       0.985 |          0.995 |           8.339 | 10.342 |      3643.61 | 1500 |
| lumpy           | lgbm           |       0.967 |          0.982 |          11.789 |  6.333 |      6103.05 | 1500 |
| lumpy           | moving_average |       0.982 |          0.988 |          13.453 |  7.243 |      3511.23 | 1500 |
| lumpy           | seasonal_naive |       0.99  |          0.993 |          17.14  |  9.26  |      1821.16 | 1500 |
| smooth          | lgbm           |       0.996 |          0.997 |          31.297 |  5.329 |      2793.15 | 1500 |
| smooth          | moving_average |       0.996 |          0.996 |          32.355 |  5.503 |      2683.41 | 1500 |
| smooth          | seasonal_naive |       0.998 |          0.998 |          36.393 |  6.238 |      1190.87 | 1500 |

## Lead-time regime sweep (LGBM, q=0.95)

| regime   |   fill_rate |   service_days |   avg_inv_units |   DOH |   lost_value |    n |
|:---------|------------:|---------------:|----------------:|------:|-------------:|-----:|
| base     |       0.984 |          0.987 |          18.807 | 5.661 |     25614.5  | 5355 |
| long     |       0.974 |          0.98  |          25.209 | 7.52  |     41486.9  | 5355 |
| short    |       0.995 |          0.994 |          12.851 | 3.909 |      8715.17 | 5355 |