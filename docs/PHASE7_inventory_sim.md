# Phase 7 — Inventory Simulation (service-vs-inventory frontier)

> **M5 has no real lead times/inventory/costs — these are ASSUMPTIONS.** This validates the reorder MACHINERY and the frontier, NOT quotable service levels. Every row is stamped with its lead-time regime.

## Frontier @ base regime (lead mean=3d): service vs inventory

Read across q for each method: higher fill should cost more DOH. Compare methods at matched fill.

| method         |    q |   fill_rate |   service_days |   avg_inv_units |   DOH |   lost_value |    n |
|:---------------|-----:|------------:|---------------:|----------------:|------:|-------------:|-----:|
| lgbm           | 0.5  |       0.809 |          0.818 |           5.109 | 1.509 |     298272   | 5355 |
| lgbm           | 0.9  |       0.975 |          0.978 |          15.522 | 4.664 |      40347.8 | 5355 |
| lgbm           | 0.95 |       0.984 |          0.987 |          18.801 | 5.657 |      25209.7 | 5355 |
| moving_average | 0.5  |       0.877 |          0.897 |           7.377 | 2.222 |     187281   | 5355 |
| moving_average | 0.9  |       0.98  |          0.984 |          17.032 | 5.179 |      30789.1 | 5355 |
| moving_average | 0.95 |       0.988 |          0.99  |          19.994 | 6.091 |      18438.2 | 5355 |
| seasonal_naive | 0.5  |       0.877 |          0.899 |           8.472 | 2.587 |     184324   | 5355 |
| seasonal_naive | 0.9  |       0.988 |          0.99  |          19.994 | 6.157 |      18239.7 | 5355 |
| seasonal_naive | 0.95 |       0.994 |          0.995 |          23.529 | 7.251 |       9540.3 | 5355 |

## The number that matters: DOH to reach a fill target (base regime)

| method         |    q |   fill_rate |   DOH |
|:---------------|-----:|------------:|------:|
| lgbm           | 0.5  |       0.809 |  1.51 |
| lgbm           | 0.9  |       0.975 |  4.66 |
| lgbm           | 0.95 |       0.984 |  5.66 |
| moving_average | 0.5  |       0.877 |  2.22 |
| moving_average | 0.9  |       0.98  |  5.18 |
| moving_average | 0.95 |       0.988 |  6.09 |
| seasonal_naive | 0.5  |       0.877 |  2.59 |
| seasonal_naive | 0.9  |       0.988 |  6.16 |
| seasonal_naive | 0.95 |       0.994 |  7.25 |

## Intermittent verdict: per-class fill & DOH @ base, q=0.95 (value-weighted)

| intermittency   | method         |   fill_rate |   service_days |   avg_inv_units |    DOH |   lost_value |    n |
|:----------------|:---------------|------------:|---------------:|----------------:|-------:|-------------:|-----:|
| erratic         | lgbm           |       0.977 |          0.984 |          32.424 |  5.37  |      7434.71 |  855 |
| erratic         | moving_average |       0.978 |          0.983 |          33.152 |  5.544 |      7546.98 |  855 |
| erratic         | seasonal_naive |       0.99  |          0.991 |          39.125 |  6.607 |      3040.05 |  855 |
| intermittent    | lgbm           |       0.964 |          0.985 |           5.587 |  6.754 |      8596.58 | 1500 |
| intermittent    | moving_average |       0.981 |          0.992 |           6.558 |  8.039 |      4329.04 | 1500 |
| intermittent    | seasonal_naive |       0.987 |          0.995 |           8.34  | 10.349 |      3348.25 | 1500 |
| lumpy           | lgbm           |       0.967 |          0.982 |          11.766 |  6.331 |      6408.98 | 1500 |
| lumpy           | moving_average |       0.982 |          0.988 |          13.434 |  7.257 |      3584.23 | 1500 |
| lumpy           | seasonal_naive |       0.991 |          0.994 |          17.145 |  9.245 |      1836.8  | 1500 |
| smooth          | lgbm           |       0.996 |          0.997 |          31.284 |  5.324 |      2769.4  | 1500 |
| smooth          | moving_average |       0.996 |          0.995 |          32.487 |  5.523 |      2977.93 | 1500 |
| smooth          | seasonal_naive |       0.998 |          0.997 |          36.214 |  6.226 |      1315.2  | 1500 |

## Lead-time regime sweep (LGBM, q=0.95)

| regime   |   fill_rate |   service_days |   avg_inv_units |   DOH |   lost_value |    n |
|:---------|------------:|---------------:|----------------:|------:|-------------:|-----:|
| base     |       0.984 |          0.987 |          18.801 | 5.657 |     25209.7  | 5355 |
| long     |       0.974 |          0.98  |          25.253 | 7.52  |     41863.7  | 5355 |
| short    |       0.995 |          0.994 |          12.851 | 3.909 |      8715.17 | 5355 |