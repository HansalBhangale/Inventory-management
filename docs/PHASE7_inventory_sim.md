# Phase 7 — Inventory Simulation (service-vs-inventory frontier)

> **M5 has no real lead times/inventory/costs — these are ASSUMPTIONS.** This validates the reorder MACHINERY and the frontier, NOT quotable service levels. Every row is stamped with its lead-time regime.

## Frontier @ base regime (lead mean=3d): service vs inventory

Read across q for each method: higher fill should cost more DOH. Compare methods at matched fill.

| method         |    q |   fill_rate |   service_days |   avg_inv_units |   DOH |   lost_value |    n |
|:---------------|-----:|------------:|---------------:|----------------:|------:|-------------:|-----:|
| lgbm           | 0.5  |       0.809 |          0.818 |           5.112 | 1.51  |    298940    | 5355 |
| lgbm           | 0.9  |       0.967 |          0.974 |          14.877 | 4.512 |     50083.5  | 5355 |
| lgbm           | 0.95 |       0.983 |          0.987 |          18.973 | 5.726 |     26476.1  | 5355 |
| lgbm           | 0.99 |       0.995 |          0.997 |          28.374 | 8.454 |      7585.69 | 5355 |
| moving_average | 0.5  |       0.877 |          0.897 |           7.386 | 2.223 |    185671    | 5355 |
| moving_average | 0.9  |       0.98  |          0.984 |          17.018 | 5.183 |     30809    | 5355 |
| moving_average | 0.95 |       0.988 |          0.99  |          19.93  | 6.07  |     18502.6  | 5355 |
| moving_average | 0.99 |       0.995 |          0.996 |          25.557 | 7.8   |      7535.48 | 5355 |
| seasonal_naive | 0.5  |       0.876 |          0.899 |           8.464 | 2.59  |    185607    | 5355 |
| seasonal_naive | 0.9  |       0.988 |          0.99  |          20.06  | 6.176 |     18589    | 5355 |
| seasonal_naive | 0.95 |       0.994 |          0.995 |          23.583 | 7.273 |      9506.09 | 5355 |
| seasonal_naive | 0.99 |       0.998 |          0.998 |          30.272 | 9.346 |      2972.96 | 5355 |

## The number that matters: DOH to reach a fill target (base regime)

| method         |    q |   fill_rate |   DOH |
|:---------------|-----:|------------:|------:|
| lgbm           | 0.5  |       0.809 |  1.51 |
| lgbm           | 0.9  |       0.967 |  4.51 |
| lgbm           | 0.95 |       0.983 |  5.73 |
| lgbm           | 0.99 |       0.995 |  8.45 |
| moving_average | 0.5  |       0.877 |  2.22 |
| moving_average | 0.9  |       0.98  |  5.18 |
| moving_average | 0.95 |       0.988 |  6.07 |
| moving_average | 0.99 |       0.995 |  7.8  |
| seasonal_naive | 0.5  |       0.876 |  2.59 |
| seasonal_naive | 0.9  |       0.988 |  6.18 |
| seasonal_naive | 0.95 |       0.994 |  7.27 |
| seasonal_naive | 0.99 |       0.998 |  9.35 |

## Intermittent verdict: per-class fill & DOH @ base, q=0.95 (value-weighted)

| intermittency   | method         |   fill_rate |   service_days |   avg_inv_units |    DOH |   lost_value |    n |
|:----------------|:---------------|------------:|---------------:|----------------:|-------:|-------------:|-----:|
| erratic         | lgbm           |       0.973 |          0.983 |          32.561 |  5.399 |      8474.22 |  855 |
| erratic         | moving_average |       0.977 |          0.982 |          33.136 |  5.539 |      8111.42 |  855 |
| erratic         | seasonal_naive |       0.99  |          0.992 |          39.134 |  6.617 |      3104.05 |  855 |
| intermittent    | lgbm           |       0.965 |          0.986 |           5.801 |  7.064 |      8217.53 | 1500 |
| intermittent    | moving_average |       0.981 |          0.991 |           6.571 |  8.035 |      4355.94 | 1500 |
| intermittent    | seasonal_naive |       0.987 |          0.995 |           8.389 | 10.394 |      3239.55 | 1500 |
| lumpy           | lgbm           |       0.966 |          0.982 |          12.003 |  6.437 |      6380.87 | 1500 |
| lumpy           | moving_average |       0.983 |          0.988 |          13.447 |  7.268 |      3307.69 | 1500 |
| lumpy           | seasonal_naive |       0.99  |          0.993 |          17.156 |  9.283 |      1822.08 | 1500 |
| smooth          | lgbm           |       0.996 |          0.996 |          31.369 |  5.342 |      3403.46 | 1500 |
| smooth          | moving_average |       0.997 |          0.996 |          32.243 |  5.483 |      2727.55 | 1500 |
| smooth          | seasonal_naive |       0.998 |          0.997 |          36.34  |  6.241 |      1340.42 | 1500 |

## Lead-time regime sweep (LGBM, q=0.95)

| regime   |   fill_rate |   service_days |   avg_inv_units |   DOH |   lost_value |    n |
|:---------|------------:|---------------:|----------------:|------:|-------------:|-----:|
| base     |       0.983 |          0.987 |          18.973 | 5.726 |      26476.1 | 5355 |
| long     |       0.972 |          0.98  |          25.406 | 7.585 |      43825.6 | 5355 |
| short    |       0.993 |          0.994 |          13.021 | 3.975 |      10907.9 | 5355 |