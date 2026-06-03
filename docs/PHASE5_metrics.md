# Phase 5 — Model Metrics & Card

Scope: stores=['CA_1'] · horizon=14d · quantiles=[0.5, 0.9, 0.95]

## Overall (validation folds)

|   fold | segment   | model          |     n |   wape |    mae |   rmse |    bias |   smape |   mase |   pinball_q50 |   cov_q50 |   pinball_q90 |   cov_q90 |   pinball_q95 |   cov_q95 |
|-------:|:----------|:---------------|------:|-------:|-------:|-------:|--------:|--------:|-------:|--------------:|----------:|--------------:|----------:|--------------:|----------:|
|      5 | ALL       | lgbm           | 42686 | 0.7144 | 1.1386 | 2.2264 | -0.0363 |  1.3254 | 0.8129 |        0.5288 |    0.5965 |        0.3471 |    0.8899 |        0.2274 |    0.9429 |
|      5 | ALL       | seasonal_naive | 42686 | 0.8693 | 1.3854 | 2.7843 | -0.0298 |  0.8822 | 0.9891 |      nan      |  nan      |      nan      |  nan      |      nan      |  nan      |

## Per-segment (ABC, LightGBM)

|   fold | segment   | model   |     n |   wape |    mae |   rmse |    bias |   smape |   mase |   pinball_q50 |   cov_q50 |   pinball_q90 |   cov_q90 |   pinball_q95 |   cov_q95 |
|-------:|:----------|:--------|------:|-------:|-------:|-------:|--------:|--------:|-------:|--------------:|----------:|--------------:|----------:|--------------:|----------:|
|      5 | ABC=A     | lgbm    | 17038 | 0.5917 | 1.6616 | 2.9775 | -0.0234 |  1.0085 | 1.1863 |        0.8045 |    0.516  |        0.4776 |    0.895  |        0.3064 |    0.9473 |
|      5 | ABC=B     | lgbm    | 12950 | 0.9364 | 0.9543 | 1.7617 | -0.0288 |  1.3874 | 0.6813 |        0.4302 |    0.589  |        0.2947 |    0.8983 |        0.1965 |    0.95   |
|      5 | ABC=C     | lgbm    | 12698 | 1.136  | 0.6248 | 1.2659 | -0.1384 |  1.6875 | 0.4461 |        0.2594 |    0.7121 |        0.2256 |    0.8744 |        0.1529 |    0.9298 |

## Model card

- **Model:** global LightGBM, one model across all (store,SKU).
- **Objectives:** central `tweedie` + pinball heads [0.5, 0.9, 0.95] (non-crossing enforced).
- **Features:** 39 (7 native-categorical). Direct H-step, lags>=H.
- **Validation:** rolling-origin, 14d embargo.
- **Credibility gate:** MASE<1 vs seasonal-naive.
- **Known limits:** M5 has no inventory/promo; censored-demand hooks inert; intermittent SKUs better served by TSB (Phase 5.7 routing).
