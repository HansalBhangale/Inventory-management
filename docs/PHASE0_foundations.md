# Phase 0 — Foundations & Problem Framing (sign-off)

> Goal: lock down exactly what we predict and decide, so no downstream ambiguity exists.

## 0.1 Objectives (SMART)

| ID | Objective | Measure |
|----|-----------|---------|
| **O1** | Forecast daily unit demand per (store, SKU) at quantiles {0.5, 0.9, 0.95} over horizon = `max_lead_time + review_period` days | Quantile forecasts produced for every active grain |
| **O2** | Output an automated reorder recommendation (order? qty?) per (store, SKU) every day | Daily PO recommendation table |
| **O3** | Beat seasonal-naive baseline (**MASE < 1**) on ≥ 80% of A/B-class SKUs | Backtest report |
| **O4** | Hit per-segment service-level targets at lower inventory than current practice | Inventory simulation |

## 0.2 Precise problem definition

- **Grain (unit of prediction):** one row = `(store_id, sku_id, date)`.
- **Target:** `units_sold` per grain — **corrected for censored demand** (stockout days handled in Phase 3, not trained as true zeros).
- **Forecast horizon `H`:** **14 days** (covers lead time ≈ 3d + daily review, with margin). Set in [config/model.yaml](../config/model.yaml).
- **What "demand" means:** true customer intent, *not* recorded sales (sales are capped by shelf availability).
- **Decision policy:** continuous-review **(s, S)** — when inventory position ≤ reorder point `s`, order up to `S`. Defined in [config/policy.yaml](../config/policy.yaml).
- **Inventory position:** `on_hand + on_order − backorders`.

## 0.3 Scope

**In scope:** demand forecasting (probabilistic), reorder qty/timing, lead-time estimation, ABC×XYZ segmentation, hybrid continuous learning, evaluation, monitoring.

**Out of scope (v1):** price/promo *optimization* (we *use* promo info, don't set prices); multi-echelon network optimization (single store→supplier first); assortment decisions.

## 0.4 Assumptions & constraints

- Store has (or can export) **transaction-level POS data** (date, SKU, qty).
- Inventory on-hand and goods-receipt timestamps available *or* reconstructable.
- Compute budget modest → LightGBM + DuckDB/Parquet over deep learning. Single VM (8–16 GB) target.
- Indian retail calendar (festivals, salary-day cycles) materially drives demand.

## 0.5 Key risks & mitigations

| Risk | Mitigation |
|------|------------|
| No clean inventory/lead-time data | Bootstrap with public data (M5); estimate lead time from PO→GRN; simulate during dev |
| Censored demand biases model to under-order | `was_stockout` flag + latent-demand handling (Phase 3) |
| Annual events unlearnable online | Hybrid: scheduled retrain over ≥2 yrs + festival features |
| Cold-start new SKUs | Attribute/category model + aggregate-and-distribute |
| Over-trust in automation | Exception dashboard + human override + explanations |

## 0.6 Decisions locked (config)

| Decision | Value | File |
|----------|-------|------|
| Horizon | 14 days | model.yaml |
| Review period | 1 day (daily) | model.yaml |
| Quantiles | {0.5, 0.9, 0.95} | model.yaml |
| Central objective | tweedie | model.yaml |
| Quantile objective | pinball/quantile | model.yaml |
| Policy type | (s, S) continuous-review | policy.yaml |
| Safety stock method | empirical quantile (Route B) | policy.yaml |
| Service levels | A 98% / B 95% / C 88% | policy.yaml |
| Primary metric | WAPE (point), pinball + coverage (dist) | metrics.yaml |
| Credibility gate | MASE < 1 | metrics.yaml |
| Backtest | rolling-origin, 14d embargo | metrics.yaml |
| Primary bootstrap data | M5 (accuracy + uncertainty) | data_sources.yaml |

## 0.7 Deliverables — status

- [x] Problem statement & success criteria (this doc)
- [x] `config/` skeleton (horizon, quantiles, policy, metrics, features, data contract)
- [x] Tech stack confirmed; dev environment bootstrapped (`.venv`, `requirements.txt`)
- [ ] Bootstrap data downloaded (M5) — pending Kaggle credentials

---
*Sign-off:* Phase 0 complete once data download succeeds; proceed to Phase 1 ingestion + validation.
