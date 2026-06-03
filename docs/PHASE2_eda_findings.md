# Phase 2 — EDA Findings → Design Decisions (M5 bootstrap)

> Every finding below is computed from the staged M5 data (46.9M rows, 10 stores, 3,049 SKUs,
> 2011-01-29 → 2016-05-22) and maps to a concrete downstream decision (doc §2.10).
> Reproduce with [src/features/segmentation.py](../src/features/segmentation.py) and [notebooks/02_eda.ipynb](../notebooks/02_eda.ipynb).

## 2.2 Intermittency classification — *the key output*
Per (store, SKU), ADI & CV² (Syntetos-Boylan):

| Class | SKUs | % | Revenue |
|-------|------|---|---------|
| intermittent | 22,143 | 72.6% | 98.8M |
| lumpy | 5,612 | 18.4% | 31.8M |
| smooth | 1,880 | 6.2% | 46.3M |
| erratic | 855 | 2.8% | 14.7M |

**→ Decision (model routing, Phase 5.7):** ~**91% of series are intermittent/lumpy** → route to
**Croston/SBA/TSB**. Only ~9% (smooth/erratic) go to the **global LightGBM**. *But* the smooth/erratic
9% carry a disproportionate share of revenue (≈45%), so LightGBM still covers a large fraction of the
business — concentrate tuning there. This high intermittency is the defining characteristic of the data.

## 2.2b Zero-demand density
**59.6%** of active store-SKU-days have zero sales.
**→ Decision:** zero-inflated target handling — **tweedie** objective (Phase 3.7); intermittent models for
the long tail; do not treat all zeros identically (censored handling, §2.5).

## 2.3 Seasonality
**Day-of-week** mean units (clear weekend peak):

| Sun | Mon | Tue | Wed | Thu | Fri | Sat |
|-----|-----|-----|-----|-----|-----|-----|
| 1.71 | 1.37 | 1.26 | 1.25 | 1.26 | 1.42 | 1.73 |

Weekend (Sat/Sun) runs ~**35% above** the mid-week trough.
Monthly: mild, peaks in Aug/Feb/Jun (~1.49 vs ~1.43 baseline).

**→ Decision (Phase 4):** build `day_of_week`, `is_weekend`, lags at **7/14** and same-DOW rolling means.
Annual signal is mild here (M5 = US retail) but the architecture keeps a ≥24-mo window for the festival-heavy
Indian production case (Phase 6).

## 2.4 ABC × XYZ segmentation
ABC (revenue Pareto, within store): **A = 11,564 SKUs / 80% rev · B = 9,354 / 15% · C = 9,572 / 5%** — textbook Pareto.
9-box skews heavily to **Z** (erratic predictability), consistent with the high intermittency above.

**→ Decision (service levels, Phase 7.7):** apply per-cell targets from [config/policy.yaml](../config/policy.yaml)
(A 98% / B 95% / C 88%); modeling effort concentrates on A/B.

## 2.5 Stockout / censored demand
**M5 has no inventory snapshot**, so true stockouts are unobservable here. The 59.6% zeros are therefore
treated as genuine no-demand for the bootstrap, with the censored-demand machinery (`was_stockout` mask /
latent imputation, Phase 3.2) wired but **inert until store inventory data arrives**.
**→ Decision:** keep the masking hook; flag this as the #1 data gap for production (doc risk §0.5).

## 2.6 Price & promotion
M5 carries weekly `sell_price` (no explicit discount). Relative-price and price-change features are
buildable; an explicit promo flag awaits the store `promotions` table.
**→ Decision (Phase 4):** `unit_price`, `relative_price` (vs category median), price-change features.

## 2.7 External drivers — SNAP / salary window
SNAP-benefit days lift mean demand **1.55 vs 1.37 (+13%)** — a clean salary-cycle analogue.
Holidays show a slight *dip* (1.37 vs 1.43): the holiday *day* itself is not the driver; the **lead-up**
window is (festival proximity, Phase 4.B).
**→ Decision (Phase 4):** `is_salary_window` (high value), `days_to/from_festival`, `in_festival_leadup`.

## EDA → downstream decision map
| Finding | Feeds | Decision |
|---------|-------|----------|
| 91% intermittent/lumpy | Phase 5 | Croston/TSB routing; LightGBM for smooth/erratic A/B |
| 59.6% zero days | Phase 3 | tweedie objective, zero-inflation handling |
| Weekend +35% | Phase 4 | DoW / weekend / same-DOW rolling features |
| ABC 80/15/5 | Phase 7 | service level per cell, effort on A/B |
| No inventory in M5 | Phase 3/prod | censored hooks inert; #1 production data gap |
| SNAP +13% | Phase 4 | salary-window feature |

## Deliverables
- [x] `data/features/segments.parquet` — (store, SKU) → ADI, CV², intermittency, ABC, XYZ, abc_xyz
- [x] This findings memo
- [x] [notebooks/02_eda.ipynb](../notebooks/02_eda.ipynb) — plots + reproducible analyses
