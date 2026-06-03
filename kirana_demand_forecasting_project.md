# Kirana Demand Forecasting & Automated Reordering — Project Documentation

**A continuous-learning system that forecasts SKU-level demand and automatically decides *when* and *how much* to reorder for small retail (kirana) stores at chain scale.**

| | |
|---|---|
| **Version** | 1.0 |
| **Status** | Design / Build specification |
| **Primary output** | A reorder decision per (store, SKU): *order now? how many?* |
| **Core idea** | Probabilistic (quantile) demand forecast → service-level reorder policy, kept alive by a hybrid continuous-learning loop |
| **Scale target** | Many stores / chain · hundreds–thousands of SKUs per store |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Repository Structure](#4-repository-structure)
5. [Phase 0 — Foundations & Problem Framing](#phase-0--foundations--problem-framing)
6. [Phase 1 — Data Acquisition & Understanding](#phase-1--data-acquisition--understanding)
7. [Phase 2 — Exploratory Data Analysis (EDA)](#phase-2--exploratory-data-analysis-eda)
8. [Phase 3 — Data Preparation & Cleaning](#phase-3--data-preparation--cleaning)
9. [Phase 4 — Feature Engineering & Feature Importance](#phase-4--feature-engineering--feature-importance)
10. [Phase 5 — Model Building (Forecasting)](#phase-5--model-building-forecasting)
11. [Phase 6 — Hybrid Continuous-Learning Architecture](#phase-6--hybrid-continuous-learning-architecture)
12. [Phase 7 — Reorder Decision Layer](#phase-7--reorder-decision-layer)
13. [Phase 8 — Evaluation & Metrics](#phase-8--evaluation--metrics)
14. [Phase 9 — Deployment & MLOps](#phase-9--deployment--mlops)
15. [Phase 10 — Monitoring, Feedback & Continuous Improvement](#phase-10--monitoring-feedback--continuous-improvement)
16. [Appendices](#appendices)

---

## 1. Executive Summary

### 1.1 The problem
Small kirana stores manage hundreds of SKUs and reorder by gut feel. The result is the classic two-sided failure: **stockouts** on fast movers (lost sales, lost customers) and **overstock** on slow movers (dead capital, waste on perishables). They lack affordable predictive analytics that learn from their own daily sales.

### 1.2 The solution
A system with two coupled halves:

1. **Forecasting half** — a *global* machine-learning model that predicts, for every (store, SKU, day), not a single number but a **demand distribution** (P50 / P90 / P95). The upper quantiles are what protect against stockouts.
2. **Reorder half** — a policy engine that turns those quantiles, plus measured supplier **lead time**, into a concrete **reorder point** and **order quantity**, respecting MOQ, pack size, shelf life, and a per-segment service-level target.

### 1.3 Why "continuous learning" (and what it actually means here)
The model must keep learning from daily sales — but *pure* online learning cannot capture once-a-year events (Diwali, Holi, monsoon). The incumbents (RELEX, ToolsGroup, Netstock, StockIQ) don't do pure streaming either; they do **scheduled recomputation on rolling windows + dynamic parameters + drift detection + exception management**. We adopt the same pragmatic **hybrid**:

- a **global base model** retrained on a schedule over a rolling window (captures seasonality + cross-SKU structure);
- a **fast online layer** that adapts daily to recent shifts, new SKUs, and promotions;
- a **drift detector** that triggers off-cycle retraining when the world changes;
- feeding the **reorder policy** that produces the actual decision.

### 1.4 Design principles
- **Global over per-series.** One model across all (store, SKU) pairs → cross-learning, cheap cold-start, scales to chain size.
- **Distributions, not point estimates.** Reorder points are *quantiles*, so we forecast quantiles directly (pinball loss).
- **Segment, then differentiate.** ABC × XYZ segmentation drives which forecasting method and which service level each SKU gets.
- **Decision-first.** A beautiful forecast that can't place an order is worthless. Lead time, MOQ, cost, and live inventory are first-class citizens.
- **Affordable & open-source.** Python + LightGBM + DuckDB/Parquet runs on a single modest VM and scales out only when needed.

---

## 2. System Architecture

### 2.1 Layered view
```
┌─────────────────────────────────────────────────────────────────────┐
│  SOURCE SYSTEMS                                                       │
│  POS/billing · product master · inventory snapshots · PO/GRN ·        │
│  supplier master · promo calendar · external (festivals, weather)     │
└───────────────┬───────────────────────────────────────────────────────┘
                │  (nightly batch + optional live txn stream)
┌───────────────▼───────────────┐
│  DATA INGESTION & VALIDATION   │  landing → raw → staged (Parquet/DuckDB)
│  pandera/Great Expectations    │  schema + quality gates
└───────────────┬───────────────┘
┌───────────────▼───────────────┐
│  FEATURE PIPELINE / STORE      │  panel build · calendar · lags · rolling
│  (Feast optional)              │  price/promo · external · stockout flags
└───────────────┬───────────────┘
┌───────────────▼───────────────────────────────────────────┐
│  FORECASTING LAYER                                         │
│  ┌──────────────────┐   ┌───────────────────────────────┐ │
│  │ Global base model│ + │ Online adaptation layer       │ │
│  │ LightGBM quantile│   │ (River residual correction)   │ │
│  │ scheduled retrain│   │ daily update                  │ │
│  └──────────────────┘   └───────────────────────────────┘ │
│  Segment routing: intermittent→Croston/TSB · cold-start→cat│
│  Output: P50 / P90 / P95 demand per (store, SKU, horizon)  │
└───────────────┬───────────────────────────────────────────┘
┌───────────────▼───────────────┐      ┌────────────────────┐
│  REORDER POLICY ENGINE         │◀────▶│ Lead-time model     │
│  safety stock · ROP · order-up │      │ (PO→GRN history)    │
│  EOQ / (s,S) / newsvendor      │      └────────────────────┘
│  MOQ · pack size · shelf life  │
│  Output: PO recommendations    │
└───────────────┬───────────────┘
┌───────────────▼───────────────┐
│  ACTION & UI                   │  PO recommendations + explanations
│  exception dashboard           │  human-in-the-loop overrides
└───────────────┬───────────────┘
┌───────────────▼───────────────┐
│  MONITORING & FEEDBACK         │  drift (Evidently) · accuracy decay ·
│  → triggers retrain            │  actuals logged → error → drift check
└────────────────────────────────┘
```

### 2.2 The continuous-learning loop (daily)
```
1. Ingest yesterday's sales + inventory + any receipts (GRN)
2. Update features (append lags/rolling for the new day)
3. Online layer updates on yesterday's residuals  ← daily adaptation
4. Score: produce P50/P90/P95 for the horizon
5. Reorder engine computes ROP & order qty → PO recommendations
6. Log actuals (when next day arrives) → compute forecast error
7. Drift detector checks the error stream per segment
8. IF (scheduled cadence) OR (drift fired): retrain global base model
9. Champion/Challenger: promote new model only if it beats current
10. Repeat
```

---

## 3. Technology Stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | Ecosystem, hiring, libraries |
| Dataframes | **pandas** (+ **Polars**/**DuckDB** for scale) | DuckDB queries Parquet on disk cheaply |
| Storage (dev) | **Parquet** files + **DuckDB** | Columnar, free, fast, no server |
| Storage (prod) | **PostgreSQL** (operational) + Parquet (analytics) | Reliable txns + cheap history |
| Baselines & intermittent | **statsforecast** (Nixtla) | Croston, TSB, ETS, seasonal-naive, fast |
| Primary model | **LightGBM** | Gradient boosting, quantile objective, global model, fast |
| Online layer | **River** | Incremental/streaming learning, drift detectors |
| Hierarchical reconciliation | **hierarchicalforecast** (Nixtla) | Reconcile SKU↔category forecasts |
| Hyperparameter tuning | **Optuna** | Efficient, supports TS CV |
| Explainability | **SHAP** + permutation importance | Feature importance & trust |
| Experiment tracking & registry | **MLflow** | Versioning, champion/challenger |
| Orchestration | **Airflow** or **Prefect** | Schedule the daily/weekly DAG |
| Feature store (optional) | **Feast** | Train/serve consistency |
| Data validation | **pandera** / **Great Expectations** | Quality gates before training |
| Drift & monitoring | **Evidently** | Data/prediction drift, dashboards |
| Serving | **FastAPI** + batch tables | Lightweight; store app reads results |
| Viz / reports | **matplotlib**, **plotly** | EDA + monitoring |

> **Cost note:** every component above is open-source. A single VM (8–16 GB RAM) handles a multi-store dev/prototype; scale out only when SKU×store×day volume demands it.

---

## 4. Repository Structure

```
kirana-forecasting/
├── README.md
├── pyproject.toml / requirements.txt
├── config/
│   ├── data_contract.yaml         # expected schemas
│   ├── features.yaml              # feature definitions
│   ├── model.yaml                 # hyperparams, objectives, quantiles
│   ├── policy.yaml                # service levels per segment, MOQ rules
│   └── metrics.yaml               # metric targets / acceptance thresholds
├── data/
│   ├── raw/                       # immutable source dumps
│   ├── staged/                    # cleaned Parquet
│   └── features/                  # engineered panels
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_segmentation.ipynb
│   └── 04_feature_importance.ipynb
├── src/
│   ├── ingest/                    # connectors + validation
│   ├── features/                  # panel build, feature transforms
│   ├── models/
│   │   ├── baselines.py
│   │   ├── global_lgbm.py
│   │   ├── intermittent.py        # Croston/TSB
│   │   ├── online_layer.py        # River
│   │   └── reconcile.py
│   ├── continuous/
│   │   ├── drift.py               # ADWIN/PageHinkley
│   │   ├── retrain.py
│   │   └── registry.py            # champion/challenger
│   ├── reorder/
│   │   ├── leadtime.py            # PO→GRN estimation
│   │   ├── safety_stock.py
│   │   └── policy.py              # ROP, order-up-to, EOQ, newsvendor
│   ├── evaluate/
│   │   ├── backtest.py            # rolling-origin
│   │   ├── forecast_metrics.py
│   │   └── inventory_sim.py       # service level, turns, waste
│   └── serve/
│       └── api.py                 # FastAPI
├── pipelines/                     # Airflow/Prefect DAGs
└── tests/
```

---

## Phase 0 — Foundations & Problem Framing

> **Goal:** lock down exactly what we are predicting and deciding, so no downstream ambiguity exists.

### 0.1 Objectives (SMART)
- **O1.** Forecast daily unit demand per (store, SKU) at quantiles {0.5, 0.9, 0.95} over a horizon of `max_lead_time + review_period` days.
- **O2.** Output an automated reorder recommendation (order? quantity?) per (store, SKU) every day.
- **O3.** Beat the seasonal-naive baseline (**MASE < 1**) on ≥ 80% of A/B-class SKUs.
- **O4.** Hit per-segment service-level targets (Phase 8) at lower inventory than the store's current practice.

### 0.2 Precise problem definition
- **Unit of prediction (the grain):** one row = (store_id, sku_id, date).
- **Target:** `units_sold` per grain — but corrected for censored demand (Phase 3).
- **Forecast horizon `H`:** must cover lead time + review period. If lead time ≈ 3 days and you review daily, forecast at least 7–14 days to be safe.
- **Decision policy:** continuous-review **(s, S)** — when inventory position ≤ reorder point `s`, order up to `S`. (Phase 7 defines s and S.)
- **What "demand" means:** true customer intent, *not* recorded sales (which are capped by what was on the shelf).

### 0.3 Scope
- **In scope:** demand forecasting, reorder quantity/timing, lead-time estimation, segmentation, continuous learning, evaluation, monitoring.
- **Out of scope (v1):** price/promo *optimization* (we *use* promo info, we don't set prices), multi-echelon network optimization (single store→supplier first; chain DC echelon is a roadmap item), assortment decisions.

### 0.4 Assumptions & constraints
- Store has (or can export) **transaction-level POS data** with date, SKU, quantity.
- Inventory on-hand and goods-receipt timestamps are available *or* can be reconstructed.
- Compute budget is modest → favor LightGBM + DuckDB over deep learning.
- Indian retail calendar (festivals, salary-day cycles) materially drives demand.

### 0.5 Key risks & mitigations
| Risk | Mitigation |
|---|---|
| No clean inventory/lead-time data | Bootstrap with public data; estimate lead time from PO→GRN; simulate during dev |
| Censored demand biases model to under-order | Stockout flag + latent-demand handling (Phase 3) |
| Annual events unlearnable online | Hybrid: scheduled retrain over ≥2 yrs + festival features |
| Cold-start new SKUs | Attribute/category model + aggregate-and-distribute |
| Over-trust in automation | Exception dashboard + human override + explanations |

### 0.6 Deliverables
- Signed-off problem statement & success criteria.
- `config/` skeleton with horizon, quantiles, policy type.
- Tech-stack confirmed and dev environment bootstrapped.

---

## Phase 1 — Data Acquisition & Understanding

> **Goal:** get the right data from the right place, with a contract the rest of the system can rely on.

### 1.1 Bootstrap / development datasets (build the pipeline before live data exists)

| Dataset | Source | Role | What it gives |
|---|---|---|---|
| **Store Item Demand Forecasting** | Kaggle (`demand-forecasting-kernels-only`) | First prototype (small) | 10 stores × 50 items × 5 yrs daily — get the pipeline end-to-end fast |
| **M5 Forecasting** (Walmart) | Kaggle (`m5-forecasting-accuracy` + `-uncertainty`) | **Primary bootstrap** | Hierarchical (item→dept→cat→store→state), **native quantile track**, prices, calendar/events, SNAP |
| **Corporación Favorita** | Kaggle (`favorita-grocery-sales-forecasting`) | Grocery realism | SKU×store×date, promo flag, **perishable flag**, store metadata, transactions, oil, holidays |
| **UCI Hierarchical Sales** | archive.ics.uci.edu (#611) | Lightweight real grocery | Real SKU-level daily series + promo flags |

> **Download:** use the Kaggle CLI (`kaggle competitions download -c <name>` / `kaggle datasets download -d <slug>`) after placing `kaggle.json` API token. Respect each competition's rules/licence.

**Why M5 primary:** its uncertainty track is literally the quantile-forecasting + service-level problem we're solving; its hierarchy lets sparse SKUs borrow strength. **Favorita** then adds grocery texture (perishable + promo). **The public data teaches the *forecasting* half; the *reorder* half (lead time, cost, MOQ, live inventory) it barely contains — those come from the store.**

### 1.2 Production data sources (the real, continuous fuel)

| Source table | Granularity | Key fields | Used for |
|---|---|---|---|
| `sales_transactions` | line-item / receipt | date, store_id, sku_id, qty, unit_price, discount | target, price/promo signals |
| `product_master` | per SKU | sku_id, name, category, family, brand, pack_size, perishable, shelf_life_days, unit_cost, sell_price | hierarchy, constraints, economics |
| `inventory_snapshot` | daily per (store, SKU) | date, store_id, sku_id, on_hand_qty, on_order_qty | stockout flag, inventory position |
| `purchase_orders` | per PO line | po_id, sku_id, supplier_id, order_date, order_qty | lead-time start, MOQ history |
| `goods_receipts` (GRN) | per receipt line | po_id, sku_id, receipt_date, received_qty | **lead-time end** → lead-time model |
| `suppliers` | per supplier | supplier_id, name, moq, order_cycle | order constraints |
| `promotions` | per promo | sku_id, start, end, type, discount_depth | promo features |
| `external_calendar` | per date (+region) | date, is_holiday, festival_name, festival_intensity, salary_window, temp, rain_mm, fuel_index | demand drivers |

> **External data:** build an **Indian festival calendar** (Diwali, Holi, Eid, Raksha Bandhan, Onam, Pongal, regional festivals) with an intensity score and lead-up days. Pull **weather** from a free API (e.g. Open-Meteo historical + forecast) keyed by store location.

### 1.3 Minimum data contract
Define in `config/data_contract.yaml` and enforce with `pandera`:
- `sales_transactions`: non-null date/store/sku; qty integer (allow negatives = returns, flagged); date ≤ today.
- `product_master`: every sku_id in sales must exist here; pack_size ≥ 1; shelf_life_days null ⇒ non-perishable.
- `inventory_snapshot`: one row per active (store, SKU, day); on_hand ≥ 0.
- Referential integrity: PO ↔ GRN joinable on po_id+sku_id.

### 1.4 Ingestion pipeline
1. **Landing:** raw dumps copied immutably to `data/raw/` (dated).
2. **Validation gate:** `pandera` schema check; reject/quarantine bad batches; log.
3. **Staging:** type-cast, deduplicate, normalize units, write partitioned Parquet (`store_id`, `date`) to `data/staged/`.
4. **Cadence:** nightly batch is the baseline. Optional: stream live transactions (Kafka/webhook) for same-day adaptation of fast movers.
5. **Idempotency:** re-running a date reproduces identical staged output.

### 1.5 Deliverables
- Connectors + validation for each source.
- Staged Parquet datasets (bootstrap + a sample of live schema).
- `data_contract.yaml` and a generated **data dictionary** (Appendix A).

---

## Phase 2 — Exploratory Data Analysis (EDA)

> **Goal:** understand demand deeply enough that every feature and modeling choice downstream is *justified by a finding here*. EDA is not decoration — it is the design input for Phases 4–7. Each subsection ends with **→ Decision** showing what it feeds.

### 2.1 Data-quality audit
- Missingness per column; duplicate receipts; negative quantities (returns); date gaps per (store, SKU); SKU churn (when each SKU enters/exits).
- Plot active-SKU count over time; count of stores reporting per day.
- **→ Decision:** which SKUs/stores have enough history to model vs. need cold-start treatment; where to fill calendar gaps (Phase 3).

### 2.2 Demand profiling & intermittency classification
- Per-SKU: histogram of daily units, % of zero-demand days, mean/variance.
- Compute **ADI** (average inter-demand interval) and **CV²** (squared coefficient of variation of nonzero demand). Classify each SKU (Syntetos–Boylan):
  - **Smooth:** ADI < 1.32 and CV² < 0.49
  - **Erratic:** ADI < 1.32 and CV² ≥ 0.49
  - **Intermittent:** ADI ≥ 1.32 and CV² < 0.49
  - **Lumpy:** ADI ≥ 1.32 and CV² ≥ 0.49
- **→ Decision:** **model routing** — Smooth/Erratic → LightGBM; Intermittent/Lumpy → Croston/TSB; very sparse → category model. This is the single most important EDA output.

### 2.3 Seasonality & trend
- **Weekly:** mean demand by day-of-week (kirana spikes on weekends / specific days).
- **Monthly / salary-day:** demand around month-start and mid-month salary windows.
- **Annual / festival:** overlay festival dates; quantify lift in the lead-up window.
- **STL decomposition** (trend/seasonal/residual) on aggregate and top SKUs.
- **ACF/PACF** to confirm lag structure (expect spikes at 7, 14, 365).
- **→ Decision:** which calendar, festival-proximity, and lag features to build (Phase 4); whether annual seasonality justifies a ≥2-year training window (Phase 6).

### 2.4 ABC × XYZ segmentation
- **ABC** (value): cumulative revenue Pareto → A (~top 80% revenue), B (next ~15%), C (last ~5%).
- **XYZ** (predictability): by demand CV → X (stable), Y (variable), Z (erratic).
- Build the **9-box** (AX … CZ).
- **→ Decision:** **service-level policy per cell** (Phase 7) — e.g. AX gets 99%, CZ gets 85%; and modeling effort concentrates on A/B.

### 2.5 Stockout / censored-demand detection (critical)
- Flag days where `units_sold == 0` AND `on_hand == 0` (or hit zero mid-day): demand was likely > 0 but unobservable.
- Quantify share of zeros that are stockout-induced vs genuine no-demand.
- Inspect demand just before/after stockouts to estimate suppressed demand.
- **→ Decision:** stockout handling in Phase 3 (don't train naively on these zeros) and a `was_stockout` feature in Phase 4. Ignoring this makes the model under-order exactly the items that keep selling out.

### 2.6 Price & promotion effects
- Sales lift during promo windows vs baseline; distribution of discount depths.
- Price-vs-units scatter / elasticity by category.
- Post-promo dip (pull-forward) check.
- **→ Decision:** price/promo features and whether to model promo lift explicitly (Phase 4/5).

### 2.7 External drivers
- Correlate daily demand with rainfall/temperature (rain suppresses footfall) and festival proximity.
- **→ Decision:** include weather + festival features; quantify expected gain.

### 2.8 Cross-SKU & hierarchy
- Category-level seasonality (cleaner signal than sparse SKUs); substitution/cannibalization and halo around promotions.
- **→ Decision:** hierarchical modeling/reconciliation (Phase 5); target-encoding of category.

### 2.9 Lead-time analysis (PO → GRN)
- Per supplier: distribution of (receipt_date − order_date); mean and std; trend over time.
- **→ Decision:** dynamic lead-time model (Phase 7) instead of a fixed number — directly tightens the reorder point.

### 2.10 EDA → downstream decision map
| EDA finding | Feeds | Decision |
|---|---|---|
| Intermittency class (ADI/CV²) | Phase 5 | Model routing per SKU |
| DoW / salary / festival peaks | Phase 4 | Calendar + festival features |
| Lag spikes at 7/14/365 | Phase 4 | Lag & rolling windows |
| ABC×XYZ box | Phase 7 | Service-level per segment |
| Stockout share | Phase 3/4 | Censored-demand handling + flag |
| Promo lift & dip | Phase 4/5 | Promo features |
| Weather/festival corr | Phase 4 | External features |
| Lead-time mean/std per supplier | Phase 7 | Dynamic lead-time + safety stock |

### 2.11 Deliverables
- `02_eda.ipynb` + an auto-profiling report.
- **Segmentation table** (sku_id → ABC, XYZ, intermittency class) saved to `data/features/segments.parquet`.
- Lead-time summary per supplier.
- A short "EDA findings → design decisions" memo (the table above, expanded).

---

## Phase 3 — Data Preparation & Cleaning

> **Goal:** turn raw transactions into a clean, leakage-free modeling panel where the target reflects *demand*, not just *sales*.

### 3.1 Build the continuous panel
- Create the full **(store_id × sku_id × date)** grid over each SKU's active window (first sale → last sale, or store-defined active range).
- Left-join sales; **fill missing dates with 0** units (no sale recorded = 0), but mark them so genuine zeros and stockout zeros differ.
- Do **not** fabricate rows before a SKU existed or after delisting.

### 3.2 Censored-demand handling (the make-or-break step)
- Add `was_stockout` (1 if on_hand reached 0 that day).
- Three policy options (configurable):
  1. **Mask:** exclude stockout days from the *loss* (sample weight = 0) so the model isn't taught a false zero.
  2. **Impute latent demand:** replace stockout-day sales with an estimate (e.g. expected demand from neighboring non-stockout same-DOW days).
  3. **Censored likelihood:** treat the observation as a lower bound (advanced; quantile models tolerate option 1/2 well).
- Default: **option 1 + a `was_stockout` feature**, revisit with option 2 if bias persists.

### 3.3 Outlier treatment
- Distinguish **explained** spikes (promo, festival, documented event) — *keep* them; the model should learn them.
- **Unexplained** extreme spikes → winsorize/cap at a high percentile (e.g. 99th per SKU) or flag as anomalies.
- Never blanket-remove high values; they often *are* the demand you must serve.

### 3.4 Returns / negative units
- Net them within the day or move to a separate `returns` signal; never let negative "sales" leak into demand.

### 3.5 Calendar & external enrichment
- Join `external_calendar` (festival, holiday, salary window, weather) onto the panel by date (+ region).

### 3.6 Train / validation / test split — **time-based, no leakage**
- **Rolling-origin / walk-forward** evaluation (never random K-fold on time series).
- Example: train on months 1–24 → validate next 14 days; roll forward; repeat over multiple origins.
- Apply an **embargo gap** equal to the horizon between train end and validation start so lag features can't peek.
- Final **hold-out test** = most recent period, untouched until the end.

### 3.7 Target definition & transforms
- Target = corrected daily `units`.
- For skewed/zero-inflated demand choose one:
  - **`tweedie` objective** (LightGBM) — natural for non-negative, zero-inflated counts; **recommended default**.
  - or **`log1p(units)`** transform with regression objective (back-transform at inference; correct for bias).
- For quantile heads, train directly on `units` with **quantile (pinball) objective** per quantile.

### 3.8 Deliverables
- `data/staged/panel.parquet` (clean, gap-filled, flagged).
- Documented split config + embargo in `config/`.
- Data-prep unit tests (no future leakage, no negative demand, grid completeness).

---

## Phase 4 — Feature Engineering & Feature Importance

> **Goal:** build the features the EDA justified, then *prove* which ones matter and prune the rest. Strict rule: **every feature must be computable using only information available at prediction time.**

### 4.1 Feature families (the full set to build)

**A. Calendar (from date)**
- `day_of_week`, `is_weekend`, `week_of_year`, `month`, `quarter`, `day_of_month`
- `is_month_start`, `is_salary_window` (configurable salary-day band)
- `is_holiday`

**B. Festival proximity (Indian retail driver)**
- `days_to_next_festival`, `days_since_last_festival`
- `festival_intensity` (0–1 weight per festival), `in_festival_leadup` (e.g. 7 days before Diwali)

**C. Lags** (respect horizon — for a direct H-step forecast use lags ≥ H)
- `lag_1, lag_7, lag_14, lag_28, lag_365`

**D. Rolling-window statistics** (shifted to avoid leakage)
- `roll_mean/median/std/min/max` over **7, 14, 28** days
- **same-day-of-week** rolling mean (e.g. last 4 same weekdays)
- expanding mean; **EWMA** (recent-weighted level)

**E. Trend / momentum**
- ratio `roll_mean_7 / roll_mean_28` (short vs long → rising/falling)
- week-over-week and year-over-year change

**F. Price & promotion**
- `unit_price`, `relative_price` (vs category median), `discount_depth`
- `on_promo`, `days_since_promo_start`, `days_to_promo_end`, `promo_in_next_7d`

**G. Stockout / availability**
- `was_stockout` (yesterday/last 7d), `days_since_stockout`, `stockout_rate_28d`
- *(used as features, but the target on those days is masked — keep the two roles separate)*

**H. Hierarchy & categorical**
- `store_id`, `category`, `family`, `brand`, `region`, `store_type`, `store_cluster`
- encode via LightGBM native categorical or **target/ordinal encoding** (fit on train only)

**I. SKU attributes**
- `pack_size`, `perishable`, `shelf_life_days`, `unit_cost`, `margin`

**J. External**
- `temp`, `rain_mm`, `fuel_index` / inflation proxy

**K. Intermittency descriptors**
- `adi`, `cv2`, `prob_of_sale_28d` (share of recent days with a sale)

### 4.2 Leakage rules (enforce mechanically)
- All lags/rolling **shifted by ≥1 day** (and ≥H for direct multi-step).
- Target/category encodings fit **only** on training folds, then applied out-of-fold.
- No feature may use same-day or future actuals.
- Unit-test: shuffle the future → metrics must not improve.

### 4.3 Feature-importance methodology (the EDA→features bridge the brief asked for)
Run in this order and reconcile:
1. **LightGBM gain & split importance** — fast first pass; rank candidates.
2. **Permutation importance** on the *validation* fold — model-agnostic, less biased toward high-cardinality features. Drop features whose permutation importance ≈ 0 or negative.
3. **SHAP values** — global ranking *and* direction of effect; SHAP interaction values reveal feature pairs (e.g. promo × festival). Produce per-segment SHAP (A-items vs intermittent) since drivers differ.
4. **Drop-column importance** on the shortlist — retrain without each top feature; confirms true contribution.
5. **Stability check** — importance must be consistent across rolling folds; unstable features are suspect.
6. **Redundancy pruning** — cluster highly correlated features (e.g. many rolling means); keep the most predictive per cluster.

**Output:** a ranked, pruned **final feature list** in `config/features.yaml` with a short rationale per feature (or per dropped feature). Aim for the smallest set that retains accuracy — fewer features = faster retrains and less drift surface.

### 4.4 Reproducibility
- Implement features as pure, versioned transforms in `src/features/`.
- Optionally register in **Feast** so the online layer and training use identical definitions (train/serve consistency).

### 4.5 Deliverables
- `data/features/train_panel.parquet` with the final feature set.
- `04_feature_importance.ipynb` (gain + permutation + SHAP + drop-column).
- Final feature list + rationale committed to config.

---

## Phase 5 — Model Building (Forecasting)

> **Goal:** produce calibrated **quantile** demand forecasts per (store, SKU, horizon), using the right model for each demand type. We borrow ToolsGroup's probabilistic philosophy and RELEX's "combination of methods."

### 5.1 Baselines (the bar every model must clear)
Implement with **statsforecast**:
- **Naive** (last value), **Seasonal Naive** (value 7 days ago) ← the key benchmark for MASE.
- **Moving Average**, **ETS / Holt-Winters** (level+trend+seasonality).
- **Croston / SBA / TSB** for intermittent & lumpy SKUs.

> A model only earns its place if it beats Seasonal Naive (**MASE < 1**). Compute **Forecast Value Add** vs naive for every candidate.

### 5.2 Primary model — global LightGBM
- **One model across all (store, SKU) series.** Rationale: cross-learning (sparse SKUs borrow from similar ones), trivial cold-start (predict from attributes), and it scales to chain size far better than per-series models.
- **Objective:** `tweedie` (or `regression_l1`/MAE for robustness) for the central forecast.
- Native categorical handling; optional **monotonic constraints** (e.g. demand non-decreasing in promo depth).
- Early stopping on a time-based validation fold.

### 5.3 Probabilistic / quantile forecasting (the core for reordering)
- Train **separate LightGBM heads per quantile** {0.5, 0.9, 0.95} with the **quantile (pinball) objective**, OR a multi-quantile approach.
- Enforce **non-crossing** quantiles (sort, or train jointly) so P95 ≥ P90 ≥ P50.
- These quantiles are exactly what Phase 7 turns into safety stock — high quantile ⇒ high service level. *This replaces the old "assume a normal distribution" safety-stock shortcut with an empirical, learned distribution.*

### 5.4 Hierarchical reconciliation (recommended)
- Forecast at **SKU** and **category** (and optionally store-group) levels.
- Reconcile with **hierarchicalforecast** (bottom-up or **MinT**) so SKU forecasts sum to the more-stable category forecast — improves sparse-SKU accuracy and coherence.

### 5.5 Cold-start / new SKU
- No history → predict from **attributes + category base rate** (a model trained on `category, brand, pack_size, price, festival, …`).
- **Aggregate-and-distribute:** forecast the category total, split to the new SKU by its expected share. Transition to the SKU-level model as history accumulates.

### 5.6 Hyperparameter optimization
- **Optuna** with **time-series CV** (rolling origin), optimizing the validation pinball/WAPE.
- Search: `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`, `max_depth`.
- Always pair with **early stopping**; log every trial to MLflow.

### 5.7 Per-segment model routing (combination of methods)
| Segment (from Phase 2) | Model |
|---|---|
| Smooth / Erratic, enough history | Global LightGBM (quantile) |
| Intermittent / Lumpy | Croston / SBA / **TSB** |
| Very sparse / brand-new | Category model + aggregate-and-distribute |
| High-value A-items | LightGBM + extra features + tighter tuning |

A **router** assigns each (store, SKU) to its model based on the segmentation table; the reorder layer consumes whichever quantiles result.

### 5.8 Deliverables
- Trained baselines + global quantile models + intermittent models, all in MLflow.
- Cross-validated metric table per segment (Phase 8 metrics).
- **Model card** (data window, features, objectives, quantiles, known limits).

---

## Phase 6 — Hybrid Continuous-Learning Architecture

> **Goal:** keep the system learning from daily data *without* forgetting annual patterns. This is the heart of the project and the explicit reason it isn't "train once, use forever."

### 6.1 Why hybrid (not pure online)
Pure online learning sees each year's Diwali **once** and tends to forget it. Pure scheduled batch reacts slowly to sudden shifts. The incumbents resolve this with rolling retrains + dynamic parameters + drift detection. We combine three timescales:

### 6.2 Component 1 — Global base model (slow timescale)
- Retrain the global LightGBM on a **rolling window of ≥24–36 months** on a **schedule** (e.g. weekly).
- Captures seasonality, festival effects, and cross-SKU structure.
- Each retrain is versioned in the MLflow registry.

### 6.3 Component 2 — Online adaptation layer (fast timescale)
- A lightweight **River** model updated **daily** that **corrects the base model's residuals** using the most recent features (recent level, last error, current promo/weather).
- Equivalent options: per-series adaptive level via **EWMA/Kalman** for fast movers.
- Absorbs: recent demand drift, new-SKU ramp, promo reactions, local shocks — *between* base retrains.
- Final forecast = `base_quantile + online_residual_correction` (kept non-negative, quantiles re-sorted).

### 6.4 Component 3 — Drift detection (event-driven retrain trigger)
- Maintain a per-segment stream of forecast errors.
- Run **ADWIN** / **Page-Hinkley** / **KSWIN** (River) on that stream.
- On a drift alarm for a slice → **trigger off-cycle retraining** of the affected slice (or globally if widespread). This is how the system reacts to regime changes faster than the weekly cadence.

### 6.5 Champion / Challenger & promotion
- A freshly trained model is a **challenger**; it runs in **shadow** and is scored on a rolling backtest.
- **Promote only if** it beats the champion on the primary metric (WAPE/pinball) by a margin and passes calibration checks.
- Otherwise keep the champion. All transitions logged; rollback supported.

### 6.6 The loop (pseudocode)
```python
# DAILY
ingest(yesterday)                       # sales, inventory, GRN
features = update_panel(yesterday)      # append lags/rolling
online_layer.learn_one(yesterday_residuals)   # fast adaptation
q = base_model.predict_quantiles(features)     # P50/P90/P95
q = online_layer.adjust(q)
recs = reorder_engine.decide(q, inventory, leadtime, policy)
publish(recs)

# WHEN ACTUALS LAND
err = error(actuals, forecast)
log_metrics(err)
if drift_detector.update(err).drift_detected:
    trigger_retrain(scope=affected_segment)

# SCHEDULED (e.g. weekly) OR on drift
challenger = train_global(rolling_window)
if beats(challenger, champion, on=backtest):
    registry.promote(challenger)
```

### 6.7 Concept-drift taxonomy & response
| Drift type | Example | Response |
|---|---|---|
| Gradual | slow taste shift | rolling-window retrain handles it |
| Sudden | new competitor, lockdown | drift detector → off-cycle retrain + online layer cushions |
| Seasonal | festivals, monsoon | festival features + ≥2-yr window |
| New-product | SKU launch | cold-start model → migrate to SKU model |

### 6.8 Deliverables
- Implemented base + online + drift components (`src/continuous/`, `src/models/online_layer.py`).
- MLflow registry with champion/challenger workflow.
- Loop wired into the orchestration DAG (Phase 9).

---

## Phase 7 — Reorder Decision Layer

> **Goal:** convert quantile forecasts + lead time into the actual decision — *order now?* and *how many?* This is the system's real output.

### 7.1 Dynamic lead-time model (StockIQ-style)
- From PO→GRN history per supplier, compute **mean lead time `L̄`** and **std `σ_L`**.
- Recompute on a rolling window so lead time tracks reality instead of a fixed guess.
- Fallback to a supplier-declared default when history is thin.

### 7.2 Demand over the protection period
- Protection period `P = L̄ + review_period`.
- Aggregate the daily quantile forecasts over `P` to get the **distribution of demand during lead time** (sum of daily P50s for the mean; use the high-quantile path or convolve daily quantiles for the upper tail).

### 7.3 Safety stock (two equivalent routes)
**Route A — formula (demand + lead-time variability):**
```
SS = z * sqrt( L̄ * σ_d²  +  d̄² * σ_L² )
```
where `z` = service-level factor, `d̄`/`σ_d` = mean/std of daily demand, `L̄`/`σ_L` = mean/std of lead time.

**Route B — empirical quantile (preferred, uses our probabilistic forecast):**
```
SS = (forecast quantile of demand over P at target service level) − (expected demand over P)
```
Route B inherits the *learned* demand distribution (skew, fat tails) instead of assuming normality; combine with `σ_L` term for supply-side risk.

### 7.4 Reorder point and order-up-to level
```
Reorder point   s = d̄ * L̄ + SS
Order-up-to     S = demand over (P + order_cycle) at service level + SS
Inventory position IP = on_hand + on_order − backorders
Trigger: if IP ≤ s  →  order
```

### 7.5 Order quantity
- **(s, S) policy:** `order_qty = S − IP`.
- **EOQ** (for steady items, to balance ordering vs holding cost):
```
EOQ = sqrt( 2 * D * S_cost / H )
```
`D` = annual demand, `S_cost` = cost per order, `H` = holding cost per unit per year.
- **Round** up to MOQ and to pack/case multiples.

### 7.6 Perishables — newsvendor constraint
- For short-shelf-life items, the optimal stocking quantile is the **critical fractile**:
```
critical_ratio = Cu / (Cu + Co)
```
`Cu` = underage cost (lost margin on a stockout), `Co` = overage cost (cost of spoiled unit). Order the demand quantile equal to `critical_ratio`, and cap the order so quantity ≤ expected sales within `shelf_life_days`.

### 7.7 Service-level targeting by segment
Set `z` per ABC×XYZ cell in `config/policy.yaml`:

| Segment | Target service level | z (≈) |
|---|---|---|
| A-items (high value, stable) | 97–99% | 1.88–2.33 |
| B-items | 95% | 1.65 |
| C-items (low value) | 85–90% | 1.04–1.28 |
| Perishables | set by newsvendor (waste-aware) | — |

### 7.8 Output (per store, SKU)
A PO recommendation row: `should_order (bool)`, `order_qty`, `target_supplier`, `expected_stockout_risk`, and a short **explanation** ("inventory position 12 ≤ reorder point 18; forecast P95 over 5-day lead time = 22; ordering up to 30, rounded to case of 6"). Group by supplier into draft POs.

### 7.9 Exception management (Netstock-style)
- Auto-approve routine recommendations; **surface only exceptions** (large orders, low-confidence forecasts, items newly drifting, perishables near expiry) for human review. Keeps the chain operable with few planners.

### 7.10 Deliverables
- `src/reorder/` (leadtime, safety_stock, policy) + `policy.yaml`.
- Draft-PO output table consumed by the store app / UI.

---

## Phase 8 — Evaluation & Metrics

> **Goal:** measure both **forecast accuracy** and **inventory/business outcome**, with explicit target values and a go-live bar. A good forecast that produces bad orders is a failure — so we simulate the full loop.

### 8.1 Backtesting protocol
- **Rolling-origin / walk-forward** over multiple origins; report per-horizon and per-segment.
- **Inventory simulation:** replay forecasts → reorder engine → simulated inventory to compute realized service level, turns, and waste (not just forecast error).
- Always compare against the **seasonal-naive baseline** and the store's current practice.

### 8.2 Forecast-accuracy metrics

| Metric | Definition | Why | Target |
|---|---|---|---|
| **WAPE** (primary) | `Σ|y−ŷ| / Σ y` | Scale-free, robust to zeros, weights by volume | A-items < 20–25%; category-level < 15% |
| **MASE** | `MAE / MAE_seasonal_naive` | The honest "are we beating naive?" test | **< 1** (target ≤ 0.8) on ≥80% of A/B SKUs |
| **Bias (MPE)** | `mean(ŷ − y)` (signed) | Detect systematic over/under-forecast | within **±5%** |
| **RMSE** | `sqrt(mean((y−ŷ)²))` | Penalizes large misses | minimize; track vs baseline |
| **sMAPE** | `mean( 2|y−ŷ| / (|y|+|ŷ|) )` | Symmetric % error (use cautiously near 0) | < 30% (smooth SKUs) |
| **Pinball loss** | `Σ max(q·(y−ŷ), (q−1)·(y−ŷ))` | Correct loss for quantile forecasts | minimize per quantile vs baseline |
| **Quantile coverage** | empirical P(y ≤ q̂) vs nominal | Calibration of the distribution | P95 coverage **93–97%**; P90 **88–92%** |

> **Primary decision metric:** WAPE for point accuracy, **pinball loss + coverage** for the distribution (because the distribution drives reordering). MASE is the credibility gate.

### 8.3 Inventory / business metrics (from the simulation & production)

| Metric | Definition | Target |
|---|---|---|
| **Cycle service level** | % of cycles with no stockout | meet per-segment target (97–99% A, 85–90% C) |
| **Fill rate** | units shipped ÷ units demanded | ≥ 95% on A/B |
| **Stockout rate** | % of (SKU·day) out of stock | **−30–50%** vs current |
| **Inventory turns** | COGS ÷ average inventory | **+15–30%** vs current |
| **Days of inventory (DOH)** | avg inventory ÷ avg daily COGS | reduce while holding service |
| **Overstock / excess units** | units above target stock | **−20–30%** total inventory |
| **Waste %** (perishables) | spoiled ÷ received | **−10–30%** |
| **GMROI** | gross margin ÷ avg inventory cost | improve quarter over quarter |
| **Forecast Value Add (FVA)** | accuracy gain vs naive | positive on every modeled segment |

> Inventory-reduction and waste targets are calibrated to publicly reported ranges from comparable systems (e.g. ~20–30% inventory reduction, 10–30% waste reduction) and should be re-baselined against the specific store's starting numbers.

### 8.4 Go-live acceptance criteria (concrete)
The system ships when, on the hold-out + simulation:
1. **MASE < 1** on ≥ 80% of A/B SKUs.
2. **P95 coverage** within 93–97% (calibrated).
3. Simulated **service level ≥ target per segment** at **inventory ≤ current**.
4. **Bias** within ±5% (no systematic under-ordering).
5. Positive **FVA** vs seasonal naive across segments.

### 8.5 Production (rolling) metrics
Track WAPE, bias, coverage, service level, and turns on rolling windows; alert on degradation (ties into Phase 10).

### 8.6 Deliverables
- `src/evaluate/` (backtest, forecast_metrics, inventory_sim).
- Evaluation report + a metrics dashboard.
- `config/metrics.yaml` with targets and acceptance thresholds.

---

## Phase 9 — Deployment & MLOps

> **Goal:** run the whole loop reliably, cheaply, and reproducibly in production.

### 9.1 Serving pattern
- **Batch-first:** nightly DAG scores all (store, SKU) and writes reorder recommendations to a table the store app reads. Simple, cheap, sufficient for daily reordering.
- **Optional near-real-time:** a **FastAPI** service for on-demand re-scoring of fast movers when live transactions stream in.

### 9.2 Orchestration (Airflow / Prefect DAG)
```
ingest → validate → build_features → online_update → score_quantiles →
reorder_decide → publish_recommendations → log_actuals → drift_check →
[weekly or on-drift] retrain → evaluate → promote
```

### 9.3 Train/serve consistency
- Same feature code path (ideally **Feast**) for training and scoring; prevents skew.

### 9.4 Model registry & CI/CD
- **MLflow** for experiments + registry (champion/challenger, versions, lineage).
- CI runs data-prep tests, leakage tests, and a mini-backtest before any promotion.

### 9.5 Data validation gates
- **pandera / Great Expectations** block bad data from reaching training or scoring; quarantine + alert.

### 9.6 Infrastructure & cost
- Start on a single VM (8–16 GB) with **DuckDB + Parquet**; PostgreSQL for operational state.
- Scale out (Polars/Dask/Spark, partitioned scoring) only when SKU×store×day volume requires it.

### 9.7 Deliverables
- Deployed DAG + serving layer.
- CI/CD with validation + backtest gates.
- Runbook (retrain, rollback, on-call).

---

## Phase 10 — Monitoring, Feedback & Continuous Improvement

> **Goal:** make the system self-aware — detect when it's drifting or decaying and close the human feedback loop.

### 10.1 Monitoring (Evidently)
- **Data drift:** input feature distributions vs training reference.
- **Prediction drift:** forecast distribution shifts.
- **Accuracy decay:** rolling WAPE/bias/coverage vs targets.
- **Operational:** data freshness/SLA, pipeline failures, scoring latency.

### 10.2 Drift → retrain automation
- Monitoring alarms feed the same trigger as Phase 6.4 (off-cycle retrain of affected slices).

### 10.3 Human-in-the-loop feedback
- Planner overrides on recommendations are **captured as signals** (e.g. known upcoming local event) and fed back to improve features/policy.
- Periodic review of exception decisions to refine thresholds.

### 10.4 Exception dashboard & alerting
- Surface low-confidence forecasts, drifting SKUs, perishables near expiry, unusually large orders. Alert channels for stockout-risk and overstock-risk.

### 10.5 Review cadence
| Cadence | Activity |
|---|---|
| Daily | scoring, reorder, monitoring alerts |
| Weekly | base-model retrain + challenger evaluation |
| Monthly | re-segment (ABC/XYZ), review policy/service levels, lead-time refresh |
| Quarterly | architecture review, target re-baselining, roadmap |

### 10.6 Roadmap (post-v1)
- **Multi-echelon** (chain DC → stores) optimization.
- **Promotion/price-aware** demand and ordering.
- **Agentic** assistant for planners (natural-language exception triage).
- Substitution/cannibalization modeling; assortment optimization.

### 10.7 Deliverables
- Monitoring dashboards + alerts.
- Feedback-capture mechanism.
- Documented review cadence and roadmap.

---

## Appendices

### Appendix A — Data Dictionary (core fields)
| Table | Field | Type | Notes |
|---|---|---|---|
| sales_transactions | date | date | transaction date |
| | store_id | str | store key |
| | sku_id | str | product key |
| | qty | int | units (negative = return, flagged) |
| | unit_price | float | actual selling price |
| | discount | float | discount applied |
| product_master | sku_id | str | PK |
| | category / family / brand | str | hierarchy |
| | pack_size | int | units per case |
| | perishable | bool | drives newsvendor logic |
| | shelf_life_days | int | null ⇒ non-perishable |
| | unit_cost / sell_price | float | economics |
| inventory_snapshot | date, store_id, sku_id | — | daily grain |
| | on_hand_qty / on_order_qty | int | inventory position inputs |
| purchase_orders | po_id, sku_id, supplier_id | — | order keys |
| | order_date, order_qty | — | lead-time start, MOQ history |
| goods_receipts | po_id, sku_id, receipt_date | — | lead-time end |
| suppliers | supplier_id, moq, order_cycle | — | order constraints |
| promotions | sku_id, start, end, type, discount_depth | — | promo features |
| external_calendar | date, festival_name, festival_intensity, salary_window, is_holiday, temp, rain_mm, fuel_index | — | demand drivers |

### Appendix B — Formula reference
```
WAPE            = Σ|y−ŷ| / Σ y
MASE            = MAE / MAE_seasonal_naive
sMAPE           = mean( 2|y−ŷ| / (|y|+|ŷ|) )
Pinball(q)      = Σ max( q·(y−ŷ), (q−1)·(y−ŷ) )
Bias (MPE)      = mean(ŷ − y)
ADI             = (#periods) / (#periods with demand)
CV²             = (std_nonzero / mean_nonzero)²
Safety stock    = z · sqrt( L̄·σ_d² + d̄²·σ_L² )
Reorder point   = d̄·L̄ + SS
EOQ             = sqrt( 2·D·S_cost / H )
Newsvendor      = order the demand quantile = Cu / (Cu + Co)
Inventory turns = COGS / average inventory
GMROI           = gross margin / average inventory cost
Fill rate       = units shipped / units demanded
```

### Appendix C — Metric targets at a glance
| Metric | Target |
|---|---|
| MASE (A/B SKUs) | < 1 (aim ≤ 0.8) |
| WAPE (A-items) | < 20–25% |
| WAPE (category) | < 15% |
| Bias | ±5% |
| P95 quantile coverage | 93–97% |
| Fill rate (A/B) | ≥ 95% |
| Stockout rate | −30–50% vs current |
| Inventory turns | +15–30% |
| Total inventory | −20–30% |
| Perishable waste | −10–30% |

### Appendix D — Glossary
- **SKU** — stock-keeping unit (a single product).
- **Lead time** — days from placing an order to receiving it.
- **Service level** — probability of *not* stocking out in a cycle.
- **Safety stock** — buffer beyond expected demand for uncertainty.
- **Reorder point (s)** — inventory level that triggers an order.
- **Order-up-to (S)** — target level to refill to.
- **Quantile / pinball loss** — loss for predicting a specific percentile.
- **Censored demand** — true demand hidden because the shelf was empty.
- **Intermittent demand** — many zero-sales days (slow movers).
- **ADI / CV²** — intermittency and variability measures for SKU classification.
- **Drift** — change in data/relationships over time that degrades a model.
- **Champion/Challenger** — keep the live model unless a new one provably beats it.
- **MEIO** — multi-echelon inventory optimization (network-level, roadmap).

### Appendix E — Phase → deliverable → tool quick map
| Phase | Key deliverable | Core tools |
|---|---|---|
| 0 Foundations | Problem spec, config skeleton | — |
| 1 Data | Staged Parquet, data contract | Kaggle API, pandera, DuckDB |
| 2 EDA | Segmentation table, findings memo | pandas, matplotlib, statsforecast |
| 3 Prep | Clean panel, splits | pandas, Polars |
| 4 Features | Feature panel, importance report | LightGBM, SHAP |
| 5 Modeling | Quantile models, model card | LightGBM, statsforecast, Optuna, MLflow |
| 6 Continuous | Hybrid loop, registry | River, MLflow, Airflow |
| 7 Reorder | Reorder engine, policy config | Python, policy.yaml |
| 8 Evaluation | Backtest + inventory sim report | custom + Evidently |
| 9 Deployment | DAG, serving, CI/CD | Airflow/Prefect, FastAPI, Feast |
| 10 Monitoring | Dashboards, feedback loop | Evidently, MLflow |

### Appendix F — Suggested timeline (indicative)
| Phase | Effort |
|---|---|
| 0 Foundations | 0.5 week |
| 1 Data | 1–2 weeks |
| 2 EDA | 1 week |
| 3 Prep | 1 week |
| 4 Features | 1–2 weeks |
| 5 Modeling | 2–3 weeks |
| 6 Continuous learning | 2 weeks |
| 7 Reorder layer | 1–2 weeks |
| 8 Evaluation | 1 week |
| 9 Deployment | 1–2 weeks |
| 10 Monitoring | ongoing |

---

*End of document. This specification is intended to be detailed enough to build from end to end; refine the target values against the specific store's baseline once live data is available.*
