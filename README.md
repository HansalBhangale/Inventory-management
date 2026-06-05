# Kirana Demand Forecasting & Automated Reordering

A continuous-learning system that forecasts SKU-level **demand distributions** (P50/P90/P95)
and automatically decides **when** and **how much** to reorder for small retail (kirana) stores
at chain scale.

> Full specification: [kirana_demand_forecasting_project.md](kirana_demand_forecasting_project.md)

## Core idea
Probabilistic (quantile) demand forecast → service-level reorder policy, kept alive by a
hybrid continuous-learning loop (scheduled global retrain + daily online adaptation + drift-triggered retrain).

## Status
- **Phases 0–5:** ✅ data → features → global LightGBM quantile model (champion), all 10 M5 stores.
- **Phase 7 (reorder + inventory sim):** ✅ — [docs/PHASE7_findings.md](docs/PHASE7_findings.md)
- **Phase 8 (acceptance):** ✅ **official gate = inventory-frontier dominance** — [docs/PHASE8_acceptance.md](docs/PHASE8_acceptance.md)
- **Phase 6 (continuous learning):** ✅ validated machinery — [docs/PHASE6_hybrid.md](docs/PHASE6_hybrid.md)

### Go-live verdict (M5, assumed lead times)
The official gate is **service per unit of inventory vs seasonal-naive** (not point accuracy),
graded **out-of-sample** (routing decided on one series split, scored on a disjoint one).
**SEGMENTED GATE: PASS** — at 95% service the model saves **~5% inventory in aggregate** and
**13–17%** on the cash-tying B/C and lumpy/intermittent tail (~90% of the catalog), and **matches
the baseline by design** on the easy fast-movers (it routes there to naive, so it ties rather than
beats). The one underperformance — **A-items at 99% service** — is a shock-free-data artifact
(magnitude −7.1% full-sample vs −0.8% held-out; expected to reverse on real shock data, not yet
measured).

> The earlier per-SKU MASE<1-on-80%-A/B test (≈0.71) is a **diagnostic**
> ([acceptance.py](src/evaluate/acceptance.py)), **not** the gate — point accuracy is a proxy the
> inventory simulation supersedes. See [docs/PHASE5_findings.md](docs/PHASE5_findings.md) and the
> full [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

## Quickstart

```bash
# 1. Create environment (Python 3.11+)
python -m venv .venv
.\.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. Add Kaggle credentials
#    kaggle.com -> Settings -> API -> Create New Token  ->  ~/.kaggle/kaggle.json
#    Accept competition rules for m5-forecasting-accuracy & -uncertainty

# 3. Download bootstrap data (M5)
python -m src.ingest.download_data --dataset m5_accuracy m5_uncertainty
```

## Repository layout
```
config/      # data contract, features, model, policy, metrics (Phase 0 outputs)
data/        # raw / staged / features (gitignored)
notebooks/   # 01_data_audit, 02_eda, 03_segmentation, 04_feature_importance
src/
  ingest/    # connectors + validation (Phase 1)
  features/  # panel build, feature transforms (Phase 4)
  models/    # baselines, global_lgbm, intermittent, online_layer, reconcile (Phase 5/6)
  continuous/# drift, retrain, registry (Phase 6)
  reorder/   # leadtime, safety_stock, policy (Phase 7)
  evaluate/  # backtest, forecast_metrics, inventory_sim (Phase 8)
  serve/     # FastAPI (Phase 9)
pipelines/   # Airflow/Prefect DAGs (Phase 9)
tests/
docs/
```

## Tech stack
Python 3.12 · pandas/Polars/DuckDB · LightGBM (quantile) · statsforecast (Croston/TSB) ·
River (online + drift) · Optuna · SHAP · MLflow · Evidently · FastAPI. All open-source.
