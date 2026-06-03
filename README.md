# Kirana Demand Forecasting & Automated Reordering

A continuous-learning system that forecasts SKU-level **demand distributions** (P50/P90/P95)
and automatically decides **when** and **how much** to reorder for small retail (kirana) stores
at chain scale.

> Full specification: [kirana_demand_forecasting_project.md](kirana_demand_forecasting_project.md)

## Core idea
Probabilistic (quantile) demand forecast → service-level reorder policy, kept alive by a
hybrid continuous-learning loop (scheduled global retrain + daily online adaptation + drift-triggered retrain).

## Status
- **Phase 0 — Foundations:** ✅ complete (config locked, env bootstrapped) — see [docs/PHASE0_foundations.md](docs/PHASE0_foundations.md)
- **Phase 1 — Data acquisition:** in progress

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
