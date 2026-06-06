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
- **Phase 6 (continuous learning):** next.

### Go-live verdict (M5, assumed lead times)
The official gate is **service per unit of inventory vs seasonal-naive** (not point accuracy).
**CORE GATE: PASS** — LGBM dominates the frontier in aggregate (+6.5% inventory saved at matched
fill) and on every demand class incl. the intermittent tail (+17%). **A-items @ q99: LOSS** until
the dedicated q0.99 head is trained (currently a normal extrapolation).

> The earlier per-SKU MASE<1-on-80%-A/B test (≈0.71) is a **diagnostic**
> ([acceptance.py](src/evaluate/acceptance.py)), **not** the gate — point accuracy is a proxy the
> inventory simulation supersedes. See [docs/PHASE5_findings.md](docs/PHASE5_findings.md).

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
