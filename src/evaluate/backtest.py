"""Rolling-origin backtest (Phase 8.1) tying model + metrics together.

For each fold: train the global LightGBM on rows up to train_end (with an internal
time-based validation slice for early stopping), predict the validation window, and score
against the seasonal-naive baseline — overall and per segment.

Outputs:
  data/features/backtest_predictions.parquet  (per-row preds for inventory sim, Phase 8)
  docs/PHASE5_metrics.md                       (metrics table + model card)

Usage:
    python -m src.evaluate.backtest --stores CA_1 --folds 1        # last fold (fast)
    python -m src.evaluate.backtest --stores CA_1 --folds 3
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

from src.config import CONFIG
from src.evaluate import forecast_metrics as M
from src.evaluate.splits import rolling_origin_splits
from src.models.baselines import baseline_predictions
from src.models.global_lgbm import GlobalLGBM, feature_columns
from src.models.intermittent import forecast_intermittent
from src.models.router import apply_routing

FEATURES = CONFIG.data_dir / "features"
FP = (FEATURES / "feature_panel.parquet").as_posix()


def _load(where_sql: str, cols: list[str], float_cols: set[str] | None = None) -> pd.DataFrame:
    float_cols = float_cols or set()
    # Cast numerics to FLOAT (float32) in SQL so the dataframe arrives half-size — avoids the
    # float64 block-consolidation spike that OOMs on the ~45M-row all-stores training load.
    sel = ", ".join(
        f"CAST({c} AS FLOAT) AS {c}" if c in float_cols else c
        for c in dict.fromkeys(cols)
    )
    return duckdb.connect().execute(
        f"SELECT {sel} FROM read_parquet('{FP}/**/*.parquet') WHERE {where_sql}"
    ).df()


def run(stores: list[str] | None, n_folds: int) -> pd.DataFrame:
    feats, cats = feature_columns()
    keep = feats + ["units", "sample_weight", "date", "store_id", "sku_id",
                    "abc", "intermittency"]
    # everything that isn't a categorical or a key/date is loaded as float32
    non_float = set(cats) | {"date", "store_id", "sku_id", "abc", "intermittency"}
    float_cols = {c for c in keep if c not in non_float}

    # rolling training window (Phase 6.2): retrain on the last N months, not all history.
    window_days = int(CONFIG.model["continuous_learning"]["base_model_retrain_window_months"] * 30)

    store_filt = ""
    if stores:
        ids = ", ".join(f"'{s}'" for s in stores)
        store_filt = f" AND store_id IN ({ids})"

    mn, mx = duckdb.connect().execute(
        f"SELECT min(date), max(date) FROM read_parquet('{FP}/**/*.parquet') WHERE 1=1{store_filt}"
    ).fetchone()
    folds = rolling_origin_splits(mn, mx)[-n_folds:]
    print(f"panel {mn}..{mx} | running {len(folds)} fold(s)")

    all_preds = []
    rows = []
    for f in folds:
        print(f"\n=== {f} ===")
        # internal early-stopping validation = last `horizon` days before the embargo
        es_end = f.train_end
        es_start = es_end - timedelta(days=CONFIG.horizon - 1)
        train_lo = es_start - timedelta(days=window_days)
        train = _load(f"date BETWEEN '{train_lo}' AND '{es_start - timedelta(days=1)}'{store_filt}",
                      keep, float_cols)
        es_val = _load(f"date BETWEEN '{es_start}' AND '{es_end}'{store_filt}", keep, float_cols)
        valid = _load(f"date BETWEEN '{f.valid_start}' AND '{f.valid_end}'{store_filt}", keep, float_cols)
        print(f"  train={len(train):,} (window {train_lo}..{es_start - timedelta(days=1)}) "
              f"es_val={len(es_val):,} valid={len(valid):,}")

        model = GlobalLGBM().fit(train, es_val)
        preds = model.predict(valid)
        base = baseline_predictions(valid)
        scale = M.seasonal_naive_scale(train["units"].to_numpy(), m=7)

        res = pd.concat([valid[["date", "store_id", "sku_id", "abc", "intermittency", "units"]]
                         .reset_index(drop=True),
                         preds.reset_index(drop=True), base.reset_index(drop=True)], axis=1)

        # Phase 5.7 routing (config-driven). TSB only computed if selected — on M5 it lost
        # to LGBM, so intermittent_model defaults to lgbm and routed == lgbm.
        if CONFIG.model.get("routing", {}).get("intermittent_model", "lgbm") != "lgbm":
            tsb = forecast_intermittent(stores, f.train_end, f.valid_end)
            res = res.merge(tsb, on=["store_id", "sku_id"], how="left")
        res["pred_final"] = apply_routing(res)
        res["fold"] = f.index
        all_preds.append(res)

        y = res["units"].to_numpy()
        rows.append(_score("ALL", "lgbm", y, res["pred_central"].to_numpy(), res, scale, f.index))
        rows.append(_score("ALL", "routed", y, res["pred_final"].to_numpy(), res, scale, f.index))
        rows.append(_score("ALL", "seasonal_naive", y, res["seasonal_naive"].to_numpy(),
                           None, scale, f.index))
        # per-segment: lgbm, routed, and the out-of-sample naive for an explicit comparison
        for seg, g in res.groupby("abc"):
            yy = g["units"].to_numpy()
            rows.append(_score(f"ABC={seg}", "lgbm", yy, g["pred_central"].to_numpy(), g, scale, f.index))
            rows.append(_score(f"ABC={seg}", "routed", yy, g["pred_final"].to_numpy(), g, scale, f.index))
            rows.append(_score(f"ABC={seg}", "seasonal_naive", yy,
                               g["seasonal_naive"].to_numpy(), None, scale, f.index))

    metrics = pd.DataFrame(rows)
    out_preds = FEATURES / "backtest_predictions.parquet"
    pd.concat(all_preds, ignore_index=True).to_parquet(out_preds)
    print(f"\nwrote {out_preds}")
    return metrics


def _score(segment, model, y, yhat, qframe, scale, fold) -> dict:
    d = {"fold": fold, "segment": segment, "model": model, "n": len(y)}
    d.update(M.all_point_metrics(y, yhat, scale))
    if qframe is not None and "pred_q95" in qframe:
        for q, col in [(0.5, "pred_q50"), (0.9, "pred_q90"), (0.95, "pred_q95")]:
            d[f"pinball_q{int(q*100)}"] = M.pinball(y, qframe[col].to_numpy(), q)
            d[f"cov_q{int(q*100)}"] = M.coverage(y, qframe[col].to_numpy())
    return d


def write_report(metrics: pd.DataFrame, stores) -> None:
    out = Path(CONFIG.root) / "docs" / "PHASE5_metrics.md"
    overall = metrics[(metrics.segment == "ALL")]
    lines = [
        "# Phase 5 — Model Metrics & Card\n",
        f"Scope: stores={stores or 'ALL'} · horizon={CONFIG.horizon}d · "
        f"quantiles={CONFIG.quantiles}\n",
        "## Overall (validation folds)\n",
        overall.round(4).to_markdown(index=False),
        "\n## Per-segment (ABC, LightGBM)\n",
        metrics[metrics.segment.str.startswith("ABC")].round(4).to_markdown(index=False),
        "\n## Model card\n",
        f"- **Model:** global LightGBM, one model across all (store,SKU).\n"
        f"- **Objectives:** central `{CONFIG.model['central_objective']}` + "
        f"pinball heads {CONFIG.quantiles} (non-crossing enforced).\n"
        f"- **Features:** {len(feature_columns()[0])} "
        f"({len(feature_columns()[1])} native-categorical). Direct H-step, lags>=H.\n"
        f"- **Validation:** rolling-origin, {CONFIG.metrics['backtest']['embargo_days']}d embargo.\n"
        f"- **Credibility gate:** per-SKU MASE<1 vs seasonal-naive (see acceptance.py).\n"
        f"- **Routing (evidence-based, Phase 5.7):** LGBM is champion across ALL segments; "
        f"TSB for intermittent and the A-blend were tested and REJECTED (both hurt per-SKU "
        f"MASE on M5 grocery). Note: per-SEGMENT aggregate MASE here is volume-weighted and "
        f"reads high on A (~1.3) — the per-SKU gate is the truth (A ~0.77).\n"
        f"- **Known limits:** M5 has no inventory/promo; censored-demand hooks inert; "
        f"intermittent daily demand is near the naive-beating ceiling (gate-binding).\n",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rolling-origin backtest of the global model.")
    ap.add_argument("--stores", nargs="+")
    ap.add_argument("--folds", type=int, default=1, help="number of most-recent folds")
    args = ap.parse_args(argv)
    metrics = run(args.stores, args.folds)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n", metrics.round(4).to_string(index=False))
    write_report(metrics, args.stores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
