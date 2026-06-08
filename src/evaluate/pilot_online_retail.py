"""End-to-end PILOT REHEARSAL on real data (UCI Online Retail) — the runbook loop for real.

ingest -> data contract (reject/fix) -> features -> train -> score -> reorder -> SHADOW review.
Real transactions with real prices; NOT kirana and still no real lead times/inventory (assumed),
so this validates the loop + recommendation sanity, NOT a real service/inventory number.

Scoped to one "store" (United Kingdom = 91% of rows) to mirror a single-shop pilot.

    python -m src.evaluate.pilot_online_retail --store "United Kingdom"
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import duckdb
import pandas as pd

from src.config import CONFIG
from src.ingest.load_online_retail import product_master, staged_sales
from src.ingest.validation import gate, validate_sales
from src.models.global_lgbm import GlobalLGBM
from src.reorder.policy import recommend
from src.serve.shadow import run_shadow

H = CONFIG.horizon
NUM = ["lag_14", "lag_21", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_28",
       "day_of_week", "week_of_year", "month", "unit_price", "relative_price"]
CAT = ["sku_id"]
LEAD = int(CONFIG.policy["lead_time"]["default_days"])          # assumed, no PO/GRN in this data


def step_contract(store: str) -> pd.DataFrame:
    print("=== STEP 1: data contract (runbook reject/fix already applied in the adapter) ===")
    pm, clean = product_master(True), staged_sales()
    clean = clean[clean["store_id"] == store].copy()
    r = gate([validate_sales(clean, pm)], raise_on_block=False)
    print(f"  {store}: {len(clean):,} daily rows | {clean.sku_id.nunique()} SKUs | "
          f"passed={r['passed']} | warnings={len(r['warnings'])}")
    return clean


def step_features(sales: pd.DataFrame) -> pd.DataFrame:
    print("=== STEP 2: continuous panel + leak-safe features (lags>=H, rolling end t-H) ===")
    con = duckdb.connect(); con.register("sales", sales)
    df = con.execute(f"""
        WITH span AS (SELECT sku_id, min(date) f, max(date) l FROM sales GROUP BY 1),
        grid AS (SELECT sku_id, CAST(unnest(generate_series(f, l, INTERVAL 1 DAY)) AS DATE) date
                 FROM span),
        panel AS (
            SELECT g.sku_id, g.date, greatest(coalesce(s.qty,0),0) AS units,
                   s.unit_price
            FROM grid g LEFT JOIN sales s USING (sku_id, date)
        ),
        priced AS (  -- forward-fill price within sku; category(=sku here) median for relative
            SELECT p.*, median(unit_price) OVER (PARTITION BY date) AS day_med_price FROM panel p
        )
        SELECT sku_id, date, units,
            coalesce(unit_price, 0) AS unit_price,
            coalesce(unit_price / nullif(day_med_price,0), 1) AS relative_price,
            dayofweek(date) AS day_of_week, week(date) AS week_of_year, month(date) AS month,
            lag(units,{H})    OVER w AS lag_14,
            lag(units,{H+7})  OVER w AS lag_21,
            lag(units,{H+14}) OVER w AS lag_28,
            avg(units) OVER (w ROWS BETWEEN {H+6}  PRECEDING AND {H} PRECEDING) AS roll_mean_7,
            avg(units) OVER (w ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_mean_28,
            stddev_samp(units) OVER (w ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_std_28
        FROM priced WINDOW w AS (PARTITION BY sku_id ORDER BY date)
    """).df()
    df = df[df["lag_28"].notna()].copy()
    df["sample_weight"] = 1.0
    print(f"  panel: {len(df):,} rows | {df.sku_id.nunique()} SKUs | {df.date.min().date()}..{df.date.max().date()}")
    return df


def step_rolling_origin(df: pd.DataFrame, n_folds: int = 4):
    """ROLLING-ORIGIN accuracy (multi-fold, like M5) — a single 14-day window can flatter or bury
    a result by luck of which fortnight you land on; we never trusted single-origin on M5 and
    won't here. Walk-forward folds with an H-day embargo; report per-fold metrics + variance.
    Returns (metrics_df, scored_predictions_of_most_recent_fold) for the shadow demo."""
    from src.evaluate import forecast_metrics as M
    mx = df["date"].max()
    rows, last_scored = [], None
    for i in range(n_folds):
        val_end = mx - timedelta(days=H * i)
        val_start = val_end - timedelta(days=H - 1)
        train_end = val_start - timedelta(days=H + 1)          # embargo = H
        tr = df[df["date"] <= train_end].copy()
        va = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        if len(tr) < 5000 or va.empty:
            break
        for d in (tr, va):
            d["units"] = d["units"].astype(float)
        m = GlobalLGBM(quantiles=[0.5, 0.9, 0.95], feats=NUM + CAT, cats=CAT)
        m.params = {**m.params, "n_estimators": 200}
        m.fit(tr, va)
        preds = m.predict(va)
        y = va["units"].to_numpy()
        scale = M.seasonal_naive_scale(tr["units"].to_numpy(), m=7)
        naive = va["lag_14"].fillna(0).clip(lower=0).to_numpy()
        rows.append({"fold": n_folds - i, "n": len(va),
                     "wape": round(M.wape(y, preds["pred_central"]), 3),
                     "naive_wape": round(M.wape(y, naive), 3),
                     "mase": round(M.mase(y, preds["pred_central"], scale), 3),
                     "bias": round(M.bias(y, preds["pred_central"]), 3),
                     "cov_p90": round(M.coverage(y, preds["pred_q90"]), 3),
                     "cov_p95": round(M.coverage(y, preds["pred_q95"]), 3)})
        if i == 0:
            last_scored = pd.concat([va[["sku_id", "date", "units", "roll_mean_7"]].reset_index(drop=True),
                                     preds.reset_index(drop=True)], axis=1)
    md = pd.DataFrame(rows).sort_values("fold")
    print("=== STEP 3-4: ROLLING-ORIGIN accuracy (multi-fold, vs seasonal-naive) ===", flush=True)
    for r in md.itertuples(index=False):
        print(f"  fold {r.fold}: n={r.n:,} wape={r.wape} (naive {r.naive_wape}) "
              f"mase={r.mase} bias={r.bias:+} cov_p90={r.cov_p90} cov_p95={r.cov_p95}", flush=True)
    s = md[["wape", "naive_wape", "mase", "cov_p90", "cov_p95"]].agg(["mean", "std"]).round(3)
    print("  across folds:\n" + s.to_string(), flush=True)
    return md, last_scored


def step_reorder_shadow(scored: pd.DataFrame) -> None:
    print("=== STEP 5-6: (s,S) recommendations + SHADOW review (assumed lead time) ===")
    P = LEAD + 1
    # demand over the protection period per SKU = sum of the next P daily quantile forecasts
    g = (scored.sort_values("date").groupby("sku_id")
         .head(P).groupby("sku_id")
         .agg(exp_P=("pred_q50", "sum"), s_lvl=("pred_q95", "sum"),
              roll7=("roll_mean_7", "first")).reset_index())
    g["order_up_to"] = g["s_lvl"] + g["exp_P"] / max(P, 1)        # S = s + ~one day mean
    g["inventory_position"] = g["roll7"].fillna(0)               # ASSUMED: ~recent daily level on hand
    recs = []
    for r in g.itertuples():
        rec = recommend(r.inventory_position, r.s_lvl, r.order_up_to, moq=1, pack_size=1,
                        service_q=0.95, protection_days=P)
        recs.append({"sku_id": r.sku_id, "should_order": rec.should_order,
                     "order_qty": rec.order_qty, "order_up_to": rec.order_up_to,
                     "inventory_position": r.inventory_position,
                     "expected_demand_protection": max(r.exp_P, 0.1), "moq": 1, "pack_size": 1})
    recs = pd.DataFrame(recs)
    rep = run_shadow(recs)
    print(f"  {len(recs):,} SKU recommendations | would-order {int(recs.should_order.sum()):,}")
    print(f"  SHADOW reject rate (week-one signal): {rep.reject_rate:.1%}")
    if rep.flag_counts:
        for f, c in sorted(rep.flag_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {c:5d}  {f}")
    else:
        print("    no recommendations flagged as obviously wrong.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Online Retail end-to-end pilot rehearsal.")
    ap.add_argument("--store", default="United Kingdom")
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args(argv)
    sales = step_contract(args.store)
    panel = step_features(sales)
    _metrics, scored = step_rolling_origin(panel, args.folds)
    step_reorder_shadow(scored)
    print("\nNOTE: real prices, but assumed lead time / no real inventory feed -> validates the "
          "loop + recommendation sanity, not a real service/inventory number. Rehearsal, not pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
