"""Favorita dress rehearsal (out-of-distribution) — exercise the promo + perishable paths M5
could never trigger, on real data. The goal is to FIND where they no-op, not to get a green run.

Five steps:
  1. Contract collision — run the data contract on RAW Favorita sales (fractional unit_sales ->
     BLOCK; negatives -> returns WARN). Proves the contract correctly meets real grocery mess.
  2. Adapter — map Favorita -> canonical, with the deliberate decisions: round weight-item sales
     to whole units (documented), keep returns for the WARN path, carry real on_promo + perishable.
  3. Build features INCLUDING the now-implemented promo family, on a tractable store subset.
  4. Train the real GlobalLGBM and check the PROMO features actually get USED (gain importance).
     Near-zero importance when promos are dense would mean the path is still silently broken.
  5. Confirm the PERISHABLE path FIRES — dispatch_reorder routes perishable items to newsvendor.

Favorita has no prices/costs (like M5 has no lead times) -> newsvendor economics are ASSUMED.

    python -m src.evaluate.favorita_exercise --stores 1 2 3 44 --since 2016-01-01
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.models.global_lgbm import GlobalLGBM
from src.reorder.policy import dispatch_reorder

TRAIN = "data/raw/favorita/train.csv"
ITEMS = "data/raw/favorita/items.csv"
H = CONFIG.horizon

PROMO_FEATS = ["on_promo", "promo_in_next_7d", "promo_in_last_7d"]
NUM_FEATS = ["lag_14", "lag_21", "lag_28", "roll_mean_7", "roll_mean_28",
             "day_of_week", "month", "perishable"] + PROMO_FEATS
CAT_FEATS = ["store_id", "sku_id", "family"]


def step1_contract_collision(stores, since) -> None:
    from src.ingest.validation import validate_sales, gate
    con = duckdb.connect()
    raw = con.execute(f"""
        SELECT CAST(date AS DATE) AS date, CAST(store_nbr AS VARCHAR) AS store_id,
               CAST(item_nbr AS VARCHAR) AS sku_id, unit_sales AS qty
        FROM read_csv_auto('{TRAIN}')
        WHERE store_nbr IN ({','.join(map(str, stores))}) AND date >= DATE '{since}'
        LIMIT 200000
    """).df()
    pm = con.execute(f"SELECT CAST(item_nbr AS VARCHAR) sku_id, 1 pack_size, "
                     f"CAST(perishable AS BOOLEAN) perishable FROM read_csv_auto('{ITEMS}')").df()
    print("\n=== STEP 1: data contract on RAW Favorita (expect fractional BLOCK + returns WARN) ===")
    res = validate_sales(raw, pm)
    summary = gate([res], raise_on_block=False)
    for b in summary["blocks"]:
        print("  [BLOCK]", b)
    for w in summary["warnings"]:
        print("  [warn] ", w)


def step2_3_build_panel(stores, since) -> pd.DataFrame:
    print("\n=== STEP 2-3: adapter + feature build (round weight items, carry promo+perishable) ===")
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    store_in = ",".join(map(str, stores))
    df = con.execute(f"""
        WITH base AS (
            SELECT CAST(date AS DATE) AS date,
                   CAST(store_nbr AS VARCHAR) AS store_id,
                   CAST(item_nbr AS VARCHAR) AS sku_id,
                   greatest(CAST(round(unit_sales) AS INTEGER), 0) AS units,   -- weight items rounded
                   CASE WHEN lower(CAST(onpromotion AS VARCHAR))='true' THEN 1 ELSE 0 END AS on_promo
            FROM read_csv_auto('{TRAIN}')
            WHERE store_nbr IN ({store_in}) AND date >= DATE '{since}'
        ),
        w AS (
            SELECT b.*, it.family, CAST(it.perishable AS INTEGER) AS perishable,
                dayofweek(date) AS day_of_week, month(date) AS month,
                lag(units, {H})    OVER s AS lag_14,
                lag(units, {H+7})  OVER s AS lag_21,
                lag(units, {H+14}) OVER s AS lag_28,
                avg(units) OVER (s ROWS BETWEEN {H+6}  PRECEDING AND {H} PRECEDING) AS roll_mean_7,
                avg(units) OVER (s ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_mean_28,
                max(on_promo) OVER (s ROWS BETWEEN 1 FOLLOWING AND 7 FOLLOWING) AS promo_in_next_7d,
                max(on_promo) OVER (s ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS promo_in_last_7d
            FROM base b
            JOIN read_csv_auto('{ITEMS}') it ON b.sku_id = CAST(it.item_nbr AS VARCHAR)
            WINDOW s AS (PARTITION BY b.store_id, b.sku_id ORDER BY b.date)
        )
        SELECT * FROM w WHERE lag_28 IS NOT NULL
    """).df()
    for c in ("promo_in_next_7d", "promo_in_last_7d"):
        df[c] = df[c].fillna(0).astype(int)
    print(f"  panel: {len(df):,} rows · {df['sku_id'].nunique()} items · "
          f"promo rows {int(df['on_promo'].sum()):,} ({df['on_promo'].mean():.1%}) · "
          f"perishable rows {df['perishable'].mean():.1%}")
    return df


def step4_promo_importance(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STEP 4: do the PROMO features actually get USED? (gain importance) ===")
    cut = df["date"].quantile(0.85)
    tr, va = df[df["date"] <= cut].copy(), df[df["date"] > cut].copy()
    for d in (tr, va):
        d["units"] = d["units"].astype(float)
        d["sample_weight"] = 1.0
    m = GlobalLGBM(quantiles=[0.5, 0.9, 0.95], feats=NUM_FEATS + CAT_FEATS, cats=CAT_FEATS)
    m.params = {**m.params, "n_estimators": 200}
    print(f"  train {len(tr):,} / valid {len(va):,}")
    m.fit(tr, va)
    imp = m.importance("gain")
    imp["rank"] = np.arange(1, len(imp) + 1)
    promo_rows = imp[imp["feature"].isin(PROMO_FEATS)]
    print(imp.to_string(index=False))
    print("\n  PROMO features:")
    print(promo_rows.to_string(index=False))
    used = promo_rows["gain"].sum() > 0
    print(f"  -> promo path {'ENGAGED (features used by the model)' if used else 'NO-OP (not used!)'}")


def step5_perishable_fires(df: pd.DataFrame) -> None:
    print("\n=== STEP 5: does the PERISHABLE path FIRE? (dispatch routing) ===")
    # assumed economics (Favorita has no prices): perishable shelf-life demand from recent mean
    sample = df.sample(min(3000, len(df)), random_state=0)
    fired = {"newsvendor": 0, "sS": 0}
    for r in sample.itertuples():
        q = {0.5: r.roll_mean_7 or 1, 0.9: (r.roll_mean_7 or 1) * 1.5, 0.95: (r.roll_mean_7 or 1) * 2}
        branch, _ = dispatch_reorder(
            perishable=bool(r.perishable), inventory_position=0.0, quantiles=q,
            sell_price=100.0, unit_cost=40.0, shelf_life_demand=(r.roll_mean_7 or 1) * 3,
            s=q[0.95], S=q[0.95] * 1.2)
        fired[branch] += 1
    n_per = int(sample["perishable"].sum())
    print(f"  of {len(sample):,} items ({n_per} perishable): newsvendor fired {fired['newsvendor']}, "
          f"(s,S) {fired['sS']}")
    ok = fired["newsvendor"] == n_per and fired["newsvendor"] > 0
    print(f"  -> perishable path {'FIRES for exactly the perishable items' if ok else 'MISROUTED'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Favorita OOD exercise: promo + perishable paths.")
    ap.add_argument("--stores", nargs="+", type=int, default=[1, 2, 3, 44])
    ap.add_argument("--since", default="2016-01-01")
    args = ap.parse_args(argv)
    step1_contract_collision(args.stores, args.since)
    panel = step2_3_build_panel(args.stores, args.since)
    step4_promo_importance(panel)
    step5_perishable_fires(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
