"""Per-SKU MASE acceptance gate (Phase 8.4) — the authoritative go-live check.

The gate is per-SKU, not aggregate: MASE < 1 on >= 80% of A/B SKUs. Segment-aggregate MASE
(volume-weighted, single global scale) can mislead — it made A-items look like they lost to
naive when, per-SKU, they are the strongest segment. So we compute the *distribution* of
per-SKU MASE (each SKU vs its OWN in-sample seasonal-naive scale) across all backtest folds.

Evaluates three prediction columns from backtest_predictions.parquet:
  - lgbm           : global LightGBM only (pred_central)
  - routed         : Phase 5.7 routing — LGBM for smooth/erratic/A, TSB for intermittent/lumpy
  - seasonal_naive : the credibility baseline

`routed` is reconstructed here (TSB computed on the fly) so the gate can be re-checked
without retraining LightGBM.

Usage:
    python -m src.evaluate.acceptance --stores CA_1
    python -m src.evaluate.acceptance --stores CA_1 --blend-sweep
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.models.intermittent import forecast_intermittent
from src.models.router import apply_routing, blend

PRED = (CONFIG.data_dir / "features" / "backtest_predictions.parquet").as_posix()
PANEL = (CONFIG.data_dir / "features" / "panel.parquet").as_posix()
GATE_SHARE = 0.80
EMBARGO = CONFIG.metrics["backtest"]["embargo_days"]


def _per_sku_scale(first_valid_date, stores) -> pd.DataFrame:
    sf = ""
    if stores:
        ids = ", ".join(f"'{s}'" for s in stores)
        sf = f"AND store_id IN ({ids})"
    q = f"""
        WITH s AS (
            SELECT store_id, sku_id, units,
                   lag(units, 7) OVER (PARTITION BY store_id, sku_id ORDER BY date) AS l7
            FROM read_parquet('{PANEL}/**/*.parquet')
            WHERE date < DATE '{first_valid_date}' {sf}
        )
        SELECT store_id, sku_id, avg(abs(units - l7)) AS scale
        FROM s WHERE l7 IS NOT NULL GROUP BY 1, 2
    """
    return duckdb.connect().execute(q).df()


def _per_sku_mase(preds, pred_col, scale, by="abc") -> pd.DataFrame:
    mae = (preds.assign(ae=(preds["units"] - preds[pred_col]).abs())
                .groupby(["store_id", "sku_id", by], observed=True)["ae"].mean()
                .reset_index(name="mae"))
    m = mae.merge(scale, on=["store_id", "sku_id"], how="left")
    m = m[m["scale"] > 0].copy()
    m["mase"] = m["mae"] / m["scale"]
    return m


def _gate_row(mase_df, label, segs=("A", "B", "C")) -> dict:
    out = {"variant": label}
    for seg in segs:
        s = mase_df[mase_df.iloc[:, 2] == seg]["mase"]
        out[f"{seg}_med"] = round(float(s.median()), 3) if len(s) else np.nan
        out[f"{seg}_share<1"] = round(float((s < 1).mean()), 3) if len(s) else np.nan
    ab = mase_df[mase_df.iloc[:, 2].isin(["A", "B"])]["mase"]
    out["AB_share<1"] = round(float((ab < 1).mean()), 3) if len(ab) else np.nan
    out["GATE_pass"] = bool(out["AB_share<1"] >= GATE_SHARE)
    return out


def _add_routing(preds: pd.DataFrame, stores) -> pd.DataFrame:
    """Reconstruct pred_final (routed) per fold, FORCING TSB substitution (compare mode).

    apply_routing is config-gated (intermittent_model defaults to lgbm), so here we override
    it to 'tsb' to make the comparison meaningful, and report how many rows actually changed.
    """
    parts = []
    for fold, g in preds.groupby("fold"):
        valid_start, valid_end = g["date"].min(), g["date"].max()
        train_end = (valid_start - pd.Timedelta(days=EMBARGO + 1)).date()
        tsb = forecast_intermittent(stores, train_end, valid_end.date())
        # normalize keys to str so a category/object dtype mismatch can't silently drop rows
        for k in ("store_id", "sku_id"):
            g[k] = g[k].astype("string")
            tsb[k] = tsb[k].astype("string")
        g = g.merge(tsb, on=["store_id", "sku_id"], how="left")
        matched = g["pred_tsb"].notna().sum()
        g["pred_final"] = apply_routing(g, intermittent_model="tsb")  # FORCE TSB
        changed = (g["pred_final"].round(6) != g["pred_central"].round(6)).sum()
        print(f"  fold {fold}: TSB matched {matched:,} rows; "
              f"routed differs from lgbm on {changed:,} rows")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def run(stores: list[str] | None, blend_sweep: bool = False, compare_tsb: bool = False) -> None:
    preds = duckdb.connect().execute(f"SELECT * FROM read_parquet('{PRED}')").df()
    first_valid = preds["date"].min()
    print(f"predictions: {len(preds):,} rows · folds={sorted(preds['fold'].unique())} · "
          f"first_valid={first_valid}")
    scale = _per_sku_scale(first_valid, stores)

    if "pred_final" not in preds.columns:
        if compare_tsb:
            print("reconstructing routed predictions WITH TSB (comparison) ...")
            preds = _add_routing(preds, stores)
            routed_label = "routed (LGBM+TSB)"
        else:
            preds["pred_final"] = apply_routing(preds)  # config routing (default == lgbm)
            routed_label = "routed (config)"

    rows = [
        _gate_row(_per_sku_mase(preds, "pred_central", scale), "lgbm"),
        _gate_row(_per_sku_mase(preds, "pred_final", scale), routed_label),
        _gate_row(_per_sku_mase(preds, "seasonal_naive", scale), "seasonal_naive"),
    ]
    gate = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Per-SKU MASE gate (share of SKUs with MASE<1) ===")
    print(gate.to_string(index=False))

    # per-intermittency view: where routing actually moves the needle
    print("\n=== Per-intermittency median MASE & share<1 ===")
    for col, lbl in [("pred_central", "lgbm"), ("pred_final", "routed")]:
        m = _per_sku_mase(preds, col, scale, by="intermittency")
        agg = (m.groupby("intermittency")["mase"]
                 .agg(n="size", median="median", share_lt1=lambda s: (s < 1).mean()).round(3))
        print(f"\n[{lbl}]\n{agg.to_string()}")

    if blend_sweep:
        print("\n=== A-item blend sweep (evidence the blend hurts) ===")
        sweep = []
        for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p = preds.copy(); a = p["abc"] == "A"
            p["pb"] = p["pred_central"]
            p.loc[a, "pb"] = blend(p.loc[a, "pred_central"], p.loc[a, "seasonal_naive"], w)
            am = _per_sku_mase(p, "pb", scale); am = am[am.iloc[:, 2] == "A"]["mase"]
            sweep.append({"w_lgbm": w, "A_median": round(float(am.median()), 3),
                          "A_share<1": round(float((am < 1).mean()), 3)})
        print(pd.DataFrame(sweep).to_string(index=False))

    print(f"\nGo-live gate: AB_share<1 >= {GATE_SHARE}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-SKU MASE acceptance gate (LGBM vs routed).")
    ap.add_argument("--stores", nargs="+")
    ap.add_argument("--blend-sweep", action="store_true", help="show A-item blend sweep")
    ap.add_argument("--compare-tsb", action="store_true",
                    help="reconstruct routed preds with TSB to compare (slower)")
    args = ap.parse_args(argv)
    run(args.stores, args.blend_sweep, args.compare_tsb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
