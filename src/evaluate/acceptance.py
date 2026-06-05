"""Per-SKU MASE acceptance gate (Phase 8.4) — the authoritative go-live check.

The gate is per-SKU, not aggregate: MASE < 1 on >= 80% of A/B SKUs, measured across all
backtest folds. Segment-aggregate MASE (volume-weighted, single global scale) misleads — it
made A-items look like they lost to naive when, per-SKU, they are the strongest segment.

METHODOLOGY (deliberate, to avoid re-introducing volume bias):
  1. Per FOLD, per SKU: MASE = MAE(sku, fold) / in-sample seasonal-naive(7) scale, where the
     scale is computed PER FOLD over that SKU's history strictly before the fold's validation
     start (correct rolling-origin in-sample denominator, no leakage).
  2. Aggregate ACROSS folds PER SKU: take the median MASE over the folds a SKU appears in.
     (We do NOT pool all fold rows into one MASE — that volume-weights toward high-demand
     folds/SKUs, the exact error that misread A-items originally.)
  3. share<1 = fraction of SKUs whose cross-fold median MASE < 1.
We also report per-FOLD AB_share<1 with spread, so a passing mean over a volatile spread is
visible. Intermittency is the primary lens (with n per bucket); ABC is the gate's unit.

With a single fold, (2) is a no-op, so single-fold/single-store numbers stay directly
comparable to the CA_1 fold-5 reference.

Usage:
    python -m src.evaluate.acceptance                      # all stores, config routing
    python -m src.evaluate.acceptance --stores CA_1        # single-store reference
    python -m src.evaluate.acceptance --stores CA_1 --compare-tsb --blend-sweep
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
KEYS = ["store_id", "sku_id"]
GRP = ["store_id", "sku_id", "abc", "intermittency"]


def _per_sku_scale(valid_start, stores) -> pd.DataFrame:
    """In-sample seasonal-naive(7) MAE per (store,sku) over history before valid_start."""
    sf = ""
    if stores:
        ids = ", ".join(f"'{s}'" for s in stores)
        sf = f"AND store_id IN ({ids})"
    q = f"""
        WITH s AS (
            SELECT store_id, sku_id, units,
                   lag(units, 7) OVER (PARTITION BY store_id, sku_id ORDER BY date) AS l7
            FROM read_parquet('{PANEL}/**/*.parquet')
            WHERE date < DATE '{pd.Timestamp(valid_start).date()}' {sf}
        )
        SELECT store_id, sku_id, avg(abs(units - l7)) AS scale
        FROM s WHERE l7 IS NOT NULL GROUP BY 1, 2
    """
    return duckdb.connect().execute(q).df()


def _mase_long(preds: pd.DataFrame, pred_col: str, stores, scale_cache: dict) -> pd.DataFrame:
    """Per (sku, fold) MASE for one prediction column. Returns long: GRP + fold + mase."""
    parts = []
    for fold, g in preds.groupby("fold"):
        vs = g["date"].min()
        if fold not in scale_cache:
            scale_cache[fold] = _per_sku_scale(vs, stores)
        scale = scale_cache[fold]
        mae = (g.assign(ae=(g["units"] - g[pred_col]).abs())
                 .groupby(GRP, observed=True)["ae"].mean().reset_index(name="mae"))
        m = mae.merge(scale, on=KEYS, how="left")
        m = m[m["scale"] > 0].copy()
        m["mase"] = m["mae"] / m["scale"]
        m["fold"] = fold
        parts.append(m[GRP + ["fold", "mase"]])
    return pd.concat(parts, ignore_index=True)


def _cross_fold_per_sku(long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per SKU across folds (median MASE) — the gate unit."""
    return long.groupby(GRP, observed=True)["mase"].median().reset_index()


def _gate_row(per_sku: pd.DataFrame, label: str) -> dict:
    out = {"variant": label}
    for seg in ["A", "B", "C"]:
        s = per_sku[per_sku["abc"] == seg]["mase"]
        out[f"{seg}_med"] = round(float(s.median()), 3) if len(s) else np.nan
        out[f"{seg}_<1"] = round(float((s < 1).mean()), 3) if len(s) else np.nan
    ab = per_sku[per_sku["abc"].isin(["A", "B"])]["mase"]
    out["AB_<1"] = round(float((ab < 1).mean()), 3) if len(ab) else np.nan
    out["GATE"] = bool(out["AB_<1"] >= GATE_SHARE)
    return out


def _per_fold_spread(long: pd.DataFrame) -> pd.DataFrame:
    """AB_share<1 computed WITHIN each fold (per-SKU), to expose fold variance."""
    rows = []
    for fold, g in long.groupby("fold"):
        ab = g[g["abc"].isin(["A", "B"])]["mase"]
        rows.append({"fold": fold, "AB_<1": round(float((ab < 1).mean()), 3), "n_ab": len(ab)})
    df = pd.DataFrame(rows)
    return df


def _intermittency_view(per_sku: pd.DataFrame) -> pd.DataFrame:
    return (per_sku.groupby("intermittency", observed=True)["mase"]
            .agg(n="size", median="median", share_lt1=lambda s: (s < 1).mean()).round(3))


def _add_routing_tsb(preds: pd.DataFrame, stores) -> pd.DataFrame:
    """Force TSB substitution (compare mode) and report rows changed."""
    parts = []
    for fold, g in preds.groupby("fold"):
        vs, ve = g["date"].min(), g["date"].max()
        train_end = (vs - pd.Timedelta(days=EMBARGO + 1)).date()
        tsb = forecast_intermittent(stores, train_end, ve.date())
        for k in KEYS:
            g[k] = g[k].astype("string")
            tsb[k] = tsb[k].astype("string")
        g = g.merge(tsb, on=KEYS, how="left")
        g["pred_final"] = apply_routing(g, intermittent_model="tsb")
        changed = (g["pred_final"].round(6) != g["pred_central"].round(6)).sum()
        print(f"  fold {fold}: TSB matched {int(g['pred_tsb'].notna().sum()):,} rows; "
              f"routed differs from lgbm on {int(changed):,} rows")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def run(stores=None, blend_sweep=False, compare_tsb=False, agg_across_folds="median") -> None:
    preds = duckdb.connect().execute(f"SELECT * FROM read_parquet('{PRED}')").df()
    folds = sorted(int(x) for x in preds["fold"].unique())
    print(f"predictions: {len(preds):,} rows · folds={folds} · "
          f"skus={preds['sku_id'].nunique():,} · stores={preds['store_id'].nunique()}")

    if "pred_final" not in preds.columns:
        if compare_tsb:
            print("reconstructing routed predictions WITH TSB (forced) ...")
            preds = _add_routing_tsb(preds, stores)
            routed_label = "routed (LGBM+TSB)"
        else:
            preds["pred_final"] = apply_routing(preds)
            routed_label = "routed (config)"

    scale_cache: dict = {}
    variants = [("pred_central", "lgbm"), ("pred_final", routed_label),
                ("seasonal_naive", "seasonal_naive")]
    longs = {lbl: _mase_long(preds, col, stores, scale_cache) for col, lbl in variants}

    # 1) gate table: cross-fold per-SKU
    gate = pd.DataFrame([_gate_row(_cross_fold_per_sku(longs[lbl]), lbl) for _, lbl in variants])
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n=== Per-SKU MASE gate (cross-fold {agg_across_folds}; share with MASE<1) ===")
    print(gate.to_string(index=False))

    # 2) fold variance for the champion
    print("\n=== Fold variance (AB share<1 within each fold) — lgbm ===")
    spread = _per_fold_spread(longs["lgbm"])
    print(spread.to_string(index=False))
    if len(spread) > 1:
        v = spread["AB_<1"]
        print(f"  across folds: mean={v.mean():.3f} std={v.std():.3f} "
              f"min={v.min():.3f} max={v.max():.3f}")

    # 3) intermittency lens (primary), with n
    print("\n=== Per-intermittency (cross-fold per-SKU) ===")
    for col, lbl in [("pred_central", "lgbm"), ("pred_final", routed_label)]:
        print(f"\n[{lbl}]\n{_intermittency_view(_cross_fold_per_sku(longs[lbl])).to_string()}")

    if blend_sweep:
        print("\n=== A-item blend sweep (evidence the blend hurts) ===")
        sweep = []
        for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p = preds.copy(); a = p["abc"] == "A"
            p["pb"] = p["pred_central"]
            p.loc[a, "pb"] = blend(p.loc[a, "pred_central"], p.loc[a, "seasonal_naive"], w)
            ps = _cross_fold_per_sku(_mase_long(p, "pb", stores, scale_cache))
            am = ps[ps["abc"] == "A"]["mase"]
            sweep.append({"w_lgbm": w, "A_median": round(float(am.median()), 3),
                          "A_share<1": round(float((am < 1).mean()), 3)})
        print(pd.DataFrame(sweep).to_string(index=False))

    print(f"\nGo-live gate: AB share<1 >= {GATE_SHARE}  ->  "
          f"{'PASS' if bool(gate.iloc[0]['GATE']) else 'FAIL'} (lgbm)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-SKU MASE acceptance gate (cross-fold).")
    ap.add_argument("--stores", nargs="+")
    ap.add_argument("--blend-sweep", action="store_true", help="show A-item blend sweep")
    ap.add_argument("--compare-tsb", action="store_true",
                    help="force TSB substitution to compare (slower)")
    args = ap.parse_args(argv)
    run(args.stores, args.blend_sweep, args.compare_tsb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
