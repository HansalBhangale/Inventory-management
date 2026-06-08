"""Inventory simulation (Phase 8.1) — replay forecasts -> reorder engine -> simulated stock.

A forecast that produces bad orders is a failure, so we simulate the full loop and measure the
REAL objective: service level PER UNIT OF INVENTORY (a frontier), not absolute service (you can
buy any service level with enough stock).

Guards (this sim is built to be able to fail):
  1. Baseline is the protagonist. The SAME engine/costs/targets run on seasonal-naive and
     moving-average forecasts. We report service-vs-inventory frontiers, not a single number.
  2. Intermittency sliced separately, value-weighted. Aggregate fill rate is dominated by
     smooth/erratic volume; the intermittent verdict only shows in its own slice.
  3. Lead time is ASSUMED (M5 has none). Every result is stamped with its regime, and we sweep
     regimes because the value of a good upper-tail forecast grows with lead time.

Planning (continuous-review (s,S), daily review):
  P = round(lead_mean) + review.  d_bar = mean daily forecast.
  LGBM (Route B, uses its quantiles):  s = Σ q-forecast over P  +  z(q)·d_bar·σ_L
  baseline (Route A, point + normal):  s = d_bar·P            +  z(q)·sqrt(P·σ_e² + d_bar²·σ_L²)
  S = s + d_bar·order_cycle.   Realized lead time per order is sampled from the regime.

Usage:
    python -m src.evaluate.inventory_sim                 # full grid on stratified sample
    python -m src.evaluate.inventory_sim --sample 400    # faster
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import CONFIG
from src.reorder.leadtime import regimes
from src.reorder.policy import round_order

PRED = (CONFIG.data_dir / "features" / "backtest_predictions.parquet").as_posix()
PANEL = (CONFIG.data_dir / "features" / "panel.parquet").as_posix()
SIM = CONFIG.policy["simulation"]
METHODS = {  # method -> (quantile col template, point col)
    "lgbm": ("pred_q{q}", "pred_q50"),
    "seasonal_naive": (None, "seasonal_naive"),
    "moving_average": (None, "moving_average"),
}


def _load_series(sample_per_class: int) -> tuple[pd.DataFrame, dict]:
    """Load stitched contiguous prediction window + per-series avg price, stratified sample."""
    # Load every quantile head present (incl. pred_q99 once trained) so the sim uses the real
    # quantiles, not a subset.
    avail = duckdb.connect().execute(f"SELECT * FROM read_parquet('{PRED}') LIMIT 0").df().columns
    qcols_sql = ", ".join(c for c in avail if c.startswith("pred_q"))
    df = duckdb.connect().execute(f"""
        SELECT store_id, sku_id, date, intermittency, abc, units,
               {qcols_sql}, seasonal_naive, moving_average
        FROM read_parquet('{PRED}') ORDER BY store_id, sku_id, date
    """).df()
    dates = sorted(df["date"].unique())
    print(f"sim window: {dates[0].date()}..{dates[-1].date()} ({len(dates)} contiguous days)")

    keys = df[["store_id", "sku_id", "intermittency"]].drop_duplicates()
    picked = (keys.groupby("intermittency", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), sample_per_class), random_state=0)))
    df = df.merge(picked[["store_id", "sku_id"]], on=["store_id", "sku_id"])

    price = duckdb.connect().execute(f"""
        SELECT store_id, sku_id, avg(unit_price) AS price
        FROM read_parquet('{PANEL}/**/*.parquet')
        WHERE unit_price IS NOT NULL GROUP BY 1,2
    """).df()
    pmap = {(r.store_id, r.sku_id): r.price for r in price.itertuples()}
    print(f"simulating {len(picked):,} -> sampled {df[['store_id','sku_id']].drop_duplicates().shape[0]:,} series")
    return df, pmap


def plan_levels(point, daily_buffer, sigma_e, z, sigma_L, P, order_cycle):
    """Vectorized (s,S) per day. Two buffer modes, both with correct √P scaling:

      empirical (LGBM, uses the TRAINED quantiles): the per-day demand buffer is
        (q_forecast − q50); summed over P and divided by √P it equals the independent-day
        buffer z·σ̄·√P, but reads the model's actual (possibly fat-tailed) quantile — so the
        trained q0.99 head feeds the reorder point directly.
      normal (baselines, no quantiles): z(q)·σ_e·√P from the point-forecast error std.

    Supply-side risk z·d_bar·σ_L is added for both (doc 7.3).
    """
    T = len(point)
    pad = int(P + order_cycle + 2)
    csum_m = np.concatenate([[0.0], np.cumsum(np.concatenate([point, np.full(pad, point[-1])]))])
    d_bar = max(float(point.mean()), 1e-6)
    supply = z * d_bar * sigma_L

    if daily_buffer is not None:   # empirical
        csum_b = np.concatenate([[0.0], np.cumsum(np.concatenate([daily_buffer,
                                                                  np.full(pad, daily_buffer[-1])]))])
        bufP = (csum_b[np.arange(T) + P] - csum_b[:T]) / np.sqrt(P)
        bufPC = (csum_b[np.arange(T) + P + order_cycle] - csum_b[:T]) / np.sqrt(P + order_cycle)
    else:                          # normal
        bufP = z * sigma_e * np.sqrt(P)
        bufPC = z * sigma_e * np.sqrt(P + order_cycle)

    idx = np.arange(T)
    meanP = csum_m[idx + P] - csum_m[idx]
    meanPC = csum_m[idx + P + order_cycle] - csum_m[idx]
    s_arr = np.maximum(0.0, meanP + bufP + supply)
    S_arr = np.maximum(s_arr, meanPC + bufPC + supply)
    return s_arr, S_arr


def simulate_inventory(y, s_arr, S_arr, lt, moq, pack, warmup, rng) -> dict:
    """Day-by-day (s,S) inventory dynamics given precomputed daily s/S levels (lost-sales)."""
    T = len(y)
    on_hand = float(S_arr[0])
    on_order = 0.0
    pipeline = np.zeros(T + int(lt.mean + 3 * lt.std) + 5)
    served = demanded = inv_sum = lost = 0.0
    stockout_days = n_eff = 0
    for t in range(T):
        arr = pipeline[t]
        on_hand += arr
        on_order -= arr
        d = y[t]
        srv = min(on_hand, d)
        on_hand -= srv
        if t >= warmup:
            served += srv; demanded += d; inv_sum += on_hand; lost += (d - srv)
            if d > 0 and srv < d - 1e-9:
                stockout_days += 1
            n_eff += 1
        IP = on_hand + on_order
        if IP <= s_arr[t]:
            qty = round_order(S_arr[t] - IP, moq, pack)
            if qty > 0:
                arrday = t + lt.sample(rng)
                if arrday < len(pipeline):
                    pipeline[arrday] += qty
                    on_order += qty
    n_eff = max(n_eff, 1)
    return {"served": served, "demanded": demanded, "avg_inv": inv_sum / n_eff,
            "lost_units": lost, "service_days": 1 - stockout_days / n_eff,
            "avg_daily_demand": demanded / n_eff}


def run(sample_per_class: int) -> pd.DataFrame:
    df, pmap = _load_series(sample_per_class)
    regs = regimes()
    qs = SIM["service_quantiles"]
    R = SIM["review_period_days"]
    C = SIM["order_cycle_days"]
    warmup = SIM["warmup_days"]
    moq = SIM["moq_default"]
    pack = SIM["pack_size_default"]
    rng = np.random.default_rng(42)

    grouped = list(df.groupby(["store_id", "sku_id"], observed=True))
    rows = []
    for (store, sku), g in grouped:
        g = g.sort_values("date")
        y = g["units"].to_numpy("float64")
        inter = g["intermittency"].iloc[0]
        abc = g["abc"].iloc[0]
        price = float(pmap.get((store, sku), 1.0) or 1.0)
        q50 = g["pred_q50"].to_numpy("float64")
        qcols = {qq: g[f"pred_q{int(qq*100)}"].to_numpy("float64")
                 for qq in qs if f"pred_q{int(qq*100)}" in g.columns}
        points = {"lgbm": q50,
                  "seasonal_naive": g["seasonal_naive"].fillna(0).to_numpy("float64"),
                  "moving_average": g["moving_average"].fillna(0).to_numpy("float64")}
        for method, (qtmpl, pointcol) in METHODS.items():
            point = points["lgbm"] if method == "lgbm" else points[method]
            sigma_e = float(np.std(y - point))   # for baseline normal buffer
            for q in qs:
                z = max(0.0, float(norm.ppf(q)))
                if method == "lgbm":
                    if q not in qcols:           # quantile head not trained -> skip
                        continue
                    daily_buffer = np.maximum(qcols[q] - q50, 0.0)
                else:
                    daily_buffer = None
                for rname, lt in regs.items():
                    P = int(round(lt.mean)) + R
                    s_arr, S_arr = plan_levels(point, daily_buffer, sigma_e, z, lt.std, P, C)
                    m = simulate_inventory(y, s_arr, S_arr, lt, moq, pack, warmup, rng)
                    rows.append({"store_id": store, "sku_id": sku, "intermittency": inter,
                                 "abc": abc, "price": price, "method": method, "q": q,
                                 "regime": rname, **m})
    return pd.DataFrame(rows)


def _agg(d: pd.DataFrame) -> pd.Series:
    served, demanded = d["served"].sum(), d["demanded"].sum()
    inv_val = (d["avg_inv"] * d["price"]).sum()
    cogs_day = (d["avg_daily_demand"] * d["price"]).sum()
    return pd.Series({
        "fill_rate": served / max(demanded, 1e-9),
        "service_days": d["service_days"].mean(),
        "avg_inv_units": d["avg_inv"].mean(),
        "DOH": inv_val / max(cogs_day, 1e-9),               # value-weighted days-on-hand
        "lost_value": (d["lost_units"] * d["price"]).sum(),
        "n": d[["store_id", "sku_id"]].drop_duplicates().shape[0],
    })


def write_report(res: pd.DataFrame) -> None:
    out = Path(CONFIG.root) / "docs" / "PHASE7_inventory_sim.md"
    L = ["# Phase 7 — Inventory Simulation (service-vs-inventory frontier)\n",
         "> **M5 has no real lead times/inventory/costs — these are ASSUMPTIONS.** This validates "
         "the reorder MACHINERY and the frontier, NOT quotable service levels. Every row is "
         "stamped with its lead-time regime.\n"]

    # 1) Frontier per method x service-quantile at the BASE regime (the headline)
    base = res[res["regime"] == "base"]
    fr = base.groupby(["method", "q"]).apply(_agg).round(3).reset_index()
    L += ["## Frontier @ base regime (lead mean=3d): service vs inventory\n",
          "Read across q for each method: higher fill should cost more DOH. Compare methods at "
          "matched fill.\n", fr.to_markdown(index=False)]

    # 2) Matched-service comparison: inventory to reach >=95% fill, per method (base regime)
    L += ["\n## The number that matters: DOH to reach a fill target (base regime)\n"]
    pts = []
    for method, gm in base.groupby("method"):
        for q, gq in gm.groupby("q"):
            a = _agg(gq)
            pts.append({"method": method, "q": q, "fill_rate": round(a.fill_rate, 3),
                        "DOH": round(a.DOH, 2)})
    L += [pd.DataFrame(pts).to_markdown(index=False)]

    # 3) Intermittency slice (value-weighted) at base regime, q=0.95 — the real verdict
    L += ["\n## Intermittent verdict: per-class fill & DOH @ base, q=0.95 (value-weighted)\n"]
    hi = base[base["q"] == 0.95]
    islice = hi.groupby(["intermittency", "method"]).apply(_agg).round(3).reset_index()
    L += [islice.to_markdown(index=False)]

    # 4) Regime sweep for LGBM @ q=0.95 — value of upper-tail forecast grows with lead time
    L += ["\n## Lead-time regime sweep (LGBM, q=0.95)\n"]
    sweep = res[(res["method"] == "lgbm") & (res["q"] == 0.95)]
    sw = sweep.groupby("regime").apply(_agg).round(3).reset_index()
    L += [sw.to_markdown(index=False)]

    out.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inventory simulation (Phase 7/8).")
    ap.add_argument("--sample", type=int, default=SIM["sample_per_intermittency"],
                    help="series per intermittency class")
    args = ap.parse_args(argv)
    res = run(args.sample)
    res.to_parquet(CONFIG.data_dir / "features" / "sim_results.parquet")
    pd.set_option("display.width", 220, "display.max_columns", 40)
    write_report(res)
    print("\nFrontier @ base regime:")
    base = res[res["regime"] == "base"]
    print(base.groupby(["method", "q"]).apply(_agg).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
