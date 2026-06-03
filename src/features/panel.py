"""Build the clean modeling panel (Phase 3).

Turns staged sales into a continuous, leakage-free (store x sku x date) panel where the
target reflects *demand*, not just sales:

  - continuous grid over each SKU's active window (first..last sale), zeros filled
  - returns split out: qty < 0 -> `returns`; demand target = max(qty, 0)
  - censored-demand hooks: `was_stockout` + `sample_weight` (mask stockout days from loss).
    M5 has no inventory, so was_stockout=0 here, but the mechanism is wired (Phase 3.2).
  - outlier flag: per-SKU winsorization of *unexplained* extreme spikes (kept, not dropped)
  - calendar/external enrichment joined by (date, region)
  - target columns for the two objective options (Phase 3.7)

Output: data/features/panel.parquet (partitioned by store_id).

Usage:
    python -m src.features.panel                 # all stores
    python -m src.features.panel --stores CA_1   # dev subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.config import CONFIG

STAGED = CONFIG.data_dir / "staged"
FEATURES = CONFIG.data_dir / "features"

WINSOR_Q = 0.995  # cap unexplained spikes at this per-SKU quantile (flagged, value kept too)


def _sales_source(stores: list[str] | None) -> str:
    base = (STAGED / "sales_transactions").as_posix()
    if stores:
        return " UNION ALL ".join(
            f"SELECT * FROM read_parquet('{base}/store_id={s}/*.parquet')" for s in stores
        )
    return f"SELECT * FROM read_parquet('{base}/**/*.parquet')"


def build(stores: list[str] | None = None) -> Path:
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "panel.parquet"
    cal = (STAGED / "external_calendar.parquet").as_posix()
    mask_stockouts = CONFIG.model.get("forecast", {}).get("mask_stockouts", True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    con.execute(f"""CREATE TEMP TABLE sales AS
        SELECT store_id, sku_id, CAST(date AS DATE) AS date,
               CAST(qty AS INTEGER) AS qty, unit_price
        FROM ({_sales_source(stores)});""")

    # 1) continuous grid per (store, sku) over active window
    print("building continuous grid ...")
    con.execute("""CREATE TEMP TABLE grid AS
        WITH span AS (
            SELECT store_id, sku_id, min(date) AS f, max(date) AS l FROM sales GROUP BY 1,2
        )
        SELECT store_id, sku_id,
               CAST(unnest(generate_series(f, l, INTERVAL 1 DAY)) AS DATE) AS date
        FROM span;""")

    # 2) join sales onto grid; fill zeros; split returns; forward context for price
    con.execute("""CREATE TEMP TABLE panel0 AS
        SELECT g.store_id, g.sku_id, g.date,
               coalesce(s.qty, 0)                       AS qty_raw,
               greatest(coalesce(s.qty, 0), 0)          AS units,        -- demand target
               CASE WHEN s.qty < 0 THEN -s.qty ELSE 0 END AS returns,
               (s.store_id IS NULL)                     AS was_gap_filled,
               s.unit_price
        FROM grid g
        LEFT JOIN sales s USING (store_id, sku_id, date);""")

    # 3) per-SKU winsorization threshold for unexplained spikes (kept + flagged)
    con.execute(f"""CREATE TEMP TABLE caps AS
        SELECT store_id, sku_id, quantile_cont(units, {WINSOR_Q}) AS cap
        FROM panel0 WHERE units > 0 GROUP BY 1,2;""")

    # 4) final panel with censored hooks, outlier flag, calendar enrichment
    print("writing panel ...")
    weight_expr = (
        "CASE WHEN was_stockout THEN 0.0 ELSE 1.0 END" if mask_stockouts else "1.0"
    )
    con.execute(f"""
        COPY (
            SELECT
                p.store_id, p.sku_id, p.date,
                substr(p.store_id, 1, 2)                 AS region,
                p.units,
                p.returns,
                p.qty_raw,
                p.unit_price,
                p.was_gap_filled,
                FALSE                                    AS was_stockout,  -- no inventory in M5
                (p.units > coalesce(c.cap, 1e18))        AS is_outlier,
                least(p.units, coalesce(c.cap, p.units)) AS units_winsor,
                ln(1 + p.units)                          AS target_log1p,  -- Phase 3.7 option
                {weight_expr}                            AS sample_weight,
                cal.is_holiday, cal.festival_name, cal.festival_intensity,
                cal.salary_window, cal.temp, cal.rain_mm, cal.fuel_index
            FROM panel0 p
            LEFT JOIN caps c USING (store_id, sku_id)
            LEFT JOIN read_parquet('{cal}') cal
              ON p.date = cal.date AND substr(p.store_id,1,2) = cal.region
        ) TO '{out.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (store_id), OVERWRITE_OR_IGNORE 1);
    """)

    stats = con.execute(f"""SELECT count(*) n_rows, count(DISTINCT sku_id) skus,
        min(date) mn, max(date) mx,
        round(100.0*avg(CASE WHEN units=0 THEN 1 ELSE 0 END),1) pct_zero,
        sum(is_outlier::int) outliers, sum(returns) total_returns
        FROM read_parquet('{out.as_posix()}/**/*.parquet')""").df()
    print(stats.to_string(index=False))
    con.close()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build clean modeling panel (Phase 3).")
    ap.add_argument("--stores", nargs="+", help="subset of store_ids")
    args = ap.parse_args(argv)
    build(args.stores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
