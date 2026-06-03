"""Map the M5 dataset into our canonical staged tables (Phase 1.3 / 1.4).

M5 native layout (data/raw/m5_accuracy/):
  - sales_train_evaluation.csv : WIDE. one row per (item_id, store_id); columns d_1..d_1941
        plus id, item_id, dept_id, cat_id, store_id, state_id.
  - calendar.csv               : d -> date, wm_yr_wk, wday/month/year, event_name_1/2,
        event_type_1/2, snap_CA/TX/WI.
  - sell_prices.csv            : (store_id, item_id, wm_yr_wk) -> sell_price (WEEKLY).

Canonical staged outputs (data/staged/):
  - sales_transactions  : date, store_id, sku_id, qty, unit_price, discount
  - product_master      : sku_id, category, family, brand, pack_size, perishable, ...
  - external_calendar   : date, region, is_holiday, festival_name, salary_window, ...

We use DuckDB to UNPIVOT the wide sales and join calendar + prices on disk, then write
partitioned Parquet. Idempotent: re-running reproduces identical staged output.

Usage
-----
    python -m src.ingest.load_m5                 # full dataset
    python -m src.ingest.load_m5 --stores CA_1   # subset for fast dev iteration
    python -m src.ingest.load_m5 --stores CA_1 TX_1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "m5_accuracy"
STAGED = ROOT / "data" / "staged"


def _store_filter(stores: list[str] | None) -> str:
    if not stores:
        return ""
    quoted = ", ".join(f"'{s}'" for s in stores)
    return f"WHERE store_id IN ({quoted})"


def build(stores: list[str] | None = None) -> None:
    STAGED.mkdir(parents=True, exist_ok=True)
    sales_csv = (RAW / "sales_train_evaluation.csv").as_posix()
    cal_csv = (RAW / "calendar.csv").as_posix()
    price_csv = (RAW / "sell_prices.csv").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    where = _store_filter(stores)

    # 1) Wide -> long via UNPIVOT, keeping only the d_* day columns as the value.
    print("unpivoting wide sales -> long ...")
    con.execute(f"""
        CREATE TEMP TABLE sales_long AS
        SELECT id, item_id, dept_id, cat_id, store_id, state_id, d, units
        FROM (
            SELECT * FROM read_csv_auto('{sales_csv}') {where}
        )
        UNPIVOT (units FOR d IN (COLUMNS('^d_[0-9]+$')));
    """)
    n = con.execute("SELECT count(*) FROM sales_long").fetchone()[0]
    print(f"  sales_long rows: {n:,}")

    con.execute(f"CREATE TEMP TABLE calendar AS SELECT * FROM read_csv_auto('{cal_csv}');")
    con.execute(f"CREATE TEMP TABLE prices  AS SELECT * FROM read_csv_auto('{price_csv}');")

    # 2) sales_transactions (canonical). Weekly price joined via wm_yr_wk.
    print("writing sales_transactions ...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(c.date AS DATE)        AS date,
                s.store_id                  AS store_id,
                s.item_id                   AS sku_id,
                CAST(s.units AS INTEGER)    AS qty,
                p.sell_price                AS unit_price,
                0.0                         AS discount
            FROM sales_long s
            JOIN calendar c USING (d)
            LEFT JOIN prices p
              ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
            -- M5 marks a series active only once it has a price; drop leading pre-launch zeros
            WHERE p.sell_price IS NOT NULL
        ) TO '{(STAGED / 'sales_transactions').as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (store_id), OVERWRITE_OR_IGNORE 1);
    """)

    # 3) product_master (canonical). M5 has no pack/perishable/cost -> sensible defaults.
    print("writing product_master ...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                item_id                     AS sku_id,
                item_id                     AS name,
                cat_id                      AS category,
                dept_id                     AS family,
                CAST(NULL AS VARCHAR)       AS brand,
                1                           AS pack_size,
                FALSE                       AS perishable,
                CAST(NULL AS INTEGER)       AS shelf_life_days,
                CAST(NULL AS DOUBLE)        AS unit_cost,
                CAST(NULL AS DOUBLE)        AS sell_price
            FROM sales_long
        ) TO '{(STAGED / 'product_master.parquet').as_posix()}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
    """)

    # 4) external_calendar (canonical) at (date, region=state) grain.
    #    M5 events -> festival/holiday; SNAP days -> salary_window proxy (benefit payout days).
    print("writing external_calendar ...")
    con.execute(f"""
        COPY (
            WITH states AS (SELECT DISTINCT state_id FROM sales_long)
            SELECT
                CAST(c.date AS DATE)                         AS date,
                st.state_id                                  AS region,
                (c.event_name_1 IS NOT NULL)                 AS is_holiday,
                c.event_name_1                               AS festival_name,
                CASE
                    WHEN c.event_name_1 IS NULL THEN 0.0
                    WHEN c.event_type_1 = 'National' THEN 1.0
                    WHEN c.event_type_1 = 'Religious' THEN 0.8
                    WHEN c.event_type_1 = 'Cultural' THEN 0.6
                    ELSE 0.4
                END                                          AS festival_intensity,
                CASE st.state_id
                    WHEN 'CA' THEN c.snap_CA = 1
                    WHEN 'TX' THEN c.snap_TX = 1
                    WHEN 'WI' THEN c.snap_WI = 1
                    ELSE FALSE
                END                                          AS salary_window,
                CAST(NULL AS DOUBLE)                         AS temp,
                CAST(NULL AS DOUBLE)                         AS rain_mm,
                CAST(NULL AS DOUBLE)                         AS fuel_index
            FROM calendar c CROSS JOIN states st
        ) TO '{(STAGED / 'external_calendar.parquet').as_posix()}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
    """)

    con.close()
    print("\nStaged outputs written to", STAGED)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Map M5 -> canonical staged tables.")
    ap.add_argument("--stores", nargs="+", help="subset of store_ids (e.g. CA_1 TX_1)")
    args = ap.parse_args(argv)
    build(args.stores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
