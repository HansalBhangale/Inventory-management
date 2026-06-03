"""Feature engineering (Phase 4).

Builds the model-ready feature panel from data/features/panel.parquet.

LEAKAGE RULE (mechanical): we train a DIRECT H-step forecaster, so when predicting day `t`
every feature may use information only up to the origin `o = t - H` (H = horizon, 14d).
All lags are >= H and every rolling window ENDS at t-H. Categorical encodings use native
LightGBM categoricals (fit-on-train concern handled at model time).

Feature families built (config/features.yaml):
  A calendar · B festival proximity · C lags · D rolling · E trend · F price ·
  G stockout (inert on M5) · H hierarchy/categorical · I sku attrs · K intermittency

Output: data/features/feature_panel.parquet (partitioned by store_id).

Usage:
    python -m src.features.build_features --stores CA_1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.config import CONFIG

FEATURES = CONFIG.data_dir / "features"
STAGED = CONFIG.data_dir / "staged"
H = CONFIG.horizon  # 14


def build(stores: list[str] | None = None) -> Path:
    panel = (FEATURES / "panel.parquet").as_posix()
    products = (STAGED / "product_master.parquet").as_posix()
    segments = (FEATURES / "segments.parquet").as_posix()
    out = FEATURES / "feature_panel.parquet"

    where = ""
    if stores:
        ids = ", ".join(f"'{s}'" for s in stores)
        where = f"WHERE store_id IN ({ids})"

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    # 0) base panel (+ product hierarchy, + segment descriptors)
    con.execute(f"""CREATE TEMP TABLE base AS
        SELECT p.*, pm.category, pm.family, pm.pack_size, pm.perishable,
               seg.adi, seg.cv2, seg.intermittency, seg.abc, seg.xyz
        FROM read_parquet('{panel}/**/*.parquet') p
        LEFT JOIN read_parquet('{products}') pm USING (sku_id)
        LEFT JOIN read_parquet('{segments}') seg USING (store_id, sku_id)
        {where};""")

    # 1) festival-proximity calendar features per (region, date) via ASOF joins
    con.execute("""CREATE TEMP TABLE fest AS
        SELECT DISTINCT region, date FROM base WHERE is_holiday;""")
    con.execute("""CREATE TEMP TABLE cal_feat AS
        WITH dates AS (SELECT DISTINCT region, date FROM base)
        SELECT d.region, d.date,
               date_diff('day', prev.date, d.date) AS days_since_last_festival,
               date_diff('day', d.date, nxt.date)  AS days_to_next_festival
        FROM dates d
        ASOF LEFT JOIN fest prev ON d.region = prev.region AND prev.date <= d.date
        ASOF LEFT JOIN fest nxt  ON d.region = nxt.region  AND nxt.date  >= d.date;""")

    # 2) category median price per (category, date) for relative_price
    con.execute("""CREATE TEMP TABLE catprice AS
        SELECT category, date, median(unit_price) AS cat_med_price
        FROM base WHERE unit_price IS NOT NULL GROUP BY 1,2;""")

    # 3) window features (lags >= H; rolling windows END at t-H). Daily contiguous panel.
    print("computing window features ...")
    con.execute(f"""
        COPY (
            WITH w AS (
                SELECT b.*,
                    cf.days_since_last_festival, cf.days_to_next_festival,
                    (cf.days_to_next_festival <= 7) AS in_festival_leadup,
                    cp.cat_med_price,
                    -- calendar
                    dayofweek(b.date) AS day_of_week,
                    (dayofweek(b.date) IN (0,6)) AS is_weekend,
                    week(b.date) AS week_of_year,
                    month(b.date) AS month, quarter(b.date) AS quarter,
                    day(b.date) AS day_of_month,
                    (day(b.date) <= 3) AS is_month_start,
                    -- C lags (all >= H)
                    lag(b.units, {H})  OVER s AS lag_14,
                    lag(b.units, {H+7})  OVER s AS lag_21,
                    lag(b.units, {H+14}) OVER s AS lag_28,
                    lag(b.units, 365)  OVER s AS lag_365,
                    -- D rolling (window ENDS at t-H)
                    avg(b.units)   OVER (s ROWS BETWEEN {H+6}  PRECEDING AND {H} PRECEDING) AS roll_mean_7,
                    avg(b.units)   OVER (s ROWS BETWEEN {H+13} PRECEDING AND {H} PRECEDING) AS roll_mean_14,
                    avg(b.units)   OVER (s ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_mean_28,
                    stddev_samp(b.units) OVER (s ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_std_28,
                    max(b.units)   OVER (s ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_max_28,
                    min(b.units)   OVER (s ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS roll_min_28,
                    -- same-DOW mean over last 4 same weekdays before origin (lags 14,21,28,35)
                    (coalesce(lag(b.units,{H},0)   OVER s,0)
                   + coalesce(lag(b.units,{H+7},0) OVER s,0)
                   + coalesce(lag(b.units,{H+14},0)OVER s,0)
                   + coalesce(lag(b.units,{H+21},0)OVER s,0)) / 4.0 AS same_dow_mean_4,
                    -- prob of sale over 28d ending at t-H (intermittency feature K)
                    avg(CASE WHEN b.units > 0 THEN 1.0 ELSE 0.0 END)
                        OVER (s ROWS BETWEEN {H+27} PRECEDING AND {H} PRECEDING) AS prob_of_sale_28d
                FROM base b
                LEFT JOIN cal_feat cf USING (region, date)
                LEFT JOIN catprice cp USING (category, date)
                WINDOW s AS (PARTITION BY b.store_id, b.sku_id ORDER BY b.date)
            )
            SELECT *,
                -- E trend / momentum
                roll_mean_7 / nullif(roll_mean_28, 0)        AS trend_7_28,
                -- F price
                unit_price / nullif(cat_med_price, 0)        AS relative_price
            FROM w
        ) TO '{out.as_posix()}' (FORMAT PARQUET, PARTITION_BY (store_id), OVERWRITE_OR_IGNORE 1);
    """)

    cols = con.execute(f"SELECT * FROM read_parquet('{out.as_posix()}/**/*.parquet') LIMIT 0").df().columns
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}/**/*.parquet')").fetchone()[0]
    print(f"wrote {out}  ({n:,} rows, {len(cols)} cols)")
    print("columns:", list(cols))
    con.close()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build model-ready feature panel (Phase 4).")
    ap.add_argument("--stores", nargs="+", help="subset of store_ids")
    args = ap.parse_args(argv)
    build(args.stores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
