"""SKU segmentation (Phase 2.2 + 2.4): intermittency class and ABC x XYZ.

This is the single most important EDA output: it drives MODEL ROUTING (Phase 5.7) and
SERVICE-LEVEL POLICY (Phase 7.7).

Per (store_id, sku_id) we compute:
  - ADI  = #periods / #periods-with-demand              (average inter-demand interval)
  - CV^2 = (std_nonzero / mean_nonzero)^2               (variability of nonzero demand)
  - intermittency class (Syntetos-Boylan):
        smooth       : ADI < 1.32 and CV2 < 0.49   -> global LightGBM
        erratic      : ADI < 1.32 and CV2 >= 0.49  -> global LightGBM
        intermittent : ADI >= 1.32 and CV2 < 0.49  -> Croston/TSB
        lumpy        : ADI >= 1.32 and CV2 >= 0.49  -> Croston/TSB
  - ABC (value/revenue Pareto): A ~ top 80% revenue, B next ~15%, C last ~5%
  - XYZ (predictability via demand CV): X stable, Y variable, Z erratic

Output: data/features/segments.parquet  (one row per store_id, sku_id).

Usage:
    python -m src.features.segmentation
    python -m src.features.segmentation --stores CA_1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.config import CONFIG

STAGED = CONFIG.data_dir / "staged"
FEATURES = CONFIG.data_dir / "features"

# Syntetos-Boylan thresholds
ADI_CUT = 1.32
CV2_CUT = 0.49


def _sales_glob(stores: list[str] | None) -> str:
    base = (STAGED / "sales_transactions").as_posix()
    if stores:
        # union of per-store partition globs
        parts = " UNION ALL ".join(
            f"SELECT * FROM read_parquet('{base}/store_id={s}/*.parquet')" for s in stores
        )
        # add store_id back (it's the partition key, restored by read_parquet hive)
        return parts
    return f"SELECT * FROM read_parquet('{base}/**/*.parquet')"


def build(stores: list[str] | None = None) -> Path:
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "segments.parquet"
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    sales = _sales_glob(stores)

    # The panel must include zero-demand days, so we measure intermittency over the
    # SKU's ACTIVE window (first sale .. last sale) at daily grain.
    con.execute(f"""
        CREATE TEMP TABLE seg AS
        WITH sales AS ({sales}),
        -- daily units (sales rows already daily for M5, but aggregate defensively)
        daily AS (
            SELECT store_id, sku_id, date, sum(qty) AS units, max(unit_price) AS unit_price
            FROM sales GROUP BY 1,2,3
        ),
        span AS (
            SELECT store_id, sku_id,
                   min(date) AS first_date, max(date) AS last_date,
                   date_diff('day', min(date), max(date)) + 1 AS active_days
            FROM daily GROUP BY 1,2
        ),
        agg AS (
            SELECT
                d.store_id, d.sku_id,
                count(*)                                   AS obs_days,
                sum(CASE WHEN units > 0 THEN 1 ELSE 0 END) AS demand_days,
                sum(units)                                 AS total_units,
                sum(units * unit_price)                    AS revenue,
                avg(units)                                 AS mean_all,
                stddev_samp(units)                         AS std_all,
                avg(CASE WHEN units > 0 THEN units END)    AS mean_nz,
                stddev_samp(CASE WHEN units > 0 THEN units END) AS std_nz
            FROM daily d GROUP BY 1,2
        )
        SELECT
            a.store_id, a.sku_id, s.first_date, s.last_date, s.active_days,
            a.obs_days, a.demand_days, a.total_units, a.revenue,
            a.mean_all, a.std_all, a.mean_nz, a.std_nz,
            -- ADI over active window (uses calendar span, counting zero days correctly)
            CAST(s.active_days AS DOUBLE) / NULLIF(a.demand_days, 0)  AS adi,
            CASE WHEN a.mean_nz IS NULL OR a.mean_nz = 0 THEN NULL
                 ELSE pow(a.std_nz / a.mean_nz, 2) END               AS cv2,
            -- XYZ predictability: CV of daily demand over the active window
            CASE WHEN a.mean_all IS NULL OR a.mean_all = 0 THEN NULL
                 ELSE a.std_all / a.mean_all END                     AS cv_demand
        FROM agg a JOIN span s USING (store_id, sku_id);
    """)

    # Intermittency class + ABC (revenue Pareto within store) + XYZ
    con.execute(f"""
        COPY (
            WITH classed AS (
                SELECT *,
                    CASE
                        WHEN adi IS NULL THEN 'no_demand'
                        WHEN adi < {ADI_CUT} AND coalesce(cv2,0) < {CV2_CUT} THEN 'smooth'
                        WHEN adi < {ADI_CUT} AND coalesce(cv2,0) >= {CV2_CUT} THEN 'erratic'
                        WHEN adi >= {ADI_CUT} AND coalesce(cv2,0) < {CV2_CUT} THEN 'intermittent'
                        ELSE 'lumpy'
                    END AS intermittency,
                    CASE
                        WHEN cv_demand IS NULL THEN 'Z'
                        WHEN cv_demand <= 0.5 THEN 'X'
                        WHEN cv_demand <= 1.0 THEN 'Y'
                        ELSE 'Z'
                    END AS xyz
                FROM seg
            ),
            ranked AS (
                SELECT *,
                    sum(revenue) OVER (PARTITION BY store_id ORDER BY revenue DESC
                                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                        / NULLIF(sum(revenue) OVER (PARTITION BY store_id), 0) AS cum_rev_frac
                FROM classed
            )
            SELECT *,
                CASE WHEN cum_rev_frac <= 0.80 THEN 'A'
                     WHEN cum_rev_frac <= 0.95 THEN 'B'
                     ELSE 'C' END AS abc,
                CASE WHEN cum_rev_frac <= 0.80 THEN 'A'
                     WHEN cum_rev_frac <= 0.95 THEN 'B'
                     ELSE 'C' END ||
                CASE WHEN cv_demand IS NULL THEN 'Z'
                     WHEN cv_demand <= 0.5 THEN 'X'
                     WHEN cv_demand <= 1.0 THEN 'Y'
                     ELSE 'Z' END AS abc_xyz
            FROM ranked
        ) TO '{out.as_posix()}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
    """)

    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"wrote {out}  ({n:,} store-SKU rows)")
    con.close()
    return out


def summary(path: Path | None = None) -> None:
    """Print the segmentation distribution (an EDA finding)."""
    path = path or (FEATURES / "segments.parquet")
    con = duckdb.connect()
    p = path.as_posix()
    print("\n--- intermittency class distribution ---")
    print(con.execute(
        f"SELECT intermittency, count(*) n, round(100.0*count(*)/sum(count(*)) OVER (),1) pct, "
        f"round(sum(revenue),0) revenue FROM read_parquet('{p}') GROUP BY 1 ORDER BY n DESC"
    ).df().to_string(index=False))
    print("\n--- ABC x XYZ 9-box (counts) ---")
    print(con.execute(
        f"SELECT abc, xyz, count(*) n FROM read_parquet('{p}') GROUP BY 1,2 ORDER BY 1,2"
    ).df().pivot(index="abc", columns="xyz", values="n").to_string())
    print("\n--- ABC revenue share ---")
    print(con.execute(
        f"SELECT abc, count(*) skus, round(100.0*sum(revenue)/sum(sum(revenue)) OVER (),1) rev_pct "
        f"FROM read_parquet('{p}') GROUP BY 1 ORDER BY 1"
    ).df().to_string(index=False))
    con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build SKU segmentation table.")
    ap.add_argument("--stores", nargs="+", help="subset of store_ids")
    ap.add_argument("--summary", action="store_true", help="print distribution after building")
    args = ap.parse_args(argv)
    path = build(args.stores)
    if args.summary:
        summary(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
