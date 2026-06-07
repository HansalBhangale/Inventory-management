"""Adapter: UCI Online Retail -> canonical schema (real-data pilot rehearsal).

Real transactional retail (UK online gift seller, 2010-2011): InvoiceNo, StockCode, Description,
Quantity, InvoiceDate, UnitPrice, CustomerID, Country. Real prices (unlike Favorita) but still no
lead times / inventory (assumed, like every proxy so far). NOT a kirana — but real, never-seen,
messy transactions, so it genuinely exercises the ingestion + data-contract path.

Mapping: store_id=Country, sku_id=StockCode, date=InvoiceDate::date, qty=sum(Quantity) per daily
grain, unit_price=avg(UnitPrice). Deliberate cleaning decisions (documented, like Favorita's
rounding): drop cancellation invoices (C*) and non-product StockCodes (POST/DOT/M/D/...); net
returns within the day; keep only positive-price product lines.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.config import CONFIG

PARQUET = (CONFIG.data_dir / "pilot_data" / "online_retail.parquet").as_posix()
PRODUCT_CODE = r"^[0-9]{5}[A-Za-z]?$"   # real product codes are 5 digits (+ optional letter)


def raw_canonical(limit: int | None = None) -> pd.DataFrame:
    """Minimal rename to canonical columns, NO cleaning — for the contract to meet the mess."""
    lim = f"LIMIT {limit}" if limit else ""
    return duckdb.connect().execute(f"""
        SELECT CAST(InvoiceDate AS DATE) AS date, Country AS store_id,
               StockCode AS sku_id, CAST(Quantity AS INTEGER) AS qty, UnitPrice AS unit_price
        FROM read_parquet('{PARQUET}') {lim}
    """).df()


def product_master(filtered: bool = True) -> pd.DataFrame:
    """Product master from StockCodes. If filtered, only real product codes (excludes POST/M/...)."""
    where = f"WHERE regexp_matches(StockCode, '{PRODUCT_CODE}')" if filtered else ""
    return duckdb.connect().execute(f"""
        SELECT DISTINCT StockCode AS sku_id, any_value(Description) AS name,
               1 AS pack_size, FALSE AS perishable
        FROM read_parquet('{PARQUET}') {where} GROUP BY StockCode
    """).df()


def staged_sales() -> pd.DataFrame:
    """Cleaned, daily-grain canonical sales: drop cancellations + non-product codes, net returns."""
    return duckdb.connect().execute(f"""
        WITH lines AS (
            SELECT CAST(InvoiceDate AS DATE) AS date, Country AS store_id,
                   StockCode AS sku_id, Quantity AS q, UnitPrice AS price
            FROM read_parquet('{PARQUET}')
            WHERE InvoiceNo NOT LIKE 'C%'                       -- drop cancellations
              AND regexp_matches(StockCode, '{PRODUCT_CODE}')   -- real products only
              AND UnitPrice > 0
        )
        SELECT date, store_id, sku_id,
               CAST(sum(q) AS INTEGER) AS qty,                  -- net returns within the day
               avg(price) AS unit_price
        FROM lines GROUP BY 1,2,3 HAVING sum(q) IS NOT NULL
    """).df()
