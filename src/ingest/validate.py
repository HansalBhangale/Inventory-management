"""Data-contract validation gate (Phase 1.3 / 9.5).

Enforces the canonical schemas with pandera before data is trusted downstream.
Bad batches are rejected (raises) so they never poison the staged layer.

    from src.ingest.validate import validate_sales_transactions
    validate_sales_transactions(df)          # raises pandera.errors.SchemaError on failure

CLI:
    python -m src.ingest.validate            # validate all staged tables, print a report
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

ROOT = Path(__file__).resolve().parents[2]
STAGED = ROOT / "data" / "staged"

# --- Canonical schemas (mirror config/data_contract.yaml) ---------------------

sales_transactions_schema = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", Check.le(pd.Timestamp(date.today())), nullable=False),
        "store_id": Column(str, nullable=False),
        "sku_id": Column(str, nullable=False),
        "qty": Column(int, nullable=False),  # negatives allowed (returns), flagged elsewhere
        "unit_price": Column(float, Check.ge(0), nullable=True),
        "discount": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

product_master_schema = DataFrameSchema(
    {
        "sku_id": Column(str, nullable=False, unique=True),
        "category": Column(str, nullable=True),
        "family": Column(str, nullable=True),
        "brand": Column(str, nullable=True),
        "pack_size": Column(int, Check.ge(1), nullable=False),
        "perishable": Column(bool, nullable=False),
        "shelf_life_days": Column("Int64", Check.ge(0), nullable=True),
        "unit_cost": Column(float, Check.ge(0), nullable=True),
        "sell_price": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

external_calendar_schema = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", nullable=False),
        "region": Column(str, nullable=True),
        "is_holiday": Column(bool, nullable=True),
        "festival_name": Column(str, nullable=True),
        "festival_intensity": Column(float, Check.in_range(0, 1), nullable=True),
        "salary_window": Column(bool, nullable=True),
        "temp": Column(float, nullable=True),
        "rain_mm": Column(float, Check.ge(0), nullable=True),
        "fuel_index": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)

SCHEMAS = {
    "sales_transactions": sales_transactions_schema,
    "product_master": product_master_schema,
    "external_calendar": external_calendar_schema,
}


def validate_sales_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return sales_transactions_schema.validate(df, lazy=True)


def validate_product_master(df: pd.DataFrame) -> pd.DataFrame:
    return product_master_schema.validate(df, lazy=True)


def validate_external_calendar(df: pd.DataFrame) -> pd.DataFrame:
    return external_calendar_schema.validate(df, lazy=True)


def _read_staged(name: str) -> pd.DataFrame | None:
    """Read a staged table whether it's a single parquet file or a partitioned dir."""
    file = STAGED / f"{name}.parquet"
    folder = STAGED / name
    if file.exists():
        return pd.read_parquet(file)
    if folder.is_dir():
        return pd.read_parquet(folder)
    return None


def validate_all(sample_rows: int | None = 200_000) -> int:
    """Validate every staged table. Returns process exit code (0 = all pass)."""
    rc = 0
    for name, schema in SCHEMAS.items():
        df = _read_staged(name)
        if df is None:
            print(f"[skip] {name}: not found in {STAGED}")
            continue
        check_df = df.sample(min(sample_rows, len(df)), random_state=0) if sample_rows else df
        try:
            schema.validate(check_df, lazy=True)
            print(f"[pass] {name}: {len(df):,} rows (checked {len(check_df):,})")
        except pa.errors.SchemaErrors as exc:
            rc = 1
            print(f"[FAIL] {name}:\n{exc.failure_cases.head(20)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(validate_all())
