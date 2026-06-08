"""SQLite -> canonical Parquet bridge (Milestone 0) + contract gate.

This is the seam between the POS and the already-validated engine. It reads the day's operational
rows from SQLite and emits them in the EXACT canonical shape the engine's data contract
(src/ingest/validation.py) expects — line-item sales, product master, inventory snapshot,
suppliers, goods receipts — then runs that contract as a blocking gate. Bad shop data never
reaches scoring; the engine stays untouched.

    from src.pos.bridge import export_and_validate
    summary = export_and_validate("shop.db", "data/raw/shop")   # raises if BLOCKED
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.ingest.validation import (gate, validate_inventory, validate_receipts, validate_sales,
                                   validate_suppliers)
from src.pos.schema import connect


# --- flatteners: SQLite rows -> canonical DataFrames -----------------------------------------

def flatten_sales(conn: sqlite3.Connection) -> pd.DataFrame:
    """line_items + transactions -> canonical sales_transactions (line-item grain)."""
    df = pd.read_sql_query(
        """SELECT date(t.datetime) AS date, t.store_id, li.sku_id,
                  li.qty, li.unit_price, li.discount
           FROM line_items li JOIN transactions t ON li.txn_id = t.txn_id""", conn)
    df["date"] = pd.to_datetime(df["date"])
    df["qty"] = df["qty"].astype("int64")
    return df


def flatten_products(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT sku_id, name, category, pack_size, perishable, shelf_life_days,
                  unit_cost, sell_price FROM products""", conn)
    df["pack_size"] = df["pack_size"].astype("int64")
    df["perishable"] = df["perishable"].astype(bool)
    return df


def flatten_inventory(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT date(updated_at) AS date, store_id, sku_id, on_hand_qty FROM inventory""", conn)
    df["date"] = pd.to_datetime(df["date"])
    df["on_hand_qty"] = df["on_hand_qty"].astype("int64")
    return df


def flatten_suppliers(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT supplier_id, name, moq, order_cycle, default_lead_time_days FROM suppliers", conn)


def flatten_receipts(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT receipt_id, po_id, sku_id, received_qty,
                  date(received_at) AS receipt_date FROM receipts""", conn)
    if not df.empty:
        df["receipt_date"] = pd.to_datetime(df["receipt_date"])
        df["received_qty"] = df["received_qty"].astype("int64")
    return df


def flatten_purchase_orders(conn: sqlite3.Connection) -> pd.DataFrame:
    """po_drafts -> minimal PO view for the PO<->GRN referential check."""
    return pd.read_sql_query(
        """SELECT po_id, supplier_id, date(created_at) AS order_date FROM po_drafts""", conn)


# --- export + the contract gate --------------------------------------------------------------

def export_and_validate(db_path: str | Path, out_dir: str | Path | None = None,
                        *, raise_on_block: bool = True) -> dict:
    """Flatten the shop DB to canonical tables, run the engine's contract, optionally write Parquet.

    Returns the gate summary {passed, blocks, warnings}. The seam is proven when clean shop data
    PASSES and corrupted data BLOCKs — before any UI exists (Milestone 0 done-criterion).
    """
    conn = connect(db_path)
    try:
        sales = flatten_sales(conn)
        products = flatten_products(conn)
        inventory = flatten_inventory(conn)
        suppliers = flatten_suppliers(conn)
        receipts = flatten_receipts(conn)
        pos = flatten_purchase_orders(conn)
    finally:
        conn.close()

    results = [validate_sales(sales, products)]
    if not inventory.empty:
        results.append(validate_inventory(inventory))
    if not receipts.empty:
        results.append(validate_receipts(receipts, pos if not pos.empty else None))
    if not suppliers.empty:
        results.append(validate_suppliers(suppliers, None))

    summary = gate(results, raise_on_block=raise_on_block)

    if out_dir is not None and summary["passed"]:
        out = Path(out_dir)
        (out / "sales_transactions").mkdir(parents=True, exist_ok=True)
        # partition sales by store/date the way the engine's staged layout expects
        for (store, day), g in sales.groupby(["store_id", sales["date"].dt.date]):
            p = out / "sales_transactions" / f"store_id={store}" / f"date={day}"
            p.mkdir(parents=True, exist_ok=True)
            g.drop(columns=["store_id"]).to_parquet(p / "part.parquet", index=False)
        products.to_parquet(out / "product_master.parquet", index=False)
        if not inventory.empty:
            inventory.to_parquet(out / "inventory_snapshot.parquet", index=False)
        if not receipts.empty:
            receipts.to_parquet(out / "goods_receipts.parquet", index=False)
        if not suppliers.empty:
            suppliers.to_parquet(out / "suppliers.parquet", index=False)
    return summary
