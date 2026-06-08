"""POS operational SQLite schema (Milestone 0).

The schema mirrors the engine's CANONICAL columns so the existing data contract (validation.py)
passes on the bridge output without any change to the engine. Reliability is enforced at the DB
level: NOT NULL on keys, FK integrity (line_items -> products/transactions), pack_size >= 1, and
INTEGER quantities (the int-dtype lesson from the Online Retail run — quantities are whole units;
negatives are allowed = returns, which the contract handles as a WARN).

SQLite is the deliberate choice (see pos_integration.md §3): a sale is an atomic, relational event
(transaction + line items + inventory + payment commit together or not at all) — the textbook job
of an ACID embedded DB, offline-first, and a clean flatten to the engine's Parquet/DuckDB path.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DDL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;          -- reads don't block writes (cashier never waits)

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id            TEXT PRIMARY KEY,
    name                   TEXT,
    contact                TEXT,
    moq                    INTEGER,
    order_cycle            INTEGER,
    default_lead_time_days INTEGER
);

CREATE TABLE IF NOT EXISTS products (
    sku_id              TEXT PRIMARY KEY,
    name                TEXT,
    category            TEXT,
    pack_size           INTEGER NOT NULL DEFAULT 1 CHECK (pack_size >= 1),
    perishable          INTEGER NOT NULL DEFAULT 0 CHECK (perishable IN (0, 1)),
    shelf_life_days     INTEGER,                -- NULL => non-perishable
    unit_cost           REAL,
    sell_price          REAL,
    primary_supplier_id TEXT REFERENCES suppliers(supplier_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id       TEXT PRIMARY KEY,
    store_id     TEXT NOT NULL,
    datetime     TEXT NOT NULL,                 -- ISO8601; <= now enforced at bridge/contract
    payment_type TEXT,
    total        REAL
);

CREATE TABLE IF NOT EXISTS line_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id     TEXT NOT NULL REFERENCES transactions(txn_id),
    sku_id     TEXT NOT NULL REFERENCES products(sku_id),
    qty        INTEGER NOT NULL,                -- whole units; <0 = return (contract WARNs)
    unit_price REAL,
    discount   REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS inventory (
    store_id     TEXT NOT NULL,
    sku_id       TEXT NOT NULL REFERENCES products(sku_id),
    on_hand_qty  INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT,
    PRIMARY KEY (store_id, sku_id)
);

-- Goods receipts (GRN) — the lead-time source every proxy dataset lacked.
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id   TEXT PRIMARY KEY,
    po_id        TEXT,
    sku_id       TEXT NOT NULL REFERENCES products(sku_id),
    received_qty INTEGER NOT NULL CHECK (received_qty >= 0),
    received_at  TEXT NOT NULL
);

-- Written BACK by the engine (recommendations) and by procurement (po_drafts).
CREATE TABLE IF NOT EXISTS recommendations (
    sku_id        TEXT NOT NULL,
    run_date      TEXT NOT NULL,
    p50 REAL, p90 REAL, p95 REAL, p99 REAL,
    should_order  INTEGER,
    order_qty     INTEGER,
    reorder_point REAL,
    reason        TEXT,
    status        TEXT DEFAULT 'pending',       -- pending|accepted|rejected|modified
    PRIMARY KEY (sku_id, run_date)
);

CREATE TABLE IF NOT EXISTS po_drafts (
    po_id       TEXT PRIMARY KEY,
    supplier_id TEXT REFERENCES suppliers(supplier_id),
    created_at  TEXT,
    status      TEXT DEFAULT 'draft',           -- draft|approved|dispatched
    payload     TEXT
);

-- Audit trail for stock adjustments (history reconstructable, not silent overwrites — M2).
CREATE TABLE IF NOT EXISTS inventory_adjustments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id   TEXT NOT NULL,
    sku_id     TEXT NOT NULL,
    delta      INTEGER NOT NULL,
    reason     TEXT,
    at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_line_items_txn ON line_items(txn_id);
CREATE INDEX IF NOT EXISTS ix_txn_store_dt   ON transactions(store_id, datetime);
CREATE INDEX IF NOT EXISTS ix_receipts_sku   ON receipts(sku_id, received_at);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with FK enforcement + WAL (durability/concurrency for a POS)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def create_db(db_path: str | Path) -> sqlite3.Connection:
    """Create the operational schema (idempotent) and return an open connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(DDL)
    conn.commit()
    return conn


TABLES = ["suppliers", "products", "transactions", "line_items", "inventory", "receipts",
          "recommendations", "po_drafts", "inventory_adjustments"]
