"""POS Milestone 0 tests: the SQLite->canonical->contract seam, proven before any UI exists.

Done-criterion (pos_integration.md M0): synthetic clean shop data flows SQLite -> bridge ->
data contract -> PASS, and corrupted data correctly BLOCKs. Plus the DB-level reliability
constraints (FK integrity, pack_size, integer-shaped quantities)."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from src.ingest.validation import DataContractError
from src.pos.bridge import export_and_validate, flatten_products, flatten_sales
from src.pos.schema import TABLES, connect, create_db
from src.pos.seed import seed_shop


@pytest.fixture
def shop_db(tmp_path):
    db = tmp_path / "shop.db"
    conn = create_db(db)
    seed_shop(conn, days=40, n_products=15)
    conn.close()
    return str(db)


# --- schema / reliability constraints ----------------------------------------

def test_schema_creates_all_tables(tmp_path):
    conn = create_db(tmp_path / "s.db")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES).issubset(names)
    conn.close()


def test_foreign_key_blocks_orphan_line_item(tmp_path):
    conn = create_db(tmp_path / "s.db")
    conn.execute("INSERT INTO transactions VALUES ('T1','SHOP01','2024-01-01T10:00','cash',0)")
    with pytest.raises(sqlite3.IntegrityError):       # sku_id not in products -> FK rejects
        conn.execute("INSERT INTO line_items (txn_id, sku_id, qty, unit_price) "
                     "VALUES ('T1','GHOST',1,5.0)")
    conn.close()


def test_pack_size_check(tmp_path):
    conn = create_db(tmp_path / "s.db")
    with pytest.raises(sqlite3.IntegrityError):       # pack_size >= 1 enforced
        conn.execute("INSERT INTO products (sku_id, pack_size) VALUES ('K1', 0)")
    conn.close()


# --- the bridge produces canonical shape -------------------------------------

def test_bridge_canonical_columns(shop_db):
    conn = connect(shop_db)
    s, p = flatten_sales(conn), flatten_products(conn)
    conn.close()
    assert {"date", "store_id", "sku_id", "qty", "unit_price"}.issubset(s.columns)
    assert {"sku_id", "pack_size", "perishable"}.issubset(p.columns)
    assert s["qty"].dtype.kind == "i"                 # whole-unit quantities
    assert p["perishable"].dtype == bool


# --- the contract gate: PASS on clean, BLOCK on corrupted --------------------

def test_clean_shop_data_passes_contract(shop_db):
    summary = export_and_validate(shop_db, raise_on_block=False)
    assert summary["passed"] is True                  # the seam works end-to-end


def test_returns_are_warned_not_blocked(shop_db):
    # seed_shop injects ~1% returns (negative qty); contract should WARN, not BLOCK.
    summary = export_and_validate(shop_db, raise_on_block=False)
    assert summary["passed"] is True
    assert any("returns_present" in w for w in summary["warnings"])


def test_future_dated_sale_blocks(shop_db):
    conn = connect(shop_db)
    sku = conn.execute("SELECT sku_id FROM products LIMIT 1").fetchone()[0]
    future = (datetime.now() + timedelta(days=400)).isoformat(timespec="seconds")
    conn.execute("INSERT INTO transactions VALUES ('TX_F','SHOP01',?,'cash',0)", (future,))
    conn.execute("INSERT INTO line_items (txn_id, sku_id, qty, unit_price) VALUES ('TX_F',?,2,9.0)",
                 (sku,))
    conn.commit(); conn.close()
    with pytest.raises(DataContractError):            # future date -> BLOCK -> quarantine
        export_and_validate(shop_db, raise_on_block=True)
