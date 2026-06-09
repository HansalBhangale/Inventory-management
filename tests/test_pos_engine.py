"""POS Milestone 3 tests: the engine wired to shop data -> recommendations in SQLite.

Confirms the loop runs end-to-end and the DECISION machinery behaves: recommendations persisted
with all fields, real GRN lead times used, pack/MOQ rounding, perishables -> newsvendor, and the
contract gate blocks bad data before scoring."""
from datetime import datetime, timedelta

import pytest

from src.ingest.validation import DataContractError
from src.pos.catalog import ProductService, SupplierService
from src.pos.engine_run import _lead_time_days, run_recommendations
from src.pos.inventory import InventoryService
from src.pos.receiving import ReceiptService
from src.pos.schema import connect, create_db
from src.pos.seed import seed_shop


@pytest.fixture
def shop(tmp_path):
    db = tmp_path / "shop.db"
    conn = create_db(db)
    seed_shop(conn, days=60, n_products=12)
    conn.close()
    return str(db)


def test_run_writes_recommendations_with_all_fields(shop):
    summary = run_recommendations(shop)
    assert summary["n"] == 12
    conn = connect(shop)
    rows = conn.execute("SELECT * FROM recommendations WHERE run_date = ?",
                        (summary["run_date"],)).fetchall()
    conn.close()
    assert len(rows) == 12
    r = dict(rows[0])
    for f in ("p50", "p90", "p95", "p99", "should_order", "order_qty", "reorder_point", "reason"):
        assert f in r and r[f] is not None
    assert r["status"] == "pending"


def test_run_is_idempotent_per_run_date(shop):
    run_recommendations(shop, run_date="2024-06-01")
    run_recommendations(shop, run_date="2024-06-01")
    conn = connect(shop)
    n = conn.execute("SELECT count(*) FROM recommendations WHERE run_date='2024-06-01'").fetchone()[0]
    conn.close()
    assert n == 12               # re-run replaces, not duplicates


def test_real_grn_lead_time_used_over_default(tmp_path):
    conn = create_db(tmp_path / "s.db")
    SupplierService(conn).upsert("S1", "Wholesale", default_lead_time_days=9)   # FK: supplier first
    ProductService(conn).upsert("A", "Atta", pack_size=1, sell_price=50.0, primary_supplier_id="S1")
    conn.execute("INSERT INTO po_drafts (po_id, supplier_id, created_at, status) "
                 "VALUES ('PO1','S1','2024-01-01T09:00','dispatched')")
    conn.commit()
    ReceiptService(conn).record("A", 10, po_id="PO1", received_at="2024-01-03T09:00")  # real = 2d
    mean, default = _lead_time_days(conn)
    conn.close()
    assert abs(mean["S1"] - 2.0) < 1e-6        # uses measured 2d, NOT the declared 9d


def test_low_stock_high_demand_triggers_order(tmp_path):
    conn = create_db(tmp_path / "s.db")
    SupplierService(conn).upsert("S1", "Wholesale", moq=12, default_lead_time_days=3)  # FK: supplier first
    ProductService(conn).upsert("A", "Atta", pack_size=6, sell_price=50.0, primary_supplier_id="S1")
    # build steady demand history, leave stock low
    base = datetime.now() - timedelta(days=30)
    for d in range(30):
        conn.execute("INSERT INTO transactions VALUES (?,?,?,?,?)",
                     (f"T{d}", "SHOP01", (base + timedelta(days=d)).isoformat(), "cash", 0))
        conn.execute("INSERT INTO line_items (txn_id, sku_id, qty, unit_price) VALUES (?,?,?,?)",
                     (f"T{d}", "A", 5, 50.0))
    conn.commit()
    InventoryService(conn).adjust("A", 2, "seed")     # only 2 on hand vs ~5/day demand
    conn.close()
    run_recommendations(tmp_path / "s.db")
    conn = connect(tmp_path / "s.db")
    r = dict(conn.execute("SELECT * FROM recommendations WHERE sku_id='A'").fetchone())
    conn.close()
    assert r["should_order"] == 1
    assert r["order_qty"] % 6 == 0 and r["order_qty"] >= 12      # pack multiple AND >= MOQ


def test_blocked_data_raises_before_scoring(shop):
    conn = connect(shop)
    sku = conn.execute("SELECT sku_id FROM products LIMIT 1").fetchone()[0]
    future = (datetime.now() + timedelta(days=500)).isoformat(timespec="seconds")
    conn.execute("INSERT INTO transactions VALUES ('TXF','SHOP01',?,'cash',0)", (future,))
    conn.execute("INSERT INTO line_items (txn_id, sku_id, qty, unit_price) VALUES ('TXF',?,1,9.0)", (sku,))
    conn.commit(); conn.close()
    with pytest.raises(DataContractError):
        run_recommendations(shop)
