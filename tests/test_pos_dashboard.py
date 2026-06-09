"""POS Milestone 4 tests: Morning Dashboard + shadow-mode decision capture.

The in-product pilot: recommendations grouped by vendor, accept/reject captured, the human
reject rate computed, and the system sanity flags (reused from shadow.py) available as diagnostic."""
import pytest

from src.pos.catalog import ProductService, SupplierService
from src.pos.dashboard import DashboardService
from src.pos.engine_run import run_recommendations
from src.pos.schema import create_db
from src.pos.seed import seed_shop


@pytest.fixture
def shop(tmp_path):
    db = tmp_path / "shop.db"
    conn = create_db(db)
    seed_shop(conn, days=60, n_products=12)
    conn.close()
    run_recommendations(str(db))            # populate recommendations
    return str(db)


def _conn(shop):
    from src.pos.schema import connect
    return connect(shop)


def test_grouped_by_vendor_with_value(shop):
    conn = _conn(shop)
    groups = DashboardService(conn).grouped()
    conn.close()
    assert groups                                   # at least one vendor group
    for g in groups:
        assert g["n"] == len(g["items"]) and g["value"] >= 0
        assert all(it["should_order"] == 1 for it in g["items"])   # dashboard shows to-order


def test_accept_reject_updates_status_and_reject_rate(shop):
    conn = _conn(shop)
    d = DashboardService(conn)
    rd = d.latest_run_date()
    items = [it for g in d.grouped() for it in g["items"]]
    assert len(items) >= 3
    d.set_decision(items[0]["sku_id"], rd, "accepted")
    d.set_decision(items[1]["sku_id"], rd, "rejected")
    d.set_decision(items[2]["sku_id"], rd, "modified", order_qty=999)
    s = d.summary(rd)
    conn.close()
    assert s["accepted"] == 1 and s["rejected"] == 1 and s["modified"] == 1
    assert s["decided"] == 3 and abs(s["reject_rate"] - 1 / 3) < 1e-9


def test_modify_changes_order_qty(shop):
    conn = _conn(shop)
    d = DashboardService(conn); rd = d.latest_run_date()
    sku = d.grouped()[0]["items"][0]["sku_id"]
    d.set_decision(sku, rd, "modified", order_qty=42)
    q = conn.execute("SELECT order_qty, status FROM recommendations WHERE sku_id=? AND run_date=?",
                     (sku, rd)).fetchone()
    conn.close()
    assert q[0] == 42 and q[1] == "modified"


def test_bad_status_rejected(shop):
    conn = _conn(shop)
    with pytest.raises(ValueError):
        DashboardService(conn).set_decision("X", "2024-01-01", "approved")
    conn.close()


def test_sanity_flags_reuse_shadow_runner(tmp_path):
    # craft an obviously-wrong recommendation -> shadow must flag it
    db = tmp_path / "s.db"
    conn = create_db(db)
    SupplierService(conn).upsert("S1", "W", moq=1)
    ProductService(conn).upsert("A", "Atta", pack_size=1, sell_price=10.0, primary_supplier_id="S1")
    conn.execute(
        "INSERT INTO recommendations (sku_id, run_date, p50,p90,p95,p99, should_order, order_qty, "
        "reorder_point, reason, status, order_up_to, inventory_position, "
        "expected_demand_protection, moq, pack_size) "
        "VALUES ('A','2024-06-01',1,2,3,4,1,5000,3,'x','pending',10,2,5,1,1)")
    conn.commit()
    flags = DashboardService(conn).sanity_flags("2024-06-01")
    conn.close()
    assert "implausibly_large" in flags["flag_counts"]    # 5000 >> order-up-to 10
