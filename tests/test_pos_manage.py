"""POS Milestone 2 tests: catalog, inventory (audited), and GRN lead-time capture."""
import pytest

from src.pos.catalog import ProductService, SupplierService
from src.pos.inventory import InventoryService
from src.pos.receiving import ReceiptService
from src.pos.schema import create_db


@pytest.fixture
def conn(tmp_path):
    return create_db(tmp_path / "shop.db")


# --- catalog -----------------------------------------------------------------

def test_product_upsert_and_list(conn):
    ps = ProductService(conn)
    ps.upsert("A", "Atta", pack_size=5, sell_price=250.0)
    ps.upsert("A", "Atta 5kg", pack_size=5, sell_price=260.0)   # update, not duplicate
    rows = ps.list()
    assert len(rows) == 1 and rows[0]["name"] == "Atta 5kg" and rows[0]["sell_price"] == 260.0


def test_product_bad_pack_size_rejected(conn):
    with pytest.raises(ValueError):
        ProductService(conn).upsert("X", pack_size=0)


def test_supplier_crud(conn):
    ss = SupplierService(conn)
    ss.upsert("S1", "Dairy", moq=12, order_cycle=1, default_lead_time_days=1)
    assert ss.get("S1")["moq"] == 12
    ss.delete("S1")
    assert ss.list() == []


# --- inventory (audited) -----------------------------------------------------

def test_adjust_logs_and_updates(conn):
    ProductService(conn).upsert("A", "Atta")
    inv = InventoryService(conn)
    assert inv.adjust("A", 20, "delivery") == 20
    assert inv.adjust("A", -3, "damage") == 17
    hist = inv.history("A")
    assert [h["delta"] for h in hist] == [-3, 20]               # full audit trail, newest first


def test_stock_take_records_delta_not_blind_set(conn):
    ProductService(conn).upsert("A", "Atta")
    inv = InventoryService(conn)
    inv.adjust("A", 10, "delivery")
    assert inv.stock_take("A", 7) == 7                          # counted 7 vs 10 -> delta -3
    assert inv.history("A")[0]["reason"] == "stocktake"
    assert inv.history("A")[0]["delta"] == -3


def test_list_inventory_flags_low_stock(conn):
    ps = ProductService(conn); ps.upsert("A", "Atta"); ps.upsert("B", "Milk")
    inv = InventoryService(conn); inv.adjust("A", 100, "delivery"); inv.adjust("B", 2, "delivery")
    rows = {r["sku_id"]: r for r in inv.list_inventory(low_stock_threshold=5)}
    assert rows["A"]["low_stock"] is False and rows["B"]["low_stock"] is True


# --- GRN (real lead-time source) ---------------------------------------------

def test_grn_records_and_increases_stock(conn):
    ProductService(conn).upsert("A", "Atta")
    rs = ReceiptService(conn)
    rid = rs.record("A", 24, po_id="PO1")
    assert rid.startswith("R")
    assert InventoryService(conn).on_hand("A") == 24           # arriving goods -> stock
    assert rs.list()[0]["received_qty"] == 24


def test_grn_negative_qty_rejected(conn):
    ProductService(conn).upsert("A", "Atta")
    with pytest.raises(ValueError):
        ReceiptService(conn).record("A", -5)


def test_lead_time_samples_from_po_to_grn(conn):
    ProductService(conn).upsert("A", "Atta")
    SupplierService(conn).upsert("S1", "Wholesale")
    conn.execute("INSERT INTO po_drafts (po_id, supplier_id, created_at, status) "
                 "VALUES ('PO1','S1','2024-01-01T09:00','dispatched')")
    conn.commit()
    ReceiptService(conn).record("A", 10, po_id="PO1", received_at="2024-01-04T09:00")
    samples = ReceiptService(conn).lead_time_samples()
    assert samples and abs(samples[0]["lead_days"] - 3.0) < 1e-6   # real PO->GRN = 3 days
