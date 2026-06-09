"""POS Milestone 5 tests: PO split by vendor + offline-tolerant idempotent dispatch."""
import json

import pytest

from src.pos.catalog import ProductService, SupplierService
from src.pos.procurement import ProcurementService
from src.pos.schema import create_db


def _rec(conn, sku, qty, status):
    conn.execute("INSERT INTO recommendations (sku_id, run_date, should_order, order_qty, status) "
                 "VALUES (?,?,1,?,?)", (sku, "2024-06-01", qty, status))


@pytest.fixture
def shop(tmp_path):
    conn = create_db(tmp_path / "s.db")
    SupplierService(conn).upsert("S1", "Wholesale"); SupplierService(conn).upsert("S2", "Dairy")
    ps = ProductService(conn)
    ps.upsert("A", "Atta", primary_supplier_id="S1")
    ps.upsert("B", "Rice", primary_supplier_id="S1")
    ps.upsert("M", "Milk", primary_supplier_id="S2")
    _rec(conn, "A", 12, "accepted")
    _rec(conn, "B", 6, "accepted")
    _rec(conn, "M", 24, "accepted")
    _rec(conn, "X", 5, "rejected")        # rejected -> excluded
    conn.commit()
    return conn


def test_build_drafts_groups_accepted_by_vendor(shop):
    po_ids = ProcurementService(shop).build_drafts("2024-06-01")
    assert set(po_ids) == {"PO-2024-06-01-S1", "PO-2024-06-01-S2"}
    s1 = json.loads(ProcurementService(shop).get("PO-2024-06-01-S1")["payload"])
    assert {i["sku_id"] for i in s1} == {"A", "B"}      # rejected X excluded


def test_build_drafts_idempotent(shop):
    p = ProcurementService(shop)
    p.build_drafts("2024-06-01"); p.build_drafts("2024-06-01")
    assert len(p.list_drafts()) == 2                    # no duplicates


def test_po_document_lists_items(shop):
    p = ProcurementService(shop); p.build_drafts("2024-06-01")
    doc = p.po_document("PO-2024-06-01-S1")
    assert "PURCHASE ORDER" in doc and "Atta" in doc and "Total units: 18" in doc


def test_dispatch_writes_outbox_and_marks_dispatched(shop, tmp_path):
    from src.pos.procurement import FileOutboxDispatcher
    p = ProcurementService(shop); p.build_drafts("2024-06-01"); p.approve("PO-2024-06-01-S2")
    disp = FileOutboxDispatcher(tmp_path / "outbox")
    assert p.dispatch("PO-2024-06-01-S2", disp) is True
    assert (tmp_path / "outbox" / "PO-2024-06-01-S2.txt").exists()
    assert p.get("PO-2024-06-01-S2")["status"] == "dispatched"


def test_dispatch_idempotent_no_double_send(shop, tmp_path):
    from src.pos.procurement import FileOutboxDispatcher
    p = ProcurementService(shop); p.build_drafts("2024-06-01")
    disp = FileOutboxDispatcher(tmp_path / "outbox")
    assert p.dispatch("PO-2024-06-01-S1", disp) is True
    assert p.dispatch("PO-2024-06-01-S1", disp) is False    # already sent -> no-op


def test_failed_send_keeps_po_queued(shop):
    class Offline:
        def send(self, *a): raise OSError("no internet")
    p = ProcurementService(shop); p.build_drafts("2024-06-01"); p.approve("PO-2024-06-01-S1")
    assert p.dispatch("PO-2024-06-01-S1", Offline()) is False
    assert p.get("PO-2024-06-01-S1")["status"] == "approved"   # queued, not lost
