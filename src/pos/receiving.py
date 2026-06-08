"""Goods receipt / GRN capture (Milestone 2) — the headline that unlocks REAL lead times.

Every proxy dataset (M5, Favorita, Online Retail) lacked real supplier lead times; we always had
to assume them. Recording what was received against which order, with a timestamp, is exactly the
PO->GRN signal the engine's lead-time model needs. So this is treated as a first-class feature.

Receipts are immutable once entered (correct via reversal/adjustment, never edit). Recording a
receipt also increases on-hand stock through the audited InventoryService.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

from src.pos.inventory import InventoryService


class ReceiptService:
    def __init__(self, conn: sqlite3.Connection, store_id: str = "SHOP01"):
        self.conn = conn
        self.inv = InventoryService(conn, store_id)

    def record(self, sku_id: str, received_qty: int, po_id: str | None = None,
               received_at: str | None = None) -> str:
        """Record a goods receipt (immutable) and add the stock (audited). Returns receipt_id."""
        if received_qty < 0:
            raise ValueError("received_qty must be >= 0")
        rid = f"R{uuid.uuid4().hex[:12]}"
        ts = received_at or datetime.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO receipts (receipt_id, po_id, sku_id, received_qty, received_at) "
                "VALUES (?,?,?,?,?)", (rid, po_id, sku_id, int(received_qty), ts))
        self.inv.adjust(sku_id, int(received_qty), reason=f"GRN {rid}")   # arriving goods -> stock
        return rid

    def list(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT receipt_id, po_id, sku_id, received_qty, received_at "
            "FROM receipts ORDER BY received_at DESC")]

    def lead_time_samples(self) -> list[dict]:
        """Per-supplier realized lead times (days) where a receipt links to a PO with an order
        date — the real PO->GRN signal the engine's lead-time estimator consumes."""
        rows = self.conn.execute(
            """SELECT pd.supplier_id,
                      julianday(r.received_at) - julianday(pd.created_at) AS lead_days
               FROM receipts r JOIN po_drafts pd ON r.po_id = pd.po_id
               WHERE pd.created_at IS NOT NULL""").fetchall()
        return [{"supplier_id": r[0], "lead_days": float(r[1])} for r in rows if r[1] is not None]
