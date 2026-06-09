"""Procurement & dispatch (Milestone 5) — turn ACCEPTED recommendations into per-vendor POs.

Only runs on recommendations the shopkeeper accepted (M4). Splits them by supplier into draft POs,
renders a PO document, and dispatches it through an offline-tolerant, idempotent dispatcher:
- a PO is created and stored BEFORE any send attempt;
- dispatch never double-sends (idempotent on po_id);
- a failed/offline send leaves the PO queued (status stays 'approved'), never silently dropped.
The default dispatcher writes to a local outbox (safe, no credentials); SMTP/WhatsApp plug into the
same interface and only fire in LIVE mode with real config.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import CONFIG


class FileOutboxDispatcher:
    """Default dispatcher: writes the PO to data/outbox/<po_id>.txt. Offline-safe + idempotent."""
    def __init__(self, outbox: str | Path | None = None):
        self.outbox = Path(outbox or (CONFIG.data_dir / "outbox"))

    def send(self, po_id: str, supplier_id: str, document: str) -> bool:
        self.outbox.mkdir(parents=True, exist_ok=True)
        (self.outbox / f"{po_id}.txt").write_text(document, encoding="utf-8")
        return True


class ProcurementService:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def build_drafts(self, run_date: str) -> list[str]:
        """Group the run's ACCEPTED recommendations by supplier into draft POs. Idempotent: a
        re-run refreshes only still-'draft' POs (never clobbers an approved/dispatched one)."""
        rows = self.conn.execute(
            """SELECT r.sku_id, r.order_qty, COALESCE(p.primary_supplier_id, 'UNASSIGNED') AS sup
               FROM recommendations r LEFT JOIN products p ON r.sku_id = p.sku_id
               WHERE r.run_date = ? AND r.status = 'accepted' AND r.order_qty > 0""",
            (run_date,)).fetchall()
        by_sup: dict[str, list] = {}
        for r in rows:
            by_sup.setdefault(r["sup"], []).append({"sku_id": r["sku_id"], "qty": r["order_qty"]})
        now = datetime.now().isoformat(timespec="seconds")
        po_ids = []
        with self.conn:
            for sup, items in by_sup.items():
                po_id = f"PO-{run_date}-{sup}"
                self.conn.execute(
                    """INSERT INTO po_drafts (po_id, supplier_id, created_at, status, payload)
                       VALUES (?,?,?,'draft',?)
                       ON CONFLICT(po_id) DO UPDATE SET payload=excluded.payload,
                           created_at=excluded.created_at WHERE status='draft'""",
                    (po_id, sup, now, json.dumps(items)))
                po_ids.append(po_id)
        return po_ids

    def list_drafts(self, status: str | None = None) -> list[dict]:
        q = "SELECT * FROM po_drafts"
        args = ()
        if status:
            q += " WHERE status = ?"; args = (status,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY created_at DESC", args)]

    def get(self, po_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM po_drafts WHERE po_id = ?", (po_id,)).fetchone()
        return dict(r) if r else None

    def approve(self, po_id: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE po_drafts SET status='approved' WHERE po_id=? AND status='draft'",
                              (po_id,))

    def po_document(self, po_id: str) -> str:
        po = self.get(po_id)
        if not po:
            raise ValueError(f"no PO {po_id}")
        sup = self.conn.execute("SELECT name, contact FROM suppliers WHERE supplier_id=?",
                                (po["supplier_id"],)).fetchone()
        items = json.loads(po["payload"])
        lines = [f"PURCHASE ORDER  {po_id}", f"Supplier: {po['supplier_id']}"
                 + (f" ({sup['name']})" if sup else ""), f"Date: {po['created_at']}", "-" * 40,
                 f"{'SKU':<24}{'Qty':>8}"]
        for it in items:
            name = self.conn.execute("SELECT name FROM products WHERE sku_id=?", (it["sku_id"],)).fetchone()
            label = f"{it['sku_id']} {name['name'] if name else ''}".strip()[:24]
            lines.append(f"{label:<24}{it['qty']:>8}")
        lines += ["-" * 40, f"Total lines: {len(items)}  Total units: {sum(i['qty'] for i in items)}"]
        return "\n".join(lines)

    def dispatch(self, po_id: str, dispatcher=None) -> bool:
        """Send a PO (idempotent). Returns True if dispatched now; False if already sent or the
        send failed (then it stays queued for retry). Document is built before the send attempt."""
        dispatcher = dispatcher or FileOutboxDispatcher()
        po = self.get(po_id)
        if not po:
            raise ValueError(f"no PO {po_id}")
        if po["status"] == "dispatched":
            return False                       # idempotent: never double-send
        document = self.po_document(po_id)
        try:
            ok = dispatcher.send(po_id, po["supplier_id"], document)
        except Exception:
            ok = False                         # offline / transient -> leave queued
        if ok:
            with self.conn:
                self.conn.execute("UPDATE po_drafts SET status='dispatched', dispatched_at=? "
                                  "WHERE po_id=?", (datetime.now().isoformat(timespec="seconds"), po_id))
        return bool(ok)
