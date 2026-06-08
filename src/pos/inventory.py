"""Inventory management (Milestone 2): on-hand view, audited adjustments, stock-take, low-stock.

Stock changes are LOGGED to inventory_adjustments (history is reconstructable, not silent
overwrites) and applied atomically. A stock-take records the delta to reach a counted quantity,
not a blind overwrite — so you can always explain how on-hand got to its current value.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime


class InventoryService:
    def __init__(self, conn: sqlite3.Connection, store_id: str = "SHOP01"):
        self.conn = conn
        self.store_id = store_id

    def on_hand(self, sku_id: str) -> int:
        r = self.conn.execute(
            "SELECT on_hand_qty FROM inventory WHERE store_id = ? AND sku_id = ?",
            (self.store_id, sku_id)).fetchone()
        return int(r[0]) if r else 0

    def adjust(self, sku_id: str, delta: int, reason: str = "manual") -> int:
        """Apply a signed delta atomically and log it. Returns the new on-hand."""
        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO inventory (store_id, sku_id, on_hand_qty, updated_at) VALUES (?,?,0,?) "
                "ON CONFLICT(store_id, sku_id) DO NOTHING", (self.store_id, sku_id, now))
            self.conn.execute(
                "UPDATE inventory SET on_hand_qty = on_hand_qty + ?, updated_at = ? "
                "WHERE store_id = ? AND sku_id = ?", (int(delta), now, self.store_id, sku_id))
            self.conn.execute(
                "INSERT INTO inventory_adjustments (store_id, sku_id, delta, reason, at) "
                "VALUES (?,?,?,?,?)", (self.store_id, sku_id, int(delta), reason, now))
        return self.on_hand(sku_id)

    def stock_take(self, sku_id: str, counted_qty: int) -> int:
        """Reconcile to a physically counted quantity via a logged delta (never a blind set)."""
        delta = int(counted_qty) - self.on_hand(sku_id)
        return self.adjust(sku_id, delta, reason="stocktake")

    def list_inventory(self, low_stock_threshold: int = 5) -> list[dict]:
        rows = self.conn.execute(
            """SELECT p.sku_id, p.name, p.category,
                      COALESCE(i.on_hand_qty, 0) AS on_hand_qty, p.perishable
               FROM products p
               LEFT JOIN inventory i ON i.sku_id = p.sku_id AND i.store_id = ?
               ORDER BY p.sku_id""", (self.store_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["low_stock"] = d["on_hand_qty"] <= low_stock_threshold
            out.append(d)
        return out

    def history(self, sku_id: str) -> list[dict]:
        # order by id (autoincrement), not `at` — second-resolution timestamps can tie
        return [dict(r) for r in self.conn.execute(
            "SELECT delta, reason, at FROM inventory_adjustments "
            "WHERE store_id = ? AND sku_id = ? ORDER BY id DESC", (self.store_id, sku_id))]
