"""Sale service (Milestone 1) — the boring, reliable core everything depends on.

Rings up a sale and commits it ATOMICALLY: the transaction row, its line items, and the inventory
decrement either all persist or none do (`with conn:` is a single SQLite transaction that rolls
back on any error). The sale is durable on disk BEFORE the caller is told it succeeded
(persist-before-confirm) — if the machine dies mid-sale, the DB is never half-updated and no sale
is silently lost.

No hardware: product lookup is by code/name (a barcode scanner is just keyboard input, so typing
a code works identically); the receipt is digital (see invoice.py), no thermal printer required.
Returns/voids are first-class (qty < 0 / 0), which the data contract already expects.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CartLine:
    sku_id: str
    qty: int
    unit_price: float
    discount: float = 0.0

    @property
    def line_total(self) -> float:
        return self.qty * self.unit_price - self.discount


@dataclass
class Cart:
    store_id: str = "SHOP01"
    payment_type: str = "cash"
    lines: list[CartLine] = field(default_factory=list)

    def add(self, sku_id: str, qty: int, unit_price: float, discount: float = 0.0) -> None:
        self.lines.append(CartLine(sku_id, int(qty), float(unit_price), float(discount)))


@dataclass
class SaleResult:
    txn_id: str
    subtotal: float
    tax: float
    total: float
    n_lines: int


class SaleService:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def lookup_product(self, query: str) -> list[dict]:
        """Find by exact SKU (what a scanner types) or by name substring (manual search)."""
        rows = self.conn.execute(
            "SELECT sku_id, name, sell_price, pack_size FROM products "
            "WHERE sku_id = ? OR name LIKE ? ORDER BY (sku_id = ?) DESC LIMIT 20",
            (query, f"%{query}%", query)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def totals(cart: Cart, tax_rate: float = 0.0) -> tuple[float, float, float]:
        subtotal = round(sum(ln.line_total for ln in cart.lines), 2)
        tax = round(subtotal * tax_rate, 2)
        return subtotal, tax, round(subtotal + tax, 2)

    def commit_sale(self, cart: Cart, tax_rate: float = 0.0) -> SaleResult:
        """Persist the whole sale atomically. Returns only AFTER the commit succeeds."""
        if not cart.lines:
            raise ValueError("cannot commit an empty cart")
        subtotal, tax, total = self.totals(cart, tax_rate)
        txn_id = f"T{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:                       # BEGIN .. COMMIT (auto-ROLLBACK on any exception)
            self.conn.execute(
                "INSERT INTO transactions (txn_id, store_id, datetime, payment_type, total) "
                "VALUES (?,?,?,?,?)", (txn_id, cart.store_id, now, cart.payment_type, total))
            for ln in cart.lines:
                self.conn.execute(
                    "INSERT INTO line_items (txn_id, sku_id, qty, unit_price, discount) "
                    "VALUES (?,?,?,?,?)", (txn_id, ln.sku_id, ln.qty, ln.unit_price, ln.discount))
                # ensure an inventory row exists, then decrement (negative qty = return -> increases)
                self.conn.execute(
                    "INSERT INTO inventory (store_id, sku_id, on_hand_qty, updated_at) "
                    "VALUES (?,?,0,?) ON CONFLICT(store_id, sku_id) DO NOTHING",
                    (cart.store_id, ln.sku_id, now))
                self.conn.execute(
                    "UPDATE inventory SET on_hand_qty = on_hand_qty - ?, updated_at = ? "
                    "WHERE store_id = ? AND sku_id = ?", (ln.qty, now, cart.store_id, ln.sku_id))
        return SaleResult(txn_id, subtotal, tax, total, len(cart.lines))

    def on_hand(self, sku_id: str, store_id: str = "SHOP01") -> int:
        r = self.conn.execute(
            "SELECT on_hand_qty FROM inventory WHERE store_id = ? AND sku_id = ?",
            (store_id, sku_id)).fetchone()
        return int(r[0]) if r else 0
