"""Checkout controller (Milestone 1) — UI-agnostic cart/checkout logic.

Deliberately separated from any Qt code so the checkout flow is unit-tested without a display, and
the GUI (app/checkout_app.py) stays a thin shell that never holds business logic. This is the
"never block the UI / logic out of the widget" discipline made concrete.
"""
from __future__ import annotations

import sqlite3

from src.pos.invoice import receipt_text
from src.pos.sales import Cart, SaleResult, SaleService


class CheckoutController:
    def __init__(self, conn: sqlite3.Connection, store_id: str = "SHOP01",
                 tax_rate: float = 0.0, store_name: str = "Kirana Store"):
        self.conn = conn
        self.svc = SaleService(conn)
        self.store_id = store_id
        self.tax_rate = tax_rate
        self.store_name = store_name
        self.cart = Cart(store_id=store_id)

    # --- cart building ---
    def search(self, query: str) -> list[dict]:
        return self.svc.lookup_product(query)

    def add(self, sku_id: str, qty: int = 1, unit_price: float | None = None) -> None:
        if unit_price is None:
            r = self.conn.execute("SELECT sell_price FROM products WHERE sku_id = ?",
                                  (sku_id,)).fetchone()
            if r is None:
                raise KeyError(f"unknown SKU {sku_id}")
            unit_price = float(r[0] or 0.0)
        self.cart.add(sku_id, int(qty), float(unit_price))

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.cart.lines):
            self.cart.lines.pop(index)

    def clear(self) -> None:
        self.cart = Cart(store_id=self.store_id)

    def preview(self) -> tuple[float, float, float]:
        return SaleService.totals(self.cart, self.tax_rate)

    # --- commit ---
    def complete_sale(self) -> tuple[SaleResult, str]:
        """Atomically commit, then build the receipt from the persisted sale, then reset the cart.
        Returns (result, receipt_text). Caller shows 'done' only after this returns (persist-first)."""
        res = self.svc.commit_sale(self.cart, self.tax_rate)
        receipt = receipt_text(self.conn, res.txn_id, store_name=self.store_name)
        self.clear()
        return res, receipt
