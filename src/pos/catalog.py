"""Catalog management (Milestone 2): products + suppliers CRUD.

UI-agnostic services (logic out of the widgets, like M1). Products carry the constraint fields the
engine's reorder layer uses — pack_size, perishable, shelf_life_days, unit_cost, sell_price,
primary_supplier_id — so adding a product in the POS feeds the engine directly.
"""
from __future__ import annotations

import sqlite3


class ProductService:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, sku_id: str, name: str = "", category: str | None = None,
               pack_size: int = 1, perishable: bool = False, shelf_life_days: int | None = None,
               unit_cost: float | None = None, sell_price: float | None = None,
               primary_supplier_id: str | None = None) -> None:
        if not sku_id:
            raise ValueError("sku_id required")
        if pack_size < 1:
            raise ValueError("pack_size must be >= 1")
        with self.conn:
            self.conn.execute(
                """INSERT INTO products (sku_id, name, category, pack_size, perishable,
                       shelf_life_days, unit_cost, sell_price, primary_supplier_id)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(sku_id) DO UPDATE SET
                       name=excluded.name, category=excluded.category, pack_size=excluded.pack_size,
                       perishable=excluded.perishable, shelf_life_days=excluded.shelf_life_days,
                       unit_cost=excluded.unit_cost, sell_price=excluded.sell_price,
                       primary_supplier_id=excluded.primary_supplier_id""",
                (sku_id, name, category, int(pack_size), int(bool(perishable)), shelf_life_days,
                 unit_cost, sell_price, primary_supplier_id))

    def list(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM products ORDER BY sku_id").fetchall()]

    def get(self, sku_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM products WHERE sku_id = ?", (sku_id,)).fetchone()
        return dict(r) if r else None

    def delete(self, sku_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM products WHERE sku_id = ?", (sku_id,))


class SupplierService:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, supplier_id: str, name: str = "", contact: str | None = None,
               moq: int | None = None, order_cycle: int | None = None,
               default_lead_time_days: int | None = None) -> None:
        if not supplier_id:
            raise ValueError("supplier_id required")
        with self.conn:
            self.conn.execute(
                """INSERT INTO suppliers (supplier_id, name, contact, moq, order_cycle,
                       default_lead_time_days) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(supplier_id) DO UPDATE SET
                       name=excluded.name, contact=excluded.contact, moq=excluded.moq,
                       order_cycle=excluded.order_cycle,
                       default_lead_time_days=excluded.default_lead_time_days""",
                (supplier_id, name, contact, moq, order_cycle, default_lead_time_days))

    def list(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM suppliers ORDER BY supplier_id").fetchall()]

    def get(self, supplier_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM suppliers WHERE supplier_id = ?",
                              (supplier_id,)).fetchone()
        return dict(r) if r else None

    def delete(self, supplier_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM suppliers WHERE supplier_id = ?", (supplier_id,))
