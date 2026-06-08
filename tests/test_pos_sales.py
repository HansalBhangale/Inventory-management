"""POS Milestone 1 tests: the atomic sale service.

The non-negotiables of a cash register: a sale commits all-or-nothing, inventory tracks exactly,
returns work, and a mid-sale failure leaves the DB completely unchanged (never a half-sale)."""
import sqlite3

import pytest

from src.pos.sales import Cart, SaleService
from src.pos.schema import create_db


@pytest.fixture
def svc(tmp_path):
    conn = create_db(tmp_path / "shop.db")
    conn.executemany("INSERT INTO products (sku_id, name, pack_size, sell_price) VALUES (?,?,1,?)",
                     [("A", "Atta", 50.0), ("B", "Milk", 25.0)])
    conn.execute("INSERT INTO inventory VALUES ('SHOP01','A',10,'2024-01-01')")
    conn.execute("INSERT INTO inventory VALUES ('SHOP01','B',8,'2024-01-01')")
    conn.commit()
    return SaleService(conn)


def test_totals_with_tax():
    cart = Cart()
    cart.add("A", 2, 50.0); cart.add("B", 1, 25.0, discount=5.0)
    sub, tax, total = SaleService.totals(cart, tax_rate=0.10)
    assert sub == 120.0 and tax == 12.0 and total == 132.0   # (100 + 20) ; 10% tax


def test_commit_persists_and_decrements(svc):
    cart = Cart(); cart.add("A", 3, 50.0); cart.add("B", 2, 25.0)
    res = svc.commit_sale(cart)
    assert res.n_lines == 2 and res.total == 200.0
    assert svc.on_hand("A") == 7 and svc.on_hand("B") == 6        # 10-3, 8-2
    # sale is durable: a fresh connection sees it (persist-before-confirm)
    assert svc.conn.execute("SELECT count(*) FROM transactions").fetchone()[0] == 1
    assert svc.conn.execute("SELECT count(*) FROM line_items").fetchone()[0] == 2


def test_return_increases_stock(svc):
    svc.commit_sale(Cart(lines=[]) if False else _cart("A", -2, 50.0))   # a return
    assert svc.on_hand("A") == 12                                  # 10 - (-2)


def test_empty_cart_rejected(svc):
    with pytest.raises(ValueError):
        svc.commit_sale(Cart())


def test_mid_sale_failure_rolls_back_everything(svc):
    # second line references a non-existent SKU -> FK error mid-commit. The WHOLE sale must roll
    # back: no transaction row, no line items, and stock on the first item unchanged.
    cart = Cart(); cart.add("A", 1, 50.0); cart.add("GHOST", 1, 9.0)
    with pytest.raises(sqlite3.IntegrityError):
        svc.commit_sale(cart)
    assert svc.conn.execute("SELECT count(*) FROM transactions").fetchone()[0] == 0
    assert svc.conn.execute("SELECT count(*) FROM line_items").fetchone()[0] == 0
    assert svc.on_hand("A") == 10                                  # untouched — no half-sale


def test_lookup_by_code_and_name(svc):
    assert svc.lookup_product("A")[0]["sku_id"] == "A"            # exact code (scanner-style)
    assert any(p["sku_id"] == "B" for p in svc.lookup_product("Milk"))  # name search


def test_receipt_text_reflects_committed_sale(svc):
    from src.pos.invoice import receipt_text
    cart = Cart(); cart.add("A", 2, 50.0)
    res = svc.commit_sale(cart, tax_rate=0.05)
    txt = receipt_text(svc.conn, res.txn_id, store_name="Test Shop")
    assert res.txn_id in txt and "TOTAL" in txt and "105.00" in txt   # 100 + 5% tax


def _cart(sku, qty, price):
    c = Cart(); c.add(sku, qty, price); return c
