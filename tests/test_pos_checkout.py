"""M1 checkout controller tests — the full checkout flow, verified with no display/Qt."""
import pytest

from src.pos.checkout import CheckoutController
from src.pos.schema import create_db


@pytest.fixture
def ctrl(tmp_path):
    conn = create_db(tmp_path / "shop.db")
    conn.executemany("INSERT INTO products (sku_id, name, pack_size, sell_price) VALUES (?,?,1,?)",
                     [("A", "Atta", 50.0), ("B", "Milk", 25.0)])
    conn.execute("INSERT INTO inventory VALUES ('SHOP01','A',10,'2024-01-01')")
    conn.commit()
    return CheckoutController(conn, tax_rate=0.05)


def test_add_uses_catalog_price_and_previews_total(ctrl):
    ctrl.add("A", 2)                      # price pulled from catalog (50)
    ctrl.add("B", 1)                      # 25
    sub, tax, total = ctrl.preview()
    assert sub == 125.0 and tax == 6.25 and total == 131.25


def test_add_unknown_sku_raises(ctrl):
    with pytest.raises(KeyError):
        ctrl.add("GHOST", 1)


def test_complete_sale_persists_and_resets_cart(ctrl):
    ctrl.add("A", 3)
    res, receipt = ctrl.complete_sale()
    assert res.total == 157.5            # 150 + 5%
    assert res.txn_id in receipt and "TOTAL" in receipt
    assert ctrl.cart.lines == []         # cart reset after a completed sale
    assert ctrl.svc.on_hand("A") == 7    # inventory decremented and durable


def test_remove_line(ctrl):
    ctrl.add("A", 1); ctrl.add("B", 1)
    ctrl.remove(0)
    assert len(ctrl.cart.lines) == 1 and ctrl.cart.lines[0].sku_id == "B"
