"""POS Milestone 7 tests: the copilot explains (never decides) and works offline."""
from src.pos.catalog import ProductService, SupplierService
from src.pos.copilot import explain, rule_based_explanation
from src.pos.schema import create_db


def _rec(conn, **kw):
    cols = ", ".join(kw); ph = ",".join("?" * len(kw))
    conn.execute(f"INSERT INTO recommendations ({cols}) VALUES ({ph})", tuple(kw.values()))
    conn.commit()


def _shop(tmp_path):
    conn = create_db(tmp_path / "s.db")
    SupplierService(conn).upsert("S1", "Wholesale", default_lead_time_days=3)
    ProductService(conn).upsert("A", "Atta 5kg", primary_supplier_id="S1")
    ProductService(conn).upsert("M", "Milk 1L", perishable=True, shelf_life_days=3,
                                unit_cost=20, sell_price=30, primary_supplier_id="S1")
    return conn


def test_explains_order_with_numbers(tmp_path):
    conn = _shop(tmp_path)
    _rec(conn, sku_id="A", run_date="2024-06-01", should_order=1, order_qty=12,
         reorder_point=18, p95=9, inventory_position=6)
    txt = explain(conn, "A", "2024-06-01", use_gemini=False)
    conn.close()
    assert "Atta 5kg" in txt and "12" in txt and "18" in txt and "Reorder" in txt


def test_explains_no_order(tmp_path):
    conn = _shop(tmp_path)
    _rec(conn, sku_id="A", run_date="2024-06-01", should_order=0, order_qty=0,
         reorder_point=5, p95=2, inventory_position=20)
    txt = explain(conn, "A", "2024-06-01", use_gemini=False)
    conn.close()
    assert "No reorder needed" in txt and "Atta 5kg" in txt


def test_perishable_mentions_spoilage(tmp_path):
    conn = _shop(tmp_path)
    _rec(conn, sku_id="M", run_date="2024-06-01", should_order=1, order_qty=10,
         reorder_point=8, p95=12, inventory_position=2)
    txt = explain(conn, "M", "2024-06-01", use_gemini=False)
    conn.close()
    assert "perishable" in txt.lower() or "spoil" in txt.lower()


def test_missing_recommendation_is_graceful(tmp_path):
    conn = _shop(tmp_path)
    assert "No recommendation found" in explain(conn, "GHOST", "2024-06-01", use_gemini=False)
    conn.close()


def test_rule_based_is_pure_from_context():
    ctx = {"sku_id": "A", "name": "Atta", "should_order": 1, "order_qty": 12,
           "reorder_point": 18, "p95": 9, "on_hand": 6, "supplier_name": "Wholesale"}
    assert "Reorder 12 of Atta" in rule_based_explanation(ctx)
