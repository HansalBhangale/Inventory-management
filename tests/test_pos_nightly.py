"""POS Milestone 6 tests: the nightly loop is observable + interruptible (never publishes garbage)."""
from datetime import datetime, timedelta

from src.pos.nightly import last_runs, run_nightly
from src.pos.schema import connect, create_db
from src.pos.seed import seed_shop


def _seed(tmp_path, days=40):
    db = tmp_path / "shop.db"
    conn = create_db(db); seed_shop(conn, days=days, n_products=10); conn.close()
    return str(db)


def test_nightly_runs_and_logs_ok(tmp_path):
    db = _seed(tmp_path)
    res = run_nightly(db)
    assert res["status"] == "ok" and res["n"] == 10
    runs = last_runs(db)
    assert runs and runs[0]["status"] == "ok"


def test_blocked_run_keeps_previous_recommendations(tmp_path):
    db = _seed(tmp_path)
    run_nightly(db)                                   # produce a good run
    conn = connect(db)
    before = conn.execute("SELECT count(*) FROM recommendations").fetchone()[0]
    # corrupt: a future-dated sale -> contract BLOCK on next run
    sku = conn.execute("SELECT sku_id FROM products LIMIT 1").fetchone()[0]
    future = (datetime.now() + timedelta(days=500)).isoformat(timespec="seconds")
    conn.execute("INSERT INTO transactions VALUES ('TXF','SHOP01',?,'cash',0)", (future,))
    conn.execute("INSERT INTO line_items (txn_id, sku_id, qty, unit_price) VALUES ('TXF',?,1,9)", (sku,))
    conn.commit(); conn.close()

    res = run_nightly(db)
    assert res["status"] == "blocked"                 # never raises; records the block
    conn = connect(db)
    after = conn.execute("SELECT count(*) FROM recommendations").fetchone()[0]
    assert last_runs(db)[0]["status"] == "blocked"
    conn.close()
    assert after == before                            # yesterday's recs left intact


def test_nightly_builds_drafts_for_accepted(tmp_path):
    db = _seed(tmp_path)
    run_nightly(db, build_drafts=False)
    conn = connect(db)
    rd = conn.execute("SELECT max(run_date) FROM recommendations").fetchone()[0]
    # accept one to-order item, then re-run -> a PO draft should appear
    sku = conn.execute("SELECT sku_id FROM recommendations WHERE run_date=? AND should_order=1 LIMIT 1",
                       (rd,)).fetchone()[0]
    conn.execute("UPDATE recommendations SET status='accepted' WHERE sku_id=? AND run_date=?", (sku, rd))
    conn.commit(); conn.close()
    res = run_nightly(db)
    assert res["status"] == "ok" and res["po_drafts"] >= 1
