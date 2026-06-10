"""Autonomous nightly loop (Milestone 6) — runs the daily sequence so the morning list is ready.

Sequence: data-contract gate -> engine scoring + reorder (engine_run) -> draft POs from any
recommendations already accepted (a prior day's, in shadow this is usually none). The loop is
OBSERVABLE and INTERRUPTIBLE: every run is logged to `nightly_runs`, and a FAILED run leaves
yesterday's recommendations untouched (it never publishes nothing or garbage).

Discipline kept from Phase 9 / the plan:
  - manual-trigger by default; only runs unattended once the manual path is trusted on this shop;
  - the contract gates every run (engine_run raises -> we record 'blocked', keep prior recs);
  - human still approves dispatch (M5) — automation prepares the list, it does not send orders;
  - promotion/retrain stays on the frontier metric (engine side), not wired to auto-fire here.

Scheduling is left to APScheduler or the OS scheduler calling run_nightly(); we don't embed a
server-grade orchestrator on the shop machine.
"""
from __future__ import annotations

from datetime import datetime

from src.ingest.validation import DataContractError
from src.pos.engine_run import run_recommendations
from src.pos.procurement import ProcurementService
from src.pos.schema import connect, create_db

_RUN_LOG_DDL = """
CREATE TABLE IF NOT EXISTS nightly_runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    status     TEXT NOT NULL,         -- ok | blocked | error
    detail     TEXT
);
"""


def _ensure_log(conn):
    conn.executescript(_RUN_LOG_DDL); conn.commit()


def run_nightly(db_path, store_id: str = "SHOP01", *, build_drafts: bool = True) -> dict:
    """Run the nightly sequence once. Always returns a status dict; never raises — a failure is
    recorded and yesterday's recommendations are left in place."""
    create_db(db_path).close()                       # ensure schema + migrations
    conn = connect(db_path)
    _ensure_log(conn)
    started = datetime.now().isoformat(timespec="seconds")

    try:
        summary = run_recommendations(db_path, store_id)   # gate + score + reorder + write
    except DataContractError as e:
        _log(conn, started, "blocked", str(e)[:300]); conn.close()
        return {"status": "blocked", "detail": str(e), "recommendations": "kept previous"}
    except Exception as e:                              # any failure -> keep prior recs, record
        _log(conn, started, "error", f"{type(e).__name__}: {e}"[:300]); conn.close()
        return {"status": "error", "detail": str(e), "recommendations": "kept previous"}

    drafted = []
    if build_drafts:
        # prepare per-vendor POs for anything already accepted (shadow: usually none until trusted)
        drafted = ProcurementService(conn).build_drafts(summary["run_date"])

    _log(conn, started, "ok", f"{summary['ordered']}/{summary['n']} to-order; {len(drafted)} PO drafts")
    conn.close()
    return {"status": "ok", **summary, "po_drafts": len(drafted)}


def _log(conn, started, status, detail):
    with conn:
        conn.execute("INSERT INTO nightly_runs (started_at, status, detail) VALUES (?,?,?)",
                     (started, status, detail))


def last_runs(db_path, n: int = 10) -> list[dict]:
    conn = connect(db_path); _ensure_log(conn)
    rows = [dict(r) for r in conn.execute(
        "SELECT started_at, status, detail FROM nightly_runs ORDER BY id DESC LIMIT ?", (n,))]
    conn.close()
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run the POS nightly reorder loop once.")
    ap.add_argument("--db", default="shop.db")
    ap.add_argument("--no-drafts", action="store_true", help="skip building PO drafts")
    args = ap.parse_args(argv)
    res = run_nightly(args.db, build_drafts=not args.no_drafts)
    print(f"nightly: {res['status']}", {k: v for k, v in res.items() if k != "status"})
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
