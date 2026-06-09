"""Morning Dashboard service (Milestone 4) — the in-product pilot, SHADOW mode.

Surfaces the engine's recommendations grouped by vendor, captures the shopkeeper's
accept/reject/modify, and computes the **reject rate on real usage** — the week-one signal the
whole project was blocked on getting from a dataset. SHADOW means it suggests and acts on nothing.

Two distinct signals (kept separate, both useful):
  - HUMAN reject rate: shopkeeper rejections / decisions — the decisive trust signal.
  - SYSTEM sanity flags: reuses src/serve/shadow.py reject_flags ("implausibly large",
    "order-despite-ample-stock", "below-MOQ"...) so a high human reject rate can be traced to a
    data/config cause, not a vague "model bad".
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from src.serve.shadow import run_shadow

VALID_STATUS = {"pending", "accepted", "rejected", "modified"}


class DashboardService:
    def __init__(self, conn: sqlite3.Connection, store_id: str = "SHOP01"):
        self.conn = conn
        self.store_id = store_id

    def latest_run_date(self) -> str | None:
        r = self.conn.execute("SELECT max(run_date) FROM recommendations").fetchone()
        return r[0] if r and r[0] else None

    def _recs(self, run_date: str | None, only_to_order: bool) -> list[dict]:
        run_date = run_date or self.latest_run_date()
        if not run_date:
            return []
        where = "r.run_date = ?" + (" AND r.should_order = 1" if only_to_order else "")
        rows = self.conn.execute(
            f"""SELECT r.*, p.name, p.unit_cost, p.sell_price, p.primary_supplier_id,
                       COALESCE(s.name, p.primary_supplier_id, 'Unassigned') AS supplier_name
                FROM recommendations r
                LEFT JOIN products p ON r.sku_id = p.sku_id
                LEFT JOIN suppliers s ON p.primary_supplier_id = s.supplier_id
                WHERE {where} ORDER BY supplier_name, r.order_qty DESC""", (run_date,)).fetchall()
        return [dict(r) for r in rows]

    def grouped(self, run_date: str | None = None, only_to_order: bool = True) -> list[dict]:
        """Recommendations grouped by vendor, with a per-vendor item count and order value."""
        groups: dict[str, dict] = {}
        for r in self._recs(run_date, only_to_order):
            g = groups.setdefault(r["supplier_name"], {"supplier": r["supplier_name"],
                                                        "items": [], "n": 0, "value": 0.0})
            g["items"].append(r)
            g["n"] += 1
            g["value"] += (r["order_qty"] or 0) * float(r["unit_cost"] or r["sell_price"] or 0)
        return sorted(groups.values(), key=lambda x: -x["value"])

    def set_decision(self, sku_id: str, run_date: str, status: str,
                     order_qty: int | None = None) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"bad status {status}")
        with self.conn:
            if order_qty is not None:
                self.conn.execute(
                    "UPDATE recommendations SET status = ?, order_qty = ? "
                    "WHERE sku_id = ? AND run_date = ?", (status, int(order_qty), sku_id, run_date))
            else:
                self.conn.execute(
                    "UPDATE recommendations SET status = ? WHERE sku_id = ? AND run_date = ?",
                    (status, sku_id, run_date))

    def summary(self, run_date: str | None = None) -> dict:
        """Human decision summary incl. the week-one reject rate (rejected / decided)."""
        run_date = run_date or self.latest_run_date()
        counts = {s: 0 for s in VALID_STATUS}
        for row in self.conn.execute(
                "SELECT status, count(*) FROM recommendations WHERE run_date = ? AND should_order = 1 "
                "GROUP BY status", (run_date,)):
            counts[row[0]] = row[1]
        decided = counts["accepted"] + counts["rejected"] + counts["modified"]
        return {"run_date": run_date, **counts, "decided": decided,
                "reject_rate": (counts["rejected"] / decided) if decided else 0.0}

    def sanity_flags(self, run_date: str | None = None) -> dict:
        """System reject-flag check via the existing shadow runner (diagnostic, not the gate)."""
        recs = self._recs(run_date, only_to_order=False)
        if not recs:
            return {"reject_rate": 0.0, "flag_counts": {}}
        df = pd.DataFrame(recs)
        df["should_order"] = df["should_order"].astype(bool)
        rep = run_shadow(df)            # reuses src/serve/shadow.py exactly
        return {"reject_rate": rep.reject_rate, "flag_counts": rep.flag_counts}
