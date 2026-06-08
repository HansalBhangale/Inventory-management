"""Digital receipt / invoice (Milestone 1) — no thermal printer required.

Reads the COMMITTED sale back from the DB (the persisted truth, not the in-memory cart) and formats
a plain-text receipt — always available, zero dependencies. A PDF version is optional and only used
if reportlab is installed; the text receipt is the dependable path for a no-hardware setup.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def receipt_text(conn: sqlite3.Connection, txn_id: str, store_name: str = "Kirana Store") -> str:
    t = conn.execute(
        "SELECT store_id, datetime, payment_type, total FROM transactions WHERE txn_id = ?",
        (txn_id,)).fetchone()
    if t is None:
        raise ValueError(f"no transaction {txn_id}")
    lines = conn.execute(
        "SELECT li.sku_id, p.name, li.qty, li.unit_price, li.discount "
        "FROM line_items li LEFT JOIN products p ON li.sku_id = p.sku_id WHERE li.txn_id = ?",
        (txn_id,)).fetchall()

    w = 40
    out = [store_name.center(w), "-" * w, f"Receipt: {txn_id}", f"Date:    {t['datetime']}", "-" * w,
           f"{'Item':<22}{'Qty':>4}{'Amt':>14}"]
    subtotal = 0.0
    for ln in lines:
        amt = ln["qty"] * ln["unit_price"] - (ln["discount"] or 0)
        subtotal += amt
        name = (ln["name"] or ln["sku_id"])[:22]
        out.append(f"{name:<22}{ln['qty']:>4}{amt:>14.2f}")
    tax = round(t["total"] - subtotal, 2)
    out += ["-" * w, f"{'Subtotal':<26}{subtotal:>14.2f}"]
    if abs(tax) > 1e-9:
        out.append(f"{'Tax':<26}{tax:>14.2f}")
    out += [f"{'TOTAL':<26}{t['total']:>14.2f}", f"Paid: {t['payment_type']}", "-" * w,
            "Thank you!".center(w)]
    return "\n".join(out)


def save_text(conn: sqlite3.Connection, txn_id: str, path: str | Path, **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(receipt_text(conn, txn_id, **kw), encoding="utf-8")
    return p


def save_pdf(conn: sqlite3.Connection, txn_id: str, path: str | Path, **kw) -> Path:
    """Optional PDF — only if reportlab is installed; otherwise use save_text()."""
    try:
        from reportlab.lib.pagesizes import A6
        from reportlab.pdfgen import canvas
    except ImportError as e:
        raise RuntimeError("PDF needs reportlab (pip install reportlab); use save_text instead") from e
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(p), pagesize=A6)
    c.setFont("Courier", 8)
    y = A6[1] - 20
    for line in receipt_text(conn, txn_id, **kw).splitlines():
        c.drawString(15, y, line)
        y -= 11
    c.save()
    return p
