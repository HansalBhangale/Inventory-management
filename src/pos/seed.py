"""Synthetic clean shop data for proving the M0 seam (and as a demo before any real shop exists).

Generates a realistic small kirana: suppliers, products (some perishable), daily sales with
returns, an inventory snapshot, and a few goods receipts (real lead-time source). The data is
clean by construction so the contract PASSES — tests then deliberately corrupt it to prove BLOCK.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import numpy as np


def seed_shop(conn: sqlite3.Connection, store_id: str = "SHOP01", days: int = 60,
              n_products: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cur = conn.cursor()

    suppliers = [("SUP_A", "Wholesale A", "a@x.com", 6, 7, 2),
                 ("SUP_B", "Dairy B", "b@x.com", 12, 1, 1)]
    cur.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?,?)", suppliers)

    products = []
    for i in range(n_products):
        sku = f"SKU{i:03d}"
        perishable = 1 if i % 4 == 0 else 0
        products.append((sku, f"Item {i}", "FOOD" if i % 2 else "HOME",
                         int(rng.choice([1, 6, 12])), perishable,
                         3 if perishable else None,
                         float(round(rng.uniform(5, 40), 2)), float(round(rng.uniform(10, 80), 2)),
                         "SUP_B" if perishable else "SUP_A"))
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", products)

    start = datetime.now() - timedelta(days=days)
    txn_n = 0
    for d in range(days):
        day = start + timedelta(days=d)
        n_txn = int(rng.integers(8, 20))
        for _ in range(n_txn):
            txn_id = f"T{txn_n:06d}"; txn_n += 1
            ts = (day + timedelta(hours=int(rng.integers(8, 20)))).isoformat(timespec="seconds")
            cur.execute("INSERT INTO transactions VALUES (?,?,?,?,?)",
                        (txn_id, store_id, ts, "cash", 0.0))
            for _ in range(int(rng.integers(1, 5))):
                p = products[int(rng.integers(0, n_products))]
                qty = int(rng.integers(1, 6))
                if rng.random() < 0.01:           # ~1% returns
                    qty = -1
                cur.execute("INSERT INTO line_items (txn_id, sku_id, qty, unit_price, discount) "
                            "VALUES (?,?,?,?,?)", (txn_id, p[0], qty, p[7], 0.0))

    now = datetime.now().isoformat(timespec="seconds")
    cur.executemany("INSERT INTO inventory VALUES (?,?,?,?)",
                    [(store_id, p[0], int(rng.integers(0, 50)), now) for p in products])

    # a few goods receipts (the lead-time source) against orders placed a few days earlier
    receipts = [(f"R{j:03d}", f"PO{j:03d}", products[j][0], int(rng.integers(6, 24)),
                 (start + timedelta(days=10 + j)).isoformat(timespec="seconds")) for j in range(3)]
    cur.executemany("INSERT INTO receipts VALUES (?,?,?,?,?)", receipts)

    conn.commit()
