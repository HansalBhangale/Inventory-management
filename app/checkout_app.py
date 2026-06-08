"""Thin PySide6 checkout shell (Milestone 1 scaffold) — run on the shop machine.

Boring and reliable, not pretty (polish comes later). No hardware: search/add by typing a code or
name (a barcode scanner just types the code), digital receipt shown on screen — no thermal printer.
All business logic lives in src/pos/checkout.py; this file only wires widgets to it and never holds
state of its own. The sale is committed atomically by the controller BEFORE "Sale complete" shows.

Run:
    pip install pyside6           # shop-machine dependency (not needed by the engine)
    python -m app.checkout_app --db shop.db
First run seeds a demo catalog if the DB is empty so you can click around immediately.
"""
from __future__ import annotations

import argparse
import sys

from src.pos.checkout import CheckoutController
from src.pos.schema import create_db
from src.pos.seed import seed_shop


def _ensure_catalog(conn):
    if conn.execute("SELECT count(*) FROM products").fetchone()[0] == 0:
        seed_shop(conn, days=30, n_products=20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kirana POS — checkout (M1 scaffold).")
    ap.add_argument("--db", default="shop.db")
    ap.add_argument("--tax", type=float, default=0.0)
    args = ap.parse_args(argv)

    conn = create_db(args.db)
    _ensure_catalog(conn)
    ctrl = CheckoutController(conn, tax_rate=args.tax)

    try:
        from PySide6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                                       QListWidget, QMessageBox, QPushButton, QTableWidget,
                                       QTableWidgetItem, QVBoxLayout, QWidget)
    except ImportError:
        print("PySide6 not installed. On the shop machine: pip install pyside6")
        return 1

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Kirana POS — Checkout")
            self.resize(820, 560)
            root = QHBoxLayout(self)

            # left: search + results
            left = QVBoxLayout()
            self.search_box = QLineEdit(placeholderText="Type product code or name, Enter to search")
            self.results = QListWidget()
            self.search_box.returnPressed.connect(self.do_search)
            self.results.itemDoubleClicked.connect(self.add_selected)
            left.addWidget(QLabel("Find product")); left.addWidget(self.search_box)
            left.addWidget(self.results)
            add_btn = QPushButton("Add to cart"); add_btn.clicked.connect(self.add_selected)
            left.addWidget(add_btn)
            root.addLayout(left, 1)

            # right: cart + total + complete
            right = QVBoxLayout()
            self.cart_tbl = QTableWidget(0, 3)
            self.cart_tbl.setHorizontalHeaderLabels(["Item", "Qty", "Amount"])
            self.cart_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.total_lbl = QLabel("Total: 0.00"); self.total_lbl.setStyleSheet("font-size:20px;")
            rm_btn = QPushButton("Remove selected"); rm_btn.clicked.connect(self.remove_selected)
            done_btn = QPushButton("Complete Sale (F2)")
            done_btn.clicked.connect(self.complete); done_btn.setShortcut("F2")
            right.addWidget(QLabel("Cart")); right.addWidget(self.cart_tbl)
            right.addWidget(rm_btn); right.addWidget(self.total_lbl); right.addWidget(done_btn)
            root.addLayout(right, 1)

        def do_search(self):
            self.results.clear()
            for p in ctrl.search(self.search_box.text().strip()):
                self.results.addItem(f"{p['sku_id']} | {p['name']} | ₹{p['sell_price']}")

        def add_selected(self):
            it = self.results.currentItem()
            if not it:
                return
            sku = it.text().split(" | ")[0]
            ctrl.add(sku, 1)
            self.refresh()

        def remove_selected(self):
            row = self.cart_tbl.currentRow()
            if row >= 0:
                ctrl.remove(row); self.refresh()

        def refresh(self):
            self.cart_tbl.setRowCount(len(ctrl.cart.lines))
            for i, ln in enumerate(ctrl.cart.lines):
                self.cart_tbl.setItem(i, 0, QTableWidgetItem(ln.sku_id))
                self.cart_tbl.setItem(i, 1, QTableWidgetItem(str(ln.qty)))
                self.cart_tbl.setItem(i, 2, QTableWidgetItem(f"{ln.line_total:.2f}"))
            _, _, total = ctrl.preview()
            self.total_lbl.setText(f"Total: {total:.2f}")

        def complete(self):
            if not ctrl.cart.lines:
                return
            try:
                res, receipt = ctrl.complete_sale()         # atomic commit happens here
            except Exception as e:                           # never crash the register
                QMessageBox.critical(self, "Sale failed", f"Not saved: {e}")
                return
            QMessageBox.information(self, "Sale complete", receipt)  # shown only after it's durable
            self.refresh()

    app = QApplication(sys.argv)
    win = Window(); win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
