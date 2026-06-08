"""Kirana POS — professional desktop app (Milestones 1–2 surface).

Sidebar navigation over five pages — Checkout, Products, Inventory, Vendors, Receiving (GRN) —
all wired to the tested service layer (src/pos/*). The GUI holds no business logic; every action
calls a service and shows a toast. Subtle fade animation on page switch.

Run:
    pip install pyside6
    python -m app.main --db shop.db
"""
from __future__ import annotations

import argparse
import sys

from src.pos.catalog import ProductService, SupplierService
from src.pos.checkout import CheckoutController
from src.pos.inventory import InventoryService
from src.pos.receiving import ReceiptService
from src.pos.schema import create_db

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                                   QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                                   QMainWindow, QMessageBox, QPushButton, QSpinBox, QStackedWidget,
                                   QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
    from app.theme import STYLESHEET, fade_in
    _QT = True
except ImportError:
    _QT = False


# ----------------------------------------------------------------------------- helpers
def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    return t


def _card(title):
    f = QFrame(); f.setObjectName("card")
    lay = QVBoxLayout(f); lay.setContentsMargins(18, 16, 18, 18); lay.setSpacing(10)
    h = QLabel(title); h.setObjectName("h1"); lay.addWidget(h)
    return f, lay


# ----------------------------------------------------------------------------- pages
class CheckoutPage(QWidget):
    def __init__(self, conn, toast):
        super().__init__()
        self.ctrl = CheckoutController(conn, tax_rate=0.0)
        self.toast = toast
        root = QHBoxLayout(self); root.setSpacing(16)

        left_card, left = _card("Find product")
        self.search = QLineEdit(placeholderText="Type code or name, Enter to search")
        self.results = QListWidget()
        self.search.returnPressed.connect(self.do_search)
        self.results.itemDoubleClicked.connect(self.add_selected)
        add = QPushButton("Add to cart"); add.clicked.connect(self.add_selected)
        left.addWidget(self.search); left.addWidget(self.results); left.addWidget(add)
        root.addWidget(left_card, 1)

        right_card, right = _card("Cart")
        self.tbl = _table(["Item", "Qty", "Amount"])
        self.total = QLabel("Total: 0.00"); self.total.setObjectName("total")
        rm = QPushButton("Remove selected"); rm.clicked.connect(self.remove_selected)
        done = QPushButton("Complete Sale  (F2)"); done.setObjectName("primary")
        done.setShortcut("F2"); done.clicked.connect(self.complete)
        right.addWidget(self.tbl); right.addWidget(rm)
        right.addWidget(self.total); right.addWidget(done)
        root.addWidget(right_card, 1)

    def do_search(self):
        self.results.clear()
        for p in self.ctrl.search(self.search.text().strip()):
            self.results.addItem(f"{p['sku_id']}  |  {p['name']}  |  ₹{p['sell_price']}")

    def add_selected(self):
        it = self.results.currentItem()
        if it:
            self.ctrl.add(it.text().split("  |  ")[0], 1); self.refresh()

    def remove_selected(self):
        r = self.tbl.currentRow()
        if r >= 0:
            self.ctrl.remove(r); self.refresh()

    def refresh(self):
        self.tbl.setRowCount(len(self.ctrl.cart.lines))
        for i, ln in enumerate(self.ctrl.cart.lines):
            self.tbl.setItem(i, 0, QTableWidgetItem(ln.sku_id))
            self.tbl.setItem(i, 1, QTableWidgetItem(str(ln.qty)))
            self.tbl.setItem(i, 2, QTableWidgetItem(f"{ln.line_total:.2f}"))
        self.total.setText(f"Total: {self.ctrl.preview()[2]:.2f}")

    def complete(self):
        if not self.ctrl.cart.lines:
            return
        try:
            res, receipt = self.ctrl.complete_sale()
        except Exception as e:
            self.toast(f"Sale failed: {e}", error=True); return
        self.refresh()
        self.toast(f"Sale saved · ₹{res.total:.2f}")
        QMessageBox.information(self, "Receipt", receipt)


class ProductsPage(QWidget):
    def __init__(self, conn, toast):
        super().__init__()
        self.ps = ProductService(conn); self.ss = SupplierService(conn); self.toast = toast
        root = QHBoxLayout(self); root.setSpacing(16)
        tcard, tlay = _card("Products")
        self.tbl = _table(["SKU", "Name", "Category", "Pack", "Perish", "Cost", "Price"])
        self.tbl.itemSelectionChanged.connect(self.load_row)
        tlay.addWidget(self.tbl)
        root.addWidget(tcard, 2)

        fcard, f = _card("Add / edit product")
        form = QFormLayout(); form.setSpacing(8)
        self.sku = QLineEdit(); self.name = QLineEdit(); self.cat = QLineEdit()
        self.pack = QSpinBox(); self.pack.setRange(1, 10000); self.pack.setValue(1)
        self.perish = QCheckBox("Perishable")
        self.shelf = QSpinBox(); self.shelf.setRange(0, 3650)
        self.cost = QDoubleSpinBox(); self.cost.setRange(0, 1e6); self.cost.setDecimals(2)
        self.price = QDoubleSpinBox(); self.price.setRange(0, 1e6); self.price.setDecimals(2)
        self.supplier = QComboBox()
        for w, lab in [(self.sku, "SKU code"), (self.name, "Name"), (self.cat, "Category"),
                       (self.pack, "Pack size"), (self.perish, ""), (self.shelf, "Shelf life (days)"),
                       (self.cost, "Unit cost"), (self.price, "Sell price"),
                       (self.supplier, "Primary supplier")]:
            form.addRow(lab, w)
        f.addLayout(form)
        save = QPushButton("Save product"); save.setObjectName("primary"); save.clicked.connect(self.save)
        new = QPushButton("Clear / new"); new.clicked.connect(self.clear)
        dele = QPushButton("Delete"); dele.setObjectName("danger"); dele.clicked.connect(self.delete)
        for b in (save, new, dele):
            f.addWidget(b)
        root.addWidget(fcard, 1)

    def refresh(self):
        self.supplier.clear(); self.supplier.addItem("", None)
        for s in self.ss.list():
            self.supplier.addItem(f"{s['supplier_id']} — {s['name']}", s["supplier_id"])
        rows = self.ps.list()
        self.tbl.setRowCount(len(rows))
        for i, p in enumerate(rows):
            vals = [p["sku_id"], p["name"], p["category"] or "", str(p["pack_size"]),
                    "Y" if p["perishable"] else "", f"{p['unit_cost'] or 0:.2f}",
                    f"{p['sell_price'] or 0:.2f}"]
            for j, v in enumerate(vals):
                self.tbl.setItem(i, j, QTableWidgetItem(v))

    def load_row(self):
        r = self.tbl.currentRow()
        if r < 0:
            return
        p = self.ps.get(self.tbl.item(r, 0).text())
        if not p:
            return
        self.sku.setText(p["sku_id"]); self.name.setText(p["name"] or "")
        self.cat.setText(p["category"] or ""); self.pack.setValue(p["pack_size"] or 1)
        self.perish.setChecked(bool(p["perishable"])); self.shelf.setValue(p["shelf_life_days"] or 0)
        self.cost.setValue(p["unit_cost"] or 0); self.price.setValue(p["sell_price"] or 0)
        ix = self.supplier.findData(p["primary_supplier_id"]); self.supplier.setCurrentIndex(max(ix, 0))

    def save(self):
        if not self.sku.text().strip():
            self.toast("SKU code required", error=True); return
        try:
            self.ps.upsert(self.sku.text().strip(), self.name.text().strip(),
                           self.cat.text().strip() or None, self.pack.value(),
                           self.perish.isChecked(), self.shelf.value() or None,
                           self.cost.value() or None, self.price.value() or None,
                           self.supplier.currentData())
        except Exception as e:
            self.toast(str(e), error=True); return
        self.refresh(); self.toast(f"Saved {self.sku.text().strip()}")

    def clear(self):
        for w in (self.sku, self.name, self.cat):
            w.clear()
        self.pack.setValue(1); self.perish.setChecked(False); self.shelf.setValue(0)
        self.cost.setValue(0); self.price.setValue(0); self.supplier.setCurrentIndex(0)

    def delete(self):
        r = self.tbl.currentRow()
        if r < 0:
            return
        sku = self.tbl.item(r, 0).text()
        self.ps.delete(sku); self.refresh(); self.clear(); self.toast(f"Deleted {sku}")


class InventoryPage(QWidget):
    def __init__(self, conn, toast):
        super().__init__()
        self.inv = InventoryService(conn); self.toast = toast
        card, lay = _card("Inventory")
        self.tbl = _table(["SKU", "Name", "On hand", "Status"])
        btns = QHBoxLayout()
        adj = QPushButton("Adjust stock"); adj.clicked.connect(self.adjust)
        st = QPushButton("Stock-take"); st.setObjectName("primary"); st.clicked.connect(self.stocktake)
        btns.addWidget(adj); btns.addWidget(st); btns.addStretch(1)
        lay.addWidget(self.tbl); lay.addLayout(btns)
        QVBoxLayout(self).addWidget(card)

    def refresh(self):
        rows = self.inv.list_inventory()
        self.tbl.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cells = [r["sku_id"], r["name"] or "", str(r["on_hand_qty"]),
                     "LOW" if r["low_stock"] else "ok"]
            for j, v in enumerate(cells):
                it = QTableWidgetItem(v)
                if j == 3 and r["low_stock"]:
                    it.setForeground(Qt.red)
                self.tbl.setItem(i, j, it)

    def _sel_sku(self):
        r = self.tbl.currentRow()
        return self.tbl.item(r, 0).text() if r >= 0 else None

    def adjust(self):
        sku = self._sel_sku()
        if not sku:
            self.toast("select an item", error=True); return
        from PySide6.QtWidgets import QInputDialog
        delta, ok = QInputDialog.getInt(self, "Adjust stock", f"Delta for {sku} (+/-):", 0, -100000, 100000)
        if ok and delta:
            self.inv.adjust(sku, delta, "manual"); self.refresh(); self.toast(f"{sku} adjusted {delta:+}")

    def stocktake(self):
        sku = self._sel_sku()
        if not sku:
            self.toast("select an item", error=True); return
        from PySide6.QtWidgets import QInputDialog
        counted, ok = QInputDialog.getInt(self, "Stock-take", f"Counted qty for {sku}:", 0, 0, 100000)
        if ok:
            self.inv.stock_take(sku, counted); self.refresh(); self.toast(f"{sku} set to {counted}")


class VendorsPage(QWidget):
    def __init__(self, conn, toast):
        super().__init__()
        self.ss = SupplierService(conn); self.toast = toast
        root = QHBoxLayout(self); root.setSpacing(16)
        tcard, tlay = _card("Suppliers")
        self.tbl = _table(["ID", "Name", "MOQ", "Cycle", "Lead(d)"])
        self.tbl.itemSelectionChanged.connect(self.load_row)
        tlay.addWidget(self.tbl); root.addWidget(tcard, 2)

        fcard, f = _card("Add / edit supplier")
        form = QFormLayout(); form.setSpacing(8)
        self.sid = QLineEdit(); self.name = QLineEdit(); self.contact = QLineEdit()
        self.moq = QSpinBox(); self.moq.setRange(0, 100000)
        self.cycle = QSpinBox(); self.cycle.setRange(0, 365)
        self.lead = QSpinBox(); self.lead.setRange(0, 365)
        for w, lab in [(self.sid, "Supplier ID"), (self.name, "Name"), (self.contact, "Contact"),
                       (self.moq, "MOQ"), (self.cycle, "Order cycle (days)"),
                       (self.lead, "Declared lead time (days)")]:
            form.addRow(lab, w)
        f.addLayout(form)
        save = QPushButton("Save supplier"); save.setObjectName("primary"); save.clicked.connect(self.save)
        dele = QPushButton("Delete"); dele.setObjectName("danger"); dele.clicked.connect(self.delete)
        f.addWidget(save); f.addWidget(dele)
        root.addWidget(fcard, 1)

    def refresh(self):
        rows = self.ss.list()
        self.tbl.setRowCount(len(rows))
        for i, s in enumerate(rows):
            vals = [s["supplier_id"], s["name"] or "", str(s["moq"] or ""),
                    str(s["order_cycle"] or ""), str(s["default_lead_time_days"] or "")]
            for j, v in enumerate(vals):
                self.tbl.setItem(i, j, QTableWidgetItem(v))

    def load_row(self):
        r = self.tbl.currentRow()
        if r < 0:
            return
        s = self.ss.get(self.tbl.item(r, 0).text())
        self.sid.setText(s["supplier_id"]); self.name.setText(s["name"] or "")
        self.contact.setText(s["contact"] or ""); self.moq.setValue(s["moq"] or 0)
        self.cycle.setValue(s["order_cycle"] or 0); self.lead.setValue(s["default_lead_time_days"] or 0)

    def save(self):
        if not self.sid.text().strip():
            self.toast("Supplier ID required", error=True); return
        self.ss.upsert(self.sid.text().strip(), self.name.text().strip(),
                       self.contact.text().strip() or None, self.moq.value() or None,
                       self.cycle.value() or None, self.lead.value() or None)
        self.refresh(); self.toast(f"Saved {self.sid.text().strip()}")

    def delete(self):
        r = self.tbl.currentRow()
        if r >= 0:
            sid = self.tbl.item(r, 0).text(); self.ss.delete(sid); self.refresh()
            self.toast(f"Deleted {sid}")


class ReceivingPage(QWidget):
    """Goods receipt (GRN) — captures real lead times (the headline of M2)."""
    def __init__(self, conn, toast):
        super().__init__()
        self.rs = ReceiptService(conn); self.ps = ProductService(conn); self.toast = toast
        root = QHBoxLayout(self); root.setSpacing(16)

        fcard, f = _card("Record goods receipt (GRN)")
        sub = QLabel("Recording deliveries is how the system learns real supplier lead times.")
        sub.setObjectName("muted"); sub.setWordWrap(True); f.addWidget(sub)
        form = QFormLayout(); form.setSpacing(8)
        self.sku = QComboBox(); self.qty = QSpinBox(); self.qty.setRange(1, 100000)
        self.po = QLineEdit(placeholderText="optional PO id (links lead time)")
        form.addRow("Product", self.sku); form.addRow("Received qty", self.qty)
        form.addRow("PO id", self.po)
        f.addLayout(form)
        rec = QPushButton("Record receipt"); rec.setObjectName("primary"); rec.clicked.connect(self.record)
        f.addWidget(rec)
        self.lead_lbl = QLabel("Real lead-time samples: 0"); self.lead_lbl.setObjectName("muted")
        f.addWidget(self.lead_lbl)
        root.addWidget(fcard, 1)

        tcard, tlay = _card("Recent receipts")
        self.tbl = _table(["Receipt", "SKU", "Qty", "When", "PO"])
        tlay.addWidget(self.tbl); root.addWidget(tcard, 1)

    def refresh(self):
        self.sku.clear()
        for p in self.ps.list():
            self.sku.addItem(f"{p['sku_id']} — {p['name']}", p["sku_id"])
        rows = self.rs.list()
        self.tbl.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r["receipt_id"][:8], r["sku_id"], str(r["received_qty"]),
                    (r["received_at"] or "")[:16], r["po_id"] or ""]
            for j, v in enumerate(vals):
                self.tbl.setItem(i, j, QTableWidgetItem(v))
        self.lead_lbl.setText(f"Real lead-time samples: {len(self.rs.lead_time_samples())}")

    def record(self):
        sku = self.sku.currentData()
        if not sku:
            self.toast("add a product first", error=True); return
        self.rs.record(sku, self.qty.value(), self.po.text().strip() or None)
        self.refresh(); self.toast(f"Received {self.qty.value()} of {sku}")
        self.po.clear()


# ----------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, conn):
        super().__init__()
        self.setWindowTitle("Kirana POS")
        self.resize(1100, 700)
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # sidebar
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(210)
        sl = QVBoxLayout(side); sl.setContentsMargins(0, 18, 0, 18); sl.setSpacing(2)
        brand = QLabel("  Kirana POS"); brand.setObjectName("h1"); brand.setContentsMargins(16, 0, 0, 14)
        sl.addWidget(brand)
        self.stack = QStackedWidget()
        self.pages = {"Checkout": CheckoutPage(conn, self.toast),
                      "Products": ProductsPage(conn, self.toast),
                      "Inventory": InventoryPage(conn, self.toast),
                      "Vendors": VendorsPage(conn, self.toast),
                      "Receiving": ReceivingPage(conn, self.toast)}
        self.nav_btns = []
        for name, page in self.pages.items():
            self.stack.addWidget(page)
            b = QPushButton(name); b.setObjectName("nav"); b.setCheckable(True)
            b.clicked.connect(lambda _=False, n=name: self.go(n))
            sl.addWidget(b); self.nav_btns.append((name, b))
        sl.addStretch(1)
        root.addWidget(side); root.addWidget(self.stack, 1)

        # toast (floating)
        self._toast = QLabel("", self); self._toast.setObjectName("toast")
        self._toast.setAlignment(Qt.AlignCenter); self._toast.hide()
        self._toast_timer = QTimer(self); self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._toast.hide)

        self.go("Checkout")

    def go(self, name):
        page = self.pages[name]
        if hasattr(page, "refresh"):
            page.refresh()
        self.stack.setCurrentWidget(page)
        fade_in(page)
        for n, b in self.nav_btns:
            b.setChecked(n == name)

    def toast(self, msg, error=False):
        self._toast.setObjectName("toastErr" if error else "toast")
        self._toast.setStyleSheet(STYLESHEET)            # re-apply for the new objectName
        self._toast.setText(msg); self._toast.adjustSize()
        self._toast.move((self.width() - self._toast.width()) // 2, self.height() - 70)
        self._toast.show(); fade_in(self._toast, 150)
        self._toast_timer.start(2200)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kirana POS desktop app.")
    ap.add_argument("--db", default="shop.db")
    args = ap.parse_args(argv)
    create_db(args.db).close()
    if not _QT:
        print("PySide6 not installed. On the shop machine: pip install pyside6")
        return 1
    conn = create_db(args.db)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(conn); win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
