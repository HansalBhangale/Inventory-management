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
from src.pos.dashboard import DashboardService
from src.pos.engine_run import run_recommendations
from src.pos.inventory import InventoryService
from src.pos.procurement import ProcurementService
from src.pos.receiving import ReceiptService
from src.pos.schema import create_db
from src.serve.settings import SETTINGS

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
        # quantity entry for the selected product (negative = return)
        qty_row = QHBoxLayout()
        qty_row.addWidget(QLabel("Qty"))
        self.qty = QSpinBox(); self.qty.setRange(-9999, 9999); self.qty.setValue(1)
        self.qty.setFixedWidth(90); self.qty.lineEdit().returnPressed.connect(self.add_selected)
        qty_row.addWidget(self.qty); qty_row.addStretch(1)
        add = QPushButton("Add to cart"); add.setObjectName("primary")
        add.clicked.connect(self.add_selected); qty_row.addWidget(add)
        left.addWidget(self.search); left.addWidget(self.results); left.addLayout(qty_row)
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
        if not it:
            self.toast("select a product first", error=True); return
        qty = self.qty.value()
        if qty == 0:
            self.toast("quantity can't be 0", error=True); return
        self.ctrl.add(it.text().split("  |  ")[0], qty)
        self.qty.setValue(1)          # reset for the next item
        self.refresh()

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


class DashboardPage(QWidget):
    """Morning Dashboard — recommendations grouped by vendor, SHADOW mode (suggest, never order).
    Captures accept/reject and shows the reject rate (the in-product pilot signal)."""
    def __init__(self, conn, db_path, toast):
        super().__init__()
        self.svc = DashboardService(conn); self.db_path = db_path; self.toast = toast
        card, lay = _card("Morning Dashboard")

        banner = QLabel(f"  SHADOW MODE — suggestions only. Nothing is ordered.  (mode: {SETTINGS.mode.value})")
        banner.setStyleSheet("background:#1e3a34; color:#5eead4; border-radius:8px; padding:8px;")
        lay.addWidget(banner)

        bar = QHBoxLayout()
        run = QPushButton("Generate today's list"); run.setObjectName("primary")
        run.clicked.connect(self.run_engine)
        self.stat = QLabel("—"); self.stat.setObjectName("muted")
        self.flags = QLabel(""); self.flags.setObjectName("muted")
        bar.addWidget(run); bar.addStretch(1); bar.addWidget(self.flags); bar.addSpacing(16)
        bar.addWidget(self.stat)
        lay.addLayout(bar)

        self.tbl = _table(["Vendor", "Item", "Qty", "Reorder pt", "Why", "Status", "Decision"])
        lay.addWidget(self.tbl)
        QVBoxLayout(self).addWidget(card)

    def run_engine(self):
        try:
            s = run_recommendations(self.db_path)        # bridge -> contract gate -> engine
        except Exception as e:
            self.toast(f"Run blocked: {e}", error=True); return
        self.toast(f"{s['ordered']} of {s['n']} items suggested for reorder")
        self.refresh()

    def _decision_cell(self, sku, run_date):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        a = QPushButton("Accept"); a.setObjectName("primary")
        r = QPushButton("Reject"); r.setObjectName("danger")
        a.clicked.connect(lambda: self.decide(sku, run_date, "accepted"))
        r.clicked.connect(lambda: self.decide(sku, run_date, "rejected"))
        h.addWidget(a); h.addWidget(r)
        return w

    def decide(self, sku, run_date, status):
        self.svc.set_decision(sku, run_date, status)
        self.toast(f"{sku}: {status}")
        self.refresh()

    def refresh(self):
        rd = self.svc.latest_run_date()
        groups = self.svc.grouped()
        flat = [(g["supplier"], it) for g in groups for it in g["items"]]
        self.tbl.setRowCount(len(flat))
        for i, (vendor, it) in enumerate(flat):
            cells = [vendor, f"{it['sku_id']} — {it.get('name') or ''}", str(it["order_qty"]),
                     f"{it['reorder_point']:.0f}", it["reason"], it["status"]]
            for j, v in enumerate(cells):
                self.tbl.setItem(i, j, QTableWidgetItem(v))
            self.tbl.setCellWidget(i, 6, self._decision_cell(it["sku_id"], rd))
        self.tbl.resizeRowsToContents()
        if rd:
            s = self.svc.summary(rd)
            self.stat.setText(f"run {rd} · to-order {len(flat)} · decided {s['decided']} · "
                              f"reject rate {s['reject_rate']:.0%}")
            fl = self.svc.sanity_flags(rd)["flag_counts"]
            self.flags.setText("sanity flags: " + (", ".join(f"{k}×{v}" for k, v in fl.items()) or "none"))
        else:
            self.stat.setText("No run yet — click 'Generate today's list'.")


class ProcurementPage(QWidget):
    """M5 — turn ACCEPTED recommendations into per-vendor POs and dispatch (offline-queued)."""
    def __init__(self, conn, toast):
        super().__init__()
        self.conn = conn; self.svc = ProcurementService(conn); self.toast = toast
        card, lay = _card("Procurement — Purchase Orders")
        bar = QHBoxLayout()
        build = QPushButton("Build POs from accepted"); build.setObjectName("primary")
        build.clicked.connect(self.build)
        bar.addWidget(build); bar.addStretch(1)
        note = QLabel("Dispatch writes to data/outbox/ (offline-safe). Real email/WhatsApp = LIVE mode.")
        note.setObjectName("muted"); bar.addWidget(note)
        lay.addLayout(bar)
        self.tbl = _table(["PO", "Vendor", "Units", "Status", "Action"])
        lay.addWidget(self.tbl)
        QVBoxLayout(self).addWidget(card)

    def _run_date(self):
        r = self.conn.execute("SELECT max(run_date) FROM recommendations").fetchone()
        return r[0] if r and r[0] else None

    def build(self):
        rd = self._run_date()
        if not rd:
            self.toast("no recommendations yet — Dashboard → Generate", error=True); return
        ids = self.svc.build_drafts(rd)
        self.toast(f"{len(ids)} vendor PO(s) drafted from accepted items" if ids
                   else "no accepted items to order"); self.refresh()

    def _action(self, po):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        if po["status"] == "draft":
            a = QPushButton("Approve"); a.setObjectName("primary")
            a.clicked.connect(lambda: self._do(self.svc.approve, po["po_id"], "approved"))
            h.addWidget(a)
        if po["status"] in ("draft", "approved"):
            d = QPushButton("Dispatch")
            d.clicked.connect(lambda: self.dispatch(po["po_id"]))
            h.addWidget(d)
        if po["status"] == "dispatched":
            h.addWidget(QLabel("✓ sent"))
        return w

    def _do(self, fn, po_id, label):
        fn(po_id); self.toast(f"{po_id}: {label}"); self.refresh()

    def dispatch(self, po_id):
        ok = self.svc.dispatch(po_id)
        self.toast(f"{po_id} dispatched → outbox" if ok else f"{po_id} queued (offline/sent)",
                   error=not ok)
        self.refresh()

    def refresh(self):
        import json
        rows = self.svc.list_drafts()
        self.tbl.setRowCount(len(rows))
        for i, po in enumerate(rows):
            units = sum(it["qty"] for it in json.loads(po["payload"] or "[]"))
            for j, v in enumerate([po["po_id"], po["supplier_id"], str(units), po["status"]]):
                self.tbl.setItem(i, j, QTableWidgetItem(v))
            self.tbl.setCellWidget(i, 4, self._action(po))


# ----------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, conn, db_path):
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
        self.pages = {"Dashboard": DashboardPage(conn, db_path, self.toast),
                      "Checkout": CheckoutPage(conn, self.toast),
                      "Products": ProductsPage(conn, self.toast),
                      "Inventory": InventoryPage(conn, self.toast),
                      "Vendors": VendorsPage(conn, self.toast),
                      "Receiving": ReceivingPage(conn, self.toast),
                      "Procurement": ProcurementPage(conn, self.toast)}
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
    win = MainWindow(conn, args.db); win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
