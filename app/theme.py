"""Professional dark theme (QSS) + small animation helpers for the POS GUI.

Kept separate so pages stay logic-light. Animations are deliberately subtle (fast fades, hover
states) — a register must feel responsive, never flashy or slow.
"""
from __future__ import annotations

ACCENT = "#2dd4bf"      # teal
BG = "#0f172a"          # slate-900
PANEL = "#1e293b"       # slate-800
PANEL2 = "#334155"      # slate-700
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DANGER = "#f87171"
OK = "#34d399"

STYLESHEET = f"""
* {{ font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 14px; color: {TEXT}; }}
QWidget {{ background: {BG}; }}
QLabel#h1 {{ font-size: 22px; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#total {{ font-size: 26px; font-weight: 800; color: {ACCENT}; }}

/* Sidebar */
QWidget#sidebar {{ background: {PANEL}; }}
QPushButton#nav {{
    text-align: left; padding: 12px 18px; border: none; border-radius: 10px;
    color: {MUTED}; font-size: 15px; margin: 2px 10px;
}}
QPushButton#nav:hover {{ background: {PANEL2}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {ACCENT}; color: #06281f; font-weight: 700; }}

/* Cards / panels */
QFrame#card {{ background: {PANEL}; border-radius: 14px; }}

/* Inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {PANEL2}; border: 1px solid #475569; border-radius: 8px; padding: 8px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}

/* Buttons */
QPushButton {{ background: {PANEL2}; border: none; border-radius: 9px; padding: 9px 16px; }}
QPushButton:hover {{ background: #41506a; }}
QPushButton#primary {{ background: {ACCENT}; color: #06281f; font-weight: 700; }}
QPushButton#primary:hover {{ background: #5eead4; }}
QPushButton#danger {{ background: transparent; color: {DANGER}; border: 1px solid {DANGER}; }}
QPushButton#danger:hover {{ background: rgba(248,113,113,0.12); }}

/* Tables */
QTableWidget, QListWidget {{
    background: {PANEL}; border: none; border-radius: 12px; gridline-color: #334155;
}}
QHeaderView::section {{
    background: {PANEL2}; color: {MUTED}; padding: 8px; border: none; font-weight: 600;
}}
QTableWidget::item {{ padding: 6px; }}
QTableWidget::item:selected {{ background: {PANEL2}; color: {TEXT}; }}

/* Toast */
QLabel#toast {{
    background: {OK}; color: #053026; border-radius: 10px; padding: 12px 20px; font-weight: 700;
}}
QLabel#toastErr {{ background: {DANGER}; color: #3b0a0a; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {PANEL2}; border-radius: 5px; }}
"""


def fade_in(widget, duration: int = 180):
    """Quick opacity fade-in for a widget (used on page switch)."""
    from PySide6.QtCore import QPropertyAnimation
    from PySide6.QtWidgets import QGraphicsOpacityEffect
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.start(QPropertyAnimation.DeleteWhenStopped)
    widget._fade_anim = anim          # keep a ref so it isn't GC'd mid-animation
    return anim
