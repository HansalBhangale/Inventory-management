"""M3 — wire the forecasting engine to the shop's OWN data.

Runs the daily loop: SQLite -> bridge + data-contract gate -> per-SKU demand quantiles -> the
validated REORDER DECISION machinery -> write recommendations back to SQLite. No UI (M4 surfaces
them); confirm the numbers are sane in the DB first, exactly as the shadow discipline demands.

What's genuinely "the engine" here — and validated in Phases 7–8 — is the *decision* layer:
  - REAL supplier lead times from PO->GRN (receipts), not assumed (the input every proxy lacked);
  - safety stock from the demand quantiles (Route B), protection period P = lead + review;
  - the perishable newsvendor branch (critical fractile, capped to shelf-life demand);
  - pack-size / MOQ rounding.
The forecast SOURCE is pluggable (`forecaster`). Default = per-SKU empirical recent quantiles,
which is the right tool for a single shop's volume; the global LightGBM is the multi-store/scale
path and drops into the same seam once a shop has rich history.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import CONFIG
from src.pos.bridge import export_and_validate, flatten_sales
from src.pos.catalog import ProductService, SupplierService
from src.pos.inventory import InventoryService
from src.pos.receiving import ReceiptService
from src.pos.schema import connect
from src.reorder.newsvendor import newsvendor_order, quantile_interpolator
from src.reorder.policy import round_order

QUANTILES = (0.5, 0.9, 0.95, 0.99)
DEFAULT_SERVICE_Q = 0.95
REVIEW_DAYS = 1


def empirical_quantiles(sales: pd.DataFrame, as_of: date, lookback: int = 28) -> dict[str, dict]:
    """Per-SKU daily-demand quantiles from the recent window (incl. zero-sale days).

    Robust from day one for a single shop. Returns {sku_id: {q50,q90,q95,q99,dbar,sigma}}."""
    if sales.empty:
        return {}
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.date
    lo = as_of - timedelta(days=lookback)
    s = s[(s["date"] > lo) & (s["date"] <= as_of)]
    out: dict[str, dict] = {}
    span_days = max((as_of - lo).days, 1)
    for sku, g in s.groupby("sku_id"):
        daily = g.groupby("date")["qty"].sum().clip(lower=0)
        # include zero-demand days so quantiles reflect intermittency
        series = np.zeros(span_days)
        for i, day in enumerate(pd.date_range(lo + timedelta(days=1), as_of)):
            series[i] = float(daily.get(day.date(), 0.0))
        out[str(sku)] = {
            "q50": float(np.quantile(series, 0.50)),
            "q90": float(np.quantile(series, 0.90)),
            "q95": float(np.quantile(series, 0.95)),
            "q99": float(np.quantile(series, 0.99)),
            "dbar": float(series.mean()),
            "sigma": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        }
    return out


def _lead_time_days(conn: sqlite3.Connection) -> tuple[dict[str, float], float]:
    """Per-supplier mean lead time: REAL PO->GRN where available, else declared default, else the
    config fallback. Returns (per_supplier_mean, global_default)."""
    default = float(CONFIG.policy["lead_time"]["default_days"])
    per: dict[str, list[float]] = {}
    for s in ReceiptService(conn).lead_time_samples():       # real receipts vs PO order date
        per.setdefault(s["supplier_id"], []).append(s["lead_days"])
    mean = {k: float(np.mean(v)) for k, v in per.items() if v}
    for sup in SupplierService(conn).list():                 # fill declared defaults
        sid = sup["supplier_id"]
        if sid not in mean and sup.get("default_lead_time_days"):
            mean[sid] = float(sup["default_lead_time_days"])
    return mean, default


def _reason(sku, branch, P, on_hand, s, qty, q95) -> str:
    return (f"{branch}: on-hand {on_hand:.0f} vs reorder point {s:.0f} "
            f"(P95 daily demand {q95:.1f} over {P}-day protection); order {qty}.")


def run_recommendations(db_path, store_id: str = "SHOP01", *, lookback: int = 28,
                        service_q: float = DEFAULT_SERVICE_Q, run_date: str | None = None,
                        forecaster=empirical_quantiles) -> dict:
    """Produce reorder recommendations from the shop's own data and persist them. Returns a
    summary {n, ordered, run_date, contract}. Raises DataContractError if the batch is BLOCKED."""
    conn = connect(db_path)
    contract = export_and_validate(db_path, raise_on_block=True)   # gate: bad data never scores

    sales = flatten_sales(conn)
    as_of = (pd.to_datetime(sales["date"]).dt.date.max() if not sales.empty
             else datetime.now().date())
    run_date = run_date or str(as_of)
    qmap = forecaster(sales, as_of, lookback)
    lead_mean, lead_default = _lead_time_days(conn)
    suppliers = {s["supplier_id"]: s for s in SupplierService(conn).list()}
    inv = InventoryService(conn, store_id)
    z = max(0.0, float(norm.ppf(service_q)))

    rows, ordered = [], 0
    for p in ProductService(conn).list():
        sku = p["sku_id"]
        q = qmap.get(sku, {"q50": 0, "q90": 0, "q95": 0, "q99": 0, "dbar": 0.0, "sigma": 0.0})
        on_hand = inv.on_hand(sku)
        sup = suppliers.get(p.get("primary_supplier_id"))
        L = lead_mean.get(p.get("primary_supplier_id"), lead_default)
        P = int(round(L)) + REVIEW_DAYS
        moq = int(sup["moq"]) if sup and sup.get("moq") else 1
        pack = int(p.get("pack_size") or 1)

        if p.get("perishable") and p.get("sell_price") and p.get("unit_cost") is not None:
            # newsvendor: stock to the critical fractile, capped to what sells before spoiling
            qfun = quantile_interpolator({0.5: q["q50"], 0.9: q["q90"], 0.95: q["q95"], 0.99: q["q99"]})
            shelf_demand = q["dbar"] * (p.get("shelf_life_days") or P)
            o = newsvendor_order(qfun, p["sell_price"], p["unit_cost"], shelf_demand, on_hand, moq, pack)
            order_qty, s_lvl, branch = o.order_qty, o.target_quantile_units, "newsvendor"
        else:
            mean_P = q["dbar"] * P
            buffer = z * q["sigma"] * math.sqrt(P)
            s_lvl = mean_P + buffer                              # reorder point (Route B safety stock)
            S = s_lvl + q["dbar"] * REVIEW_DAYS                  # order-up-to
            order_qty = round_order(S - on_hand, moq, pack) if on_hand <= s_lvl else 0
            branch = "(s,S)"

        should = int(order_qty > 0)
        ordered += should
        rows.append((sku, run_date, q["q50"], q["q90"], q["q95"], q["q99"], should, int(order_qty),
                     round(float(s_lvl), 2), _reason(sku, branch, P, on_hand, s_lvl, int(order_qty), q["q95"]),
                     "pending"))

    with conn:
        conn.execute("DELETE FROM recommendations WHERE run_date = ?", (run_date,))
        conn.executemany(
            "INSERT INTO recommendations (sku_id, run_date, p50, p90, p95, p99, should_order, "
            "order_qty, reorder_point, reason, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.close()
    return {"n": len(rows), "ordered": ordered, "run_date": run_date, "contract": contract}
