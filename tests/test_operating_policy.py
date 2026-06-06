"""Phase 8 tests: segmented operating-policy router logic (pure functions)."""
import numpy as np
import pandas as pd

from src.evaluate import operating_policy as OPm


def _slice(method, q, fill_doh):
    """Build a tiny sim-results slice: one series per (method,q) hitting (served,demanded,inv)."""
    rows = []
    for (m, qq, served, demanded, inv) in fill_doh:
        rows.append({"store_id": "S", "sku_id": f"{m}{int(qq*100)}", "abc": "B",
                     "intermittency": "lumpy", "price": 1.0, "method": m, "q": qq,
                     "served": served, "demanded": demanded, "avg_inv": inv,
                     "avg_daily_demand": demanded / 10, "lost_units": demanded - served})
    return pd.DataFrame(rows)


def test_split_is_deterministic_and_disjoint():
    df = pd.DataFrame({"store_id": ["S"] * 6,
                       "sku_id": [f"x{i}" for i in range(6)], "regime": "base"})
    a1, b1 = OPm._split(df)
    a2, b2 = OPm._split(df)
    assert set(a1["sku_id"]) == set(a2["sku_id"])          # deterministic
    assert set(a1["sku_id"]).isdisjoint(set(b1["sku_id"]))  # disjoint
    assert len(a1) + len(b1) == len(df)                     # exhaustive


def test_decide_routing_keeps_lgbm_on_tie():
    # lgbm and naive at the SAME (fill, inv) -> TIE -> must KEEP lgbm (not route to naive).
    df = _slice("x", 0, [])
    pts = [("lgbm", 0.95, 9.5, 10, 5.0),
           ("seasonal_naive", 0.5, 8.0, 10, 4.0),
           ("seasonal_naive", 0.95, 9.5, 10, 5.0),
           ("seasonal_naive", 0.99, 9.9, 10, 7.0)]
    df = pd.DataFrame([{"store_id": "S", "sku_id": "k", "abc": "B", "intermittency": "lumpy",
                        "price": 1.0, "method": m, "q": q, "served": s, "demanded": d,
                        "avg_inv": inv, "avg_daily_demand": d / 10, "lost_units": d - s}
                       for (m, q, s, d, inv) in pts])
    routing = OPm.decide_routing(df)
    assert routing["B"] == "lgbm"     # TIE -> keep lgbm, never forfeit to naive


def test_decide_routing_routes_naive_on_loss():
    # lgbm strictly worse: same fill, MORE inventory than naive -> LOSS -> route to naive.
    pts = [("lgbm", 0.95, 9.5, 10, 9.0),               # fill .95 but inv 9 (expensive)
           ("seasonal_naive", 0.5, 8.0, 10, 3.0),
           ("seasonal_naive", 0.95, 9.5, 10, 5.0),     # naive hits .95 fill at inv 5
           ("seasonal_naive", 0.99, 9.9, 10, 7.0)]
    df = pd.DataFrame([{"store_id": "S", "sku_id": "k", "abc": "B", "intermittency": "lumpy",
                        "price": 1.0, "method": m, "q": q, "served": s, "demanded": d,
                        "avg_inv": inv, "avg_daily_demand": d / 10, "lost_units": d - s}
                       for (m, q, s, d, inv) in pts])
    assert OPm.decide_routing(df)["B"] == "seasonal_naive"
