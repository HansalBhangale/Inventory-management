"""Promotion features (Phase 4.F) — declared in config/features.yaml, never actually built.

M5 has no promotions, so this path sat empty. Favorita (and any real kirana store) has an
`on_promo` signal, so we build the family the config promised. Promo CALENDARS are known ahead of
time, so forward-looking promo features (days_to_promo_end, promo_in_next_7d) are legitimately
available at prediction time — unlike demand lags, they don't leak.

    add_promo_features(df)  # df: store_id, sku_id, date, on_promo[, discount_depth]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROMO_FEATURES = ["on_promo", "discount_depth", "days_since_promo_start",
                  "days_to_promo_end", "promo_in_next_7d"]


def add_promo_features(df: pd.DataFrame, horizon_safe: bool = True) -> pd.DataFrame:
    """Add the promo feature family from a daily `on_promo` flag per (store, sku).

    If `on_promo` is absent (e.g. M5), all promo features default to 0 — the path runs and stays
    inert rather than silently being skipped. With real promo data it populates.
    """
    out = df.copy()
    if "on_promo" not in out.columns:
        for f in PROMO_FEATURES:
            out[f] = 0.0
        return out

    out["on_promo"] = out["on_promo"].fillna(0).astype(int)
    if "discount_depth" not in out.columns:
        out["discount_depth"] = 0.0
    out["discount_depth"] = out["discount_depth"].fillna(0.0).astype(float)
    out = out.sort_values(["store_id", "sku_id", "date"])

    def _per_series(g: pd.DataFrame) -> pd.DataFrame:
        p = g["on_promo"].to_numpy()
        n = len(p)
        # rising edge = promo start; days since the most recent start (0 on the start day)
        start = (p == 1) & (np.r_[0, p[:-1]] == 0)
        since = np.full(n, 0)
        last = -1
        for i in range(n):
            if start[i]:
                last = i
            since[i] = (i - last) if last >= 0 else 999
        g["days_since_promo_start"] = since
        # days until the current promo ends (0 if not on promo); forward scan
        to_end = np.zeros(n, dtype=int)
        run_end = -1
        for i in range(n - 1, -1, -1):
            if p[i] == 1:
                if run_end < i:
                    run_end = i  # find end of this run
                    j = i
                    while j + 1 < n and p[j + 1] == 1:
                        j += 1
                    run_end = j
                to_end[i] = run_end - i
            else:
                run_end = -1
        g["days_to_promo_end"] = to_end
        # promo in the next 7 days (calendar known ahead -> not leakage)
        nxt = np.zeros(n, dtype=int)
        for i in range(n):
            nxt[i] = int(p[i + 1:i + 8].max()) if i + 1 < n else 0
        g["promo_in_next_7d"] = nxt
        return g

    out = out.groupby(["store_id", "sku_id"], group_keys=False, observed=True).apply(_per_series)
    return out
