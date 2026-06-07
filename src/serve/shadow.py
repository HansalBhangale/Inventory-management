"""Shadow-mode runner (Phase 9) — built to embarrass the model, not rubber-stamp it.

Shadow mode that only logs "here's what we'd recommend" is theatre. This logs, for each
recommendation: what we'd order, what the store actually did (when that feed exists), the
divergence, AND a set of "would a shopkeeper look at this and say 'that's obviously wrong'?"
reject flags. The metric that matters in shadow mode is NOT accuracy — it's the reject-flag rate:
how often the engine proposes something a human would immediately veto.

On M5 there are no real store orders, so the actual-order feed is stubbed; but the divergence and
reject-flag logic are built now, because that is exactly the part a pilot plugs into and the part
that tells you whether the recommendations are sane against real behaviour or quietly insane.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Each recommendation row is expected to carry the decision + the context to sanity-check it.
REQUIRED = ["sku_id", "should_order", "order_qty", "order_up_to", "inventory_position",
            "expected_demand_protection", "moq", "pack_size"]


@dataclass
class ShadowConfig:
    implausible_factor: float = 5.0     # order_qty > factor * order-up-to => implausibly large
    ample_stock_factor: float = 2.0     # ordering while IP > factor * expected demand over P


def reject_flags(row: pd.Series, cfg: ShadowConfig) -> list[str]:
    """Sanity checks a shopkeeper would catch — independent of any real-order feed."""
    flags = []
    q = float(row["order_qty"])
    if row["should_order"] and q <= 0:
        flags.append("order_flagged_but_zero_qty")
    if q > 0:
        if q < float(row["moq"]):
            flags.append("below_moq")
        ps = float(row.get("pack_size", 1) or 1)
        if ps > 1 and abs(q % ps) > 1e-9:
            flags.append("not_pack_multiple")
        if q > cfg.implausible_factor * max(float(row["order_up_to"]), 1.0):
            flags.append("implausibly_large")
    if row["should_order"] and float(row["inventory_position"]) > \
            cfg.ample_stock_factor * max(float(row["expected_demand_protection"]), 1.0):
        flags.append("order_despite_ample_stock")
    if bool(row.get("perishable", False)) and "shelf_life_demand" in row and \
            q > float(row["shelf_life_demand"]) + 1e-9:
        flags.append("exceeds_shelf_life_demand")
    return flags


@dataclass
class ShadowReport:
    n: int
    reject_rate: float
    flag_counts: dict
    divergence: dict | None = None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_shadow(recs: pd.DataFrame, actual_orders: pd.DataFrame | None = None,
               cfg: ShadowConfig | None = None) -> ShadowReport:
    """Score a batch of recommendations in shadow mode. `actual_orders` (sku_id, ordered_qty) is
    optional — STUBBED on M5; when a real feed is present, divergence is computed too."""
    cfg = cfg or ShadowConfig()
    missing = [c for c in REQUIRED if c not in recs.columns]
    if missing:
        raise ValueError(f"shadow recs missing required columns: {missing}")

    out = recs.copy()
    out["reject_flags"] = out.apply(lambda r: reject_flags(r, cfg), axis=1)
    out["rejected"] = out["reject_flags"].map(bool)

    flag_counts: dict[str, int] = {}
    for fl in out["reject_flags"]:
        for f in fl:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    divergence = None
    if actual_orders is not None:
        m = out.merge(actual_orders[["sku_id", "ordered_qty"]], on="sku_id", how="left")
        d = m["order_qty"].to_numpy(float) - m["ordered_qty"].fillna(0).to_numpy(float)
        divergence = {
            "mean_abs_divergence": float(np.mean(np.abs(d))),
            "n_matched": int(m["ordered_qty"].notna().sum()),
            "over_order_rate": float(np.mean(d > 0)),
        }

    return ShadowReport(n=len(out), reject_rate=float(out["rejected"].mean()),
                        flag_counts=flag_counts, divergence=divergence, rows=out)
