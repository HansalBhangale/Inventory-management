"""Reorder policy (Phase 7.4–7.5): convert protection-period demand into (s, S) and an order.

Continuous-review (s, S): when inventory position IP = on_hand + on_order − backorders falls
to/below the reorder point s, order up to S.

  s = demand over protection period P (= lead time + review) at the service quantile  (Route B)
  S = demand over (P + order_cycle) at the service quantile
  order_qty = round_up(S − IP) to MOQ and pack multiples

The reorder point already embeds safety stock because it is an upper quantile of lead-time
demand (see safety_stock.ss_empirical).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def round_order(qty: float, moq: int = 1, pack_size: int = 1) -> int:
    """Round an order up to MOQ then to a whole number of packs."""
    if qty <= 0:
        return 0
    qty = max(qty, moq)
    if pack_size > 1:
        qty = math.ceil(qty / pack_size) * pack_size
    return int(math.ceil(qty))


def reorder_levels(demand_over_P_q: float, demand_over_PC_q: float) -> tuple[float, float]:
    """(s, S) from protection-period and (protection+cycle) demand at the service quantile."""
    s = max(0.0, demand_over_P_q)
    S = max(s, demand_over_PC_q)
    return s, S


@dataclass
class Recommendation:
    should_order: bool
    order_qty: int
    reorder_point: float
    order_up_to: float
    inventory_position: float
    explanation: str


def recommend(inventory_position: float, s: float, S: float,
              moq: int = 1, pack_size: int = 1,
              service_q: float = 0.95, protection_days: float = 4.0) -> Recommendation:
    """Single (store, SKU) PO recommendation with a human-readable explanation (Phase 7.8)."""
    if inventory_position > s:
        return Recommendation(False, 0, s, S, inventory_position,
                              f"IP {inventory_position:.0f} > reorder point {s:.0f}; no order.")
    qty = round_order(S - inventory_position, moq, pack_size)
    expl = (f"IP {inventory_position:.0f} <= reorder point {s:.0f} "
            f"(P{int(service_q*100)} demand over {protection_days:.0f}d protection period); "
            f"ordering up to {S:.0f} -> qty {qty} (MOQ {moq}, pack {pack_size}).")
    return Recommendation(True, qty, s, S, inventory_position, expl)


def dispatch_reorder(*, perishable: bool, inventory_position: float,
                     quantiles: dict[float, float] | None = None,
                     sell_price: float = 0.0, unit_cost: float = 0.0,
                     shelf_life_demand: float = 0.0,
                     s: float = 0.0, S: float = 0.0,
                     moq: int = 1, pack_size: int = 1) -> tuple[str, int]:
    """Route an item to the correct reorder branch and report WHICH branch fired.

    Perishable -> newsvendor (critical fractile, capped to shelf-life demand); everything else
    -> continuous-review (s, S). The returned branch name is the diagnostic that proves the
    perishable path actually engages instead of silently defaulting to (s, S).
    """
    if perishable and quantiles:
        from src.reorder.newsvendor import newsvendor_order, quantile_interpolator
        o = newsvendor_order(quantile_interpolator(quantiles), sell_price, unit_cost,
                             shelf_life_demand, inventory_position, moq, pack_size)
        return "newsvendor", o.order_qty
    return "sS", recommend(inventory_position, s, S, moq, pack_size).order_qty
