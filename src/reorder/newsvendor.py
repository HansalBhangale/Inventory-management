"""Newsvendor / perishable reorder branch (Phase 7.6) — previously declared in config, never built.

For short-shelf-life items the service-level quantile is the WRONG target: over-ordering spoils
stock. The optimal stocking quantile is the critical fractile:

    critical_ratio = Cu / (Cu + Co)
        Cu = underage cost  = lost margin on a stockout      (sell_price - unit_cost)
        Co = overage  cost  = cost of a spoiled unit         (unit_cost)

Order up to the demand quantile equal to critical_ratio, and CAP the order so we never stock more
than can sell within shelf_life_days (anything beyond that spoils). This replaces the per-segment
service quantile for items flagged perishable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.reorder.policy import round_order


def critical_ratio(cu: float, co: float) -> float:
    """Cu / (Cu + Co), clamped to (0, 1). Higher margin-vs-spoilage -> stock more aggressively."""
    cu, co = max(cu, 0.0), max(co, 0.0)
    if cu + co <= 0:
        return 0.5
    return min(max(cu / (cu + co), 1e-6), 1 - 1e-6)


def critical_ratio_from_prices(sell_price: float, unit_cost: float) -> float:
    return critical_ratio(cu=sell_price - unit_cost, co=unit_cost)


@dataclass
class PerishableOrder:
    order_qty: int
    critical_ratio: float
    target_quantile_units: float
    shelf_life_cap: float
    capped: bool


def newsvendor_order(
    demand_quantile,                # callable: fractile in (0,1) -> forecast demand over protection period
    sell_price: float,
    unit_cost: float,
    shelf_life_demand: float,       # expected units that will SELL within shelf_life_days
    inventory_position: float,
    moq: int = 1,
    pack_size: int = 1,
) -> PerishableOrder:
    """Order up to the critical-fractile demand, capped at shelf-life demand, net of stock on hand."""
    cr = critical_ratio_from_prices(sell_price, unit_cost)
    target = float(demand_quantile(cr))
    capped = target > shelf_life_demand
    target = min(target, float(shelf_life_demand))          # don't stock what will spoil
    qty = round_order(max(0.0, target - inventory_position), moq, pack_size)
    return PerishableOrder(qty, cr, float(demand_quantile(cr)), float(shelf_life_demand), capped)


def quantile_interpolator(quantiles: dict[float, float]):
    """Build a demand_quantile(fractile) callable from a sparse set of forecast quantiles
    (e.g. {0.5: 3, 0.9: 7, 0.95: 9, 0.99: 12}) by monotone interpolation."""
    qs = np.array(sorted(quantiles))
    vs = np.array([quantiles[q] for q in qs])
    vs = np.maximum.accumulate(vs)                          # enforce non-crossing

    def f(p: float) -> float:
        return float(np.interp(min(max(p, qs[0]), qs[-1]), qs, vs))

    return f
