"""Safety stock (Phase 7.3) — two routes.

Route B (preferred, used for the quantile model): the safety stock is implicit in the learned
demand distribution. Demand over the protection period at the target service quantile already
includes the buffer, so SS = q-quantile(demand over P) − E[demand over P].

Route A (formula, used for point baselines that lack quantiles): assume normal errors,
SS = z(q) · sqrt(L·σ_d² + d̄²·σ_L²), combining demand and lead-time variability.
"""
from __future__ import annotations

import math

from scipy.stats import norm


def ss_empirical(quantile_demand_over_P: float, mean_demand_over_P: float) -> float:
    """Route B: buffer beyond expected demand, taken from the learned distribution."""
    return max(0.0, quantile_demand_over_P - mean_demand_over_P)


def ss_formula(q: float, sigma_d: float, lead_mean: float,
               d_bar: float, sigma_L: float) -> float:
    """Route A: z(q)·sqrt(L·σ_d² + d̄²·σ_L²) — demand + lead-time variability."""
    z = norm.ppf(q)
    var = lead_mean * sigma_d ** 2 + (d_bar ** 2) * (sigma_L ** 2)
    return max(0.0, z * math.sqrt(max(var, 0.0)))
