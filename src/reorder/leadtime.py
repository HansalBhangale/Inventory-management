"""Lead-time model (Phase 7.1).

In production: estimate mean L̄ and std σ_L per supplier from PO→GRN history on a rolling
window, falling back to a supplier-declared default when history is thin.

On M5 there is NO PO/GRN data, so we run on ASSUMED lead-time regimes (config/policy.yaml).
Every simulation result must be stamped with the regime it used — a synthetic lead time can
flatter or bury the value of a good upper-tail forecast, so we sweep regimes instead of
picking one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import CONFIG


@dataclass(frozen=True)
class LeadTime:
    name: str
    mean: float
    std: float

    def sample(self, rng: np.random.Generator) -> int:
        """One realized lead time (clipped Normal, >= 1 day)."""
        if self.std <= 0:
            return int(round(self.mean))
        return int(max(1, round(rng.normal(self.mean, self.std))))


def regimes() -> dict[str, LeadTime]:
    cfg = CONFIG.policy["simulation"]["lead_time_regimes"]
    return {k: LeadTime(k, float(v["mean"]), float(v["std"])) for k, v in cfg.items()}


def estimate_from_po_grn(*_args, **_kwargs) -> LeadTime:
    """Placeholder: PO→GRN estimation. Unavailable on M5 — returns the configured default."""
    lt = CONFIG.policy["lead_time"]
    return LeadTime("default", float(lt["default_days"]), float(lt["default_std_days"]))
