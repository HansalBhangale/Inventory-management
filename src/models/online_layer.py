"""Online adaptation layer (Phase 6.3) — the FAST timescale of the hybrid.

A lightweight River model updated daily that **corrects the global base model's residuals**
using recent features. It absorbs drift, new-SKU ramp, and local shocks *between* scheduled base
retrains. Final forecast = base_quantile + online_residual_correction (kept non-negative,
quantiles re-sorted).

CAVEAT (M5): a fixed historical extract has no live distribution shift, so this proves the
mechanism (the corrector learns and removes a moving residual) rather than a production lift.
The win only materializes on a real store's evolving stream.
"""
from __future__ import annotations

import numpy as np
from river import linear_model, optim, preprocessing


class OnlineResidualCorrector:
    """Per-stream online regressor predicting (actual − base_forecast) from recent features.

    Usage (daily loop):
        c = OnlineResidualCorrector()
        corr = c.predict_one(features)              # today's residual correction
        q_adj = c.adjust(base_quantiles, features)  # base + corr, clipped, re-sorted
        ... observe actual ...
        c.learn_one(features, actual - base_central)
    """

    def __init__(self, lr: float = 0.03):
        self.model = preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(lr)
        )
        self.n_seen = 0

    def predict_one(self, features: dict) -> float:
        return float(self.model.predict_one(features) or 0.0)

    def learn_one(self, features: dict, residual: float) -> "OnlineResidualCorrector":
        self.model.learn_one(features, float(residual))
        self.n_seen += 1
        return self

    def adjust(self, base_quantiles: dict[str, float], features: dict) -> dict[str, float]:
        """Apply the residual correction to every quantile, clip >=0, re-sort non-crossing."""
        corr = self.predict_one(features)
        cols = list(base_quantiles)
        vals = np.clip(np.array([base_quantiles[c] for c in cols], float) + corr, 0, None)
        vals = np.sort(vals)  # enforce non-crossing after correction
        return dict(zip(cols, vals))


class EWMALevelCorrector:
    """Lightweight alternative (Phase 6.3): EWMA of recent residuals — a per-series adaptive
    bias term for fast movers, no features required."""

    def __init__(self, halflife: float = 7.0):
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.level = 0.0
        self.n_seen = 0

    def predict_one(self, *_a) -> float:
        return self.level

    def learn_one(self, residual: float) -> "EWMALevelCorrector":
        self.level += self.alpha * (float(residual) - self.level)
        self.n_seen += 1
        return self
