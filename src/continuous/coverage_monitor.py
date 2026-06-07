"""Tail-coverage & magnitude-drift monitors (Phase 6/10) — the guard for the blind window.

The hazard (verified in test_continuous.py): during a MAGNITUDE surge the online layer adapts the
level instantly, so point error / MAE / the median all look healthy — while P95 coverage quietly
collapses (0.95 -> ~0.84) and the reorder engine under-buffers the very spike that most needs
protection. The drift detector fires only after a latency, then the base retrain lags on top.

So tail coverage must be a FIRST-CLASS production signal, watched directly (not inferred from
point error), and a magnitude surge must trigger a protective widened buffer (see
src/reorder/protective.py) to bridge the window until the retrain lands.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.config import CONFIG

# Coverage target bands (config has P90/P95; P99 defaulted to a tight upper band).
_CFG_COV = CONFIG.metrics["forecast_accuracy_targets"]["coverage"]
DEFAULT_BANDS = {
    "pred_q90": tuple(_CFG_COV["p90"]),
    "pred_q95": tuple(_CFG_COV["p95"]),
    "pred_q99": (0.97, 0.999),
}


class RollingCoverageMonitor:
    """Rolling empirical coverage of each quantile head — the direct calibration signal.

        m = RollingCoverageMonitor()
        m.update(actual, {"pred_q95": 9.0, ...})
        m.breaches()   # quantiles whose rolling coverage has fallen BELOW band (under-buffering)
    """

    def __init__(self, bands: dict[str, tuple] | None = None, window: int = 200, min_n: int = 60):
        self.bands = bands or DEFAULT_BANDS
        self.min_n = min_n
        self.hits = {q: deque(maxlen=window) for q in self.bands}

    def update(self, y: float, preds: dict[str, float]) -> "RollingCoverageMonitor":
        for q in self.bands:
            if q in preds:
                self.hits[q].append(1.0 if y <= preds[q] else 0.0)
        return self

    def coverage(self) -> dict[str, float]:
        return {q: float(np.mean(h)) for q, h in self.hits.items() if len(h) >= self.min_n}

    def breaches(self) -> list[str]:
        """Quantiles UNDER-covering (coverage below the band floor) — the dangerous direction."""
        out = []
        for q, cov in self.coverage().items():
            if cov < self.bands[q][0]:
                out.append(q)
        return out

    def healthy(self) -> bool:
        return not self.breaches()


class MagnitudeShiftMonitor:
    """Early-warning for an upward LEVEL surge (festival / salary-day): recent mean rising well
    above a longer reference mean. Fires before coverage fully collapses, so the protective buffer
    can engage proactively rather than only after the tail breach is measurable."""

    def __init__(self, ref_window: int = 120, recent_window: int = 14, threshold: float = 0.5,
                 min_ref: int = 30):
        self.ref = deque(maxlen=ref_window)
        self.recent = deque(maxlen=recent_window)
        self.threshold = threshold
        self.min_ref = min_ref

    def update(self, y: float) -> "MagnitudeShiftMonitor":
        self.ref.append(float(y))
        self.recent.append(float(y))
        return self

    def surge_ratio(self) -> float:
        if len(self.ref) < self.min_ref or not self.recent:
            return 1.0
        ref_mean = max(float(np.mean(self.ref)), 1e-6)
        return float(np.mean(self.recent)) / ref_mean

    def surging(self) -> bool:
        return self.surge_ratio() > 1.0 + self.threshold
