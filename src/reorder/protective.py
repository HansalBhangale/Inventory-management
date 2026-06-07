"""Protective widened-buffer fallback (Phase 7/6) — bridges the magnitude-drift blind window.

When demand magnitude surges, the global model's learned quantiles are stale-narrow (they haven't
re-widened) and the online layer can only shift level, not spread — so P95 coverage collapses and
the reorder engine under-buffers the spike. Until the drift-triggered base retrain restores the
tail, we widen the buffer from RECENT OBSERVED VOLATILITY (which already reflects the surge):

    protective_quantile(q) = recent_mean + z(q) * recent_std    (Gaussian approx, Route A)

The served quantile is then max(model_quantile, protective_quantile) — a floor that never lets the
buffer be narrower than recent volatility justifies. This is deliberately conservative: it can
slightly OVER-stock during the window, which is the correct error to make on a festival spike.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import norm


class ProtectiveBuffer:
    """Rolling recent-volatility quantile estimator used as a buffer floor during surges."""

    def __init__(self, window: int = 28, min_n: int = 7):
        self.buf: deque[float] = deque(maxlen=window)
        self.min_n = min_n

    def update(self, y: float) -> "ProtectiveBuffer":
        self.buf.append(float(y))
        return self

    def ready(self) -> bool:
        return len(self.buf) >= self.min_n

    def quantiles(self, nominal: dict[str, float]) -> dict[str, float]:
        """nominal: {colname: quantile level}. Returns protective quantile values."""
        a = np.asarray(self.buf, dtype="float64")
        m = float(a.mean())
        s = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        return {c: max(0.0, m + norm.ppf(q) * s) for c, q in nominal.items()}


def guarded_quantiles(model_quantiles: dict[str, float],
                      protective: ProtectiveBuffer,
                      nominal: dict[str, float],
                      engage: bool) -> dict[str, float]:
    """Compose the served quantiles: model output, floored by the protective buffer when a
    magnitude surge / tail-coverage breach is active. Returns a non-crossing (sorted) dict."""
    q = dict(model_quantiles)
    if engage and protective.ready():
        prot = protective.quantiles(nominal)
        q = {c: max(q.get(c, 0.0), prot.get(c, 0.0)) for c in q}
    cols = list(q)
    vals = np.sort(np.clip([q[c] for c in cols], 0, None))
    return dict(zip(cols, vals))
