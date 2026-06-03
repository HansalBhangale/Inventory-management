"""Forecast-accuracy metrics (Phase 8.2). All operate on numpy arrays.

Primary: WAPE (point) + pinball/coverage (distribution). MASE is the credibility gate.
See config/metrics.yaml for targets and Appendix B for formulas.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def wape(y, yhat) -> float:
    y, yhat = _arr(y), _arr(yhat)
    return float(np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + EPS))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y, yhat) -> float:
    y, yhat = _arr(y), _arr(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y, yhat) -> float:
    """Signed mean error (yhat - y), normalized by mean demand -> relative bias (MPE-like)."""
    y, yhat = _arr(y), _arr(yhat)
    return float(np.mean(yhat - y) / (np.mean(y) + EPS))


def smape(y, yhat) -> float:
    y, yhat = _arr(y), _arr(yhat)
    return float(np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + EPS)))


def pinball(y, yhat_q, q: float) -> float:
    """Pinball (quantile) loss for a single quantile q."""
    y, yhat_q = _arr(y), _arr(yhat_q)
    d = y - yhat_q
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, yhat_q) -> float:
    """Empirical P(y <= yhat_q) — should approach the nominal quantile if calibrated."""
    y, yhat_q = _arr(y), _arr(yhat_q)
    return float(np.mean(y <= yhat_q))


def seasonal_naive_scale(y_train, m: int = 7) -> float:
    """In-sample MAE of the m-seasonal naive — the MASE denominator."""
    y = _arr(y_train)
    if len(y) <= m:
        return EPS
    return float(np.mean(np.abs(y[m:] - y[:-m])) + EPS)


def mase(y, yhat, scale: float) -> float:
    """MASE = MAE(model) / in-sample seasonal-naive MAE. < 1 means beating naive."""
    return float(mae(y, yhat) / (scale + EPS))


def all_point_metrics(y, yhat, scale: float | None = None) -> dict:
    out = {
        "wape": wape(y, yhat),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "bias": bias(y, yhat),
        "smape": smape(y, yhat),
    }
    if scale is not None:
        out["mase"] = mase(y, yhat, scale)
    return out
