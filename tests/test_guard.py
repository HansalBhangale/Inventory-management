"""Phase 6/7 tests: the magnitude-drift guard closes the blind window.

Verifies the hazard is caught and bridged: (1) the rolling coverage monitor flags the silent P95
collapse that point-error metrics miss; (2) the magnitude monitor fires on an upward surge and is
quiet when stationary; (3) the protective recent-volatility buffer restores P95 coverage during
the stale-base window — the festival/salary-spike case a kirana store lives or dies on.
"""
import numpy as np
from scipy.stats import poisson

from src.continuous.coverage_monitor import MagnitudeShiftMonitor, RollingCoverageMonitor
from src.models.online_layer import OnlineResidualCorrector
from src.reorder.protective import ProtectiveBuffer, guarded_quantiles

L0, L1 = 5.0, 15.0
BASE = {"pred_q90": poisson.ppf(.9, L0), "pred_q95": poisson.ppf(.95, L0),
        "pred_q99": poisson.ppf(.99, L0)}
NOMINAL = {"pred_q90": .9, "pred_q95": .95, "pred_q99": .99}


def _magnitude_stream(T=8000, seed=7):
    rng = np.random.default_rng(seed)
    for t in range(T):
        yield t, float(rng.poisson(L0 if t < T // 2 else L1))


def test_coverage_monitor_flags_silent_tail_collapse():
    # Stale base + online level correction: MAE looks fine but P95 must be flagged under-covering.
    c = OnlineResidualCorrector(lr=0.05)
    cov = RollingCoverageMonitor(window=150, min_n=50)
    for t, a in _magnitude_stream():
        q = c.adjust(BASE, {"dow": float(t % 7)})
        cov.update(a, q)
        c.learn_one({"dow": float(t % 7)}, a - L0)
    assert "pred_q95" in cov.breaches()                 # the silent collapse is caught
    assert cov.coverage()["pred_q95"] < 0.93


def test_magnitude_monitor_fires_on_surge_quiet_when_stationary():
    rng = np.random.default_rng(1)
    m = MagnitudeShiftMonitor(ref_window=120, recent_window=14, threshold=0.5)
    for _ in range(200):
        m.update(5 + rng.normal(0, 0.3))
    assert not m.surging()                               # quiet at steady level
    fired = False
    for _ in range(20):
        m.update(15.0)                                   # upward surge vs recent baseline
        fired = fired or m.surging()
    assert fired


def test_protective_guard_restores_p95_coverage():
    c = OnlineResidualCorrector(lr=0.05)
    prot = ProtectiveBuffer(window=28)
    cov = RollingCoverageMonitor(window=150, min_n=50)
    g95 = u95 = n = 0
    for t, a in _magnitude_stream():
        feat = {"dow": float(t % 7)}
        q = c.adjust(BASE, feat)
        cov.update(a, q)                                 # monitor the BASE calibration
        served = guarded_quantiles(q, prot, NOMINAL, engage=bool(cov.breaches()))
        if t >= 6400:                                    # post-shift, post-adaptation
            u95 += a <= q["pred_q95"]
            g95 += a <= served["pred_q95"]
            n += 1
        c.learn_one(feat, a - L0)
        prot.update(a)
    assert u95 / n < 0.90                                # unguarded under-covers (the hazard)
    assert g95 / n >= 0.92                               # guard restores ~P95 band
    assert g95 / n - u95 / n > 0.05                      # and it's a real improvement
