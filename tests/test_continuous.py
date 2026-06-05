"""Phase 6 tests: the hybrid machinery is VALIDATED (not demonstrated on M5).

We prove: the online corrector learns/removes a moving residual; the drift detector fires on
injected drift and stays quiet on a stationary stream; the registry promotes only on a
frontier-metric improvement beyond the margin.
"""
import numpy as np

from src.continuous.drift import DriftMonitor, SegmentDriftMonitors
from src.continuous.registry import ModelVersion, Registry
from src.continuous.retrain import decide_retrain
from src.models.online_layer import EWMALevelCorrector, OnlineResidualCorrector


# --- online residual corrector -----------------------------------------------

def test_online_corrector_reduces_error_on_biased_stream():
    rng = np.random.default_rng(0)
    c = OnlineResidualCorrector(lr=0.05)
    base = 5.0
    err_base, err_corr = 0.0, 0.0
    for t in range(800):
        feat = {"level": base, "dow": float(t % 7)}
        actual = 10.0 + rng.normal(0, 0.5)        # base is biased low by ~5
        corr = c.predict_one(feat)
        if t >= 400:                               # measure after it has learned
            err_base += abs(actual - base)
            err_corr += abs(actual - (base + corr))
        c.learn_one(feat, actual - base)
    assert err_corr < 0.5 * err_base               # correction roughly halves error or better


def test_ewma_corrector_tracks_level():
    c = EWMALevelCorrector(halflife=5)
    for _ in range(200):
        c.learn_one(3.0)
    assert abs(c.predict_one() - 3.0) < 0.1


def test_adjust_keeps_quantiles_nonneg_and_sorted():
    c = OnlineResidualCorrector()
    for _ in range(50):
        c.learn_one({"x": 1.0}, -100.0)            # push a large negative correction
    out = c.adjust({"pred_q50": 2.0, "pred_q90": 4.0, "pred_q95": 6.0}, {"x": 1.0})
    vals = list(out.values())
    assert all(v >= 0 for v in vals) and vals == sorted(vals)


# --- drift detection ---------------------------------------------------------

def test_drift_fires_after_injected_shift():
    m = DriftMonitor("adwin")
    rng = np.random.default_rng(1)
    for _ in range(400):
        m.update(rng.normal(0, 0.1))               # stationary
    pre = len(m.alarms)
    for _ in range(400):
        m.update(rng.normal(8, 0.1))               # clear regime shift
    assert pre == 0                                 # quiet while stationary
    assert len(m.alarms) >= 1                       # fires after the shift
    assert m.alarms[-1] > 400


def test_segment_monitors_report_only_drifting_segment():
    segs = SegmentDriftMonitors(["A", "B"], "adwin")
    fired_any = []
    for t in range(600):
        a = 0.0 if t < 300 else 9.0                # A drifts mid-stream
        fired_any += segs.update({"A": a, "B": 0.0})
    assert "A" in fired_any and "B" not in fired_any


# --- champion / challenger registry -----------------------------------------

def test_registry_promotes_only_on_frontier_improvement():
    r = Registry()
    assert r.consider(ModelVersion("v1", score=0.05)) is True      # first champion
    assert r.consider(ModelVersion("v2", score=0.051)) is False    # within margin -> keep
    assert r.consider(ModelVersion("v3", score=0.10)) is True      # clear win -> promote
    assert r.champion.name == "v3"


def test_registry_rejects_metric_mismatch():
    r = Registry()
    r.consider(ModelVersion("v1", score=0.05, metric="frontier_inventory_saved"))
    import pytest
    with pytest.raises(ValueError):
        r.consider(ModelVersion("bad", score=0.9, metric="wape"))


# --- retrain trigger logic ---------------------------------------------------

def test_decide_retrain_triggers():
    assert decide_retrain(False, []).should_retrain is False
    assert decide_retrain(True, []).scope == "global"              # scheduled
    assert decide_retrain(False, ["intermittent"]).scope == "intermittent"
    assert decide_retrain(False, ["A", "B", "C"]).scope == "global"  # widespread -> global
