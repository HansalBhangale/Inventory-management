"""Phase 8 tests: frontier-dominance gate logic (pure functions)."""
import numpy as np

from src.evaluate import frontier_gate as G


def test_classify_win_tie_loss():
    m = G.WIN_MARGIN
    assert G._classify(m + 0.01) == "WIN"
    assert G._classify(-(m + 0.01)) == "LOSS"
    assert G._classify(0.0) == "TIE"
    assert G._classify(m / 2) == "TIE"


def test_doh_interpolation_within_range():
    f = np.array([0.80, 0.95])
    d = np.array([4.0, 8.0])
    # halfway in fill -> halfway in DOH
    assert abs(G._doh_for_fill(f, d, 0.875) - 6.0) < 1e-9


def test_doh_extrapolates_above_reach():
    # baseline can't reach target fill -> extrapolate upward (not free)
    f = np.array([0.80, 0.90])
    d = np.array([4.0, 6.0])
    doh = G._doh_for_fill(f, d, 0.95)   # slope 20 per unit fill, +0.05 -> +1.0
    assert doh > 6.0 and abs(doh - 7.0) < 1e-6


def test_doh_clamps_below_range():
    f = np.array([0.80, 0.95]); d = np.array([4.0, 8.0])
    assert G._doh_for_fill(f, d, 0.5) == 4.0
