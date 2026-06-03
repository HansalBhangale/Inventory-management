"""Phase 5/8 tests: forecast metrics correctness (pure functions, no data needed)."""
import numpy as np

from src.evaluate import forecast_metrics as M


def test_perfect_forecast():
    y = np.array([0, 1, 5, 10, 3])
    assert M.wape(y, y) == 0
    assert M.mae(y, y) == 0
    assert M.rmse(y, y) == 0
    assert abs(M.bias(y, y)) < 1e-9


def test_wape_known_value():
    y = np.array([10.0, 10.0])
    yhat = np.array([8.0, 13.0])  # errors 2 + 3 = 5 over sum 20
    assert abs(M.wape(y, yhat) - 0.25) < 1e-9


def test_bias_sign():
    y = np.array([5.0, 5.0])
    assert M.bias(y, np.array([7.0, 7.0])) > 0   # over-forecast
    assert M.bias(y, np.array([3.0, 3.0])) < 0   # under-forecast


def test_pinball_asymmetry():
    # For q=0.9, under-prediction is penalized ~9x more than over-prediction.
    y = np.array([10.0])
    under = M.pinball(y, np.array([9.0]), 0.9)   # yhat below y
    over = M.pinball(y, np.array([11.0]), 0.9)   # yhat above y
    assert abs(under - 0.9) < 1e-9
    assert abs(over - 0.1) < 1e-9
    assert under > over


def test_coverage():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat_q = np.array([2.0, 2.0, 2.0, 2.0])      # y<=2 for 2 of 4
    assert abs(M.coverage(y, yhat_q) - 0.5) < 1e-9


def test_mase_vs_naive():
    # Model equal to seasonal-naive scale -> MASE ~ 1.
    y_train = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0])  # period-2 swing
    scale = M.seasonal_naive_scale(y_train, m=2)
    assert scale > 0
    y = np.array([2.0, 2.0]); yhat = np.array([2.0, 2.0])
    assert M.mase(y, yhat, scale) == 0
