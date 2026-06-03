"""Baseline forecasts (Phase 5.1) — the bar every model must clear.

For a direct H-step (H=14) forecast, the values known at the origin t-H give us:
  - seasonal_naive : y_{t-14}  (= lag_14; 14 is a multiple of the weekly period 7)
  - naive          : last known level = lag_14 as well at this horizon
  - moving_average : 28-day rolling mean ending at the origin (= roll_mean_28)

These are already columns in the feature panel, so baselines are zero-cost to evaluate
and give us the MASE denominator and Forecast Value Add reference.
"""
from __future__ import annotations

import pandas as pd

SEASONAL_NAIVE_COL = "lag_14"
MOVING_AVG_COL = "roll_mean_28"


def baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["seasonal_naive"] = df[SEASONAL_NAIVE_COL].fillna(0).clip(lower=0)
    out["moving_average"] = df[MOVING_AVG_COL].fillna(0).clip(lower=0)
    return out
