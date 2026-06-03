"""Per-segment model routing & blending (Phase 5.7).

The global LightGBM under-fits high-signal A-items (clean weekly seasonality is a hard
naive baseline to beat), and is the wrong tool for intermittent/lumpy series. The router
produces `pred_final` from per-row component predictions:

  - A-items            -> blend  w*lgbm + (1-w)*seasonal_naive   (closes the A gap)
  - intermittent/lumpy -> TSB/Croston when available, else seasonal_naive fallback
  - everything else    -> lgbm (the global model)

Components expected as columns: pred_central (lgbm), seasonal_naive, optional pred_tsb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG

_ROUTING = CONFIG.model.get("routing", {})
A_BLEND_WEIGHT = float(_ROUTING.get("a_blend_weight", 1.0))
INTERMITTENT_MODEL = str(_ROUTING.get("intermittent_model", "lgbm"))  # lgbm|tsb|sba
INTERMITTENT = {"intermittent", "lumpy"}

# Evidence (CA_1, per-SKU MASE): on M5 grocery the global LGBM beats both the A-blend and
# TSB in every segment (intermittent items keep day-of-week structure that a flat TSB rate
# discards, and the MASE baseline is seasonal-naive). Defaults below therefore keep LGBM
# everywhere; switch intermittent_model/a_blend_weight in config to re-test on more data.


def apply_routing(
    df: pd.DataFrame,
    a_blend_weight: float = A_BLEND_WEIGHT,
    intermittent_model: str = INTERMITTENT_MODEL,
) -> pd.Series:
    lgbm = df["pred_central"].astype("float64").clip(lower=0)
    naive = df["seasonal_naive"].astype("float64").fillna(0).clip(lower=0)
    pred = lgbm.copy()

    if a_blend_weight < 1.0:
        is_a = df["abc"].astype("string") == "A"
        pred[is_a] = a_blend_weight * lgbm[is_a] + (1 - a_blend_weight) * naive[is_a]

    if intermittent_model != "lgbm" and "pred_tsb" in df.columns:
        is_int = df["intermittency"].astype("string").isin(INTERMITTENT)
        pred[is_int] = df.loc[is_int, "pred_tsb"].astype("float64").clip(lower=0)

    return pred.clip(lower=0)


def blend(lgbm, naive, w: float) -> np.ndarray:
    lgbm = np.clip(np.asarray(lgbm, "float64"), 0, None)
    naive = np.clip(np.nan_to_num(np.asarray(naive, "float64")), 0, None)
    return np.clip(w * lgbm + (1 - w) * naive, 0, None)
