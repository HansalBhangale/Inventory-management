"""Global LightGBM forecaster (Phase 5.2 / 5.3).

ONE model across all (store, SKU) series — cross-learning + cheap cold-start + scales.
Trains a central forecast (tweedie) plus separate quantile heads {0.5, 0.9, 0.95}
(pinball objective). Quantiles are sorted at predict time to enforce non-crossing
(P95 >= P90 >= P50). These quantiles feed the reorder policy (Phase 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import CONFIG


def feature_columns() -> tuple[list[str], list[str]]:
    """Return (all_features, categorical_features) from config/features.yaml."""
    mf = CONFIG.features["model_features"]
    feats = list(mf["numeric"]) + list(mf["boolean"]) + list(mf["categorical"])
    return feats, list(mf["categorical"])


def _prep(df: pd.DataFrame, feats: list[str], cats: list[str]) -> pd.DataFrame:
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    for c in feats:
        if c not in cats:
            # float32 halves memory vs float64 — material for the all-stores run (~47M rows)
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


@dataclass
class GlobalLGBM:
    quantiles: list[float] = field(default_factory=lambda: CONFIG.quantiles)
    params: dict = field(default_factory=lambda: dict(CONFIG.model["lightgbm"]))
    central_objective: str = field(default_factory=lambda: CONFIG.model["central_objective"])
    feats: list[str] = field(default_factory=list)
    cats: list[str] = field(default_factory=list)
    central_: lgb.Booster | None = None
    quantile_: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.feats:
            self.feats, self.cats = feature_columns()

    def _train_with(self, dtr, dva, objective: str, alpha=None):
        p = dict(self.params)
        n_est = p.pop("n_estimators", 2000)
        early = p.pop("early_stopping_rounds", 100)
        p["objective"] = objective
        if alpha is not None:
            p["alpha"] = alpha
        if objective == "tweedie":
            p["tweedie_variance_power"] = CONFIG.model.get("tweedie_variance_power", 1.2)
        return lgb.train(
            p, dtr, num_boost_round=n_est, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(0)],
        )

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "GlobalLGBM":
        # Build & bin the dataset ONCE and reuse across all 4 heads (central + 3 quantiles).
        # construct() then free_raw_data=True drops the raw float matrix after binning, so we
        # don't hold 4 copies — critical for the ~27M-row all-stores training load.
        import gc
        Xtr = _prep(train, self.feats, self.cats)
        Xva = _prep(valid, self.feats, self.cats)
        dtr = lgb.Dataset(Xtr, label=train["units"].to_numpy(),
                          weight=train["sample_weight"].to_numpy(),
                          categorical_feature=self.cats, free_raw_data=True)
        dva = lgb.Dataset(Xva, label=valid["units"].to_numpy(),
                          categorical_feature=self.cats, reference=dtr, free_raw_data=True)
        dtr.construct(); dva.construct()
        del Xtr, Xva; gc.collect()

        print(f"  training central ({self.central_objective}) ...")
        self.central_ = self._train_with(dtr, dva, self.central_objective)
        for q in self.quantiles:
            print(f"  training quantile head q={q} ...")
            self.quantile_[q] = self._train_with(dtr, dva, "quantile", alpha=q)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _prep(df, self.feats, self.cats)
        out = pd.DataFrame(index=df.index)
        out["pred_central"] = np.clip(self.central_.predict(X), 0, None)
        qcols = []
        for q in self.quantiles:
            col = f"pred_q{int(q * 100)}"
            out[col] = np.clip(self.quantile_[q].predict(X), 0, None)
            qcols.append(col)
        # enforce non-crossing: sort quantile predictions row-wise ascending
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
        return out

    def importance(self, kind: str = "gain") -> pd.DataFrame:
        imp = self.central_.feature_importance(importance_type=kind)
        return (pd.DataFrame({"feature": self.central_.feature_name(), kind: imp})
                .sort_values(kind, ascending=False).reset_index(drop=True))
