"""Intermittent-demand models (Phase 5.1 / 5.7).

Croston / SBA / TSB for intermittent & lumpy SKUs — the 90% of SKUs where a single global
continuous model under-performs. These methods model demand as (size x interval) and emit a
near-constant rate, which is the right shape for sparse series.

We use statsforecast (Numba-fast, fits thousands of series quickly). Because TSB/Croston
forecasts are flat, the embargo gap between train_end and the validation window doesn't
matter: the rate for day train_end+1 equals the rate for any later day, so we forecast far
enough to cover the gap and reuse the per-series rate across the validation window.

    rates = forecast_intermittent(stores, train_end, valid_end)   # -> store_id,sku_id,pred_tsb
"""
from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from src.config import CONFIG

PANEL = (CONFIG.data_dir / "features" / "panel.parquet").as_posix()
SEGMENTS = (CONFIG.data_dir / "features" / "segments.parquet").as_posix()
SEP = "__"


def _intermittent_keys_sql(classes: tuple[str, ...]) -> str:
    cls = ", ".join(f"'{c}'" for c in classes)
    return (f"SELECT store_id, sku_id FROM read_parquet('{SEGMENTS}') "
            f"WHERE intermittency IN ({cls})")


def forecast_intermittent(
    stores: list[str] | None,
    train_end: date,
    valid_end: date,
    classes: tuple[str, ...] = ("intermittent", "lumpy"),
    model: str = "tsb",
) -> pd.DataFrame:
    """Return a per-(store,sku) forecast rate `pred_tsb` for intermittent/lumpy series."""
    from statsforecast import StatsForecast
    from statsforecast.models import TSB, CrostonSBA

    sf_model = (TSB(alpha_d=0.1, alpha_p=0.1) if model == "tsb"
                else CrostonSBA())

    store_filt = ""
    if stores:
        ids = ", ".join(f"'{s}'" for s in stores)
        store_filt = f"AND p.store_id IN ({ids})"

    # long format for intermittent series, training history up to train_end
    long = duckdb.connect().execute(f"""
        SELECT p.store_id || '{SEP}' || p.sku_id AS unique_id,
               p.date AS ds, p.units AS y
        FROM read_parquet('{PANEL}/**/*.parquet') p
        JOIN ({_intermittent_keys_sql(classes)}) k USING (store_id, sku_id)
        WHERE p.date <= DATE '{train_end}' {store_filt}
        ORDER BY 1, 2
    """).df()
    if long.empty:
        return pd.DataFrame(columns=["store_id", "sku_id", "pred_tsb"])

    long["ds"] = pd.to_datetime(long["ds"])
    h = (valid_end - train_end).days  # cover the embargo gap + validation window
    print(f"  TSB: {long['unique_id'].nunique():,} intermittent series, h={h}")

    sf = StatsForecast(models=[sf_model], freq="D", n_jobs=-1)
    fcst = sf.forecast(df=long, h=h)
    col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
    rate = fcst.groupby("unique_id")[col].mean().clip(lower=0).reset_index(name="pred_tsb")

    rate[["store_id", "sku_id"]] = rate["unique_id"].str.split(SEP, n=1, expand=True)
    return rate[["store_id", "sku_id", "pred_tsb"]]
