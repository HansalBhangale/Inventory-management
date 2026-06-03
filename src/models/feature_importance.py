"""Feature importance (Phase 4.3): gain/split + permutation + SHAP.

Trains the global LightGBM once on a recent training window, then reconciles three
importance views to produce a ranked list. Permutation is computed on a held-out slice
(less biased toward high-cardinality features); SHAP gives direction of effect.

Usage:
    python -m src.models.feature_importance --stores CA_1 --months 12
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluate import forecast_metrics as M
from src.models.global_lgbm import GlobalLGBM, feature_columns

FP = (CONFIG.data_dir / "features" / "feature_panel.parquet").as_posix()


def _load(where: str, cols: list[str]) -> pd.DataFrame:
    sel = ", ".join(dict.fromkeys(cols))
    return duckdb.connect().execute(
        f"SELECT {sel} FROM read_parquet('{FP}/**/*.parquet') WHERE {where}"
    ).df()


def run(stores: list[str] | None, months: int) -> pd.DataFrame:
    feats, cats = feature_columns()
    keep = feats + ["units", "sample_weight", "date", "store_id", "sku_id"]
    store_filt = ""
    if stores:
        ids = ", ".join(f"'{s}'" for s in stores)
        store_filt = f" AND store_id IN ({ids})"

    mx = duckdb.connect().execute(
        f"SELECT max(date) FROM read_parquet('{FP}/**/*.parquet') WHERE 1=1{store_filt}"
    ).fetchone()[0]
    val_start = mx - timedelta(days=CONFIG.horizon - 1)
    tr_end = val_start - timedelta(days=CONFIG.horizon + 1)
    tr_start = tr_end - timedelta(days=months * 30)

    train = _load(f"date BETWEEN '{tr_start}' AND '{tr_end}'{store_filt}", keep)
    valid = _load(f"date BETWEEN '{val_start}' AND '{mx}'{store_filt}", keep)
    print(f"train={len(train):,} valid={len(valid):,}")

    model = GlobalLGBM().fit(train, valid)

    gain = model.importance("gain").rename(columns={"gain": "imp"})
    gain["method"] = "gain"

    # permutation importance on the validation slice (WAPE degradation when shuffled)
    base_pred = model.predict(valid)["pred_central"].to_numpy()
    base_wape = M.wape(valid["units"].to_numpy(), base_pred)
    perm = []
    rng = np.random.default_rng(0)
    for col in feats:
        saved = valid[col].copy()
        valid[col] = rng.permutation(valid[col].to_numpy())
        w = M.wape(valid["units"].to_numpy(), model.predict(valid)["pred_central"].to_numpy())
        valid[col] = saved
        perm.append({"feature": col, "imp": w - base_wape, "method": "permutation_wape_delta"})
    perm = pd.DataFrame(perm).sort_values("imp", ascending=False)

    # SHAP (sampled for speed)
    shap_df = _shap(model, valid, feats, cats)

    out = Path(CONFIG.root) / "docs" / "PHASE4_feature_importance.md"
    lines = [
        "# Phase 4.3 — Feature Importance\n",
        f"Scope: stores={stores or 'ALL'} · train window ~{months} months · base WAPE={base_wape:.4f}\n",
        "## Gain (top 20)\n", gain.head(20).round(2).to_markdown(index=False),
        "\n## Permutation (validation WAPE delta, top 20)\n", perm.head(20).round(5).to_markdown(index=False),
    ]
    if shap_df is not None:
        lines += ["\n## SHAP mean|value| (top 20)\n", shap_df.head(20).round(4).to_markdown(index=False)]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return perm


def _shap(model, valid, feats, cats):
    try:
        import shap
    except ImportError:
        return None
    from src.models.global_lgbm import _prep
    sample = valid.sample(min(20000, len(valid)), random_state=0)
    X = _prep(sample, feats, cats)
    expl = shap.TreeExplainer(model.central_)
    sv = expl.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Feature importance (gain/permutation/SHAP).")
    ap.add_argument("--stores", nargs="+")
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args(argv)
    run(args.stores, args.months)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
