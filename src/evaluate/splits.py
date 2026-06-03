"""Time-based, leakage-free splits (Phase 3.6 / 8.1).

Rolling-origin (walk-forward) evaluation — never random K-fold on time series.
An embargo gap equal to the horizon separates train-end from validation-start so that
lag/rolling features (which look back up to H days) cannot peek across the boundary.

    from src.evaluate.splits import rolling_origin_splits
    for fold in rolling_origin_splits(min_date, max_date):
        train = df[df.date <= fold.train_end]
        valid = df[(df.date >= fold.valid_start) & (df.date <= fold.valid_end)]
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.config import CONFIG


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: date          # last date used for training
    valid_start: date        # first validation date (after embargo)
    valid_end: date          # last validation date
    embargo_days: int

    def __repr__(self) -> str:
        return (f"Fold{self.index}(train<= {self.train_end}, "
                f"valid {self.valid_start}..{self.valid_end})")


def rolling_origin_splits(
    min_date: date | pd.Timestamp,
    max_date: date | pd.Timestamp,
    n_origins: int | None = None,
    horizon: int | None = None,
    embargo: int | None = None,
) -> list[Fold]:
    """Generate walk-forward folds anchored at the END of the series.

    The final hold-out test = the most recent `horizon` days (the last fold's validation),
    which should remain untouched until final evaluation.
    """
    bt = CONFIG.metrics["backtest"]
    n_origins = n_origins or bt["n_origins"]
    horizon = horizon or bt["validation_horizon_days"]
    embargo = bt["embargo_days"] if embargo is None else embargo

    min_date = pd.Timestamp(min_date).date()
    max_date = pd.Timestamp(max_date).date()

    folds: list[Fold] = []
    # Most recent origin first; step back by `horizon` for each earlier origin.
    for i in range(n_origins):
        valid_end = max_date - timedelta(days=horizon * i)
        valid_start = valid_end - timedelta(days=horizon - 1)
        train_end = valid_start - timedelta(days=embargo + 1)
        if train_end <= min_date:
            break  # not enough history for this origin
        folds.append(Fold(n_origins - i, train_end, valid_start, valid_end, embargo))
    folds.sort(key=lambda f: f.index)
    return folds


def describe_splits(min_date, max_date) -> pd.DataFrame:
    rows = [
        {"fold": f.index, "train_end": f.train_end,
         "valid_start": f.valid_start, "valid_end": f.valid_end,
         "embargo_days": f.embargo_days}
        for f in rolling_origin_splits(min_date, max_date)
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import duckdb
    panel = (CONFIG.data_dir / "features" / "panel.parquet").as_posix()
    mn, mx = duckdb.connect().execute(
        f"SELECT min(date), max(date) FROM read_parquet('{panel}/**/*.parquet')"
    ).fetchone()
    print(f"panel span: {mn} .. {mx}\n")
    print(describe_splits(mn, mx).to_string(index=False))
