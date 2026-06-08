"""Phase 3 tests: panel integrity + leak-free splits."""
from datetime import date

import duckdb
import pytest

from src.config import CONFIG
from src.evaluate.splits import rolling_origin_splits

PANEL = CONFIG.data_dir / "features" / "panel.parquet"
_has = PANEL.exists()
P = f"read_parquet('{PANEL.as_posix()}/**/*.parquet')"


# --- splits: pure logic, no data needed --------------------------------------

def test_embargo_respected():
    folds = rolling_origin_splits(date(2011, 1, 1), date(2016, 5, 22))
    for f in folds:
        gap = (f.valid_start - f.train_end).days
        assert gap >= f.embargo_days + 1, f"{f}: gap {gap} < embargo"


def test_validation_windows_are_horizon_length():
    h = CONFIG.metrics["backtest"]["validation_horizon_days"]
    for f in rolling_origin_splits(date(2011, 1, 1), date(2016, 5, 22)):
        assert (f.valid_end - f.valid_start).days + 1 == h


def test_folds_walk_forward_without_overlap():
    folds = rolling_origin_splits(date(2011, 1, 1), date(2016, 5, 22))
    for a, b in zip(folds, folds[1:]):
        assert b.valid_start > a.valid_end  # later fold validates strictly later


def test_final_fold_is_holdout_end():
    mx = date(2016, 5, 22)
    folds = rolling_origin_splits(date(2011, 1, 1), mx)
    assert folds[-1].valid_end == mx


# --- panel integrity ---------------------------------------------------------

@pytest.mark.skipif(not _has, reason="panel not built")
def test_no_negative_demand():
    con = duckdb.connect()
    assert con.execute(f"SELECT count(*) FROM {P} WHERE units < 0").fetchone()[0] == 0


@pytest.mark.skipif(not _has, reason="panel not built")
def test_sample_weight_binary():
    con = duckdb.connect()
    bad = con.execute(f"SELECT count(*) FROM {P} WHERE sample_weight NOT IN (0.0, 1.0)").fetchone()[0]
    assert bad == 0


@pytest.mark.skipif(not _has, reason="panel not built")
def test_grid_is_contiguous_daily():
    # For each (store,sku) the #rows must equal calendar span (no gaps, no dupes).
    con = duckdb.connect()
    bad = con.execute(
        f"SELECT count(*) FROM ("
        f"  SELECT store_id, sku_id, count(*) n, date_diff('day', min(date), max(date))+1 span "
        f"  FROM {P} GROUP BY 1,2) WHERE n <> span"
    ).fetchone()[0]
    assert bad == 0
