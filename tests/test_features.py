"""Phase 4 tests: feature leak-safety. Skip if feature panel not built."""
import duckdb
import pytest

from src.config import CONFIG

FP = CONFIG.data_dir / "features" / "feature_panel.parquet"
_has = FP.exists()
F = f"read_parquet('{FP.as_posix()}/**/*.parquet')"
H = CONFIG.horizon

pytestmark = pytest.mark.skipif(not _has, reason="feature_panel not built")


@pytest.fixture(scope="module")
def series():
    con = duckdb.connect()
    sku = con.execute(
        f"SELECT sku_id, store_id FROM {F} WHERE units>0 GROUP BY 1,2 "
        f"ORDER BY sum(units) DESC LIMIT 1"
    ).fetchone()
    df = con.execute(
        f"SELECT date, units, lag_14, roll_mean_7 FROM {F} "
        f"WHERE sku_id='{sku[0]}' AND store_id='{sku[1]}' ORDER BY date"
    ).df()
    return df


def test_lag_references_past_only(series):
    # lag_14[t] must equal units[t-H]; any deviation means leakage or wrong shift.
    manual = series["units"].shift(H)
    mask = manual.notna()
    mismatches = (series["lag_14"][mask].round(6) != manual[mask].round(6)).sum()
    assert mismatches == 0


def test_rolling_window_ends_before_origin(series):
    # roll_mean_7[t] == mean(units[t-H-6 .. t-H]); compare on interior rows.
    manual = series["units"].shift(H).rolling(7).mean()
    both = series["roll_mean_7"].notna() & manual.notna()
    diff = (series["roll_mean_7"][both].round(4) != manual[both].round(4)).sum()
    assert diff == 0


def test_no_target_leak_columns_in_features():
    # The same-day target proxies must never appear shifted by < H.
    cols = duckdb.connect().execute(f"SELECT * FROM {F} LIMIT 0").df().columns
    for c in cols:
        if c.startswith("lag_"):
            assert int(c.split("_")[1]) >= H, f"{c} violates lag>=H"
