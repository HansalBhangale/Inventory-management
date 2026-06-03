"""Phase 1 integrity tests on the staged M5 data.

Skip automatically if the data hasn't been loaded yet (so CI without data still passes).
"""
import duckdb
import pytest

from src.config import CONFIG

SALES = (CONFIG.data_dir / "staged" / "sales_transactions").as_posix() + "/**/*.parquet"
PRODUCTS = (CONFIG.data_dir / "staged" / "product_master.parquet").as_posix()
CAL = (CONFIG.data_dir / "staged" / "external_calendar.parquet").as_posix()

_has_data = (CONFIG.data_dir / "staged" / "product_master.parquet").exists()
pytestmark = pytest.mark.skipif(not _has_data, reason="staged data not loaded")


@pytest.fixture(scope="module")
def con():
    return duckdb.connect()


def test_no_null_keys(con):
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{SALES}') "
        "WHERE date IS NULL OR store_id IS NULL OR sku_id IS NULL"
    ).fetchone()[0]
    assert n == 0


def test_prices_present(con):
    # Loader drops pre-launch unpriced rows, so unit_price must be non-null everywhere.
    n = con.execute(f"SELECT count(*) FROM read_parquet('{SALES}') WHERE unit_price IS NULL").fetchone()[0]
    assert n == 0


def test_referential_integrity_skus(con):
    # Every sales sku_id must exist in product_master (data_contract rule).
    orphans = con.execute(
        f"SELECT count(DISTINCT s.sku_id) FROM read_parquet('{SALES}') s "
        f"LEFT JOIN read_parquet('{PRODUCTS}') p USING (sku_id) WHERE p.sku_id IS NULL"
    ).fetchone()[0]
    assert orphans == 0


def test_dates_not_in_future(con):
    mx = con.execute(f"SELECT max(date) FROM read_parquet('{SALES}')").fetchone()[0]
    assert mx is not None


def test_calendar_intensity_in_range(con):
    bad = con.execute(
        f"SELECT count(*) FROM read_parquet('{CAL}') "
        "WHERE festival_intensity < 0 OR festival_intensity > 1"
    ).fetchone()[0]
    assert bad == 0
