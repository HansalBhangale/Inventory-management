"""Phase 2 tests on the segmentation table. Skip if not built yet."""
import duckdb
import pytest

from src.config import CONFIG

SEG_PATH = CONFIG.data_dir / "features" / "segments.parquet"
pytestmark = pytest.mark.skipif(not SEG_PATH.exists(), reason="segments.parquet not built")
SEG = f"read_parquet('{SEG_PATH.as_posix()}')"


@pytest.fixture(scope="module")
def con():
    return duckdb.connect()


def test_one_row_per_store_sku(con):
    total, distinct = con.execute(
        f"SELECT count(*), count(DISTINCT (store_id, sku_id)) FROM {SEG}"
    ).fetchone()
    assert total == distinct


def test_intermittency_classes_valid(con):
    classes = {r[0] for r in con.execute(f"SELECT DISTINCT intermittency FROM {SEG}").fetchall()}
    assert classes <= {"smooth", "erratic", "intermittent", "lumpy", "no_demand"}


def test_abc_xyz_valid(con):
    abc = {r[0] for r in con.execute(f"SELECT DISTINCT abc FROM {SEG}").fetchall()}
    xyz = {r[0] for r in con.execute(f"SELECT DISTINCT xyz FROM {SEG}").fetchall()}
    assert abc <= {"A", "B", "C"}
    assert xyz <= {"X", "Y", "Z"}


def test_abc_revenue_pareto(con):
    # A items should account for ~80% of revenue (Pareto), allow tolerance.
    a_share = con.execute(
        f"SELECT sum(CASE WHEN abc='A' THEN revenue ELSE 0 END)/sum(revenue) FROM {SEG}"
    ).fetchone()[0]
    assert 0.70 <= a_share <= 0.90


def test_adi_threshold_consistency(con):
    # No 'smooth' SKU may have ADI >= 1.32 (Syntetos-Boylan boundary).
    bad = con.execute(
        f"SELECT count(*) FROM {SEG} WHERE intermittency='smooth' AND adi >= 1.32"
    ).fetchone()[0]
    assert bad == 0
