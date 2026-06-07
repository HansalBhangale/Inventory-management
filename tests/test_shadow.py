"""Phase 9 tests: shadow-mode runner must be able to EMBARRASS the model.

Proves the reject-flag logic catches obviously-wrong recommendations a shopkeeper would veto,
that a sane recommendation passes clean, and that divergence-vs-actual is computed when a real
order feed is present (stubbed on M5)."""
import pandas as pd
import pytest

from src.serve.shadow import REQUIRED, run_shadow


def _rec(**over):
    base = dict(sku_id="K1", should_order=True, order_qty=12, order_up_to=30,
                inventory_position=6, expected_demand_protection=10, moq=1, pack_size=1,
                perishable=False)
    base.update(over)
    return base


def _df(rows):
    return pd.DataFrame(rows)


def test_sane_recommendation_not_flagged():
    rep = run_shadow(_df([_rec()]))
    assert rep.reject_rate == 0.0


def test_flags_implausibly_large_order():
    rep = run_shadow(_df([_rec(order_qty=1000, order_up_to=30)]))
    assert "implausibly_large" in rep.flag_counts


def test_flags_order_despite_ample_stock():
    rep = run_shadow(_df([_rec(inventory_position=500, expected_demand_protection=10)]))
    assert "order_despite_ample_stock" in rep.flag_counts


def test_flags_below_moq_and_pack_violation():
    rep = run_shadow(_df([_rec(order_qty=4, moq=10, pack_size=6)]))
    assert "below_moq" in rep.flag_counts
    assert "not_pack_multiple" in rep.flag_counts


def test_flags_order_but_zero_qty():
    rep = run_shadow(_df([_rec(should_order=True, order_qty=0)]))
    assert "order_flagged_but_zero_qty" in rep.flag_counts


def test_flags_perishable_over_shelf_life():
    row = _rec(perishable=True, order_qty=50)
    row["shelf_life_demand"] = 10
    rep = run_shadow(_df([row]))
    assert "exceeds_shelf_life_demand" in rep.flag_counts


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        run_shadow(pd.DataFrame({"sku_id": ["K1"]}))
    assert "order_qty" in REQUIRED


def test_divergence_vs_actual_when_feed_present():
    recs = _df([_rec(sku_id="K1", order_qty=12), _rec(sku_id="K2", order_qty=4)])
    actual = pd.DataFrame({"sku_id": ["K1", "K2"], "ordered_qty": [10, 4]})
    rep = run_shadow(recs, actual_orders=actual)
    assert rep.divergence is not None
    assert rep.divergence["n_matched"] == 2
    assert abs(rep.divergence["mean_abs_divergence"] - 1.0) < 1e-9   # |12-10|+|4-4| = 2 over 2
