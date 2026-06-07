"""Phase 7.6 tests: newsvendor/perishable branch (was config-declared, never implemented)."""
from src.reorder.newsvendor import (critical_ratio, critical_ratio_from_prices,
                                     newsvendor_order, quantile_interpolator)


def test_critical_ratio_bounds_and_value():
    assert abs(critical_ratio_from_prices(sell_price=100, unit_cost=40) - 0.6) < 1e-9
    assert 0 < critical_ratio(1e9, 1) < 1            # huge margin -> stock aggressively, still <1
    assert critical_ratio(0, 0) == 0.5               # degenerate -> neutral


def test_high_margin_orders_more_than_low_margin():
    qf = quantile_interpolator({0.5: 5, 0.9: 9, 0.95: 11, 0.99: 14})
    hi = newsvendor_order(qf, sell_price=100, unit_cost=10, shelf_life_demand=99,
                          inventory_position=0)
    lo = newsvendor_order(qf, sell_price=100, unit_cost=90, shelf_life_demand=99,
                          inventory_position=0)
    assert hi.order_qty > lo.order_qty               # higher Cu/(Cu+Co) -> larger order


def test_capped_at_shelf_life_demand():
    qf = quantile_interpolator({0.5: 5, 0.9: 9, 0.95: 11, 0.99: 14})
    o = newsvendor_order(qf, sell_price=100, unit_cost=10, shelf_life_demand=3,
                         inventory_position=0)
    assert o.capped is True
    assert o.order_qty <= 3                           # never stock more than sells before spoiling


def test_nets_inventory_and_rounds():
    qf = quantile_interpolator({0.5: 5, 0.9: 9, 0.95: 11, 0.99: 14})
    o = newsvendor_order(qf, sell_price=100, unit_cost=40, shelf_life_demand=99,
                         inventory_position=2, moq=1, pack_size=1)
    # cr=0.6 -> interp ~6 over the window; minus IP 2 -> ~4
    assert 3 <= o.order_qty <= 5


def test_quantile_interpolator_monotone():
    f = quantile_interpolator({0.5: 5, 0.9: 4, 0.95: 11})   # deliberately non-monotone input
    assert f(0.5) <= f(0.9) <= f(0.95)                      # enforced non-crossing


def test_dispatch_routes_perishable_to_newsvendor():
    from src.reorder.policy import dispatch_reorder
    q = {0.5: 5, 0.9: 9, 0.95: 11, 0.99: 14}
    branch, _ = dispatch_reorder(perishable=True, inventory_position=0, quantiles=q,
                                 sell_price=100, unit_cost=40, shelf_life_demand=8, s=6, S=12)
    assert branch == "newsvendor"                            # the perishable path FIRES
    branch2, _ = dispatch_reorder(perishable=False, inventory_position=2, quantiles=q,
                                  s=6, S=12)
    assert branch2 == "sS"                                   # non-perishable uses (s,S)
