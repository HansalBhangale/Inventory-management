"""Phase 7 tests: reorder policy mechanics (pure functions)."""
import numpy as np

from src.reorder.policy import recommend, reorder_levels, round_order
from src.reorder.safety_stock import ss_empirical, ss_formula
from src.reorder.leadtime import LeadTime


def test_round_order_moq_and_pack():
    assert round_order(0) == 0
    assert round_order(3, moq=5, pack_size=1) == 5          # bumped to MOQ
    assert round_order(7, moq=1, pack_size=6) == 12         # rounded up to pack multiple
    assert round_order(12, moq=1, pack_size=6) == 12


def test_reorder_levels_monotone():
    s, S = reorder_levels(demand_over_P_q=10.0, demand_over_PC_q=13.0)
    assert s == 10.0 and S == 13.0 and S >= s
    s, S = reorder_levels(8.0, 5.0)                          # S floored at s
    assert S == s == 8.0


def test_ss_empirical_nonneg():
    assert ss_empirical(12.0, 9.0) == 3.0
    assert ss_empirical(5.0, 9.0) == 0.0                    # never negative


def test_ss_formula_grows_with_service_and_variability():
    low = ss_formula(0.90, sigma_d=2.0, lead_mean=3, d_bar=5, sigma_L=1.0)
    high = ss_formula(0.99, sigma_d=2.0, lead_mean=3, d_bar=5, sigma_L=1.0)
    assert high > low > 0


def test_recommend_triggers_below_s():
    r = recommend(inventory_position=5, s=10, S=20, moq=1, pack_size=1)
    assert r.should_order and r.order_qty == 15
    r2 = recommend(inventory_position=12, s=10, S=20)
    assert not r2.should_order and r2.order_qty == 0


def test_leadtime_sample_deterministic_when_no_variance():
    lt = LeadTime("x", 3, 0.0)
    rng = np.random.default_rng(0)
    assert lt.sample(rng) == 3
