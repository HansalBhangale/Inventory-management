"""Phase 0 smoke tests: config loads and the locked decisions are present & consistent."""
from src.config import CONFIG


def test_horizon_and_quantiles():
    assert CONFIG.horizon == 14
    # 0.99 head added in Phase 8 for high-service A-items; 0.5/0.9/0.95 must remain present.
    assert {0.5, 0.9, 0.95}.issubset(set(CONFIG.quantiles))


def test_quantiles_sorted_for_non_crossing():
    qs = CONFIG.quantiles
    assert qs == sorted(qs)
    assert CONFIG.model["forecast"]["enforce_non_crossing"] is True


def test_policy_is_sS():
    assert CONFIG.policy["policy_type"] == "sS"
    for seg in ("A", "B", "C"):
        assert 0 < CONFIG.policy["service_levels"][seg]["target"] < 1


def test_lags_respect_horizon():
    # Direct H-step forecast: every lag must be >= horizon to avoid leakage.
    lags = CONFIG.features["feature_families"]["lags"]["values"]
    assert all(lag >= CONFIG.horizon for lag in lags), lags


def test_all_configs_load():
    for attr in ("model", "policy", "metrics", "features", "data_contract", "data_sources"):
        assert isinstance(getattr(CONFIG, attr), dict)
