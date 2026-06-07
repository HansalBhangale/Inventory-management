"""Phase 4.F tests: promo features (config-declared, previously unbuilt). Proves the path turns a
real on_promo signal into populated features, and stays inert (zeros) when promo data is absent."""
import pandas as pd

from src.features.promo import PROMO_FEATURES, add_promo_features


def _series(promo):
    n = len(promo)
    return pd.DataFrame({"store_id": "S1", "sku_id": "K1",
                         "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                         "on_promo": promo})


def test_inert_when_no_promo_column():
    df = pd.DataFrame({"store_id": ["S1"], "sku_id": ["K1"], "date": [pd.Timestamp("2024-01-01")]})
    out = add_promo_features(df)
    assert all(f in out for f in PROMO_FEATURES)
    assert (out[PROMO_FEATURES].fillna(0).to_numpy() == 0).all()   # path runs, stays inert


def test_promo_features_populate_from_real_signal():
    # promo on days 2,3,4 (index 2..4) then off
    out = add_promo_features(_series([0, 0, 1, 1, 1, 0, 0, 0])).reset_index(drop=True)
    assert out["on_promo"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    # days_since_promo_start: 0 at the start day (idx2), 1, 2 across the run
    assert out.loc[2, "days_since_promo_start"] == 0
    assert out.loc[4, "days_since_promo_start"] == 2
    # days_to_promo_end: 2,1,0 across the run (idx2->end at idx4)
    assert out.loc[2, "days_to_promo_end"] == 2
    assert out.loc[4, "days_to_promo_end"] == 0


def test_promo_in_next_7d_is_forward_looking():
    out = add_promo_features(_series([0, 0, 0, 0, 0, 0, 1, 0])).reset_index(drop=True)
    assert out.loc[0, "promo_in_next_7d"] == 1     # promo at idx6 is within next 7 of idx0
    assert out.loc[6, "promo_in_next_7d"] == 0     # nothing after the promo day
