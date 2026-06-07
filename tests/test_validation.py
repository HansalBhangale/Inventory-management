"""Phase 9 adversarial data-contract tests.

The point is NOT that clean M5-shaped data passes (it must, but that proves only the pipes
connect). The point is that DELIBERATELY BROKEN input — the mess M5 never contained — is caught:
quarantined (BLOCK) or flagged (WARN), never silently scored.
"""
import numpy as np
import pandas as pd
import pytest

from src.ingest.validation import (DataContractError, Severity, gate, validate_inventory,
                                    validate_receipts, validate_sales, validate_suppliers)


def _clean_sales(n=40):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "store_id": ["S1"] * n, "sku_id": ["K1"] * n,
                         "qty": np.arange(n) % 5, "unit_price": 2.0})


def _master(skus=("K1",)):
    return pd.DataFrame({"sku_id": list(skus), "pack_size": [1] * len(skus),
                         "perishable": [False] * len(skus)})


# --- clean input passes (pipes connect) --------------------------------------

def test_clean_sales_passes():
    res = validate_sales(_clean_sales(), _master())
    assert not res.blocked
    assert gate([res])["passed"] is True


# --- BLOCK: the mess that must quarantine the batch --------------------------

def test_block_null_keys():
    df = _clean_sales(); df.loc[3, "sku_id"] = None
    assert validate_sales(df, _master()).blocked


def test_block_future_dates():
    df = _clean_sales(); df.loc[2, "date"] = pd.Timestamp("2099-01-01")
    assert validate_sales(df, _master()).blocked


def test_block_non_integer_qty():
    df = _clean_sales(); df["qty"] = df["qty"].astype(float); df.loc[1, "qty"] = 2.5
    assert validate_sales(df, _master()).blocked


def test_block_sku_not_in_master():
    df = _clean_sales(); df.loc[5, "sku_id"] = "GHOST"
    res = validate_sales(df, _master())
    assert res.blocked
    assert any(f.rule == "referential_sku" for f in res.findings)


def test_block_negative_on_hand():
    inv = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "store_id": "S1",
                        "sku_id": "K1", "on_hand_qty": [5, -2, 3]})
    assert validate_inventory(inv).blocked


def test_block_duplicate_inventory_grain():
    inv = pd.DataFrame({"date": ["2024-01-01"] * 2, "store_id": "S1", "sku_id": "K1",
                        "on_hand_qty": [5, 6]})
    inv["date"] = pd.to_datetime(inv["date"])
    assert validate_inventory(inv).blocked


def test_block_grn_orphan_not_joinable_to_po():
    po = pd.DataFrame({"po_id": ["P1"], "sku_id": ["K1"]})
    grn = pd.DataFrame({"po_id": ["P1", "P_GHOST"], "sku_id": ["K1", "K1"],
                        "receipt_date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
                        "received_qty": [10, 4]})
    assert validate_receipts(grn, po).blocked


def test_gate_raises_on_block():
    df = _clean_sales(); df.loc[0, "store_id"] = None
    with pytest.raises(DataContractError):
        gate([validate_sales(df, _master())])


# --- WARN: handled, not blocked (returns, gaps, missing lead times) ----------

def test_warn_returns_present_not_blocking():
    df = _clean_sales(); df.loc[4, "qty"] = -3
    res = validate_sales(df, _master())
    assert not res.blocked
    assert any(f.rule == "returns_present" and f.severity is Severity.WARN for f in res.findings)


def test_warn_calendar_gaps():
    df = _clean_sales(20).drop(index=[7, 8, 9]).reset_index(drop=True)  # punch a hole
    res = validate_sales(df, _master())
    assert not res.blocked
    assert any(f.rule == "calendar_gaps" for f in res.findings)


def test_warn_missing_supplier_leadtime():
    sup = pd.DataFrame({"supplier_id": ["U1", "U2"], "moq": [10, None], "order_cycle": [7, 7]})
    res = validate_suppliers(sup)
    assert not res.blocked
    assert any(f.rule == "missing_moq" for f in res.findings)


def test_warn_duplicate_receipts():
    grn = pd.DataFrame({"po_id": ["P1", "P1"], "sku_id": ["K1", "K1"],
                        "receipt_date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
                        "received_qty": [10, 10]})
    res = validate_receipts(grn)
    assert not res.blocked
    assert any(f.rule == "duplicate_receipts" for f in res.findings)


def test_gate_warnings_pass_through():
    df = _clean_sales(); df.loc[4, "qty"] = -3        # a return -> WARN only
    summary = gate([validate_sales(df, _master())])
    assert summary["passed"] is True and summary["warnings"]
