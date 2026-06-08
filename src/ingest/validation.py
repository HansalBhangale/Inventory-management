"""Adversarial data-contract validation gate (Phase 1.3 / Phase 9).

The deliverable is NOT "it validates M5" — M5 is the data the engine was built on, so that is
guaranteed-green and proves only that the pipes connect. The deliverable is a layer **proven to
reject the realistic mess M5 never had**, because that mess is the first thing a real store throws
at the system, and silent acceptance of bad data is how forecasting systems produce confident
nonsense in production.

It catches, by design (see tests/test_validation.py for the deliberately-broken inputs):
  - schema violations: null keys, non-integer qty, negative on-hand, future dates
  - referential breaks: a sales SKU absent from the product master; PO/GRN not joinable
  - silent demand corruption: returns/voids (qty <= 0) mixed into the sales stream
  - structural gaps: missing calendar days inside a SKU's active window
  - duplicate goods receipts (double-counted stock)
  - missing supplier lead-time inputs (reorder would silently use a fallback)

Two severities: BLOCK (quarantine the batch — never reaches scoring) and WARN (proceed with the
documented handling, e.g. net returns, fill gaps, fall back to default lead time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors


class Severity(str, Enum):
    BLOCK = "BLOCK"      # quarantine batch; do not score
    WARN = "WARN"        # proceed with documented handling


@dataclass
class Finding:
    severity: Severity
    table: str
    rule: str
    detail: str
    n_rows: int = 0


@dataclass
class ValidationResult:
    table: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity is Severity.BLOCK for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    def add(self, sev, rule, detail, n=0):
        self.findings.append(Finding(sev, self.table, rule, detail, n))


class DataContractError(Exception):
    """Raised by gate() when a batch is BLOCKED."""


# --- pandera schemas (the canonical staged tables) -------------------------------------------

def _not_future(s: pd.Series) -> pd.Series:
    today = pd.Timestamp(datetime.now().date())
    return pd.to_datetime(s, errors="coerce") <= today


def _whole_number(s: pd.Series) -> pd.Series:
    """True where the value is a whole number, regardless of storage dtype (int32/int64/float).
    BLOCKs genuinely fractional qty (e.g. Favorita weight items) while accepting any integer
    representation — real loaders return int32, which exact-int64 typing wrongly rejected."""
    v = pd.to_numeric(s, errors="coerce")
    return v.notna() & (v == v.round())


SALES_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", pa.Check(_not_future, error="date in the future"),
                          nullable=False, coerce=True),
        "store_id": pa.Column(str, nullable=False),
        "sku_id": pa.Column(str, nullable=False),
        # whole-number, any int/float storage; genuinely fractional (weight items) -> FAIL
        "qty": pa.Column(float, pa.Check(_whole_number, error="non-integer qty"),
                         nullable=False, coerce=True),
        "unit_price": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True, required=False),
    },
    strict=False, name="sales_transactions",
)

PRODUCT_SCHEMA = pa.DataFrameSchema(
    {
        "sku_id": pa.Column(str, nullable=False, unique=True),
        "pack_size": pa.Column(int, pa.Check.ge(1), nullable=False, coerce=True),
        "perishable": pa.Column(bool, nullable=False, coerce=True),
    },
    strict=False, name="product_master",
)

INVENTORY_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "store_id": pa.Column(str, nullable=False),
        "sku_id": pa.Column(str, nullable=False),
        "on_hand_qty": pa.Column(float, [pa.Check.ge(0), pa.Check(_whole_number, error="non-integer on_hand")],
                                 nullable=False, coerce=True),  # negative or fractional => FAIL
    },
    strict=False, name="inventory_snapshot",
)

SCHEMAS = {"sales_transactions": SALES_SCHEMA, "product_master": PRODUCT_SCHEMA,
           "inventory_snapshot": INVENTORY_SCHEMA}


def _check_schema(name: str, df: pd.DataFrame, res: ValidationResult) -> None:
    schema = SCHEMAS.get(name)
    if schema is None:
        return
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as e:
        fc = e.failure_cases
        for check, grp in fc.groupby("check"):
            cols = sorted(set(grp["column"].dropna().astype(str)))
            res.add(Severity.BLOCK, f"schema:{check}",
                    f"{', '.join(cols) or 'frame'} failed [{check}]", n=len(grp))


# --- structural / cross-table checks M5 never exercises --------------------------------------

def validate_sales(df: pd.DataFrame, product_master: pd.DataFrame | None = None) -> ValidationResult:
    res = ValidationResult("sales_transactions")
    _check_schema("sales_transactions", df, res)

    # null keys (explicit, even if schema caught it — clearer message)
    nk = df[["date", "store_id", "sku_id"]].isna().any(axis=1).sum()
    if nk:
        res.add(Severity.BLOCK, "null_keys", "null date/store/sku in sales", int(nk))

    # returns / voids silently corrupting demand
    if "qty" in df:
        qty = pd.to_numeric(df["qty"], errors="coerce")
        n_ret = int((qty < 0).sum())
        n_void = int((qty == 0).sum())
        if n_ret:
            res.add(Severity.WARN, "returns_present",
                    "qty<0 (returns) must be split out, not summed into demand", n_ret)
        if n_void:
            res.add(Severity.WARN, "voids_present", "qty==0 rows present (voids/no-sale)", n_void)

    # referential integrity: every sales SKU must exist in the product master
    if product_master is not None and "sku_id" in df:
        known = set(product_master["sku_id"].astype(str))
        missing = sorted(set(df["sku_id"].astype(str)) - known)
        if missing:
            res.add(Severity.BLOCK, "referential_sku",
                    f"{len(missing)} sales SKU(s) absent from product_master "
                    f"(e.g. {missing[:3]})", n=len(missing))

    # calendar gaps inside each SKU's active window
    if {"store_id", "sku_id", "date"}.issubset(df.columns) and len(df):
        d = df.dropna(subset=["store_id", "sku_id", "date"]).copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        span = d.groupby(["store_id", "sku_id"])["date"].agg(["min", "max", "nunique"])
        span["expected"] = (span["max"] - span["min"]).dt.days + 1
        gappy = int((span["nunique"] < span["expected"]).sum())
        if gappy:
            res.add(Severity.WARN, "calendar_gaps",
                    f"{gappy} (store,SKU) series have missing days inside their active window", gappy)
    return res


def validate_inventory(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult("inventory_snapshot")
    _check_schema("inventory_snapshot", df, res)
    if {"store_id", "sku_id", "date"}.issubset(df.columns):
        dup = int(df.duplicated(["store_id", "sku_id", "date"]).sum())
        if dup:
            res.add(Severity.BLOCK, "duplicate_grain",
                    "multiple on-hand rows per (store,SKU,day)", dup)
    return res


def validate_receipts(grn: pd.DataFrame, po: pd.DataFrame | None = None) -> ValidationResult:
    res = ValidationResult("goods_receipts")
    # duplicate receipts double-count received stock
    if {"po_id", "sku_id", "receipt_date"}.issubset(grn.columns):
        dup = int(grn.duplicated(["po_id", "sku_id", "receipt_date", "received_qty"]).sum()
                  if "received_qty" in grn else grn.duplicated(["po_id", "sku_id", "receipt_date"]).sum())
        if dup:
            res.add(Severity.WARN, "duplicate_receipts",
                    "duplicate GRN lines — dedupe before counting stock", dup)
    # PO <-> GRN joinable (lead-time model depends on it)
    if po is not None and {"po_id", "sku_id"}.issubset(grn.columns) and {"po_id", "sku_id"}.issubset(po.columns):
        po_keys = set(map(tuple, po[["po_id", "sku_id"]].astype(str).values))
        orphan = sum(tuple(map(str, k)) not in po_keys
                     for k in grn[["po_id", "sku_id"]].itertuples(index=False))
        if orphan:
            res.add(Severity.BLOCK, "grn_orphans",
                    f"{orphan} GRN line(s) not joinable to any PO (lead-time would be wrong)", orphan)
    return res


def validate_suppliers(suppliers: pd.DataFrame, po: pd.DataFrame | None = None) -> ValidationResult:
    res = ValidationResult("suppliers")
    # missing lead-time inputs -> reorder would silently fall back to a default
    for col in ("moq", "order_cycle"):
        if col in suppliers:
            n = int(suppliers[col].isna().sum())
            if n:
                res.add(Severity.WARN, f"missing_{col}",
                        f"{n} supplier(s) missing {col}; reorder uses configured fallback", n)
    if po is not None and "supplier_id" in suppliers and "supplier_id" in po:
        with_history = set(po["supplier_id"].astype(str))
        thin = sorted(set(suppliers["supplier_id"].astype(str)) - with_history)
        if thin:
            res.add(Severity.WARN, "no_leadtime_history",
                    f"{len(thin)} supplier(s) have no PO history -> default lead time", len(thin))
    return res


def gate(results: list[ValidationResult], *, raise_on_block: bool = True) -> dict:
    """Aggregate results into a gate decision. BLOCK => quarantine (raise); WARN => proceed."""
    blocks = [f for r in results for f in r.findings if f.severity is Severity.BLOCK]
    warns = [f for r in results for f in r.findings if f.severity is Severity.WARN]
    summary = {
        "passed": not blocks,
        "blocks": [f"[{f.table}] {f.rule}: {f.detail} ({f.n_rows})" for f in blocks],
        "warnings": [f"[{f.table}] {f.rule}: {f.detail} ({f.n_rows})" for f in warns],
    }
    if blocks and raise_on_block:
        raise DataContractError("BATCH QUARANTINED:\n  " + "\n  ".join(summary["blocks"]))
    return summary


# --- CLI: validate a store's CSV exports before anything else (Phase 9 / pilot onboarding) ----

_DATE_COLS = {"date", "receipt_date", "order_date", "start", "end"}


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        if c in _DATE_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Validate a store's CSV exports against the data contract before ingesting.")
    ap.add_argument("--sales", help="sales export CSV (date, store_id, sku_id, qty[, unit_price])")
    ap.add_argument("--product-master", help="product master CSV (sku_id, pack_size, perishable)")
    ap.add_argument("--inventory", help="inventory snapshot CSV (date, store_id, sku_id, on_hand_qty)")
    ap.add_argument("--suppliers", help="suppliers CSV (supplier_id[, moq, order_cycle])")
    ap.add_argument("--purchase-orders", help="PO CSV (po_id, sku_id, supplier_id, order_date)")
    ap.add_argument("--goods-receipts", help="GRN CSV (po_id, sku_id, receipt_date[, received_qty])")
    args = ap.parse_args(argv)

    pm = _read_csv(args.product_master) if args.product_master else None
    po = _read_csv(args.purchase_orders) if args.purchase_orders else None
    results: list[ValidationResult] = []
    if args.sales:
        results.append(validate_sales(_read_csv(args.sales), pm))
    if args.inventory:
        results.append(validate_inventory(_read_csv(args.inventory)))
    if args.goods_receipts:
        results.append(validate_receipts(_read_csv(args.goods_receipts), po))
    if args.suppliers:
        results.append(validate_suppliers(_read_csv(args.suppliers), po))
    if not results:
        ap.error("provide at least one input (e.g. --sales sales.csv --product-master pm.csv)")

    summary = gate(results, raise_on_block=False)
    print("=" * 70)
    if summary["blocks"]:
        print(f"QUARANTINED — {len(summary['blocks'])} blocking problem(s). Fix these, re-export, "
              "re-run. Nothing is scored until they're gone:")
        for b in summary["blocks"]:
            print(f"  [BLOCK] {b}")
    else:
        print("No blocking problems — this batch can be ingested.")
    if summary["warnings"]:
        print(f"\n{len(summary['warnings'])} warning(s) — handled automatically, review when you can:")
        for w in summary["warnings"]:
            print(f"  [warn]  {w}")
    print("=" * 70)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
