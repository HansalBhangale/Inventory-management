# Pilot Onboarding Runbook

**For the person holding the store's data** (shopkeeper, their POS vendor, or a junior engineer).
You do **not** need to understand the model. Follow the steps; each one is runnable as written.

## What this pilot can and cannot tell you (read first)
- **CAN (week one):** whether the reorder recommendations look *sane* to the shopkeeper, and
  exactly where the store's data or settings are wrong.
- **CANNOT (yet):** the real stockout/inventory reduction — that needs a few months of data and
  real supplier lead times. Don't judge week one on forecast accuracy; judge it on *sanity*.
- The system **acts on nothing** during the pilot. It runs in **shadow mode**: it suggests, you
  compare, nobody's shelves change until you decide to go live.

---

## Step 1 — Send us these files (CSV is fine)

**A. Sales history** (required) — one row per item sold per day. Example:
```
date,store_id,sku_id,qty,unit_price
2025-01-03,STORE01,RICE5KG,4,520
2025-01-03,STORE01,MILK1L,18,52
2025-01-04,STORE01,RICE5KG,2,520
```
**B. Product list** (required) — one row per product you sell. Example:
```
sku_id,pack_size,perishable
RICE5KG,1,false
MILK1L,12,true
```
**C. Current stock on hand** (optional but valuable) — one row per item per day. Example:
```
date,store_id,sku_id,on_hand_qty
2025-01-04,STORE01,RICE5KG,11
```
**D. Suppliers + lead time** (see Step 3) — `supplier_id, moq, order_cycle`.

**We do NOT need:** customer names, prices you don't have, loyalty data, anything not above.
`sku_id` must be the **same code** in the sales file and the product list — that's the #1 thing
that trips first contact.

---

## Step 2 — Check the data BEFORE anything else (expect it to complain)

```
python -m src.ingest.validation --sales sales.csv --product-master products.csv --inventory stock.csv
```

> **The checker WILL flag things on first contact. That is it working, not breaking.** Real stores
> always have mess the demo data didn't. The whole first week is: *flag → fix → re-export → re-run.*

**It either QUARANTINES the batch (must fix) or WARNS (we handle it):**

| BLOCK — fix before ingesting | Almost always means | Fix |
|---|---|---|
| `referential_sku` (SKU in sales, not in product list) | new item never added to the product file, or a stale product file | add the missing items to products.csv |
| `null_keys` (blank date/store/sku) | export glitch / empty cells | remove or fill the blank rows |
| `schema:...` on `date` (future date) | wrong date format or a typo (2205 not 2025) | fix the dates |
| `schema:...` on `qty` (not a whole number) | a weight/volume item, or a decimal export | send whole units; weight items need separate handling |
| `duplicate_grain` (two stock rows, same item+day) | export ran twice / merged files | de-duplicate |
| `grn_orphans` (a receipt with no matching order) | partial PO records | only send receipts you can match to an order |

| WARN — no action needed, just so you know | What we do |
|---|---|
| `returns_present` (negative qty) | we split returns out of demand automatically |
| `voids_present` (zero qty) | ignored as no-sale |
| `calendar_gaps` (missing days) | we fill them as zero-sales days |
| `missing_moq` / `missing_order_cycle` / `no_leadtime_history` | we use the configured fallback lead time (Step 3) |
| `duplicate_receipts` | we de-duplicate before counting stock |

Re-run the command until it prints **"No blocking problems — this batch can be ingested."**

---

## Step 3 — Lead times: a rough guess is fine to start

The reorder math needs, per supplier: *how many days from ordering to arrival?* Most stores
**won't** have clean records for this. That's expected and not a blocker.

> **Just ask the shopkeeper:** "When you order from this supplier, how many days till it arrives?"
> **One number per supplier is enough for week one.** Put it in `suppliers.csv` as `order_cycle`,
> or set a single default in `config/policy.yaml` → `lead_time.default_days`.

Because these are guesses, the early numbers validate *decisions*, not absolute service levels —
the same caveat the whole project carries until real lead-time records exist.

---

## Step 4 — Week one: run shadow mode, watch the REJECT RATE (not accuracy)

Each day, after scoring produces a recommendations file:
```
python -m src.serve.shadow --recs recommendations.csv
```
It prints a **reject rate** — the fraction of recommendations a shopkeeper would look at and say
*"that's obviously wrong."* **That, not forecast accuracy, is the week-one signal.** Every flag
points at a data/config fix, not a model problem:

| Flag | Chase this |
|---|---|
| `implausibly_large` | pack size or units wrong; stale demand history |
| `order_despite_ample_stock` | the stock-on-hand file is stale or missing |
| `below_moq` / `not_pack_multiple` | supplier MOQ / pack size not loaded correctly |
| `order_flagged_but_zero_qty` | edge case — send us the row |
| `exceeds_shelf_life_demand` | perishable shelf-life set wrong |

**Daily loop:** run shadow → sit with the shopkeeper → for each flagged item ask "is this wrong,
and why?" → fix the data/config behind it → re-run. A high reject rate is a **data/config problem
to chase**, not a verdict on the model.

---

## Step 5 — Go / no-go: leaving shadow mode

The system **stays in SHADOW and acts on nothing** until a deliberate decision, gated on a modest,
stated bar — not a vibe:

> **Leave shadow only when BOTH hold:** reject rate **below ~5%** for **5 consecutive days**, AND
> the shopkeeper agrees the recommendations are reasonable.

Going live is also blocked in code until a real lead-time feed is supplied:
`KIRANA_MODE=live` will refuse to run unless `KIRANA_REAL_LEADTIME_FEED=1` is set
(see `src/serve/settings.py`). Default mode is `shadow`; you don't need to set anything to stay safe.

---

## If you get stuck
- Validation keeps quarantining after fixes → save the printed `[BLOCK]` lines and the file, send both.
- Reject rate stays high after data fixes → save the shadow output + the recommendations file.
- Everything in this runbook is meant to be doable without us in the room; if a step isn't, that's
  a runbook bug — tell us which line.
