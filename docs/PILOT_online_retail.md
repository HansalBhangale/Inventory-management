# Real-data rehearsal — UCI Online Retail

> **Rehearsal, not pilot.** Real transactions, real prices, never-seen messy data taken through the
> whole runbook loop (ingest → contract → features → train → score → reorder → shadow). It's a UK
> online gift seller, not a kirana, with **no real lead times or inventory** (assumed). It proves
> the *machinery survives real data*; it does **not** prove a real service/inventory outcome.

## The milestone here is NOT the WAPE number — it's that the contract caught our own bugs
The most valuable outcome of this run: on first contact with real mess, the **adversarial data
contract caught two bugs that were invisible on clean M5** and would otherwise have surfaced at a
real shopkeeper's data — the worst possible moment. We found them on a free dataset because the
contract was built to break, not to pass:

1. **Case-inconsistent / multi-suffix StockCodes** (`15056BL` vs `15056bl` = same item; `79323GR`).
   The adapter's product filter was too strict and case-sensitive → broadened to allow 2-letter
   colour suffixes and **uppercased** codes to merge variants.
2. **int32 over-strictness** — DuckDB returns int32, the schema demanded exactly int64, so valid
   integer quantities were wrongly BLOCKED → replaced with a **whole-number check** that accepts
   any integer storage and still BLOCKs genuinely fractional qty (Favorita weight items).

After the documented cleaning (drop cancellations + non-product codes + non-positive prices, net
returns, daily grain) the contract **PASSES** (only the expected `calendar_gaps` WARN). That is the
full reject → fix → re-ingest loop, on real data — the thesis of the last several phases paying off.

## Forecast accuracy — ROLLING-ORIGIN (4 folds, UK), not a single window
A single 14-day window can flatter or bury a result by luck; we don't trust single-origin (the M5
discipline). Across 4 walk-forward folds with an H-day embargo:

| metric | mean | std |
|---|---|---|
| **WAPE** | **0.740** | 0.048 |
| naive WAPE (out-of-sample) | 1.237 | 0.149 |
| MASE (vs in-sample naive scale) | 1.020 | 0.192 |
| P90 coverage | 0.953 | 0.015 |
| P95 coverage | 0.975 | 0.008 |

**Read it precisely (defensible):**
- The model **beats the naive it actually competes against by ~40%** (WAPE 0.74 vs 1.24), with
  **low fold-to-fold variance** (std 0.05) — so the single-origin read held up; it's trustworthy now.
- **WAPE-beats-naive and MASE≈1 are not in tension** — they measure against *different* baselines:
  WAPE is vs the *out-of-sample* naive forecast; MASE is normalized by the *in-sample* naive scale.
  On sparse gift demand the in-sample scale sits near the model's error, so MASE hovers ~1 even as
  the model clearly beats the forecast it competes with. The >1 is a normalizing-baseline artifact
  on sparse data, not a sign it loses (same intermittency signature as M5's majority).
- **Tails are calibrated and stable, slightly conservative** (P90 0.95, P95 0.975 — just above
  target, tiny std). It over-buffers a touch rather than under-buffering — the safe direction.

## Full loop + shadow
UK: 259,993 daily rows · 3,785 SKUs → panel 946k rows → global quantile model → **2,542 (s,S)
recommendations** → **shadow reject rate ~1.5%** (only `order_despite_ample_stock`, an artifact of
the *assumed* inventory stand-in, not the model). ~98.5% of recommendations pass the
"obviously-wrong?" checks.

## What this proves — and what it doesn't
- **Proves:** the ingestion + data contract survive (and were hardened by) real-world mess; the
  full forecast→reorder→shadow loop runs end-to-end on real transactions and yields a model that
  beats naive ~40% with calibrated tails and sane orders.
- **Does NOT prove:** a real stockout/inventory reduction — needs real lead times, real inventory,
  and months. Online Retail has neither (3-day lead time and a recent-mean inventory stand-in were
  assumed). The validation gap sits exactly where it was; the machinery is just more trustworthy.
- **Still a proxy:** a UK gift seller is not a kirana. The decisive pilot remains a real shop's POS
  + a lead-time guess per supplier (PILOT_RUNBOOK.md / PILOT_ONEPAGER.md).

## Reproduce
```
python -m src.evaluate.pilot_online_retail --store "United Kingdom" --folds 4
```
