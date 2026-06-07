# Favorita Dress Rehearsal — promo + perishable paths on a 2nd real dataset

> **Goal (not product validation):** exercise the two code paths M5 could never trigger —
> **promotions** and **perishables** — on structurally different real grocery data, to find where
> they silently no-op. Favorita has no prices/costs (like M5 had no lead times), so newsvendor
> economics are ASSUMED. This hardens the engine; it does **not** move us toward a real store.

## The exercise paid off before any modelling
A code audit (not Favorita) found the premise was wrong: the two paths weren't *dormant*, they
were **unimplemented** — declared in config, absent from code:
- promo features: listed in `features.yaml`, never built in `build_features.py`.
- newsvendor/perishable: `use_newsvendor: true` in `policy.yaml`, no code anywhere.

So feeding Favorita's `onpromotion`/`perishable` flags would have no-opped. We **built both first**
(`src/features/promo.py`, `src/reorder/newsvendor.py` + `dispatch_reorder`, 13 unit tests), then
used Favorita to confirm they engage on real data.

## Favorita at a glance (EDA)
125.5M rows · 54 stores · 4,036 items · 2013–2017. `onpromotion` **6.2% True** / 76.5% False /
17.3% null (pre-tracking). **24% of items perishable.** `unit_sales` is **6.5% fractional**
(weight items) and 0.006% negative (returns) — the exact mess M5 lacked.

## Results (stores 1/2/3/44, since 2016-06; 3.27M train / 575k valid rows)

**1. Data contract correctly meets the mess (raw Favorita):**
- `[BLOCK] qty non-integer` — fractional weight-item sales quarantined (the collision, as designed).
- `[WARN] returns_present` (negatives) · `[WARN] calendar_gaps` (7,378 series) — flagged, handled.
- Adapter decision: round weight-item sales to whole units; returns flow to the WARN path.

**2. Promo path ENGAGED (not silently broken):** gain importance, 14 features —

| feature | rank | note |
|---|---|---|
| roll_mean_28 / roll_mean_7 / lag_14 | 1–3 | demand history dominates (expected) |
| **on_promo** | **5** | top-5 — the model genuinely uses the promo signal |
| promo_in_next_7d / promo_in_last_7d | 12 / 13 | weak — forward/back windows add little |
| perishable | 14 | low as a *demand* feature — expected; it drives the *reorder* branch, not demand |

→ The promo feature path is real: `on_promo` outranks lag_28, day_of_week, store_id. Honest
caveat: the windowed promo features are near-noise here; the same-day flag carries it.

**3. Perishable path FIRES:** of 3,000 sampled items (830 perishable), `dispatch_reorder` routed
**all 830 perishable → newsvendor** and the 2,170 others → (s,S). The branch engages for exactly
the right items instead of defaulting through the non-perishable path.

## Verdict
Both previously-unimplemented paths now exist, are unit-tested, and **engage on a second real
grocery dataset** including promotions and perishables the engine had never met. The contract
meets real grocery mess (fractional/returns) as designed.

## Honest limits / not done
- Favorita has **no prices** → newsvendor economics (Cu/Co) are assumed; this validates the path
  *fires and is correct*, not its tuned output.
- A full **frontier-gate OOD run** on Favorita (inventory sim with assumed economics + lead times)
  was not done — it's a larger follow-up; the path-engagement diagnostics above answer the
  exercise's actual question.
- **Favorita is still a proxy.** It makes the engine more robust for a pilot; the decisive test
  remains a real store's POS (see PILOT_RUNBOOK.md).

## Reproduce
```
python -m src.evaluate.favorita_exercise --stores 1 2 3 44 --since 2016-06-01
```
