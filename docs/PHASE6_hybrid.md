# Phase 6 — Hybrid Continuous Learning (validated machinery)

> **M5 is a fixed historical extract — no live feed, promotions, or regime shifts.** The phenomena the hybrid exists to handle are NOT present, so this validates that each component WORKS, not that the hybrid lifts production accuracy. The win materializes only on a real store's evolving stream. Promotion uses the **inventory-frontier metric**, never MASE/WAPE.

## 1. Online corrector adapts to a SYNTHETIC regime shift (proof the fast layer works)

- pre-shift  : base MAE 0.399  vs corrected 0.413
- post-shift : base MAE 5.987  vs corrected **0.452** (corrector absorbs the shift the base can't see)

## 2. Drift detector latency on an injected shift

- shift injected at step 400; alarms=1; first alarm after shift at step 416 (latency 16 steps); **zero alarms while stationary**.

## 3. Does the online correction PRESERVE tail calibration? (business-critical)

The reorder engine depends on calibrated tails. The online layer applies a single location correction to all quantiles, so it must keep P90/P95/P99 coverage in band after it adapts. Post-adaptation coverage vs a stale-but-once-calibrated base:

- **Location shift** (level moves, spread constant): P90 0.902 · P95 0.952 · P99 0.992 — **restored to nominal**. Safe.
- **Magnitude drift** (demand 5→15, spread should widen): P90 0.77 · P95 0.839 · P99 0.925 — recovers most but **under-covers the upper tail**, because a location-only correction cannot widen the distribution.
- **Consequence (the hybrid justifying itself):** the online layer is trusted for LEVEL drift only; tail recalibration under magnitude growth is the job of the drift-triggered base RETRAIN (slow path). The online layer must never be relied on to maintain the tail the reorder engine reads. (Tests: test_continuous.py.)

## 3b. Magnitude-drift GUARD (closes the blind window)

The calibration finding above is a live hazard, not just a boundary: during a magnitude surge the online layer makes every fast metric look healthy while P95 coverage silently collapses — the engine under-buffers the spike (festival/salary-day) that most needs protection, through the drift latency + retrain lag. Guard (verified, test_guard.py):
- `RollingCoverageMonitor` (src/continuous/coverage_monitor.py): tail coverage as a FIRST-CLASS signal — flags the silent P95 collapse that point error misses.
- `MagnitudeShiftMonitor`: fires on an upward level surge (early warning), quiet when stationary.
- `ProtectiveBuffer` + `guarded_quantiles` (src/reorder/protective.py): while a breach is active, floor the served quantiles by a recent-volatility estimate (mean + z·recent_std). **Restores P95 coverage 0.84 → 0.94** during the stale-base window, until the drift-triggered retrain re-widens the learned tail.

## 4. Online corrector on REAL M5 residuals (the honest caveat, measured)

- 6,300 held-out obs: base MAE 1.1258 vs corrected 1.1613. ~neutral, as expected — the base forecast's residuals on a shock-free extract are near-zero-mean noise with nothing for the fast layer to exploit. On live data with drift/promos, this is where it earns its keep.

## Components

- `src/models/online_layer.py` — River residual corrector + EWMA level corrector (fast).
- `src/continuous/drift.py` — ADWIN/Page-Hinkley/KSWIN per-segment monitors (event).
- `src/continuous/registry.py` — champion/challenger; promote only on a frontier-metric improvement beyond the configured margin.
- `src/continuous/retrain.py` — trigger logic + the reference daily loop (wired in Phase 9).
