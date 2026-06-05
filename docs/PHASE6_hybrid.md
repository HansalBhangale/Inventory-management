# Phase 6 — Hybrid Continuous Learning (validated machinery)

> **M5 is a fixed historical extract — no live feed, promotions, or regime shifts.** The phenomena the hybrid exists to handle are NOT present, so this validates that each component WORKS, not that the hybrid lifts production accuracy. The win materializes only on a real store's evolving stream. Promotion uses the **inventory-frontier metric**, never MASE/WAPE.

## 1. Online corrector adapts to a SYNTHETIC regime shift (proof the fast layer works)

- pre-shift  : base MAE 0.399  vs corrected 0.413
- post-shift : base MAE 5.987  vs corrected **0.452** (corrector absorbs the shift the base can't see)

## 2. Drift detector latency on an injected shift

- shift injected at step 400; alarms=1; first alarm after shift at step 416 (latency 16 steps); **zero alarms while stationary**.

## 3. Online corrector on REAL M5 residuals (the honest caveat, measured)

- 6,300 held-out obs: base MAE 0.9353 vs corrected 0.9602. ~neutral, as expected — the base forecast's residuals on a shock-free extract are near-zero-mean noise with nothing for the fast layer to exploit. On live data with drift/promos, this is where it earns its keep.

## Components

- `src/models/online_layer.py` — River residual corrector + EWMA level corrector (fast).
- `src/continuous/drift.py` — ADWIN/Page-Hinkley/KSWIN per-segment monitors (event).
- `src/continuous/registry.py` — champion/challenger; promote only on a frontier-metric improvement beyond the configured margin.
- `src/continuous/retrain.py` — trigger logic + the reference daily loop (wired in Phase 9).
