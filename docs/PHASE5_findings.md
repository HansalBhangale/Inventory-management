# Phase 5 — Modeling Findings & Decisions (CA_1, fold 5)

> Evidence-driven record. Several doc prescriptions were **tested and rejected** on M5
> grocery data. Decisions below are backed by per-SKU MASE, the actual go-live gate.

## 1. Segment-aggregate MASE is misleading; the gate is per-SKU

The first backtest reported **A-items MASE 1.19** (worse than naive) — an artifact. That
number divided A's volume-weighted MAE by a *single global* seasonal-naive scale (the
average SKU's error). A-items have large absolute errors and the average SKU has a tiny
scale → inflated ratio.

Computed **per-SKU** (each SKU vs its own seasonal-naive scale), the order flips:

| segment | per-SKU median MASE | share MASE<1 |
|---|---|---|
| A (LGBM) | **0.79** | **0.756** |
| B (LGBM) | 0.90 | 0.621 |
| C (LGBM) | 0.99 | 0.508 |
| seasonal-naive (A) | 0.99 | 0.518 |

**LGBM beats seasonal-naive in every segment per-SKU; A is the strongest, not the weakest.**
→ The acceptance gate (`src/evaluate/acceptance.py`) is per-SKU across all folds.

## 2. The A-item blend was rejected (it hurts)

Blend `w·lgbm + (1-w)·seasonal_naive` on A, swept:

| w_lgbm | A median MASE | A share<1 |
|---|---|---|
| 0.0 (pure naive) | 0.99 | 0.518 |
| 0.5 | 0.85 | 0.671 |
| **1.0 (pure LGBM)** | **0.79** | **0.756** |

Monotonic — every step toward naive worsens A. **Decision: `a_blend_weight = 1.0`** (pure LGBM).

## 3. TSB routing for intermittent/lumpy was rejected (it hurts)

The doc routes intermittent/lumpy → TSB/Croston. Tested on this data:

| class | n | LGBM share<1 | TSB share<1 |
|---|---|---|---|
| intermittent | 2195 | **0.596** | 0.554 |
| lumpy | 546 | **0.690** | 0.658 |
| smooth | 232 | 0.888 | (LGBM) |
| erratic | 76 | 0.842 | (LGBM) |

**Why:** M5 grocery intermittent items retain day-of-week structure (weekend spikes). LGBM
captures it via lag/DOW features; TSB emits a *flat rate* and discards it — and since the
MASE baseline is *seasonal*-naive(7), the flat forecast loses. The doc's assumption
(intermittent ⇒ no exploitable pattern) does not hold for grocery here.

**Decision: `intermittent_model = lgbm`.** TSB/SBA kept config-switchable for re-test on
more data. The global LGBM is the champion across all segments.

## 3b. All-stores cross-learning result (THE decisive run)

Trained the global model on **all 10 stores (30,490 series)**, 3 rolling folds. Compared to
the CA_1 fold-5 anchor, cross-store learning helped — but **modestly**, and the intermittent
majority barely moved:

| segment / class | CA_1 fold5 share<1 | all-stores share<1 | delta |
|---|---|---|---|
| AB gate | 0.697 | **0.709** | +0.012 |
| A | 0.756 | 0.767 | +0.011 |
| B | 0.621 | 0.637 | +0.016 |
| smooth | 0.888 | 0.907 | +0.019 |
| erratic | 0.842 | 0.849 | +0.007 |
| lumpy | 0.690 | 0.709 | +0.019 |
| **intermittent (22,143 SKUs)** | 0.596 | **0.601** | **+0.005** |

**Fold variance is tight** (AB share<1 = 0.666 / 0.686 / 0.693; mean 0.682, std 0.014) — the
result is a stable signal, not a single-fold fluke. The methodology now holds up.

**Verdict:** the all-stores model is the champion (best + stable), but **more data alone does
not crack intermittent daily demand**. The 72% intermittent mass sits at median MASE ~0.92
(barely beating a seasonal-naive that is itself decent on weekly-structured grocery). Gate
still FAILs at 0.709 (need 0.80); binding constraint = B (0.637), which is intermittent-heavy.
The cross-learning thesis was directionally right but has diminishing returns on the sparse mass.

## 4. Metric-target recalibration

- **WAPE is not a gate at daily SKU grain.** The doc's "A-items <25%" is a weekly/aggregated
  target; at daily SKU level WAPE 50–70% is normal (A measured ~0.59). WAPE → sanity check.
- **sMAPE is unreliable on zero-heavy series** (zero days drive the term to 2.0). LGBM sMAPE
  (1.33) > naive (0.88) is the known pathology, not a regression. Track, don't gate.
- **Gate on per-SKU MASE + tail coverage.**

## 5. What is actually good

- **Tail calibration (drives Phase 7 reordering):** P90 coverage 0.891, P95 0.942 — both
  inside target bands. Safety-stock math runs off these and they're trustworthy.
- **Bias** −0.031 (within ±5%). No systematic under-ordering.
- LGBM beats naive on point accuracy in every segment.

## 6. Gate status & the real levers (untested)

Current (CA_1, fold5): **AB per-SKU share<1 = 0.697** (need 0.80). Binding constraint = **B**
(0.621) and the intermittent mass (72% of SKUs at 0.596). Not yet shippable.

High-probability levers, **not yet tried**, in priority order:
1. **True cross-store global model** — train on all 10 stores (~30k series). The core design
   principle (cross-learning); sparse intermittent SKUs borrow strength. Only 1 store tested.
2. **Optuna tuning** (Phase 5.6) — current params are hand-set.
3. **Online residual layer** (Phase 6) — daily adaptation.
4. **More folds** — confirm stability beyond fold 5.

## Reproduce
```
python -m src.evaluate.backtest  --stores CA_1 --folds 1     # train + backtest (champion)
python -m src.evaluate.acceptance --stores CA_1              # per-SKU gate (fast)
python -m src.evaluate.acceptance --stores CA_1 --compare-tsb --blend-sweep   # the evidence
python -m src.evaluate.backtest  --folds 1                   # THE LEVER: all-stores global
```
