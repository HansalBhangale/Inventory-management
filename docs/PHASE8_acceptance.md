# Phase 8 — Official Go-Live Gate: Inventory-Frontier Dominance

**VERDICT: PASS**  ·  regime=base  ·  baseline=seasonal_naive  ·  win_margin=0.03

> This REPLACES the per-SKU MASE gate (now a diagnostic in acceptance.py). Point accuracy is a proxy; this measures the true objective — service per unit of inventory. M5 lead times are ASSUMED, so this validates the decision quality and machinery, not absolute numbers.

## Rule

- Compare LGBM at the engine's operating quantile (A→q99, B/C→q95; per-class→q95) vs the seasonal-naive frontier, at MATCHED FILL. `inv_saved = (DOH_naive@fill − DOH_lgbm)/DOH_naive@fill`.
- WIN if inv_saved ≥ margin; LOSS if ≤ −margin (worse on both axes); else TIE.
- PASS = aggregate WIN **and** smooth/erratic/lumpy WIN **and** intermittent ≠ LOSS.

## Results

| slice              |   op_q |   fill |   DOH_lgbm |   DOH_naive@fill |   inv_saved | verdict   |
|:-------------------|-------:|-------:|-----------:|-----------------:|------------:|:----------|
| AGGREGATE          |   0.95 |  0.984 |       5.66 |             6.06 |       0.065 | WIN       |
| class=smooth       |   0.95 |  0.996 |       5.33 |             5.6  |       0.049 | WIN       |
| class=erratic      |   0.95 |  0.976 |       5.37 |             5.55 |       0.032 | WIN       |
| class=lumpy        |   0.95 |  0.967 |       6.33 |             7.47 |       0.153 | WIN       |
| class=intermittent |   0.95 |  0.962 |       6.77 |             8.19 |       0.174 | WIN       |
| ABC=A              |   0.99 |  0.994 |       7.27 |             6.73 |      -0.079 | LOSS      |
| ABC=B              |   0.95 |  0.974 |       7.14 |             8.75 |       0.184 | WIN       |
| ABC=C              |   0.95 |  0.94  |       8.15 |            10.69 |       0.237 | WIN       |

## Notes

- A-items operating quantile = q99. q0.99 head trained: **NO — q99 is currently a normal extrapolation of the q90/q95 spread; train the dedicated pinball head (config set) to refine the A tail**.

- The per-SKU MASE diagnostic (AB share<1 ≈ 0.71) is intentionally NOT a gate input.
