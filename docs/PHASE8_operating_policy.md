# Phase 8 — Segmented Operating Policy (out-of-sample re-gate)

**VERDICT: PASS**  ·  regime=base  ·  operating q=0.95  ·  baseline=seasonal_naive

> The reorder engine routes each ABC tier to the buffer that wins the inventory frontier at the common operating quantile (doc 7.7 'combination of methods'). RULE: naive only where LGBM strictly LOSES; keep LGBM on WIN and TIE. The map is decided on one series split and graded on a DISJOINT held-out split, so the PASS is not circular. PASS = aggregate WIN AND no slice is a LOSS.


**Routing map** (by ABC; decided out-of-sample on 2,712 series; graded on 2,643):

- `A` → **lgbm**
- `B` → **lgbm**
- `C` → **lgbm**


*On shock-free M5 no tier strictly loses at the operating quantile, so the router makes no overrides (all LGBM) and the segmented system equals the single global model here. The router is the standing, config-driven mechanism for when real data flips a segment.*


## Re-gate on held-out split

| slice              |   fill |   DOH_combined |   DOH_naive@fill |   inv_saved | verdict   |
|:-------------------|-------:|---------------:|-----------------:|------------:|:----------|
| AGGREGATE          |  0.986 |           5.76 |             6.05 |       0.048 | WIN       |
| ABC=A              |  0.99  |           5.54 |             5.75 |       0.036 | WIN       |
| ABC=B              |  0.973 |           7.47 |             8.56 |       0.127 | WIN       |
| ABC=C              |  0.927 |           9.06 |            10.88 |       0.167 | WIN       |
| class=smooth       |  0.997 |           5.36 |             5.53 |       0.031 | WIN       |
| class=erratic      |  0.98  |           5.45 |             5.53 |       0.014 | TIE       |
| class=lumpy        |  0.967 |           6.45 |             7.51 |       0.142 | WIN       |
| class=intermittent |  0.973 |           7.23 |             8.49 |       0.148 | WIN       |

## Reading it

- Dominates in aggregate and loses on no slice. Value concentrates where it should: the hard, cash-tying tail (lumpy/intermittent) and the B/C tiers; ties on the easy head.

## Diagnostic — A at its 99% aspirational target (NOT a gate input)

- **A @ q99: TIE** (inv_saved -0.008). LGBM over-buffers A at 99% on SHOCK-FREE M5: with no promos/festivals/stockouts the q0.99 tail buffer protects against nothing, so seasonal-naive's tail-blind buffer is tighter. On a real store A-items spike on festivals/salary days and this is expected to FLIP. Recorded as a data artifact, not a settled limitation; if A is operated at 99% on such data, the router routes A→naive (config `operating_routing`, `redecide: true`).

- M5 lead times are ASSUMED; this validates decision quality + machinery, not absolute service numbers.
