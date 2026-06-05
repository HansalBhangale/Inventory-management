"""Retrain orchestration (Phase 6.6) — wires the three timescales into one loop.

  slow   : scheduled base-model retrain on a rolling window (captures annual seasonality)
  fast   : daily online residual correction (between retrains)
  event  : drift alarm -> off-cycle retrain of the affected slice

A retrain produces a CHALLENGER, scored on the inventory frontier, promoted only if it beats the
champion (registry.py). This module is the control flow; the heavy training is src/evaluate/
backtest.py and the scoring is src/evaluate/frontier_gate.py / operating_policy.py.

CAVEAT (M5): no live feed, promotions, or regime shifts exist in a fixed extract, so the loop is
VALIDATED MACHINERY, not a demonstrated lift — the phenomena it handles aren't present in M5.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrainDecision:
    should_retrain: bool
    scope: str          # "global" or a segment name
    reason: str


def decide_retrain(scheduled_due: bool, drift_segments: list[str]) -> RetrainDecision:
    """Trigger logic: scheduled cadence OR a drift alarm (Phase 6.6 pseudocode)."""
    if drift_segments:
        scope = "global" if len(drift_segments) >= 3 else drift_segments[0]
        return RetrainDecision(True, scope, f"drift alarm on {drift_segments}")
    if scheduled_due:
        return RetrainDecision(True, "global", "scheduled cadence")
    return RetrainDecision(False, "", "no trigger")


def daily_cycle(base_model, online, drift_monitors, scorer=None):
    """Reference daily loop (Phase 6.6). Pseudocode-level: the real I/O (ingest/score/publish) is
    wired in the Phase 9 DAG. Returns the retrain decision for the day.

        for each (store, sku):
            q      = base_model.predict_quantiles(features)
            q      = online.adjust(q, features)            # fast correction
            recs   = reorder_engine.decide(q, inventory, leadtime, policy)
            publish(recs)
        # when actuals land:
            err    = error(actuals, forecast)
            online.learn_one(features, actuals - base_central)
            fired  = drift_monitors.update(per_segment_err)
            return decide_retrain(scheduled_due, fired)
    """
    raise NotImplementedError(
        "Reference control flow only — wired into the orchestration DAG in Phase 9. "
        "Components (online_layer, drift, registry) are unit-validated; see tests + PHASE6 doc."
    )
