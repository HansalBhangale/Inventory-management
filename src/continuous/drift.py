"""Concept-drift detection (Phase 6.4) — the EVENT-DRIVEN retrain trigger.

Maintains a per-segment stream of forecast errors and runs a River drift detector (ADWIN /
Page-Hinkley) on each. A drift alarm for a slice triggers off-cycle retraining of that slice —
how the system reacts to regime changes faster than the weekly cadence.

CAVEAT (M5): a fixed extract has no genuine regime shift, so this is validated against INJECTED
synthetic drift (a mean shift in the error stream) — it proves the detector fires on real drift
and stays quiet on a stationary stream, not that M5 drifts.
"""
from __future__ import annotations

from river import drift


def _make(kind: str):
    if kind == "adwin":
        return drift.ADWIN()
    if kind == "page_hinkley":
        return drift.PageHinkley()
    if kind == "kswin":
        return drift.KSWIN()
    raise ValueError(f"unknown drift detector: {kind}")


class DriftMonitor:
    """One detector over a single error stream."""

    def __init__(self, kind: str = "adwin"):
        self.kind = kind
        self.det = _make(kind)
        self.n_seen = 0
        self.alarms: list[int] = []

    def update(self, error: float) -> bool:
        self.det.update(float(error))
        self.n_seen += 1
        fired = bool(self.det.drift_detected)
        if fired:
            self.alarms.append(self.n_seen)
        return fired


class SegmentDriftMonitors:
    """A drift detector per segment (e.g. intermittency class or ABC). Returns the set of
    segments that alarmed on this step — those are the retrain scopes."""

    def __init__(self, segments: list[str], kind: str = "adwin"):
        self.monitors = {s: DriftMonitor(kind) for s in segments}

    def update(self, errors: dict[str, float]) -> list[str]:
        fired = []
        for seg, err in errors.items():
            if seg in self.monitors and self.monitors[seg].update(err):
                fired.append(seg)
        return fired
