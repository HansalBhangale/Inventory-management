"""Champion / Challenger registry & promotion (Phase 6.5).

A freshly trained model is a CHALLENGER; it runs in shadow and is scored on a rolling backtest.
Promote ONLY if it beats the champion by a margin — and the score is the **inventory-frontier
metric** (operating_policy / frontier_gate), NOT MASE/WAPE. We proved point accuracy is a proxy
that does not track business value, so promoting on it would build the wrong selection pressure
into the loop (and make drift retraining chase the wrong signal). All transitions are logged;
rollback = re-promote a prior version.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config import CONFIG

# Promotion margin reused from the champion/challenger config (Phase 6.5).
MIN_IMPROVEMENT = float(
    CONFIG.model["continuous_learning"]["champion_challenger"]["min_improvement_pct"]
) / 100.0


@dataclass
class ModelVersion:
    name: str
    score: float                       # frontier metric (e.g. aggregate inventory saved); higher is better
    metric: str = "frontier_inventory_saved"
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Registry:
    champion: ModelVersion | None = None
    history: list[str] = field(default_factory=list)

    def _log(self, msg: str) -> None:
        self.history.append(f"{datetime.now().isoformat(timespec='seconds')} {msg}")

    def consider(self, challenger: ModelVersion, min_improvement: float = MIN_IMPROVEMENT) -> bool:
        """Promote the challenger iff it beats the champion on the frontier metric by the margin.

        The gate is RELATIVE improvement on the frontier score; ties/regressions keep the champion.
        Returns True if promoted.
        """
        if self.champion is None:
            self.champion = challenger
            self._log(f"PROMOTE {challenger.name} (first champion, {challenger.metric}={challenger.score:.4f})")
            return True

        # both must be scored on the SAME metric for the comparison to be valid
        if challenger.metric != self.champion.metric:
            raise ValueError(
                f"metric mismatch: champion={self.champion.metric} challenger={challenger.metric}"
            )

        threshold = self.champion.score + abs(self.champion.score) * min_improvement
        if challenger.score > threshold:
            self._log(
                f"PROMOTE {challenger.name} ({challenger.score:.4f} > "
                f"{self.champion.name} {self.champion.score:.4f} + {min_improvement:.0%})"
            )
            self.champion = challenger
            return True

        self._log(
            f"KEEP {self.champion.name} ({self.champion.score:.4f}); "
            f"challenger {challenger.name} {challenger.score:.4f} below margin"
        )
        return False
