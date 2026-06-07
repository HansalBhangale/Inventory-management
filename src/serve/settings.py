"""Runtime/deployment settings (Phase 9) — the awaiting-real-data stub pattern.

Deployment concerns (mode, DB, object store, paths) come from the environment with safe local
defaults. M5-SPECIFIC business values (lead-time regimes, costs, margins, the routing map) are NOT
hard-coded here — they live in config/*.yaml, stubbed for M5 and swappable for a pilot. The test
for anything in this file: will it survive contact with a real store unchanged? If it's M5
statistics, it belongs in config behind an `awaiting_real_data` marker, not here.

Production swap: pydantic-settings (typed env validation). Kept dependency-free here so the
groundwork runs in CI without extra installs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    SHADOW = "shadow"   # generate recommendations, compare to actuals, act on NOTHING (pilot default)
    LIVE = "live"       # recommendations drive real POs (only after shadow is trusted)


@dataclass(frozen=True)
class Settings:
    mode: Mode = Mode(os.getenv("KIRANA_MODE", "shadow"))
    database_url: str = os.getenv("KIRANA_DB_URL", "postgresql://localhost:5432/kirana")
    artifact_store: str = os.getenv("KIRANA_ARTIFACT_STORE", "s3://kirana-artifacts")
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    data_dir: str = os.getenv("KIRANA_DATA_DIR", "data")
    # Retrain is MANUAL by default — no autonomous loop against a runtime with no real drift.
    autonomous_retrain: bool = os.getenv("KIRANA_AUTONOMOUS_RETRAIN", "false").lower() == "true"

    def assert_safe_for_live(self) -> None:
        """A real store should never flip to LIVE without the shadow run + a real lead-time feed."""
        if self.mode is Mode.LIVE and os.getenv("KIRANA_REAL_LEADTIME_FEED") != "1":
            raise RuntimeError(
                "Refusing LIVE mode: no real lead-time feed. Run shadow mode against real data "
                "and supply KIRANA_REAL_LEADTIME_FEED=1 before acting on recommendations."
            )


SETTINGS = Settings()
