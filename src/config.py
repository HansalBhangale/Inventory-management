"""Central config access. Loads the YAML files under config/ once and caches them.

    from src.config import CONFIG
    CONFIG.model["forecast"]["horizon_days"]   # 14
    CONFIG.policy["service_levels"]["A"]
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class _Config:
    root = ROOT
    config_dir = CONFIG_DIR
    data_dir = DATA_DIR

    @cached_property
    def model(self) -> dict:
        return _load("model.yaml")

    @cached_property
    def policy(self) -> dict:
        return _load("policy.yaml")

    @cached_property
    def metrics(self) -> dict:
        return _load("metrics.yaml")

    @cached_property
    def features(self) -> dict:
        return _load("features.yaml")

    @cached_property
    def data_contract(self) -> dict:
        return _load("data_contract.yaml")

    @cached_property
    def data_sources(self) -> dict:
        return _load("data_sources.yaml")

    # Convenience accessors used across the codebase.
    @property
    def horizon(self) -> int:
        return self.model["forecast"]["horizon_days"]

    @property
    def quantiles(self) -> list[float]:
        return self.model["forecast"]["quantiles"]


CONFIG = _Config()
