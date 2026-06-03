"""Download bootstrap datasets from Kaggle (Phase 1.1).

Reads the dataset registry from config/data_sources.yaml and pulls each requested
dataset/competition via the Kaggle CLI, unzipping into data/raw/<dest>.

Prerequisites
-------------
1. `pip install kaggle` (in requirements.txt).
2. Kaggle API token at ~/.kaggle/kaggle.json  (kaggle.com -> Settings -> API -> Create New Token).
3. For competitions, accept the rules on the competition page first, otherwise the
   API returns 403.

Usage
-----
    python -m src.ingest.download_data --list
    python -m src.ingest.download_data --dataset m5_accuracy m5_uncertainty
    python -m src.ingest.download_data --all
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "data_sources.yaml"


def load_registry() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _kaggle_api():
    """Import and authenticate the Kaggle API, with a friendly error if not configured."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle not installed. Run: pip install -r requirements.txt")

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        sys.exit(
            f"Kaggle token not found at {token}.\n"
            "  1. kaggle.com -> Settings -> API -> Create New Token (downloads kaggle.json)\n"
            f"  2. Move it to {token}\n"
        )
    api = KaggleApi()
    api.authenticate()
    return api


def _unzip_all(dest: Path) -> None:
    for zf in dest.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()  # remove the zip after extracting


def download_one(name: str, spec: dict, kind: str, api) -> None:
    dest = ROOT / spec["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading {kind} '{spec['slug']}' -> {dest}")
    if kind == "competition":
        api.competition_download_files(spec["slug"], path=str(dest), quiet=False)
    else:
        api.dataset_download_files(spec["slug"], path=str(dest), quiet=False, unzip=False)
    _unzip_all(dest)
    files = sorted(p.name for p in dest.iterdir())
    print(f"[{name}] done. files: {files}")


def main(argv: list[str] | None = None) -> int:
    reg = load_registry()
    comps = reg.get("kaggle", {}).get("competitions", {}) or {}
    dsets = reg.get("kaggle", {}).get("datasets", {}) or {}
    all_keys = {**{k: ("competition", v) for k, v in comps.items()},
                **{k: ("dataset", v) for k, v in dsets.items()}}

    ap = argparse.ArgumentParser(description="Download Kaggle bootstrap datasets.")
    ap.add_argument("--dataset", nargs="+", metavar="NAME",
                    help="registry keys to download (see --list)")
    ap.add_argument("--all", action="store_true", help="download everything in registry")
    ap.add_argument("--list", action="store_true", help="list available registry keys")
    args = ap.parse_args(argv)

    if args.list or (not args.dataset and not args.all):
        print("Available datasets in config/data_sources.yaml:")
        for k, (kind, spec) in all_keys.items():
            print(f"  {k:20s} [{kind}] {spec['slug']}  (role: {spec.get('role','-')})")
        return 0

    targets = list(all_keys) if args.all else args.dataset
    unknown = [t for t in targets if t not in all_keys]
    if unknown:
        sys.exit(f"Unknown dataset key(s): {unknown}. Use --list to see options.")

    api = _kaggle_api()
    for t in targets:
        kind, spec = all_keys[t]
        download_one(t, spec, kind, api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
