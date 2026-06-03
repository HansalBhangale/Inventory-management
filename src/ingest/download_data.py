"""Download bootstrap datasets from Kaggle (Phase 1.1).

Reads the dataset registry from config/data_sources.yaml and pulls each requested
dataset/competition via the Kaggle CLI, unzipping into data/raw/<dest>.

We shell out to the Kaggle CLI (`python -m kaggle ...`) rather than the Python API
because kaggle 2.x removed the `competition_download_files` Python method while keeping
the CLI stable across versions.

Prerequisites
-------------
1. `pip install kaggle` (in requirements.txt).
2. A Kaggle API token, either:
     - new style: ~/.kaggle/access_token  (a single KGAT_... line), or
     - classic:   ~/.kaggle/kaggle.json   ({"username":..., "key":...}), or
     - env var:   KAGGLE_API_TOKEN / KAGGLE_USERNAME+KAGGLE_KEY
3. For competitions, accept the rules on the competition page first, else the API 403s.

Usage
-----
    python -m src.ingest.download_data --list
    python -m src.ingest.download_data --dataset m5_accuracy m5_uncertainty
    python -m src.ingest.download_data --all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "data_sources.yaml"


def load_registry() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _check_credentials() -> None:
    kdir = Path.home() / ".kaggle"
    has_file = (kdir / "access_token").exists() or (kdir / "kaggle.json").exists()
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN")) or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    if not (has_file or has_env):
        sys.exit(
            f"No Kaggle credentials found.\n"
            f"  New token: kaggle.com -> Settings -> API -> Create New Token,\n"
            f"             then save the KGAT_... string to {kdir / 'access_token'}\n"
            f"  Classic:   save kaggle.json to {kdir / 'kaggle.json'}\n"
        )


def _unzip_all(dest: Path) -> None:
    for zf in dest.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()  # remove the zip after extracting


def download_one(name: str, spec: dict, kind: str) -> None:
    dest = ROOT / spec["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[{name}] downloading {kind} '{spec['slug']}' -> {dest}")

    base = [sys.executable, "-m", "kaggle", kind + "s", "download"]
    if kind == "competition":
        cmd = base + [spec["slug"], "-p", str(dest)]
    else:  # dataset
        cmd = base + ["-d", spec["slug"], "-p", str(dest)]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"[{name}] FAILED (exit {result.returncode}). "
            f"If 403: accept the competition rules at "
            f"https://www.kaggle.com/c/{spec['slug']} first.",
            file=sys.stderr,
        )
        return

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

    _check_credentials()
    for t in targets:
        kind, spec = all_keys[t]
        download_one(t, spec, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
