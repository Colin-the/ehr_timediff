"""Where the pipeline's Data/ directory is, and where runs go.

Resolution order, first hit wins:
  1. an explicit --data-root / --runs-root on the command line
  2. EHR_DATA_ROOT / EHR_RUNS_ROOT environment variables
  3. config/paths.yaml (copy config/paths.example.yaml and edit it)
  4. runs_root defaults to <project>/runs; data_root has no default

Relative paths from 1 and 2 are taken from the current directory, as usual.
Relative paths in paths.yaml are taken from the project folder, so a SLURM job
started from anywhere finds the same place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .arms import PROJECT_ROOT

PATHS_FILE = PROJECT_ROOT / "config" / "paths.yaml"


def _from_file(key: str) -> Optional[str]:
    if PATHS_FILE.exists():
        with open(PATHS_FILE) as f:
            value = (yaml.safe_load(f) or {}).get(key)
        if value:
            return os.path.expandvars(os.path.expanduser(str(value)))
    return None


def _lookup(cli: Optional[str], env: str, key: str) -> Optional[Path]:
    if cli:
        return Path(cli).resolve()
    if os.environ.get(env):
        return Path(os.environ[env]).resolve()
    value = _from_file(key)
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def data_root(cli: Optional[str] = None) -> Path:
    value = _lookup(cli, "EHR_DATA_ROOT", "data_root")
    if value is None or "CHANGE_ME" in str(value):
        raise SystemExit(
            "No pipeline data root configured. Pass --data-root, set EHR_DATA_ROOT, or copy "
            "config/paths.example.yaml to config/paths.yaml and set data_root to the "
            "pipeline's Data/ directory (the one containing mimic-iii/, mimic-iv/, ...)."
        )
    return value


def runs_root(cli: Optional[str] = None) -> Path:
    return _lookup(cli, "EHR_RUNS_ROOT", "runs_root") or PROJECT_ROOT / "runs"
