from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, project_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    artifact = project_path(config["output"]["artifact_dir"])
    submission = project_path(config["output"]["submission_dir"])
    paths = {
        "artifact": artifact,
        "audit": artifact / "audit",
        "folds": artifact / "folds",
        "models": artifact / "models",
        "oof": artifact / "oof",
        "predictions": artifact / "predictions",
        "experiments": artifact / "experiments",
        "reports": artifact / "reports",
        "submissions": submission,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: _json_value(value) for key, value in payload.items()}
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False, default=_json_value),
        encoding="utf-8",
    )


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                timeout=10, check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    root = run("rev-parse", "--show-toplevel")
    if not root:
        return {"is_repository": False, "branch": None, "commit": None, "dirty": None}
    status = run("status", "--porcelain")
    return {
        "is_repository": True,
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def memory_info() -> tuple[int | None, int | None]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        if platform.system() == "Windows":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.dwLength = ctypes.sizeof(status)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
            except (AttributeError, OSError):
                pass
    return None, None


def gpu_info() -> str:
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            result = subprocess.run(
                [nvidia, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().replace("\n", "; ")
        except (OSError, subprocess.SubprocessError):
            pass
    if platform.system() == "Windows":
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if powershell:
            try:
                result = subprocess.run(
                    [powershell, "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_VideoController).Name -join '; '"],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip() + " (no CUDA device detected)"
            except (OSError, subprocess.SubprocessError):
                pass
    return "Not detected"


def environment_markdown(data_paths: dict[str, Path | None]) -> str:
    git = git_info()
    total_mem, available_mem = memory_info()
    gib = 1024 ** 3
    packages = {
        "pandas": package_version("pandas"),
        "numpy": package_version("numpy"),
        "scikit-learn": package_version("scikit-learn"),
        "LightGBM": package_version("lightgbm"),
        "CatBoost": package_version("catboost"),
        "XGBoost": package_version("xgboost"),
    }
    path_lines = "\n".join(
        f"- {name}: `{path}` ({'present' if path and path.exists() else 'missing'})"
        for name, path in data_paths.items()
    )
    package_lines = "\n".join(f"- {name}: {version}" for name, version in packages.items())
    total_text = f"{total_mem / gib:.2f} GiB" if total_mem else "unknown"
    avail_text = f"{available_mem / gib:.2f} GiB" if available_mem else "unknown"
    return f"""# Environment Report

Generated: {utc_now()}

## System

- Operating system: {platform.platform()}
- Python: {platform.python_version()} (`{shutil.which('python') or 'unknown'}`)
- CPU logical cores: {os.cpu_count()}
- Total memory: {total_text}
- Available memory: {avail_text}
- GPU: {gpu_info()}

## Python packages

{package_lines}

## Data files

{path_lines}

## Git

- Repository: {git['is_repository']}
- Branch: {git['branch'] or 'N/A'}
- Commit: {git['commit'] or 'N/A'}
- Uncommitted changes: {git['dirty'] if git['dirty'] is not None else 'N/A'}

The workspace was inspected without cleaning or modifying pre-existing uncommitted work.
"""


def append_experiment_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        if row["experiment_id"] in set(existing.get("experiment_id", [])):
            raise FileExistsError(f"Experiment already recorded: {row['experiment_id']}")
        columns = list(dict.fromkeys([*existing.columns, *new.columns]))
        existing = existing.reindex(columns=columns)
        new = new.reindex(columns=columns)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(path, index=False)


def unique_experiment_id(base_id: str, output_paths: dict[str, Path]) -> str:
    collision = any(
        (output_paths[key] / name).exists()
        for key, name in (
            ("oof", f"{base_id}.csv"),
            ("predictions", f"{base_id}_test.csv"),
            ("models", base_id),
        )
    )
    if not collision:
        return base_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_id}_{stamp}"

