from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import settings


def upload_path(task_id: str, ext: str) -> Path:
    ext = ext if ext.startswith(".") else f".{ext}"
    return settings.uploads_dir / f"{task_id}{ext}"


def result_path(task_id: str) -> Path:
    return settings.results_dir / f"{task_id}.json"


def keyframes_dir(task_id: str) -> Path:
    d = settings.keyframes_dir / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def keyframe_path(task_id: str, filename: str) -> Path:
    return settings.keyframes_dir / task_id / filename


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
