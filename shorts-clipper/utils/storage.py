import json
from pathlib import Path
from typing import Any

from config import settings


def get_jobs_root() -> Path:
    root = Path(settings.jobs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_job_dir(job_id: str) -> Path:
    job_dir = get_jobs_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def get_job_subdir(job_id: str, name: str) -> Path:
    subdir = get_job_dir(job_id) / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
