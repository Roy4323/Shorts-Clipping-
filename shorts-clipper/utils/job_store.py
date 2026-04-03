import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from config import settings


class JobStore:
    """
    Job state store with three backends (in priority order):
      1. Redis (if REDIS_URL is set and reachable)
      2. Disk-backed in-memory dict (default) — persists to
         {JOBS_DIR}/{job_id}/job_state.json so jobs survive restarts
    """

    _STATE_FILE = "job_state.json"

    def __init__(self) -> None:
        self._lock  = Lock()
        self._memory: dict[str, dict[str, Any]] = {}
        self._redis: Redis | None = None

        if settings.redis_url:
            try:
                self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except RedisError:
                self._redis = None

        # Pre-load all persisted jobs from disk when Redis is unavailable
        if self._redis is None:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, job_id: str) -> str:
        return f"shorts:job:{job_id}"

    def _state_path(self, job_id: str) -> Path:
        return Path(settings.jobs_dir) / job_id / self._STATE_FILE

    def _persist(self, job_id: str, payload: dict[str, Any]) -> None:
        """Write payload to {JOBS_DIR}/{job_id}/job_state.json."""
        path = self._state_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # non-fatal — memory copy still works

    def _load_from_disk(self) -> None:
        """Scan JOBS_DIR for job_state.json files and warm the memory cache."""
        jobs_dir = Path(settings.jobs_dir)
        if not jobs_dir.exists():
            return
        loaded = 0
        for state_file in jobs_dir.glob(f"*/{self._STATE_FILE}"):
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                job_id = payload.get("job_id")
                if job_id:
                    self._memory[job_id] = payload
                    loaded += 1
            except (OSError, json.JSONDecodeError):
                continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, job_id: str, payload: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        base_payload = {
            "job_id": job_id,
            "created_at": now,
            "updated_at": now,
            **payload,
        }
        self.save(job_id, base_payload)

    def save(self, job_id: str, payload: dict[str, Any]) -> None:
        payload["updated_at"] = datetime.now(UTC).isoformat()

        if self._redis is not None:
            self._redis.set(self._key(job_id), json.dumps(payload))
            return

        with self._lock:
            self._memory[job_id] = payload
        self._persist(job_id, payload)

    def get(self, job_id: str) -> dict[str, Any] | None:
        if self._redis is not None:
            raw = self._redis.get(self._key(job_id))
            return json.loads(raw) if raw else None

        with self._lock:
            return self._memory.get(job_id)

    def update(self, job_id: str, **updates: Any) -> dict[str, Any]:
        current = self.get(job_id)
        if current is None:
            raise KeyError(f"Unknown job_id: {job_id}")

        current.update(updates)
        self.save(job_id, current)
        return current

    def list_all(self) -> list[dict[str, Any]]:
        """Return all jobs sorted newest first."""
        if self._redis is not None:
            keys = self._redis.keys("shorts:job:*")
            jobs = []
            for key in keys:
                raw = self._redis.get(key)
                if raw:
                    jobs.append(json.loads(raw))
        else:
            with self._lock:
                jobs = list(self._memory.values())

        return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


job_store = JobStore()
