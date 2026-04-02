import json
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from config import settings


class JobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._memory: dict[str, dict[str, Any]] = {}
        self._redis: Redis | None = None

        if settings.redis_url:
            try:
                self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except RedisError:
                self._redis = None

    def _key(self, job_id: str) -> str:
        return f"shorts:job:{job_id}"

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


job_store = JobStore()
