import json
from pathlib import Path
from typing import Any

import certifi
import httpx

from api.models import TranscriptResult, TranscriptSegment
from config import settings
from utils.logger import logger

SUPADATA_URL = "https://api.supadata.ai/v1/youtube/transcript"
CACHE_DIR = Path("data/transcripts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_video_id(url: str) -> str:
    """Simple video ID extractor from URL."""
    import re
    match = re.search(r"(?:v=|\/|be\/|shorts\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else "unknown"


def _mock_transcript(url: str) -> TranscriptResult:
    return TranscriptResult(
        source="mock",
        segments=[
            TranscriptSegment(
                start_sec=0,
                end_sec=8,
                text=f"Transcript fallback placeholder for {url}.",
            )
    ]
    )


def fetch_transcript(url: str) -> TranscriptResult:
    video_id = _get_video_id(url)
    cache_path = CACHE_DIR / f"{video_id}.json"

    # Check Cache
    if cache_path.exists():
        logger.info(f"💾 Loading transcript from cache: {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = [TranscriptSegment(**s) for s in data["segments"]]
            return TranscriptResult(source="cache", segments=segments)
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_path}: {e}")

    if not settings.supadata_api_key:
        logger.warning("Supadata API key missing, using mock transcript.")
        return _mock_transcript(url)

    headers = {"x-api-key": settings.supadata_api_key}
    params = {"url": url}

    logger.info(f"🌐 Fetching transcript for {url}")
    try:
        # Use certifi for SSL and a custom client to handle logging
        with httpx.Client(timeout=30.0, verify=certifi.where()) as client:
            response = client.get(SUPADATA_URL, headers=headers, params=params)
            if response.status_code != 200:
                logger.error(f"Supadata error: {response.status_code} - {response.text}")
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"Supadata HTTP error: {e}")
        return _mock_transcript(url)

    data: dict[str, Any] = response.json()
    logger.debug(f"Supadata response keys: {list(data.keys())}")

    # Supadata returns {"content": [...], "lang": "en", ...}
    # Each item: {"text": str, "offset": int (ms), "duration": int (ms)}
    # Fallback to other key names for forward-compatibility.
    items = (
        data.get("content")
        or data.get("segments")
        or data.get("transcript")
        or data.get("data")
        or []
    )
    logger.info(f"Supadata returned {len(items)} raw items.")

    segments: list[TranscriptSegment] = []
    for item in items:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        # Supadata uses offset+duration in milliseconds
        if "offset" in item:
            start = float(item["offset"]) / 1000.0
            end = start + float(item.get("duration", 0)) / 1000.0
        else:
            start = float(item.get("start") or item.get("start_sec") or 0)
            end = float(item.get("end") or item.get("end_sec") or item.get("to") or start)

        segments.append(
            TranscriptSegment(
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                text=text,
            )
        )

    if not segments:
        logger.warning(f"No transcription segments returned for {url}, using mock fallback.")
        return _mock_transcript(url)

    res = TranscriptResult(source="supadata", segments=segments)

    # Save to Cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"video_id": video_id, "segments": [s.model_dump() for s in segments]}, f, indent=2)
        logger.info(f"✅ Transcript saved to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")

    return res
