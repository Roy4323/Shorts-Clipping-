from typing import Any

import httpx

from api.models import TranscriptResult, TranscriptSegment
from config import settings


SUPADATA_URL = "https://api.supadata.ai/v1/youtube/transcript"


def _mock_transcript(url: str) -> TranscriptResult:
    return TranscriptResult(
        source="mock",
        segments=[
            TranscriptSegment(
                start_sec=0,
                end_sec=8,
                text=f"Transcript fallback placeholder for {url}.",
            )
        ],
    )


def fetch_transcript(url: str) -> TranscriptResult:
    if not settings.supadata_api_key:
        return _mock_transcript(url)

    headers = {"x-api-key": settings.supadata_api_key}
    params = {"url": url}

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(SUPADATA_URL, headers=headers, params=params)
            response.raise_for_status()
    except httpx.HTTPError:
        return _mock_transcript(url)

    data: dict[str, Any] = response.json()
    items = (
        data.get("segments")
        or data.get("transcript")
        or data.get("data")
        or []
    )

    segments: list[TranscriptSegment] = []
    for item in items:
        start = item.get("start") or item.get("start_sec") or item.get("offset") or 0
        end = item.get("end") or item.get("end_sec") or item.get("to") or start
        text = (item.get("text") or item.get("content") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_sec=float(start),
                end_sec=float(end),
                text=text,
            )
        )

    if not segments:
        return _mock_transcript(url)

    return TranscriptResult(source="supadata", segments=segments)
