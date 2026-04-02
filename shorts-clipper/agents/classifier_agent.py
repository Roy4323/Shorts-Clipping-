import json
from typing import Any

import httpx

from api.models import ClassificationResult, ContentType, VideoMetadata
from config import settings


ALLOWED_CONTENT_TYPES = {
    "podcast_interview",
    "music_lyrics",
    "music_instrumental",
    "sports_action",
    "mute_broll",
    "general",
}


def _build_prompt(metadata: VideoMetadata) -> str:
    return (
        "Classify this YouTube video into exactly one of these labels: "
        "podcast_interview, music_lyrics, music_instrumental, sports_action, "
        "mute_broll, general.\n"
        "Return JSON only with keys content_type and reason.\n\n"
        f"Title: {metadata.title}\n"
        f"Description: {metadata.description[:1000]}\n"
        f"Tags: {', '.join(metadata.tags[:20])}\n"
        f"Categories: {', '.join(metadata.categories[:10])}\n"
        f"Duration: {metadata.duration}\n"
        f"Uploader: {metadata.uploader}\n"
        f"Auto captions available: {metadata.auto_caption_available}\n"
    )


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    content_type = data.get("content_type")
    if content_type not in ALLOWED_CONTENT_TYPES:
        return None

    return data


def _heuristic_classification(metadata: VideoMetadata) -> ClassificationResult:
    haystack = " ".join(
        [
            metadata.title,
            metadata.description,
            " ".join(metadata.tags),
            " ".join(metadata.categories),
        ]
    ).lower()

    content_type: ContentType = "general"
    reason = "Defaulted to general content based on metadata."

    if any(word in haystack for word in ("podcast", "interview", "guest", "episode")):
        content_type = "podcast_interview"
        reason = "Podcast/interview keywords found in title, description, or tags."
    elif any(word in haystack for word in ("sport", "football", "basketball", "goal", "match")):
        content_type = "sports_action"
        reason = "Sports/action keywords found in metadata."
    elif any(word in haystack for word in ("lyrics", "official audio", "music video", "song")):
        content_type = "music_lyrics"
        reason = "Music/lyrics keywords found in metadata."
    elif metadata.duration is not None and metadata.duration < 60:
        content_type = "mute_broll"
        reason = "Very short duration suggests visual-first or b-roll content."

    return ClassificationResult(
        content_type=content_type,
        source="heuristic",
        reason=reason,
    )


def _openai_classification(metadata: VideoMetadata) -> ClassificationResult | None:
    if not settings.openai_api_key:
        return None

    payload = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": "You classify YouTube videos. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": _build_prompt(metadata),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 120,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None

    return ClassificationResult(
        content_type=parsed["content_type"],
        source="openai",
        reason=parsed.get("reason") or "OpenAI classification returned no reason.",
    )


def _gemini_classification(metadata: VideoMetadata) -> ClassificationResult | None:
    if not settings.gemini_api_key:
        return None

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_prompt(metadata),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 120,
            "responseMimeType": "application/json",
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    raw = "".join(part.get("text", "") for part in parts).strip()
    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None

    return ClassificationResult(
        content_type=parsed["content_type"],
        source="gemini",
        reason=parsed.get("reason") or "Gemini classification returned no reason.",
    )


def classify_content(metadata_dict: dict[str, Any]) -> ClassificationResult:
    metadata = VideoMetadata.model_validate(metadata_dict)

    if settings.llm_provider == "openai":
        return _openai_classification(metadata) or _heuristic_classification(metadata)

    if settings.llm_provider == "gemini":
        return _gemini_classification(metadata) or _heuristic_classification(metadata)

    return (
        _openai_classification(metadata)
        or _gemini_classification(metadata)
        or _heuristic_classification(metadata)
    )
