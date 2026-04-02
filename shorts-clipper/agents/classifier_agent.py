from typing import Any

from api.models import ClassificationResult, ContentType, VideoMetadata
from config import settings

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    Anthropic = None  # type: ignore[assignment]


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


def _anthropic_classification(metadata: VideoMetadata) -> ClassificationResult | None:
    if not settings.anthropic_api_key or Anthropic is None:
        return None

    client = Anthropic(api_key=settings.anthropic_api_key)
    prompt = (
        "Classify this YouTube video into exactly one of: "
        "podcast_interview, music_lyrics, music_instrumental, sports_action, "
        "mute_broll, general.\n"
        "Return compact JSON with keys content_type and reason.\n\n"
        f"Title: {metadata.title}\n"
        f"Description: {metadata.description[:1000]}\n"
        f"Tags: {', '.join(metadata.tags[:20])}\n"
        f"Categories: {', '.join(metadata.categories[:10])}\n"
        f"Duration: {metadata.duration}\n"
        f"Auto captions available: {metadata.auto_caption_available}\n"
    )

    try:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=120,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    text_parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            text_parts.append(text)

    raw = "".join(text_parts).strip()
    if not raw:
        return None

    import json

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None

    content_type = data.get("content_type")
    reason = data.get("reason") or "Anthropic classification returned no reason."
    allowed = {
        "podcast_interview",
        "music_lyrics",
        "music_instrumental",
        "sports_action",
        "mute_broll",
        "general",
    }
    if content_type not in allowed:
        return None

    return ClassificationResult(
        content_type=content_type,
        source="anthropic",
        reason=reason,
    )


def classify_content(metadata_dict: dict[str, Any]) -> ClassificationResult:
    metadata = VideoMetadata.model_validate(metadata_dict)
    anthropic_result = _anthropic_classification(metadata)
    if anthropic_result is not None:
        return anthropic_result
    return _heuristic_classification(metadata)
