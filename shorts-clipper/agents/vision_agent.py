import base64
import json
import subprocess
import time
from pathlib import Path

import httpx

from api.models import Region
from config import settings
from utils.logger import logger

# Bounding box detection prompt (shared across providers)
_DETECTION_PROMPT = """
Analyze this video frame and detect specialized regions for vertical (9:16) reframing.
Return JSON ONLY with a list called 'regions'.
Each region must have: label ("face", "screen", "gameplay"), x1, y1, x2, y2 (coordinates 0-1000).

Guidelines:
- "face": Detect human head/shoulders.
- "screen": Detect the active area of a computer monitor, slide, or code editor.
- "gameplay": Detect the primary action area in a video game.
- If multiple people are present, list all distinct "face" regions.
- Prefer the most important visible subjects over tiny background subjects.
"""

_SAMPLE_RATIOS = (0.25, 0.5, 0.75)
_MAX_RETRIES = 3
_RATE_LIMIT_BACKOFF_SEC = 2.0
_EARLY_EXIT_SCORE = (450, 0, 0)


def _probe_duration(video_path: str) -> float | None:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception as exc:
        logger.error(f"[VISION] Failed to probe duration: {exc}")
        return None


def extract_sample_frames(video_path: str, output_dir: Path) -> list[tuple[float, Path]]:
    """Extract a small set of representative frames across the clip."""
    duration = _probe_duration(video_path)
    if not duration or duration <= 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[tuple[float, Path]] = []

    for index, ratio in enumerate(_SAMPLE_RATIOS, start=1):
        timestamp = max(0.0, duration * ratio)
        frame_path = output_dir / f"vision_sample_{index:02d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            video_path,
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and frame_path.exists():
            frame_paths.append((timestamp, frame_path))

    logger.info(f"[VISION] Extracted {len(frame_paths)} sample frames from {Path(video_path).name}")
    return frame_paths


def _parse_regions(raw_text: str) -> list[Region]:
    payload = json.loads(raw_text)
    regions: list[Region] = []
    for item in payload.get("regions", []):
        try:
            regions.append(
                Region(
                    label=item["label"],
                    x1=item["x1"],
                    y1=item["y1"],
                    x2=item["x2"],
                    y2=item["y2"],
                    score=float(item.get("score", 1.0)),
                )
            )
        except KeyError:
            continue
    return regions


# ── Provider-specific detection ──────────────────────────────────────────────


def _detect_via_openai(image_data: str, frame_name: str) -> list[Region]:
    """Call OpenAI Vision API (gpt-4o-mini) with base64 image."""
    payload = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _DETECTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                raw_text = data["choices"][0]["message"]["content"]
                return _parse_regions(raw_text)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code != 429:
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                sleep_for = _RATE_LIMIT_BACKOFF_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"[VISION] OpenAI rate limited for {frame_name}; "
                    f"retry {attempt}/{_MAX_RETRIES} in {sleep_for:.1f}s."
                )
                time.sleep(sleep_for)
            except Exception as exc:
                last_error = exc
                raise

    if last_error is not None:
        raise last_error
    return []


def _detect_via_gemini(image_data: str, frame_name: str) -> list[Region]:
    """Call Gemini Vision API with base64 image."""
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _DETECTION_PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )

    last_error: Exception | None = None
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return _parse_regions(raw_text)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code != 429:
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                sleep_for = _RATE_LIMIT_BACKOFF_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"[VISION] Gemini rate limited for {frame_name}; "
                    f"retry {attempt}/{_MAX_RETRIES} in {sleep_for:.1f}s."
                )
                time.sleep(sleep_for)
            except Exception as exc:
                last_error = exc
                raise

    if last_error is not None:
        raise last_error
    return []


# ── Unified detection with provider selection ────────────────────────────────


def _detect_regions_from_frame(frame_path: Path) -> list[Region]:
    """Detect regions using the configured LLM provider (prefers OpenAI)."""
    with open(frame_path, "rb") as file:
        image_data = base64.b64encode(file.read()).decode("utf-8")

    provider = settings.llm_provider  # "auto", "openai", or "gemini"

    if provider == "openai" and settings.openai_api_key:
        logger.debug(f"[VISION] Using OpenAI for {frame_path.name}")
        return _detect_via_openai(image_data, frame_path.name)

    if provider == "gemini" and settings.gemini_api_key:
        logger.debug(f"[VISION] Using Gemini for {frame_path.name}")
        return _detect_via_gemini(image_data, frame_path.name)

    # Auto mode: prefer OpenAI, fall back to Gemini
    if settings.openai_api_key:
        logger.debug(f"[VISION] Auto-selected OpenAI for {frame_path.name}")
        return _detect_via_openai(image_data, frame_path.name)

    if settings.gemini_api_key:
        logger.debug(f"[VISION] Auto-selected Gemini for {frame_path.name}")
        return _detect_via_gemini(image_data, frame_path.name)

    logger.warning("[VISION] No vision API key available.")
    return []


def _region_selection_score(regions: list[Region]) -> tuple[int, int, int]:
    """Rank frames by how suitable they are for layout selection. Higher tuple wins."""
    face_count = sum(1 for region in regions if region.label == "face")
    has_screen = any(region.label == "screen" for region in regions)
    has_gameplay = any(region.label == "gameplay" for region in regions)

    if has_screen and face_count >= 1:
        return (500, face_count, len(regions))
    if has_gameplay and face_count >= 1:
        return (450, face_count, len(regions))
    if face_count >= 3:
        return (400, face_count, len(regions))
    if face_count == 2:
        return (350, face_count, len(regions))
    if face_count == 1:
        return (300, face_count, len(regions))
    if has_screen:
        return (200, face_count, len(regions))
    if has_gameplay:
        return (180, face_count, len(regions))
    return (0, face_count, len(regions))


def detect_regions(video_path: str) -> list[Region]:
    """Sample multiple frames and keep the most layout-informative one."""
    if not settings.openai_api_key and not settings.gemini_api_key:
        logger.warning("[VISION] No API key (OpenAI or Gemini) - skipping region detection.")
        return []

    provider = settings.llm_provider
    if provider == "auto":
        active = "OpenAI" if settings.openai_api_key else "Gemini"
    else:
        active = provider.capitalize()
    logger.info(f"[VISION] Using {active} for region detection.")

    sample_dir = Path(video_path).parent / f"{Path(video_path).stem}_vision_samples"
    sampled_frames = extract_sample_frames(video_path, sample_dir)
    if not sampled_frames:
        return []

    best_regions: list[Region] = []
    best_timestamp = 0.0
    best_score = (-1, -1, -1)

    try:
        for timestamp, frame_path in sampled_frames:
            try:
                regions = _detect_regions_from_frame(frame_path)
            except httpx.HTTPStatusError as exc:
                logger.error(f"[VISION] Detection failed for {frame_path.name}: {exc}")
                if exc.response.status_code == 429:
                    logger.warning("[VISION] Stopping further frame analysis due to rate limiting.")
                    break
                continue
            except Exception as exc:
                logger.error(f"[VISION] Detection failed for {frame_path.name}: {exc}")
                continue

            score = _region_selection_score(regions)
            labels = [region.label for region in regions]
            logger.info(
                f"[VISION] Frame {frame_path.name} @ {timestamp:.2f}s -> "
                f"regions={labels} score={score}"
            )
            if score > best_score:
                best_regions = regions
                best_timestamp = timestamp
                best_score = score
                if score >= _EARLY_EXIT_SCORE:
                    logger.info(
                        f"[VISION] Early exit at {timestamp:.2f}s after finding a strong layout candidate."
                    )
                    break

        logger.info(
            f"[VISION] Selected best frame @ {best_timestamp:.2f}s from {Path(video_path).name} "
            f"with {len(best_regions)} regions."
        )
        return best_regions
    finally:
        for _, frame_path in sampled_frames:
            if frame_path.exists():
                frame_path.unlink()
        if sample_dir.exists():
            try:
                sample_dir.rmdir()
            except OSError:
                pass
