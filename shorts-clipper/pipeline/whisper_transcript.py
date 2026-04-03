"""
Whisper fallback transcript extractor.

Uses the OpenAI Whisper API to transcribe audio extracted from a video file.
Called when the primary transcript source (Supadata) returns no segments
(e.g. Hindi videos, non-English content, or private/unlisted videos).

Returns a TranscriptResult with source="whisper", or None if transcription fails.
"""

import subprocess
import tempfile
from pathlib import Path

from api.models import TranscriptResult, TranscriptSegment
from config import settings
from utils.logger import logger


def _extract_audio_for_whisper(video_path: str, out_path: str) -> bool:
    """Extract a mono 16kHz MP3 (small, fast upload) from the video."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-q:a", "5",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(f"[WHISPER] Audio extraction failed: {result.stderr.decode(errors='ignore')[:200]}")
        return False
    size_kb = Path(out_path).stat().st_size // 1024
    logger.info(f"[WHISPER] Audio extracted for Whisper: {Path(out_path).name} ({size_kb} KB)")
    return True


def transcribe_with_whisper(video_path: str) -> TranscriptResult | None:
    """
    Transcribe video audio using the OpenAI Whisper API.

    Returns TranscriptResult with source="whisper" on success, None on failure.
    """
    if not settings.openai_api_key:
        logger.warning("[WHISPER] No OpenAI API key — skipping Whisper transcription.")
        return None

    logger.info(f"[WHISPER] Starting Whisper transcription for {Path(video_path).name}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = str(Path(tmp_dir) / "whisper_audio.mp3")

        if not _extract_audio_for_whisper(video_path, audio_path):
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)

            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            raw_segments = response.segments or []
            if not raw_segments:
                logger.warning("[WHISPER] Whisper returned no segments.")
                return None

            segments: list[TranscriptSegment] = []
            for seg in raw_segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append(
                    TranscriptSegment(
                        start_sec=round(float(seg.start), 3),
                        end_sec=round(float(seg.end), 3),
                        text=text,
                    )
                )

            if not segments:
                logger.warning("[WHISPER] Whisper returned only empty segments.")
                return None

            logger.info(f"[WHISPER] Transcription complete: {len(segments)} segments.")
            return TranscriptResult(source="whisper", segments=segments)

        except Exception as exc:
            logger.error(f"[WHISPER] Whisper API call failed: {exc}")
            return None
