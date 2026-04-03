"""TTS Engine — edge-tts wrapper.

Synthesizes a spoken hook script to an MP3 file and returns word-level
timestamps for use by hook_subtitles.py.

Uses edge-tts (Microsoft Edge TTS, free, no API key required, async).
Word boundary events give 100-nanosecond-resolution timestamps.

Supported voices:
    en-US-GuyNeural   (male, energetic)
    en-US-JennyNeural (female, warm)
    en-GB-RyanNeural  (male, British)
"""

import asyncio
from pathlib import Path

from utils.logger import logger

VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural"]
DEFAULT_VOICE = VOICES[0]

# edge-tts uses 100-nanosecond units for offset/duration
_NS100_TO_SEC = 1 / 10_000_000


async def _synthesize_async(text: str, voice: str, audio_path: str) -> list[dict]:
    """
    Async core: stream TTS audio to disk, collect WordBoundary events.

    Returns:
        List of {"word": str, "start": float, "end": float} in seconds.
    """
    import edge_tts  # lazy import — only needed when process_hooks=True

    communicate = edge_tts.Communicate(text, voice)
    word_timestamps: list[dict] = []

    with open(audio_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                start_sec = chunk["offset"] * _NS100_TO_SEC
                end_sec   = (chunk["offset"] + chunk["duration"]) * _NS100_TO_SEC
                word_timestamps.append({
                    "word":  chunk["text"],
                    "start": round(start_sec, 3),
                    "end":   round(end_sec,   3),
                })

    return word_timestamps


def synthesize_hook(
    text: str,
    audio_path: str,
    voice: str = DEFAULT_VOICE,
) -> list[dict]:
    """
    Synthesize hook text to audio and return word-level timestamps.

    Args:
        text       : Spoken hook script (30-40 words).
        audio_path : Destination path for the MP3 output.
        voice      : edge-tts voice name (default: en-US-GuyNeural).

    Returns:
        List of {"word": str, "start": float, "end": float} dicts (seconds).

    Raises:
        RuntimeError: If TTS synthesis fails or produces no audio.
    """
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[TTS] Synthesizing | voice={voice} | chars={len(text)} | out={Path(audio_path).name}")

    word_timestamps = asyncio.run(_synthesize_async(text, voice, audio_path))

    if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
        raise RuntimeError(f"[TTS] edge-tts produced no audio at {audio_path}")

    size_kb = Path(audio_path).stat().st_size // 1024
    duration = word_timestamps[-1]["end"] if word_timestamps else 0.0
    logger.info(
        f"[TTS] Done | {len(word_timestamps)} word boundaries | "
        f"~{duration:.1f}s | {size_kb} KB"
    )
    return word_timestamps
