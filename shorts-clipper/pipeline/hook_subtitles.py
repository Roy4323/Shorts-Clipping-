"""Hook Subtitles — SRT generation from TTS word timestamps + FFmpeg burn.

Converts the word-level timestamps returned by tts_engine.py into an SRT file,
then burns it onto the hook clip using the same FFmpeg ASS style used by
subtitles.py (default preset).

Words are grouped into short phrases (~5 words) to keep captions readable.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from utils.logger import logger

# Re-use the same preset styles and helpers from subtitles.py
from pipeline.subtitles import PRESET_STYLES, _escape_path_for_ffmpeg, _wrap_text


class HookSubtitleError(Exception):
    pass


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def _sec_to_srt(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm."""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_hook_srt(
    word_timestamps: list[dict],
    words_per_chunk: int = 5,
    preset: str = "default",
) -> str:
    """
    Group word-level timestamps into short phrase chunks and build an SRT string.

    Args:
        word_timestamps : List of {"word": str, "start": float, "end": float}.
        words_per_chunk : How many words to show per subtitle line.
        preset          : Style preset name (for uppercase transforms matching subtitles.py).

    Returns:
        SRT-formatted string.
    """
    if not word_timestamps:
        return ""

    entries: list[str] = []
    idx = 1

    for i in range(0, len(word_timestamps), words_per_chunk):
        chunk = word_timestamps[i : i + words_per_chunk]
        text  = " ".join(w["word"] for w in chunk)
        start = chunk[0]["start"]
        end   = chunk[-1]["end"]

        # Match uppercase behaviour of karaoke / beasty / pod_p presets
        if preset in ("karaoke", "beasty", "pod_p"):
            text = text.upper()

        wrapped = _wrap_text(text)
        entries.append(f"{idx}\n{_sec_to_srt(start)} --> {_sec_to_srt(end)}\n{wrapped}\n")
        idx += 1

    return "\n".join(entries)


# ---------------------------------------------------------------------------
# Subtitle burn
# ---------------------------------------------------------------------------

def burn_hook_subtitles(
    input_path: str,
    word_timestamps: list[dict],
    output_path: str,
    preset: str = "default",
) -> str:
    """
    Burn word-timestamp-based subtitles onto a hook clip.

    If word_timestamps is empty the clip is copied without burning (non-fatal).

    Args:
        input_path      : Path to the hook clip with TTS audio already muxed.
        word_timestamps : Word-level timing list from tts_engine.synthesize_hook().
        output_path     : Destination MP4 path.
        preset          : Subtitle style preset (default, karaoke, beasty, …).

    Returns:
        Absolute path to the finished clip (str).

    Raises:
        HookSubtitleError: If ffmpeg hard-fails on both the burn and copy fallback.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[HOOK_SUBS] Starting | words={len(word_timestamps)} | "
        f"preset={preset} | in={Path(input_path).name}"
    )

    srt_content = _build_hook_srt(word_timestamps, preset=preset)

    if not srt_content.strip():
        logger.warning("[HOOK_SUBS] No word timestamps — copying hook clip without subtitles.")
        cmd = ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HookSubtitleError(f"FFmpeg copy failed:\n{result.stderr[-400:]}")
        return str(output_path)

    tmp_fd, tmp_srt = tempfile.mkstemp(suffix=".srt")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(srt_content)

        escaped = _escape_path_for_ffmpeg(tmp_srt)
        style   = PRESET_STYLES.get(preset, PRESET_STYLES["default"])
        vf      = f"subtitles='{escaped}':force_style='{style}'"

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(
                f"[HOOK_SUBS] FFmpeg subtitle burn failed (rc={result.returncode}) — "
                f"falling back to copy.\n{result.stderr[-600:]}"
            )
            cmd_fb = ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]
            result_fb = subprocess.run(cmd_fb, capture_output=True, text=True)
            if result_fb.returncode != 0:
                raise HookSubtitleError(
                    f"Fallback copy also failed:\n{result_fb.stderr[-400:]}"
                )
        else:
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(
                f"[HOOK_SUBS] Done: {Path(output_path).name} "
                f"({size_mb:.1f} MB, {len(word_timestamps)} words)"
            )
    finally:
        try:
            os.unlink(tmp_srt)
        except OSError:
            pass

    return str(output_path)
