"""Stage 6: Generate SRT from transcript and burn it onto the reframed clip."""

import os
import subprocess
import tempfile
from pathlib import Path

from api.models import TranscriptResult
from utils.logger import logger


class SubtitleError(Exception):
    pass


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def _sec_to_srt(seconds: float) -> str:
    """Convert float seconds to SRT timestamp  HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap_text(text: str, max_chars: int = 34) -> str:
    """Wrap text to roughly 34 characters per line to avoid spilling horizontally."""
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    for word in words:
        if current_len + len(word) > max_chars and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
        else:
            current_line.append(word)
            current_len += len(word) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def _build_srt(transcript: TranscriptResult, clip_start: float, clip_end: float, preset: str = "default") -> str:
    """
    Build an SRT string containing only the segments that overlap with
    [clip_start, clip_end], with timestamps adjusted to clip-relative time.
    Prevents overlapping timestamps to ensure only one subtitle shows at a time.
    """
    valid_segs = []
    # Pre-filter relevant segments
    for seg in transcript.segments:
        if seg.end_sec <= clip_start or seg.start_sec >= clip_end:
            continue
        text = seg.text.strip().replace(">>", "").strip()
        if text:
            valid_segs.append((seg, text))

    entries: list[str] = []
    idx = 1

    for i, (seg, clean_text) in enumerate(valid_segs):
        rel_start = max(seg.start_sec - clip_start, 0.0)
        rel_end = min(seg.end_sec - clip_start, clip_end - clip_start)

        # Cap the end time to the next segment's start time to prevent overlaps
        if i < len(valid_segs) - 1:
            next_seg = valid_segs[i+1][0]
            next_rel_start = max(next_seg.start_sec - clip_start, 0.0)
            if rel_end > next_rel_start:
                # Tiny tiny gap to prevent player rendering quirks
                rel_end = max(rel_start + 0.1, next_rel_start - 0.01)

        if rel_end <= rel_start:
            continue

        text = _wrap_fancy_text(clean_text, preset)

        entries.append(
            f"{idx}\n{_sec_to_srt(rel_start)} --> {_sec_to_srt(rel_end)}\n{text}\n"
        )
        idx += 1

    return "\n".join(entries)


# ---------------------------------------------------------------------------
# FFmpeg subtitle burning
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Visual Style Presets (ASS format)
# ---------------------------------------------------------------------------

PRESET_STYLES = {
    "default": (
        "FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,Bold=1,Outline=1,Shadow=0.5,Alignment=2,MarginV=25"
    ),
    "karaoke": (
        "FontName=Arial,FontSize=24,PrimaryColour=&H0000FF00,SecondaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H80000000,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=40"
    ),
    "beasty": (
        "FontName=Arial Black,FontSize=28,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,"
        "BackColour=&H00000000,Bold=1,Outline=3,Shadow=0,Alignment=2,MarginV=50"
    ),
    "pod_p": (
        "FontName=Arial,FontSize=26,PrimaryColour=&H00FF00FF,OutlineColour=&H00FFFFFF,"
        "BackColour=&H00000000,Bold=1,Outline=1,Shadow=2,Alignment=2,MarginV=30"
    ),
    "youshaei": (
        "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H00000000,Bold=1,Outline=4,Shadow=0,Alignment=2,MarginV=25"
    ),
    "deep_diver": (
        "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H40000000,Bold=0,Outline=0.5,Shadow=0,Alignment=2,MarginV=15"
    ),
}


def _process_karaoke(text: str) -> str:
    """Add ASS karaoke highlight tags {\k} to each word."""
    words = text.split()
    if not words:
        return text
    # Simple estimate: evenly divide the segment duration among words
    # (Since total duration is known by the SRT entry, we just add the tags)
    # {\k10}word1 {\k15}word2 ... (10 and 15 are durations in centiseconds)
    # For now, we'll just return the words. The ASS filter handles basic styling.
    # To do real {\k}, we need the total duration of the segment.
    # Refined approach: just use the specialized Karaoke style which Highlights
    return " ".join(words).upper()


def _wrap_fancy_text(text: str, preset: str) -> str:
    """Apply preset-specific text transformations (like uppercase)."""
    if preset in ["beasty", "karaoke", "pod_p"]:
        text = text.upper()
    return _wrap_text(text)


def _escape_path_for_ffmpeg(path: str) -> str:
    """
    FFmpeg's subtitles filter needs forward slashes and the colon in Windows
    drive letters escaped as  C\\:/path/to/file.srt
    """
    p = path.replace("\\", "/")
    # Escape drive-letter colon: C:/ -> C\:/
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return p


def burn_subtitles(
    input_path: str,
    transcript: TranscriptResult,
    clip_start: float,
    clip_end: float,
    output_path: Path,
    preset: str = "default",
) -> str:
    """
    Burn subtitles onto the clip.

    If no transcript segments exist for this window the clip is copied
    without subtitles (no quality loss).

    Args:
        input_path:   Path to the reframed clip.
        transcript:   Full transcript (will be filtered to clip window).
        clip_start:   Start time in the original video (seconds).
        clip_end:     End time in the original video (seconds).
        output_path:  Destination for the final clip.

    Returns:
        Absolute path to the finished clip (str).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_segs = len(transcript.segments)
    logger.info(f"[SUBTITLES] Starting | clip={clip_start:.1f}s-{clip_end:.1f}s | preset={preset} | source={transcript.source}")

    srt_content = _build_srt(transcript, clip_start, clip_end, preset=preset)
    srt_entries = srt_content.strip().count("\n\n") + 1 if srt_content.strip() else 0
    logger.info(f"[SUBTITLES] SRT built: {srt_entries} entries for this clip window")

    if not srt_content.strip():
        logger.warning(f"[SUBTITLES] No transcript segments for {clip_start:.1f}s-{clip_end:.1f}s — copying without subtitles.")
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SubtitleError(f"FFmpeg copy failed:\n{result.stderr[-400:]}")
        return str(output_path)

    # Write SRT to a temp file
    tmp_fd, tmp_srt = tempfile.mkstemp(suffix=".srt")
    logger.debug(f"[SUBTITLES] SRT temp file: {tmp_srt}")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(srt_content)

        escaped = _escape_path_for_ffmpeg(tmp_srt)
        style = PRESET_STYLES.get(preset, PRESET_STYLES["default"])
        vf = f"subtitles='{escaped}':force_style='{style}'"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ]
        logger.debug(f"[SUBTITLES] CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"[SUBTITLES] FFmpeg subtitle burn FAILED (rc={result.returncode}):\n{result.stderr[-800:]}")
            logger.warning(f"[SUBTITLES] Falling back to copy without subtitles.")
            cmd_fb = ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)]
            result_fb = subprocess.run(cmd_fb, capture_output=True, text=True)
            if result_fb.returncode != 0:
                raise SubtitleError(f"Fallback copy also failed:\n{result_fb.stderr[-400:]}")
        else:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"[SUBTITLES] Done: {output_path.name} ({size_mb:.1f} MB) with {srt_entries} subtitle entries")
    finally:
        try:
            os.unlink(tmp_srt)
        except OSError:
            pass

    return str(output_path)
