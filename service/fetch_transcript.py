"""
Service: Transcript Fetcher
Fetches a YouTube transcript via Supadata API.
Saves formatted text to output_file AND returns parsed segments directly
so video_analyzer.py can consume them without re-reading from disk.

Segment format returned:
  [{"time_seconds": int, "timestamp": "[MM:SS]", "text": str}, ...]
"""

import os
from dotenv import load_dotenv

load_dotenv()


def fetch_transcript(url: str, output_file: str) -> list[dict]:
    """
    Fetch transcript for a YouTube URL via Supadata.

    Saves timestamped text to output_file (for audit / replay).
    Returns the parsed segment list directly so the caller can pass it
    straight to video_analyzer.analyze_segments() without a disk round-trip.

    Args:
        url         : YouTube video URL
        output_file : Where to write the [MM:SS] text file (e.g. output/transcript.txt)

    Returns:
        List of dicts: [{"time_seconds": int, "timestamp": "[MM:SS]", "text": str}, ...]
    """
    api_key = os.getenv("SUPADATA_API_KEY")
    if not api_key:
        raise ValueError("SUPADATA_API_KEY not found in .env")

    from supadata import Supadata
    client     = Supadata(api_key=api_key)
    print(f"[transcript] Fetching: {url}")
    transcript = client.transcript(url=url)

    segments = []
    lines    = []

    if hasattr(transcript, "content"):
        for chunk in transcript.content:
            offset_ms = getattr(chunk, "offset", 0)
            text      = getattr(chunk, "text", "").strip()
            if not text:
                continue

            ts_str, time_sec = _format_timestamp(offset_ms)
            segments.append({"time_seconds": time_sec, "timestamp": ts_str, "text": text})
            lines.append(f"{ts_str} {text}")
    else:
        # Fallback: raw string, no timestamps
        raw = str(transcript)
        lines.append(raw)
        segments.append({"time_seconds": 0, "timestamp": "[00:00]", "text": raw})

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[transcript] {len(segments)} segments saved -> {output_file}")
    return segments


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_timestamp(offset_ms: int) -> tuple[str, int]:
    """Convert millisecond offset to ('[MM:SS]', total_seconds)."""
    s     = offset_ms // 1000
    h, r  = divmod(s, 3600)
    m, sec = divmod(r, 60)
    ts    = f"[{h:02d}:{m:02d}:{sec:02d}]" if h else f"[{m:02d}:{sec:02d}]"
    return ts, s
