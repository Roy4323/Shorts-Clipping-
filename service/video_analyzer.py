"""
Service: Semantic Highlight Analyzer (Signal 1)
Uses OpenAI gpt-4o-mini to find top 5 engaging 30-60s windows.
Called by pipeline.py — all paths passed explicitly, no hardcoded constants.
"""

import os
import json
import re
from openai import OpenAI


SYSTEM_PROMPT = """You are an expert short-form video content analyst (YouTube Shorts, TikToks, Reels).

Given a timestamped video transcript, identify the TOP 5 most engaging 30-60 second windows suitable for standalone short clips.

Engagement signals to prioritize:
- Surprising revelations or counterintuitive statements ("aha moments")
- Emotional peaks: humor, tension, excitement, inspiration
- Self-contained story arcs with a clear beginning/middle/end
- Strong hooks or highly quotable one-liners
- High-energy exchanges, confrontations, or debates
- Concrete actionable tips or insights
- Controversial or strongly opinionated statements

Rules:
- Each highlight window MUST span 30-60 seconds.
- Windows must not overlap.
- Pick timestamps that exist in the transcript — do not invent times.
- Score 0.0-1.0 where 1.0 = perfect viral clip.

Return ONLY valid JSON — no markdown fences, no extra text — in this exact schema:
{
  "highlights": [
    {
      "rank": 1,
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "duration_seconds": 45,
      "score": 0.95,
      "engagement_type": "surprise|humor|tension|insight|action|advice|controversy",
      "content_type": "solo_lecture|interview_dialogue|educational_tutorial|motivational_speech|debate_panel|comedy_entertainment|music_with_lyrics|sports_commentary",
      "hook": "One punchy sentence hook for the short's caption",
      "reason": "2-3 sentences: why this window works as a standalone clip and what makes it engaging."
    }
  ],
  "analysis_summary": "1-2 sentences on the video's overall content and tone."
}"""


def _parse_transcript(transcript_file: str) -> list[dict]:
    segments = []
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\[(\d{2}):(\d{2})(?::(\d{2}))?\]\s*(.*)", line)
            if not match:
                continue
            a, b, c, text = match.groups()
            if c is not None:
                total_sec = int(a) * 3600 + int(b) * 60 + int(c)
                ts_str    = f"[{a}:{b}:{c}]"
            else:
                total_sec = int(a) * 60 + int(b)
                ts_str    = f"[{a}:{b}]"
            segments.append({"time_seconds": total_sec, "timestamp": ts_str, "text": text})
    return segments


def _seconds_to_mmss(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def analyze_segments(segments: list[dict], output_file: str,
                      source_label: str = "live") -> dict:
    """
    Analyze pre-parsed segments directly (no disk read).
    Called by scorer.py when transcript was just fetched from a URL.

    Args:
        segments     : List of {"time_seconds": int, "timestamp": str, "text": str}
        output_file  : Where to save highlight_analysis.json
        source_label : Label used in metadata (e.g. the URL or "live")
    """
    if not segments:
        raise ValueError("segments list is empty — nothing to analyze.")
    return _run_analysis(segments, output_file, source_label=source_label)


def analyze_transcript(transcript_file: str, output_file: str) -> dict:
    """
    Read transcript from file, call OpenAI, save highlights to output_file.
    Used when transcript already exists on disk (e.g. local test files).
    """
    segments = _parse_transcript(transcript_file)
    if not segments:
        raise ValueError(f"No timestamped segments found in '{transcript_file}'.")
    return _run_analysis(segments, output_file, source_label=transcript_file)


def _run_analysis(segments: list[dict], output_file: str, source_label: str) -> dict:
    """Core OpenAI call shared by both public entry points."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment / .env")

    client = OpenAI(api_key=api_key)

    total_duration = segments[-1]["time_seconds"] - segments[0]["time_seconds"]
    print(f"[video]  {len(segments)} segments | ~{total_duration // 60}m {total_duration % 60}s")

    transcript_text = "\n".join(f"{s['timestamp']} {s['text']}" for s in segments)

    print("[video]  Calling OpenAI gpt-4o-mini ...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Analyze this transcript and return the top 5 engaging 30-60 second highlight windows as JSON.\n\n{transcript_text}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    result["metadata"] = {
        "transcript_file":        source_label,
        "total_segments":         len(segments),
        "total_duration_seconds": total_duration,
        "total_duration_human":   _seconds_to_mmss(total_duration),
        "model":                  response.model,
        "tokens": {
            "prompt":     response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total":      response.usage.total_tokens,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[video]  {len(result.get('highlights', []))} highlights saved -> {output_file}")
    return result
