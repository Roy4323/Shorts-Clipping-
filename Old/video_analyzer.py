import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_transcript(transcript_file: str) -> list[dict]:
    """Parse [MM:SS] or [HH:MM:SS] timestamped transcript into segment dicts."""
    segments = []

    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match [HH:MM:SS] or [MM:SS]
            match = re.match(r"^\[(\d{2}):(\d{2})(?::(\d{2}))?\]\s*(.*)", line)
            if not match:
                continue

            a, b, c, text = match.groups()
            if c is not None:          # HH:MM:SS
                total_seconds = int(a) * 3600 + int(b) * 60 + int(c)
                timestamp_str = f"[{a}:{b}:{c}]"
            else:                       # MM:SS
                total_seconds = int(a) * 60 + int(b)
                timestamp_str = f"[{a}:{b}]"

            segments.append({
                "time_seconds": total_seconds,
                "timestamp": timestamp_str,
                "text": text,
            })

    return segments


def _seconds_to_mmss(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_transcript_for_prompt(segments: list[dict]) -> str:
    return "\n".join(f"{s['timestamp']} {s['text']}" for s in segments)


# ---------------------------------------------------------------------------
# OpenAI sub-agent
# ---------------------------------------------------------------------------

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
- Score 0.0–1.0 where 1.0 = perfect viral clip.

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


def _call_openai(client: OpenAI, transcript_text: str) -> dict:
    user_prompt = (
        "Analyze this transcript and return the top 5 engaging 30-60 second highlight windows as JSON.\n\n"
        f"{transcript_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    result["_meta"] = {
        "model": response.model,
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_transcript(transcript_file: str = "my_video_transcript.txt") -> dict:
    """
    Sub-agent: reads a timestamped transcript file, calls OpenAI, and returns
    top engaging moments in 30-60 second windows with reasoning as JSON.

    Returns a dict with keys:
      highlights       – list of ranked highlight windows
      analysis_summary – brief video summary
      metadata         – file stats + token usage
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")

    client = OpenAI(api_key=api_key)

    print(f"[analyzer] Parsing transcript: {transcript_file}")
    segments = parse_transcript(transcript_file)
    if not segments:
        raise ValueError(f"No timestamped segments found in '{transcript_file}'.")

    total_duration = segments[-1]["time_seconds"] - segments[0]["time_seconds"]
    print(f"[analyzer] {len(segments)} segments | duration ~{total_duration // 60}m {total_duration % 60}s")

    transcript_text = format_transcript_for_prompt(segments)

    print("[analyzer] Sending transcript to OpenAI gpt-4o-mini …")
    result = _call_openai(client, transcript_text)

    result["metadata"] = {
        "transcript_file": transcript_file,
        "total_segments": len(segments),
        "total_duration_seconds": total_duration,
        "total_duration_human": _seconds_to_mmss(total_duration),
        **result.pop("_meta"),
    }

    print(f"[analyzer] Done. {len(result.get('highlights', []))} highlights returned. "
          f"Tokens used: {result['metadata']['tokens']['total']}")
    return result


def save_analysis(result: dict, output_file: str = "highlight_analysis.json") -> None:
    """Persist analysis result to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[analyzer] Saved → {output_file}")


def print_highlights(result: dict) -> None:
    """Pretty-print the highlights to stdout."""
    print("\n" + "=" * 62)
    print("  TOP ENGAGING MOMENTS")
    print("=" * 62)

    summary = result.get("analysis_summary", "")
    if summary:
        print(f"\nSummary: {summary}\n")

    for h in result.get("highlights", []):
        print(f"  #{h['rank']}  {h['start_time']} → {h['end_time']}  "
              f"({h.get('duration_seconds', '?')}s)  score={h['score']}")
        print(f"       Type : {h.get('engagement_type', 'N/A')}  |  Content: {h.get('content_type', 'N/A')}")
        print(f"       Hook : {h.get('hook', '')}")
        print(f"       Why  : {h.get('reason', '')}")
        print()

    meta = result.get("metadata", {})
    if meta:
        print(f"  Model: {meta.get('model')} | "
              f"Tokens: {meta.get('tokens', {}).get('total', '?')}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    transcript_file = sys.argv[1] if len(sys.argv) > 1 else "my_video_transcript.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "highlight_analysis.json"

    result = analyze_transcript(transcript_file)
    print_highlights(result)
    save_analysis(result, output_file)
