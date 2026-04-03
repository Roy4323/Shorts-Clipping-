"""AI Hook Script Generator.

Calls OpenAI gpt-4o-mini to produce a 30-40 word spoken hook for a high-energy
but poorly-spoken video window identified by the hook detector.

Context-aware: uses the video's content_type to set the appropriate tone so
motivational videos get empowering hooks, comedy gets playful ones, etc.

Requires OPENAI_API_KEY in .env (same key already used by the scorer).
"""

import json

from config import settings
from utils.logger import logger

# Per-content-type tone guidance injected into the prompt
_CONTENT_GUIDANCE: dict[str, str] = {
    # Scorer content types
    "solo_lecture":         "Tone: authoritative and mind-opening. Use a bold claim or provocative question.",
    "interview_dialogue":   "Tone: conversational and intriguing. Tease a surprising admission or revelation.",
    "motivational_speech":  "Tone: urgent and empowering. Use a powerful challenge or declaration. Absolutely NO jokes or sarcasm.",
    "debate_panel":         "Tone: bold and provocative. State a controversial position with conviction.",
    "educational_tutorial": "Tone: curious and illuminating. Open with a counterintuitive fact or a common misconception people get wrong.",
    "comedy_entertainment": "Tone: playful and punchy. Use an absurd premise, unexpected twist, or self-aware observation.",
    "sports_commentary":    "Tone: electric and intense. Capture the drama and stakes of the moment.",
    "music_with_lyrics":    "Tone: emotional and immersive. Draw the listener in with atmosphere or a lyrical tease.",
    # Classification content types from models.py
    "podcast_interview":    "Tone: conversational and compelling. Tease a surprising insight or candid admission.",
    "music_lyrics":         "Tone: emotional and evocative. Open with a lyrical tease or raw emotional reveal.",
    "music_instrumental":   "Tone: atmospheric and descriptive. Convey the mood and energy without words.",
    "sports_action":        "Tone: high-energy and dramatic. Capture the intensity and stakes of the moment.",
    "mute_broll":           "Tone: observational and intriguing. Create curiosity through visual context.",
    "general":              "Tone: direct and engaging. Use a strong statement or intriguing question.",
}
_DEFAULT_GUIDANCE = "Tone: engaging and direct. Use a strong statement or intriguing question."

# Prompt template — do not modify the structure without product sign-off
_PROMPT_TEMPLATE = """\
Video context: {video_title}
Topic: {video_description}
Content type: {content_type}
{tone_guidance}
What was said: {weak_transcript_segment}
Why it's energetic: {signal_reason}

Write a 30-40 word spoken hook that opens with a strong statement, question, or \
surprising fact. Must work without seeing the visuals. Match the tone exactly.

Return JSON only: {{"hook": "...", "duration_estimate_sec": 12, "hook_type": "question|statement|fact"}}"""


def generate_hook(
    video_title: str,
    video_description: str,
    weak_transcript: str,
    signal_reason: str,
    content_type: str = "general",
) -> dict:
    """
    Generate a spoken hook script for a hook candidate window.

    Args:
        video_title       : Title of the source video.
        video_description : First 200 chars of the video description.
        weak_transcript   : Transcript text from the candidate window.
        signal_reason     : Why the window is energetic (engagement_type or fallback).
        content_type      : Video content type (used to set appropriate tone/style).

    Returns:
        Dict with keys: hook (str), duration_estimate_sec (int), hook_type (str).

    Raises:
        ValueError  : If OPENAI_API_KEY is not set.
        RuntimeError: If the API call fails or returns invalid JSON.
    """
    api_key = settings.openai_api_key.strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set — cannot generate hooks.")

    from openai import OpenAI  # same client used throughout the codebase

    tone_guidance = _CONTENT_GUIDANCE.get(content_type, _DEFAULT_GUIDANCE)

    prompt = _PROMPT_TEMPLATE.format(
        video_title=video_title,
        video_description=video_description[:200],
        content_type=content_type,
        tone_guidance=tone_guidance,
        weak_transcript_segment=weak_transcript[:500],
        signal_reason=signal_reason or "high audio energy",
    )

    logger.info(
        f"[HOOK_GEN] Calling OpenAI gpt-4o-mini | title='{video_title[:60]}' | "
        f"content_type={content_type}"
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=256,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.choices[0].message.content.strip()
    logger.debug(f"[HOOK_GEN] Raw response: {raw_text[:200]}")

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned non-JSON: {raw_text[:200]}") from exc

    if "hook" not in result:
        raise RuntimeError(f"OpenAI response missing 'hook' key: {result}")

    result.setdefault("duration_estimate_sec", 12)
    result.setdefault("hook_type", "statement")

    logger.info(
        f"[HOOK_GEN] Hook generated | type={result['hook_type']} | "
        f"est={result['duration_estimate_sec']}s | text='{result['hook'][:80]}'"
    )
    return result
