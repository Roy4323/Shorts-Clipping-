"""Multi-Signal Scoring Engine (Stage 3).

Integrates three signals:
  Signal 1 — Semantic highlights      (OpenAI gpt-4o-mini via service/video_analyzer.py)
  Signal 2 — Audio energy peaks       (librosa      via service/audio_analyzer.py)
  Signal 3 — Speaker turn density     (pyannote     via service/speaker_turn_density.py)

Falls back to stub scorer when:
  - OPENAI_API_KEY is not set
  - WAV file is not available
  - Any signal raises an unexpected exception

Output contract (same as stub, extended with optional fields):
    [
      {
        "start":           float,
        "end":             float,
        "score":           float,   # composite 0-1
        "hook":            str,     # AI-suggested caption (empty for stub)
        "engagement_type": str,     # surprise|humor|insight|tension|... (empty for stub)
        "reason":          str,     # why this window works (empty for stub)
      },
      ...
    ]
"""

import math
from collections import Counter
from pathlib import Path

from api.models import TranscriptResult
from config import settings
from utils.logger import logger
from utils.storage import get_job_subdir


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_DURATION = 45.0   # stub: target seconds per clip
MIN_DURATION    = 10.0   # minimum accepted clip length (Relaxed for high-energy highlights)
MAX_DURATION    = 90.0   # maximum accepted clip length

# Weight profiles keyed by content_type returned by Signal 1 (OpenAI)
_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "solo_lecture":         {"semantic": 0.55, "energy": 0.30, "dialogue": 0.15},
    "interview_dialogue":   {"semantic": 0.35, "energy": 0.30, "dialogue": 0.35},
    "motivational_speech":  {"semantic": 0.50, "energy": 0.35, "dialogue": 0.15},
    "debate_panel":         {"semantic": 0.30, "energy": 0.25, "dialogue": 0.45},
    "educational_tutorial": {"semantic": 0.45, "energy": 0.35, "dialogue": 0.20},
    "comedy_entertainment": {"semantic": 0.35, "energy": 0.45, "dialogue": 0.20},
    "sports_commentary":    {"semantic": 0.20, "energy": 0.55, "dialogue": 0.25},
    "music_with_lyrics":    {"semantic": 0.15, "energy": 0.70, "dialogue": 0.15},
    "default":              {"semantic": 0.40, "energy": 0.35, "dialogue": 0.25},
}
_VALID_TYPES = set(_WEIGHT_PROFILES.keys()) - {"default"}


# ── Public entry point ────────────────────────────────────────────────────────

def score_windows(
    transcript: TranscriptResult,
    video_duration: float,
    audio_wav_path: str | None = None,
    job_id: str | None = None,
    shorts_count: int = 3,
) -> tuple[list[dict], str | None]:
    """
    Score clip windows using the multi-signal engine.
    Falls back to stub scoring when dependencies are unavailable.

    Args:
        transcript      : Fetched transcript (already cached from Stage 2)
        video_duration  : Total video length in seconds
        audio_wav_path  : Path to extracted WAV file (needed by Signal 2 & 3)
        job_id          : Job ID for saving intermediate JSON outputs

    Returns:
        (windows, warning)
        windows : list of window dicts with start, end, score, hook, engagement_type, reason
        warning : human-readable error string if real engine failed/skipped, else None
    """
    logger.info(
        f"[SCORER] Inputs — duration={video_duration:.2f}s "
        f"| segments={len(transcript.segments)} | source={transcript.source} "
        f"| wav={'YES' if audio_wav_path and Path(audio_wav_path).exists() else 'NO'} "
        f"| shorts_count={shorts_count} | job_id={job_id}"
    )

    has_openai = bool(settings.openai_api_key.strip())
    has_hf     = bool(settings.hugging_face_token.strip())

    key = settings.openai_api_key.strip()
    logger.info(
        f"[SCORER] API key check — "
        f"prefix={key[:12] if key else 'EMPTY'} | "
        f"suffix={key[-6:] if key else 'EMPTY'} | "
        f"length={len(key)} | "
        f"has_openai={has_openai} | has_hf={has_hf}"
    )

    if not has_openai:
        msg = "OPENAI_API_KEY not set — used stub scorer (equal-time windows, no AI ranking)."
        logger.warning(f"[SCORER] {msg}")
        return _stub_score(transcript, video_duration, shorts_count), msg

    wav_ok = audio_wav_path and Path(audio_wav_path).exists()
    if not wav_ok:
        msg = "WAV audio file not available — used stub scorer (equal-time windows, no AI ranking)."
        logger.warning(f"[SCORER] {msg}")
        return _stub_score(transcript, video_duration, shorts_count), msg

    # Detect whether we have a real transcript (not a mock/empty placeholder)
    has_real_transcript = (
        transcript.source not in ("mock",)
        and len(transcript.segments) >= 3
    )
    if not has_real_transcript:
        logger.warning(
            f"[SCORER] Transcript is empty or mock (source={transcript.source}, "
            f"segments={len(transcript.segments)}) — running Signal 2+3 only (no Signal 1)."
        )

    try:
        logger.info("[SCORER] Starting multi-signal scoring engine.")
        windows = _multi_signal_score(
            transcript, video_duration, audio_wav_path, job_id, has_hf, shorts_count,
            skip_signal1=not has_real_transcript,
        )
        windows = windows[:shorts_count]
        logger.info(f"[SCORER] Trimmed to {len(windows)} windows (shorts_count={shorts_count})")
        return windows, None
    except Exception as exc:
        # Extract a clean human-readable message from the exception
        err_str = str(exc)
        if "401" in err_str or "insufficient permissions" in err_str or "Missing scopes" in err_str:
            msg = (
                "OpenAI API key error (401 — insufficient permissions). "
                "Go to platform.openai.com → API Keys → ensure the key has 'model.request' scope. "
                "Fell back to stub scorer."
            )
        elif "429" in err_str:
            msg = "OpenAI rate limit hit (429). Fell back to stub scorer."
        elif "insufficient_quota" in err_str:
            msg = "OpenAI quota exceeded. Add credits at platform.openai.com. Fell back to stub scorer."
        else:
            msg = f"Scorer error: {err_str[:200]}. Fell back to stub scorer."

        logger.error(
            f"[SCORER] Multi-signal scoring failed: {exc} — falling back to stub.",
            exc_info=True,
        )
        return _stub_score(transcript, video_duration, shorts_count), msg


# ── Multi-signal engine ───────────────────────────────────────────────────────

def _multi_signal_score(
    transcript: TranscriptResult,
    video_duration: float,
    audio_wav_path: str,
    job_id: str | None,
    has_hf: bool,
    shorts_count: int = 3,
    skip_signal1: bool = False,
) -> list[dict]:
    """Run all three signals and combine into scored windows."""

    # Per-job output directory for intermediate JSONs (useful for debugging)
    if job_id:
        score_dir = get_job_subdir(job_id, "scoring")
    else:
        score_dir = Path("data/scoring")
        score_dir.mkdir(parents=True, exist_ok=True)

    highlight_json = str(score_dir / "highlight_analysis.json")
    audio_json     = str(score_dir / "audio_analysis_output.json")
    dialogue_json  = str(score_dir / "speaker_turn_density.json")

    # ── Signal 1: OpenAI semantic highlights ─────────────────────────────────
    highlights: list[dict] = []
    if skip_signal1:
        logger.info("[SCORER] Signal 1 — Skipped (no real transcript).")
    else:
        segs = _convert_transcript(transcript)
        logger.info(f"[SCORER] Signal 1 — Semantic analysis | {len(segs)} segments ...")
        if len(segs) < 3:
            logger.warning(
                f"[SCORER] Signal 1 — Only {len(segs)} transcript segment(s). "
                "OpenAI needs a real transcript to find highlights. "
                "This video may have no captions or Supadata returned a mock transcript."
            )
        from service.video_analyzer import analyze_segments
        hl_data    = analyze_segments(segs, highlight_json, source_label="pipeline")
        highlights = hl_data.get("highlights", [])
        if not highlights:
            logger.warning(
                "[SCORER] Signal 1 returned 0 highlights — "
                "transcript may be too short or OpenAI found nothing engaging. "
                f"transcript_segments={len(segs)} source={transcript.source}"
            )
        else:
            logger.info(f"[SCORER] Signal 1 complete: {len(highlights)} highlights found.")

    # ── Signal 2: Audio energy ───────────────────────────────────────────────
    logger.info(f"[SCORER] Signal 2 — Audio energy analysis | wav={audio_wav_path} ...")
    try:
        from service.audio_analyzer import analyze_audio
        audio_data = analyze_audio(audio_wav_path, audio_json)
        windows_count = len(audio_data.get("analysis_30s_windows", []))
        logger.info(f"[SCORER] Signal 2 complete: {windows_count} energy windows analysed.")
    except Exception as exc:
        logger.error(f"[SCORER] Signal 2 (audio energy) FAILED: {exc}", exc_info=True)
        raise

    # ── Signal 3: Speaker turn density (optional — needs HuggingFace token) ──
    dialogue_data: dict | None = None
    if has_hf:
        logger.info(f"[SCORER] Signal 3 — Speaker turn density | wav={audio_wav_path} ...")
        try:
            from service.speaker_turn_density import score_speaker_turn_density, save_results
            dialogue_data = score_speaker_turn_density(audio_wav_path)
            save_results(dialogue_data, dialogue_json)
            switches = dialogue_data.get("metadata", {}).get("total_switches", "?")
            speakers = dialogue_data.get("metadata", {}).get("num_speakers", "?")
            logger.info(f"[SCORER] Signal 3 complete: {speakers} speakers, {switches} switches detected.")
        except Exception as exc:
            logger.warning(
                f"[SCORER] Signal 3 (pyannote diarization) FAILED: {exc} "
                "— dialogue score will be 0 for all windows.",
                exc_info=True,
            )
    else:
        logger.warning("[SCORER] HUGGING_FACE_TOKEN not set — Signal 3 skipped. Dialogue scores will be 0.")

    # ── Combine ───────────────────────────────────────────────────────────────
    logger.info(
        f"[SCORER] Combining signals — "
        f"highlights={len(highlights)} | "
        f"audio_windows={len(audio_data.get('analysis_30s_windows', []))} | "
        f"dialogue={'yes' if dialogue_data else 'no (skipped/failed)'}"
    )

    if not highlights:
        # No Signal 1 — score using audio (Signal 2 + 3) only
        logger.info("[SCORER] No Signal 1 highlights — scoring with Signal 2+3 only.")
        windows = _audio_only_score(audio_data, dialogue_data, video_duration, shorts_count)
    else:
        windows = _combine(highlights, audio_data, dialogue_data, video_duration)

    if not windows:
        logger.warning("[SCORER] Multi-signal engine produced no windows — falling back to stub.")
        return _stub_score(transcript, video_duration, shorts_count)

    logger.info(f"[SCORER] Multi-signal scoring complete: {len(windows)} windows selected.")
    return windows


def _audio_only_score(
    audio_data: dict,
    dialogue_data: dict | None,
    video_duration: float,
    shorts_count: int,
) -> list[dict]:
    """
    Score windows using Signal 2 (audio energy) + Signal 3 (speaker turns) only.
    Used when no transcript is available (Hindi videos, no captions, etc.).

    Groups the 30s energy buckets into TARGET_DURATION-length candidate windows,
    scores each by energy + dialogue density, then picks the top non-overlapping ones.
    """
    energy_windows = audio_data.get("analysis_30s_windows", [])
    if not energy_windows:
        return []

    # Normalize energy scores 0-1
    energies = [w["mean_energy_30s"] for w in energy_windows]
    e_min, e_max = min(energies), max(energies)
    def _norm_e(v: float) -> float:
        return (v - e_min) / (e_max - e_min) if e_max > e_min else 0.5

    # Build dialogue lookup (30s buckets)
    dial_lookup: dict[float, float] = {}
    if dialogue_data:
        for w in dialogue_data.get("windows", []):
            dial_lookup[float(w["start_seconds"])] = w["score"]

    # Weights: energy 60%, dialogue 40% (no semantic signal)
    W_E, W_D = 0.60, 0.40

    # Build candidate windows of ~TARGET_DURATION by merging consecutive 30s buckets
    buckets_per_clip = max(1, round(TARGET_DURATION / 30))
    candidates: list[dict] = []

    for i in range(len(energy_windows)):
        end_i = min(i + buckets_per_clip, len(energy_windows))
        chunk = energy_windows[i:end_i]

        c_start = float(chunk[0]["window_start"])
        c_end   = float(chunk[-1]["window_start"]) + 30.0
        c_end   = min(c_end, video_duration)
        dur     = c_end - c_start

        if dur < MIN_DURATION:
            continue

        s_eng  = sum(_norm_e(b["mean_energy_30s"]) for b in chunk) / len(chunk)
        s_dial = sum(dial_lookup.get(float(b["window_start"]), 0.0) for b in chunk) / len(chunk)
        score  = W_E * s_eng + W_D * s_dial

        candidates.append({
            "start": round(c_start, 3),
            "end":   round(c_end,   3),
            "score": round(score, 4),
            "hook":            "",
            "engagement_type": "",
            "reason":          "Audio energy + speaker activity (no transcript available)",
            # Individual signal scores — used by hook_detector.py
            "signal1":         0.0,
            "signal2":         round(s_eng,  4),
            "signal3":         round(s_dial, 4),
        })

    # Sort by score descending, pick non-overlapping
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected: list[dict] = []
    used_ends: list[float] = []
    for clip in candidates:
        if not any(clip["start"] < end + 10 for end in used_ends):
            selected.append(clip)
            used_ends.append(clip["end"])
            if len(selected) >= shorts_count:
                break

    # Return sorted by start time for a coherent timeline
    selected.sort(key=lambda x: x["start"])
    logger.info(f"[SCORER] Audio-only scoring complete: {len(selected)} windows selected.")
    return selected


def _combine(
    highlights: list[dict],
    audio_data: dict,
    dialogue_data: dict | None,
    video_duration: float,
) -> list[dict]:
    """Combine three signal outputs into ranked, non-overlapping windows."""

    if not highlights:
        return []

    # Determine content type from Signal 1 (majority vote across highlights)
    content_type = Counter(
        h.get("content_type", "default") for h in highlights
    ).most_common(1)[0][0]
    if content_type not in _VALID_TYPES:
        content_type = "default"
    weights = _WEIGHT_PROFILES[content_type]
    logger.info(
        f"[SCORER] Content type: {content_type} | "
        f"sem={weights['semantic']} eng={weights['energy']} dial={weights['dialogue']}"
    )

    # Build Signal 2 lookup: 30s window_start → normalized mean energy
    raw_energy: dict[float, float] = {
        w["window_start"]: w["mean_energy_30s"]
        for w in audio_data.get("analysis_30s_windows", [])
    }

    # Build Signal 3 lookup: window start_seconds → score (or empty → all zeros)
    if dialogue_data:
        dialogue: dict[float, float] = {
            float(w["start_seconds"]): w["score"]
            for w in dialogue_data.get("windows", [])
        }
    else:
        dialogue = {}

    # Score each highlight
    scored: list[dict] = []
    for h in highlights:
        try:
            h_start = _parse_timestamp(h["start_time"])
            h_end   = _parse_timestamp(h["end_time"])
        except Exception:
            logger.debug(f"[SCORER] Skipping highlight with unparseable timestamps: {h}")
            continue

        # Clamp to actual video length
        h_end = min(h_end, video_duration)
        dur   = h_end - h_start

        if dur < MIN_DURATION:
            logger.debug(
                f"[SCORER] Skipping {h['start_time']}-{h['end_time']}: "
                f"duration {dur:.1f}s < MIN {MIN_DURATION}s after clamping."
            )
            continue

        s_sem  = float(h.get("score", 0.0))
        s_eng  = _nearest(h_start, raw_energy)
        s_dial = _nearest(h_start, dialogue)
        comp   = (
            weights["semantic"]   * s_sem
            + weights["energy"]   * s_eng
            + weights["dialogue"] * s_dial
        )

        scored.append({
            "start":           round(h_start, 3),
            "end":             round(h_end,   3),
            "score":           round(comp, 4),
            "hook":            h.get("hook", ""),
            "engagement_type": h.get("engagement_type", ""),
            "reason":          h.get("reason", ""),
            # Individual signal scores — used by hook_detector.py
            "signal1":         round(s_sem,  4),
            "signal2":         round(s_eng,  4),
            "signal3":         round(s_dial, 4),
        })
        logger.debug(
            f"[SCORER] Highlight {h['start_time']}-{h['end_time']} | "
            f"comp={comp:.3f} (sem={s_sem:.2f} eng={s_eng:.2f} dial={s_dial:.2f})"
        )

    # Sort by composite score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Select non-overlapping windows (10s gap enforced between clips)
    selected: list[dict] = []
    used_ends: list[float] = []
    for clip in scored:
        overlaps = any(clip["start"] < end + 10 for end in used_ends)
        if not overlaps:
            selected.append(clip)
            used_ends.append(clip["end"])

    return selected


# ── Helpers ───────────────────────────────────────────────────────────────────

def _convert_transcript(transcript: TranscriptResult) -> list[dict]:
    """Convert our TranscriptResult → list[{time_seconds, timestamp, text}]."""
    return [
        {
            "time_seconds": int(seg.start_sec),
            "timestamp":    _sec_to_bracket(seg.start_sec),
            "text":         seg.text,
        }
        for seg in transcript.segments
    ]


def _sec_to_bracket(sec: float) -> str:
    """Format seconds as [HH:MM:SS] or [MM:SS] bracket timestamp."""
    s = int(sec)
    h, remainder = divmod(s, 3600)
    m, s = divmod(remainder, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]" if h else f"[{m:02d}:{s:02d}]"


def _parse_timestamp(t: str) -> float:
    """Parse MM:SS or HH:MM:SS string to total seconds."""
    parts = t.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0]) * 60 + int(parts[1])


def _nearest(sec: float, signal: dict, window: int = 30) -> float:
    """Find value for the 30s bucket containing sec, with one-bucket fallback."""
    bucket = (sec // window) * window
    return signal.get(bucket, signal.get(bucket - window, 0.0))


# ── Stub fallback ─────────────────────────────────────────────────────────────

def _nearest_boundary(seconds: float, segments: list, prefer_end: bool) -> float:
    candidates = [s.end_sec if prefer_end else s.start_sec for s in segments]
    return min(candidates, key=lambda t: abs(t - seconds))


def _stub_score(transcript: TranscriptResult, video_duration: float, shorts_count: int = 3) -> list[dict]:
    """
    Divide video into equal-duration windows snapped to transcript boundaries.
    Used as fallback when the multi-signal engine is unavailable.
    """
    logger.info(
        f"[SCORER] Stub scoring | duration={video_duration:.2f}s "
        f"| segments={len(transcript.segments)} | shorts_count={shorts_count}"
    )

    if video_duration <= 0:
        logger.warning("[SCORER] Video duration is 0 — cannot produce windows.")
        return []

    n_windows = max(1, min(shorts_count, math.ceil(video_duration / TARGET_DURATION)))
    chunk     = video_duration / n_windows
    segs      = transcript.segments

    logger.info(
        f"[SCORER] Plan: {n_windows} window(s) × ~{chunk:.1f}s each "
        f"| snapping={'yes' if segs else 'no'}"
    )

    windows: list[dict] = []
    for i in range(n_windows):
        raw_start = i * chunk
        raw_end   = (i + 1) * chunk

        if segs:
            start = _nearest_boundary(raw_start, segs, prefer_end=False)
            end   = _nearest_boundary(raw_end,   segs, prefer_end=True)
        else:
            start, end = raw_start, raw_end

        duration = end - start
        if duration < MIN_DURATION:
            start, end = raw_start, raw_end
            duration   = end - start

        if MIN_DURATION <= duration <= MAX_DURATION:
            windows.append({
                "start":           round(start, 3),
                "end":             round(end,   3),
                "score":           0.75,
                "hook":            "",
                "engagement_type": "",
                "reason":          "",
                # Individual signal scores — used by hook_detector.py (all unknown in stub)
                "signal1":         0.0,
                "signal2":         0.0,
                "signal3":         0.0,
            })
            logger.info(f"[SCORER] Stub window {i+1}: {start:.1f}s->{end:.1f}s ({duration:.1f}s)")
        else:
            logger.warning(
                f"[SCORER] Stub window {i+1} skipped: "
                f"duration {duration:.1f}s outside [{MIN_DURATION}-{MAX_DURATION}]s"
            )

    logger.info(f"[SCORER] Stub done: {len(windows)}/{n_windows} windows accepted.")
    return windows
