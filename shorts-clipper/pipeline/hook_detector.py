"""Hook Candidate Detection.

Scans the scored windows produced by Stage 3 and identifies moments that are
high-energy (Signal 2 + Signal 3 strong) but poorly-spoken (Signal 1 weak).
These windows are good candidates for audio replacement with a generated hook.

Threshold:
    (signal2 + signal3) / 2 > 0.7  AND  signal1 < 0.4
"""

from api.models import HookCandidate, TranscriptResult
from utils.logger import logger

HOOK_ENERGY_THRESHOLD = 0.7   # (s_eng + s_dial) / 2 must exceed this
HOOK_SEMANTIC_CEILING  = 0.4  # s_sem must be below this


def _get_window_text(transcript: TranscriptResult, start: float, end: float) -> str:
    """Return concatenated transcript text for the given window."""
    texts = [
        seg.text.strip()
        for seg in transcript.segments
        if seg.end_sec > start and seg.start_sec < end and seg.text.strip()
    ]
    return " ".join(texts)


def detect_hook_candidates(
    windows: list[dict],
    transcript: TranscriptResult,
) -> list[HookCandidate]:
    """
    Return windows that qualify as hook candidates.

    Args:
        windows    : Scored window dicts from score_windows() — must include
                     signal1, signal2, signal3 fields (added in scorer.py).
        transcript : Full transcript for the job (used to extract spoken text).

    Returns:
        List of HookCandidate objects, one per qualifying window.
    """
    candidates: list[HookCandidate] = []

    for idx, w in enumerate(windows):
        s1 = float(w.get("signal1", 0.0))
        s2 = float(w.get("signal2", 0.0))
        s3 = float(w.get("signal3", 0.0))

        av_energy = (s2 + s3) / 2.0
        if av_energy > HOOK_ENERGY_THRESHOLD and s1 < HOOK_SEMANTIC_CEILING:
            weak_text = _get_window_text(transcript, w["start"], w["end"])
            logger.info(
                f"[HOOK_DETECTOR] Candidate #{idx + 1} | "
                f"{w['start']:.1f}-{w['end']:.1f}s | "
                f"s1={s1:.2f} s2={s2:.2f} s3={s3:.2f} | "
                f"avg_energy={av_energy:.2f}"
            )
            candidates.append(
                HookCandidate(
                    window_index=idx,
                    start=w["start"],
                    end=w["end"],
                    signal1=s1,
                    signal2=s2,
                    signal3=s3,
                    weak_transcript=weak_text,
                    engagement_type=w.get("engagement_type", ""),
                )
            )

    logger.info(
        f"[HOOK_DETECTOR] {len(candidates)}/{len(windows)} windows qualify "
        f"(energy>{HOOK_ENERGY_THRESHOLD}, semantic<{HOOK_SEMANTIC_CEILING})"
    )
    return candidates
