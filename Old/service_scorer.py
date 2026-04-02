"""
Service: Multi-Signal Scorer (Final Stage)
Combines semantic + audio energy + speaker turn density signals into ranked shorts.
Called by pipeline.py — all paths and config passed explicitly.
"""

import json
from collections import Counter

WEIGHT_PROFILES = {
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
VALID_TYPES = set(WEIGHT_PROFILES.keys()) - {"default"}

_DURATION_MODES = {"0-10": 10, "10-20": 20}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_semantic(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    highlights   = data["highlights"]
    types        = [h.get("content_type", "default") for h in highlights]
    content_type = Counter(types).most_common(1)[0][0]
    if content_type not in VALID_TYPES:
        content_type = "default"
    return {"content_type": content_type, "highlights": highlights}


def _load_audio(path: str) -> tuple[dict, dict]:
    """Returns (raw audio_data dict, normalised energy signal {start: score})."""
    with open(path, encoding="utf-8") as f:
        audio_data = json.load(f)
    raw  = {w["window_start"]: w["mean_energy_30s"] for w in audio_data["analysis_30s_windows"]}
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo or 1.0
    energy = {start: (e - lo) / span for start, e in raw.items()}
    return audio_data, energy


def _load_dialogue(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {float(w["start_seconds"]): w["score"] for w in data["windows"]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mmss_to_sec(t: str) -> float:
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0]) * 60 + int(parts[1])


def _sec_to_mmss(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _nearest_window(sec: float, signal: dict, window_size: int = 30) -> float:
    bucket = (sec // window_size) * window_size
    return signal.get(bucket, signal.get(bucket - window_size, 0.0))


def _find_best_subwindow(h_start: float, h_end: float, audio_data: dict,
                          target_sec: int, step: float = 0.5) -> tuple[float, float]:
    """Slide a target_sec window inside [h_start, h_end] and pick peak-dense region."""
    if h_end - h_start <= target_sec:
        return h_start, h_end

    peaks = [
        p
        for w in audio_data["analysis_30s_windows"]
        for p in w.get("peaks_in_window", [])
        if h_start <= p["start_sec"] < h_end
    ]

    if not peaks:
        return h_start, h_start + target_sec

    best_start, best_score = h_start, -1.0
    t = h_start
    while round(t + target_sec, 1) <= h_end:
        score = sum(p["energy"] for p in peaks if t <= p["start_sec"] < t + target_sec)
        if score > best_score:
            best_score, best_start = score, t
        t = round(t + step, 1)

    return best_start, round(best_start + target_sec, 1)


# ── Core scorer ───────────────────────────────────────────────────────────────

def run_scorer(
    highlight_file: str,
    audio_file:     str,
    dialogue_file:  str,
    output_file:    str,
    shorts_count:   int = 2,
    duration_mode:  str = "0-10",
    min_gap_sec:    int = 10,
) -> dict:
    """
    Load all three signal files, combine them, select best sub-windows,
    save final_clips.json, and return the result dict.
    """
    if duration_mode not in _DURATION_MODES:
        raise ValueError(f"duration_mode must be one of {list(_DURATION_MODES)}, got '{duration_mode}'")
    target_sec = _DURATION_MODES[duration_mode]

    semantic             = _load_semantic(highlight_file)
    audio_data, energy   = _load_audio(audio_file)
    dialogue             = _load_dialogue(dialogue_file)

    content_type = semantic["content_type"]
    weights      = WEIGHT_PROFILES.get(content_type, WEIGHT_PROFILES["default"])
    highlights   = semantic["highlights"]

    scored = []
    for h in highlights:
        h_start = _mmss_to_sec(h["start_time"])
        h_end   = _mmss_to_sec(h["end_time"])

        s_sem  = h.get("score", 0.0)
        s_eng  = _nearest_window(h_start, energy)
        s_dial = _nearest_window(h_start, dialogue)

        composite = (weights["semantic"] * s_sem +
                     weights["energy"]   * s_eng +
                     weights["dialogue"] * s_dial)

        clip_start, clip_end = _find_best_subwindow(h_start, h_end, audio_data, target_sec)

        scored.append({
            "rank":              0,
            "highlight_start":   h["start_time"],
            "highlight_end":     h["end_time"],
            "clip_start_sec":    clip_start,
            "clip_end_sec":      clip_end,
            "clip_start_time":   _sec_to_mmss(clip_start),
            "clip_end_time":     _sec_to_mmss(clip_end),
            "clip_duration_sec": round(clip_end - clip_start, 1),
            "composite_score":   round(composite,  4),
            "semantic_score":    round(s_sem,       4),
            "energy_score":      round(s_eng,       4),
            "dialogue_score":    round(s_dial,      4),
            "hook":              h.get("hook", ""),
            "engagement_type":   h.get("engagement_type", ""),
            "reason":            h.get("reason", ""),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    selected, used_ends = [], []
    for clip in scored:
        if not any(clip["clip_start_sec"] < end + min_gap_sec for end in used_ends):
            selected.append(clip)
            used_ends.append(clip["clip_end_sec"])
        if len(selected) >= shorts_count:
            break

    for i, clip in enumerate(selected, 1):
        clip["rank"] = i

    result = {
        "content_type":       content_type,
        "duration_mode":      duration_mode,
        "target_duration_sec": target_sec,
        "weights_used":       weights,
        "total_clips":        len(selected),
        "clips":              selected,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[scorer]  Saved -> {output_file}")
    return result
