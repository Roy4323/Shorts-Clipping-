"""
scorer.py  —  Shorts Clipping Engine
--------------------------------------
Single entry point. Takes a YouTube URL, runs all three signals,
combines them, and returns ranked short clip recommendations.

Usage:
    from scorer import score
    result = score("https://youtu.be/xxx", shorts_length="0-10")

CLI:
    python scorer.py --url https://youtu.be/xxx --length 0-10 --count 2

All intermediate and final outputs are saved under output/:
    output/transcript.txt               (fetched transcript)
    output/audio.wav                    (downloaded audio)
    output/audio_analysis_output.json   (Signal 2)
    output/highlight_analysis.json      (Signal 1)
    output/speaker_turn_density.json    (Signal 3)
    output/final_clips.json             (final result)
"""

import sys
import json
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("output")

# ── Weight profiles ───────────────────────────────────────────────────────────

_WEIGHT_PROFILES = {
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
_VALID_TYPES     = set(_WEIGHT_PROFILES.keys()) - {"default"}
_DURATION_MODES  = {"0-10": 10, "10-20": 20}


# ── Public API ────────────────────────────────────────────────────────────────

def score(url: str, shorts_length: str = "0-10", shorts_count: int = 2) -> dict:
    """
    Full pipeline: fetch → download → analyze → combine → clip.

    Args:
        url           : YouTube video URL
        shorts_length : "0-10" or "10-20" (target clip duration in seconds)
        shorts_count  : number of shorts to recommend (default 2)

    Returns:
        Dict with content_type, duration_mode, weights_used, total_clips, clips[]
    """
    if shorts_length not in _DURATION_MODES:
        raise ValueError(f"shorts_length must be '0-10' or '10-20', got '{shorts_length}'")

    OUTPUT_DIR.mkdir(exist_ok=True)
    transcript_out = str(OUTPUT_DIR / "transcript.txt")
    audio_out      = str(OUTPUT_DIR / "audio_file.wav")
    audio_json     = str(OUTPUT_DIR / "audio_analysis_output.json")
    highlight_json = str(OUTPUT_DIR / "highlight_analysis.json")
    dialogue_json  = str(OUTPUT_DIR / "speaker_turn_density.json")
    final_json     = str(OUTPUT_DIR / "final_clips.json")

    # ── Step 1: Transcript (fetch → save txt → pass segments directly) ────────
    print("\n[scorer] Step 1 — Fetching transcript ...")
    from service.fetch_transcript import fetch_transcript
    segments = fetch_transcript(url, transcript_out)

    # ── Step 2: Signal 2 — Audio energy (librosa) ────────────────────────────
    # audio.wav is placed in output/ by Stage 4 before scorer runs
    print("\n[scorer] Step 2 — Audio energy analysis ...")
    from service.audio_analyzer import analyze_audio
    audio_data = analyze_audio(audio_out, audio_json)

    # ── Step 3: Signal 1 — Semantic highlights (OpenAI) ──────────────────────
    print("\n[scorer] Step 3 — Semantic highlight analysis ...")
    from service.video_analyzer import analyze_segments
    analyze_segments(segments, highlight_json, source_label=url)

    # ── Step 4: Signal 3 — Speaker turn density (pyannote) ───────────────────
    print("\n[scorer] Step 4 — Speaker turn density ...")
    from service.speaker_turn_density import score_speaker_turn_density, save_results
    density = score_speaker_turn_density(audio_out)
    save_results(density, dialogue_json)

    # ── Step 5: Combine all three signals ─────────────────────────────────────
    print("\n[scorer] Step 5 — Combining signals ...")
    result = _combine(highlight_json, audio_json, audio_data, dialogue_json,
                      final_json, shorts_count, shorts_length)

    _print_summary(result)
    return result


# ── Signal combiner ───────────────────────────────────────────────────────────

def _combine(highlight_json, audio_json, audio_data, dialogue_json,
             final_json, shorts_count, duration_mode) -> dict:

    target_sec = _DURATION_MODES[duration_mode]

    # Load signals
    with open(highlight_json, encoding="utf-8") as f:
        hl_data = json.load(f)
    highlights   = hl_data["highlights"]
    content_type = Counter(h.get("content_type", "default") for h in highlights).most_common(1)[0][0]
    if content_type not in _VALID_TYPES:
        content_type = "default"
    weights = _WEIGHT_PROFILES[content_type]

    with open(audio_json, encoding="utf-8") as f:
        raw_audio = json.load(f) if audio_data is None else audio_data
    _raw_e = {w["window_start"]: w["mean_energy_30s"] for w in raw_audio["analysis_30s_windows"]}
    _lo, _hi = min(_raw_e.values()), max(_raw_e.values())
    _span = _hi - _lo or 1.0
    energy = {s: (e - _lo) / _span for s, e in _raw_e.items()}

    with open(dialogue_json, encoding="utf-8") as f:
        dialogue = {float(w["start_seconds"]): w["score"] for w in json.load(f)["windows"]}

    # Score each highlight
    scored = []
    for h in highlights:
        h_start = _mmss(h["start_time"])
        h_end   = _mmss(h["end_time"])
        s_sem   = h.get("score", 0.0)
        s_eng   = _nearest(h_start, energy)
        s_dial  = _nearest(h_start, dialogue)
        comp    = weights["semantic"]*s_sem + weights["energy"]*s_eng + weights["dialogue"]*s_dial
        cs, ce  = _best_subwindow(h_start, h_end, raw_audio, target_sec)
        scored.append({
            "rank": 0,
            "highlight_start": h["start_time"], "highlight_end": h["end_time"],
            "clip_start_sec": cs, "clip_end_sec": ce,
            "clip_start_time": _sec_to_mmss(cs), "clip_end_time": _sec_to_mmss(ce),
            "clip_duration_sec": round(ce - cs, 1),
            "composite_score": round(comp,  4),
            "semantic_score":  round(s_sem, 4),
            "energy_score":    round(s_eng, 4),
            "dialogue_score":  round(s_dial, 4),
            "hook":            h.get("hook", ""),
            "engagement_type": h.get("engagement_type", ""),
            "reason":          h.get("reason", ""),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    selected, used_ends = [], []
    for clip in scored:
        if not any(clip["clip_start_sec"] < end + 10 for end in used_ends):
            selected.append(clip)
            used_ends.append(clip["clip_end_sec"])
        if len(selected) >= shorts_count:
            break

    for i, c in enumerate(selected, 1):
        c["rank"] = i

    result = {
        "content_type": content_type, "duration_mode": duration_mode,
        "target_duration_sec": target_sec, "weights_used": weights,
        "total_clips": len(selected), "clips": selected,
    }
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[scorer]  Saved -> {final_json}")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mmss(t: str) -> float:
    p = t.split(":")
    return int(p[0])*3600 + int(p[1])*60 + int(p[2]) if len(p) == 3 else int(p[0])*60 + int(p[1])

def _sec_to_mmss(sec: float) -> str:
    sec = int(sec); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _nearest(sec: float, signal: dict, w: int = 30) -> float:
    b = (sec // w) * w
    return signal.get(b, signal.get(b - w, 0.0))

def _best_subwindow(h_start, h_end, audio_data, target_sec, step=0.5):
    if h_end - h_start <= target_sec:
        return h_start, h_end
    peaks = [p for win in audio_data["analysis_30s_windows"]
               for p in win.get("peaks_in_window", [])
               if h_start <= p["start_sec"] < h_end]
    if not peaks:
        return h_start, h_start + target_sec
    best_s, best_sc = h_start, -1.0
    t = h_start
    while round(t + target_sec, 1) <= h_end:
        sc = sum(p["energy"] for p in peaks if t <= p["start_sec"] < t + target_sec)
        if sc > best_sc:
            best_sc, best_s = sc, t
        t = round(t + step, 1)
    return best_s, round(best_s + target_sec, 1)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(result: dict) -> None:
    w = result["weights_used"]
    print("\n" + "=" * 65)
    print("  SHORTS RECOMMENDATIONS")
    print("=" * 65)
    print(f"  Content type  : {result['content_type']}")
    print(f"  Duration      : {result['duration_mode']}s  (target={result['target_duration_sec']}s)")
    print(f"  Weights       : sem={w['semantic']} eng={w['energy']} dial={w['dialogue']}")
    print(f"  Clips         : {result['total_clips']}\n")
    for c in result["clips"]:
        bar = "#" * int(c["composite_score"] * 20)
        print(f"  [{c['rank']}] {c['clip_start_time']} - {c['clip_end_time']}  ({c['clip_duration_sec']}s)")
        print(f"       highlight {c['highlight_start']} - {c['highlight_end']}")
        print(f"       {c['composite_score']:.4f}  sem={c['semantic_score']:.2f}  "
              f"eng={c['energy_score']:.2f}  dial={c['dialogue_score']:.2f}  {bar}")
        print(f"       {c['hook']}")
        print()
    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Shorts scoring engine")
    p.add_argument("--url",    required=True,  help="YouTube URL")
    p.add_argument("--length", default="0-10", help='"0-10" or "10-20"')
    p.add_argument("--count",  default=2, type=int, help="Number of shorts")
    a = p.parse_args()
    score(a.url, a.length, a.count)
