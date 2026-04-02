"""
Test suite for speaker_turn_density.py
Run: python test_speaker_density.py
"""

import json
import os
import sys
import wave
import struct
import math

AUDIO_FILE = "audio_file.wav"
OUTPUT_FILE = "speaker_turn_density.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_wav(path: str, duration_sec: int = 10, sample_rate: int = 16000) -> None:
    """Create a minimal silent WAV file for unit tests (no real audio needed)."""
    n_samples = duration_sec * sample_rate
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # simple sine tone so it's non-zero
        data = b"".join(
            struct.pack("<h", int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(n_samples)
        )
        wf.writeframes(data)


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Unit tests (no model needed)
# ---------------------------------------------------------------------------

def test_parse_seconds_to_mmss() -> None:
    from speaker_turn_density import _seconds_to_mmss
    assert _seconds_to_mmss(0)    == "00:00"
    assert _seconds_to_mmss(59)   == "00:59"
    assert _seconds_to_mmss(60)   == "01:00"
    assert _seconds_to_mmss(3661) == "01:01:01"
    _pass("_seconds_to_mmss conversions correct")


def test_detect_switches_same_speaker() -> None:
    from speaker_turn_density import detect_speaker_switches
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "A"},
    ]
    switches = detect_speaker_switches(segments)
    assert switches == [], f"Expected no switches, got {switches}"
    _pass("No switch detected when same speaker talks consecutively")


def test_detect_switches_alternating() -> None:
    from speaker_turn_density import detect_speaker_switches
    segments = [
        {"start": 0.0,  "end": 5.0,  "speaker": "A"},
        {"start": 5.0,  "end": 10.0, "speaker": "B"},
        {"start": 10.0, "end": 15.0, "speaker": "A"},
        {"start": 15.0, "end": 20.0, "speaker": "B"},
    ]
    switches = detect_speaker_switches(segments)
    assert len(switches) == 3
    assert switches[0]["from_speaker"] == "A"
    assert switches[0]["to_speaker"]   == "B"
    _pass("3 switches detected in A-B-A-B pattern")


def test_bucket_windows() -> None:
    from speaker_turn_density import bucket_switches_into_windows
    switches = [
        {"time": 5.0},
        {"time": 10.0},
        {"time": 35.0},   # window 1
    ]
    windows = bucket_switches_into_windows(switches, total_duration=60.0, window_sec=30)
    assert windows[0]["switch_count"] == 2
    assert windows[1]["switch_count"] == 1
    assert windows[2]["switch_count"] == 0
    _pass("Switches bucketed into correct 30s windows")


def test_normalize_scores() -> None:
    from speaker_turn_density import normalize_scores
    windows = [
        {"switch_count": 0},
        {"switch_count": 4},
        {"switch_count": 8},
        {"switch_count": 2},
    ]
    result = normalize_scores(windows)
    assert result[2]["score"] == 1.0,  f"Max window should be 1.0, got {result[2]['score']}"
    assert result[0]["score"] == 0.0,  f"Zero-switch window should be 0.0"
    assert result[1]["score"] == 0.5,  f"Half-max window should be 0.5"
    _pass("Normalization: max=1.0, zero=0.0, proportional scores correct")


def test_normalize_all_zeros() -> None:
    from speaker_turn_density import normalize_scores
    windows = [{"switch_count": 0}, {"switch_count": 0}]
    result = normalize_scores(windows)
    assert all(w["score"] == 0.0 for w in result)
    _pass("All-zero switch counts -> all scores 0.0 (no division by zero)")


def test_speaker_stats() -> None:
    from speaker_turn_density import speaker_stats
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 14.0, "speaker": "SPEAKER_01"},
        {"start": 14.0, "end": 20.0, "speaker": "SPEAKER_00"},
    ]
    stats = speaker_stats(segments)
    assert stats["SPEAKER_00"] == 16.0
    assert stats["SPEAKER_01"] == 4.0
    _pass("Speaker stats: total speaking time computed correctly")


# ---------------------------------------------------------------------------
# Integration test — real model + real audio
# ---------------------------------------------------------------------------

def test_full_pipeline_real_audio() -> None:
    if not os.path.isfile(AUDIO_FILE):
        print(f"  [SKIP] {AUDIO_FILE} not found — skipping integration test")
        return

    print(f"\n  [INFO] Running full pipeline on {AUDIO_FILE} …")
    print("         (loads pyannote model — may take 30-60s on first run)\n")

    from speaker_turn_density import score_speaker_turn_density, save_results

    result = score_speaker_turn_density(AUDIO_FILE)

    # Structure checks
    assert "windows"       in result, "Missing 'windows' key"
    assert "top_windows"   in result, "Missing 'top_windows' key"
    assert "switches"      in result, "Missing 'switches' key"
    assert "speaker_stats" in result, "Missing 'speaker_stats' key"
    assert "metadata"      in result, "Missing 'metadata' key"
    _pass("Result has all required top-level keys")

    windows = result["windows"]
    assert len(windows) > 0, "No windows returned"
    for w in windows:
        assert 0.0 <= w["score"] <= 1.0, f"Score out of range: {w['score']}"
        assert w["switch_count"] >= 0
    _pass(f"{len(windows)} windows returned, all scores in [0.0, 1.0]")

    top = result["top_windows"]
    assert len(top) <= 5
    scores = [w["score"] for w in top]
    assert scores == sorted(scores, reverse=True), "top_windows not sorted by score desc"
    _pass(f"top_windows sorted correctly ({len(top)} entries)")

    meta = result["metadata"]
    assert meta["total_duration_seconds"] > 0
    assert meta["num_speakers"] >= 1
    _pass(f"Metadata valid: {meta['num_speakers']} speaker(s), "
          f"{meta['total_duration_human']} duration, "
          f"{meta['total_switches']} total switches")

    # Save and verify JSON round-trip
    save_results(result, OUTPUT_FILE)
    assert os.path.isfile(OUTPUT_FILE), "Output JSON file not created"
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["total_switches"] == meta["total_switches"]
    _pass(f"JSON saved and round-trips correctly -> {OUTPUT_FILE}")

    # Print summary
    print(f"\n  --- Integration Result Summary ---")
    print(f"  Speakers   : {meta['num_speakers']}")
    print(f"  Duration   : {meta['total_duration_human']}")
    print(f"  Switches   : {meta['total_switches']}")
    print(f"  Top window : {top[0]['start_time']}–{top[0]['end_time']} "
          f"(score={top[0]['score']}, switches={top[0]['switch_count']})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  UNIT TESTS")
    print("=" * 55)
    test_parse_seconds_to_mmss()
    test_detect_switches_same_speaker()
    test_detect_switches_alternating()
    test_bucket_windows()
    test_normalize_scores()
    test_normalize_all_zeros()
    test_speaker_stats()

    print(f"\n  All unit tests passed.\n")

    print("=" * 55)
    print("  INTEGRATION TEST (real model + audio_file.wav)")
    print("=" * 55)
    test_full_pipeline_real_audio()

    print("\n  Done.")
