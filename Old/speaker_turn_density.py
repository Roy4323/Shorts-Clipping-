"""
Stage 3 — Signal 3: Speaker Turn Density Scorer
------------------------------------------------
Uses pyannote/speaker-diarization-3.1 (local, HuggingFace gated model) to
detect speaker switches per 30-second window and normalize to a 0.0–1.0 score.

Requirements:
  pip install pyannote.audio torch

HuggingFace setup:
  1. Accept the model license at hf.co/pyannote/speaker-diarization-3.1
  2. Accept the model license at hf.co/pyannote/segmentation-3.0
  3. Set HUGGING_FACE_TOKEN in your .env file

Usage:
  python speaker_turn_density.py [audio_file.wav] [output.json]
"""

import os
import json
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

WINDOW_SECONDS = 30
AUDIO_FILE = "audio_file.wav"
OUTPUT_FILE = "speaker_turn_density.json"


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def run_diarization(audio_path: str) -> list[dict]:
    """
    Run pyannote speaker diarization on the audio file.
    Returns a list of segments sorted by start time:
      [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"}, ...]
    """
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in .env")

    # Use pyannote/segmentation-3.0 via direct Inference — no speechbrain, no embedding,
    # no community-1 restricted model. The segmentation model outputs per-frame speaker
    # activity probabilities; we threshold to get local speaker segments and detect turns.
    print("[diarizer] Loading pyannote/segmentation-3.0 …")

    import torch
    import numpy as np
    import soundfile as sf
    from pyannote.audio import Model, Inference

    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)

    # Use soundfile — avoids torchaudio's broken torchcodec backend on Windows
    print(f"[diarizer] Loading audio: {audio_path}")
    data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, samples)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    print("[diarizer] Running segmentation inference …")
    inference = Inference(seg_model, duration=10.0, step=2.5)
    seg_output = inference(audio_input)
    # seg_output: SlidingWindowFeature, shape (frames, num_local_speakers)
    # Each column = probability that a local speaker is active

    # Convert frame probabilities → speaker segments.
    # Strategy: per frame, pick the dominant speaker (argmax). If all speakers are below
    # the silence threshold, label the frame as silence. Then merge consecutive same-label
    # frames into segments. This gives clean speaker turn boundaries.
    SILENCE_THRESHOLD = 0.3
    probs = seg_output.data  # (frames, num_speakers) or (chunks, frames_per_chunk, speakers)
    if probs.ndim == 3:
        # Average frame probabilities within each chunk so that n_frames == n_chunks.
        # sw.step (2.5 s) is the step between consecutive chunks, not between individual
        # model frames — using it against n_chunks gives the correct timestamps.
        probs = probs.mean(axis=1)  # (n_chunks, n_speakers)
    n_frames = probs.shape[0]
    sw = seg_output.sliding_window
    frame_times = sw.start + np.arange(n_frames) * sw.step

    # Per-frame dominant speaker (or "silence")
    frame_max = probs.max(axis=1).tolist()      # list of Python floats
    frame_argmax = probs.argmax(axis=1).tolist()  # list of Python ints
    frame_labels = [
        f"SPEAKER_{frame_argmax[i]:02d}" if frame_max[i] > SILENCE_THRESHOLD else "SILENCE"
        for i in range(n_frames)
    ]

    # Merge consecutive same-label frames into segments
    segments = []
    seg_start = float(frame_times[0])
    seg_label = frame_labels[0]
    for i in range(1, n_frames):
        if frame_labels[i] != seg_label:
            if seg_label != "SILENCE":
                segments.append({
                    "start":   round(seg_start, 3),
                    "end":     round(float(frame_times[i]), 3),
                    "speaker": seg_label,
                })
            seg_start = float(frame_times[i])
            seg_label = frame_labels[i]
    if seg_label != "SILENCE":
        segments.append({
            "start":   round(seg_start, 3),
            "end":     round(float(frame_times[-1]), 3),
            "speaker": seg_label,
        })

    segments.sort(key=lambda s: s["start"])
    print(f"[diarizer] {len(segments)} speaker segments found.")
    return segments


# ---------------------------------------------------------------------------
# Turn counting
# ---------------------------------------------------------------------------

def detect_speaker_switches(segments: list[dict]) -> list[dict]:
    """
    Walk consecutive segments. A switch = adjacent segments with different speakers.
    Returns list of switch events: {"time": float, "from": str, "to": str}
    """
    switches = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        if prev["speaker"] != curr["speaker"]:
            switches.append({
                "time": curr["start"],
                "from_speaker": prev["speaker"],
                "to_speaker": curr["speaker"],
            })
    return switches


def bucket_switches_into_windows(
    switches: list[dict],
    total_duration: float,
    window_sec: int = WINDOW_SECONDS,
) -> list[dict]:
    """
    Assign each switch to its 30-second window bucket.
    Returns a list of window dicts sorted by window index.
    """
    num_windows = int(total_duration // window_sec) + 1
    counts: dict[int, int] = defaultdict(int)

    for sw in switches:
        bucket = int(sw["time"] // window_sec)
        counts[bucket] += 1

    windows = []
    for idx in range(num_windows):
        start_s = idx * window_sec
        end_s = min(start_s + window_sec, total_duration)
        windows.append({
            "window_index": idx,
            "start_seconds": start_s,
            "end_seconds": round(end_s, 1),
            "start_time": _seconds_to_mmss(start_s),
            "end_time": _seconds_to_mmss(int(end_s)),
            "switch_count": counts.get(idx, 0),
            "score": 0.0,  # filled in normalize step
        })

    return windows


def normalize_scores(windows: list[dict]) -> list[dict]:
    """
    Normalize switch counts: score = count / max_count.
    If all windows are silent (max=0) every score stays 0.
    """
    max_count = max((w["switch_count"] for w in windows), default=0)
    for w in windows:
        w["score"] = round(w["switch_count"] / max_count, 4) if max_count > 0 else 0.0
    return windows


# ---------------------------------------------------------------------------
# Speaker-level stats
# ---------------------------------------------------------------------------

def speaker_stats(segments: list[dict]) -> dict:
    """Total speaking time per speaker (seconds)."""
    totals: dict[str, float] = defaultdict(float)
    for s in segments:
        totals[s["speaker"]] += s["end"] - s["start"]
    return {k: round(v, 2) for k, v in sorted(totals.items())}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def score_speaker_turn_density(
    audio_path: str = "audio_file.wav",
    window_sec: int = WINDOW_SECONDS,
) -> dict:
    """
    Full pipeline: diarize → count switches → bucket → normalize.

    Returns dict with:
      windows        – per-window switch counts + 0-1 scores
      switches       – raw switch event list
      speaker_stats  – speaking time per speaker
      metadata       – file info, model, window size, totals
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments = run_diarization(audio_path)

    if not segments:
        raise ValueError("Diarization returned no segments — check the audio file.")

    total_duration = segments[-1]["end"]
    num_speakers = len({s["speaker"] for s in segments})

    print(f"[scorer]  Duration : {_seconds_to_mmss(int(total_duration))} ({total_duration:.1f}s)")
    print(f"[scorer]  Speakers : {num_speakers}")

    switches = detect_speaker_switches(segments)
    print(f"[scorer]  Switches : {len(switches)} total")

    windows = bucket_switches_into_windows(switches, total_duration, window_sec)
    windows = normalize_scores(windows)

    top = sorted(windows, key=lambda w: w["score"], reverse=True)[:5]
    print(f"[scorer]  Top window: {top[0]['start_time']}–{top[0]['end_time']} "
          f"({top[0]['switch_count']} switches, score={top[0]['score']})")

    return {
        "windows": windows,
        "top_windows": top,
        "switches": switches,
        "speaker_stats": speaker_stats(segments),
        "metadata": {
            "audio_file": audio_path,
            "model": "pyannote/segmentation-3.0",
            "window_seconds": window_sec,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_human": _seconds_to_mmss(int(total_duration)),
            "num_speakers": num_speakers,
            "total_segments": len(segments),
            "total_switches": len(switches),
        },
    }


def save_results(result: dict, output_file: str = "speaker_turn_density.json") -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[scorer]  Saved -> {output_file}")


def print_summary(result: dict) -> None:
    print("\n" + "=" * 60)
    print("  SPEAKER TURN DENSITY — TOP WINDOWS")
    print("=" * 60)
    meta = result["metadata"]
    print(f"  File     : {meta['audio_file']}")
    print(f"  Duration : {meta['total_duration_human']}")
    print(f"  Speakers : {meta['num_speakers']}  |  "
          f"Total switches : {meta['total_switches']}")
    print(f"  Window   : {meta['window_seconds']}s\n")

    print("  Speaking time:")
    for spk, secs in result["speaker_stats"].items():
        print(f"    {spk}: {secs}s")

    print("\n  Top 5 windows by switch density:")
    for w in result["top_windows"]:
        bar = "█" * int(w["score"] * 20)
        print(f"  {w['start_time']}–{w['end_time']}  "
              f"switches={w['switch_count']:3d}  score={w['score']:.4f}  {bar}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_mmss(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = score_speaker_turn_density(AUDIO_FILE)
    print_summary(result)
    save_results(result, OUTPUT_FILE)
