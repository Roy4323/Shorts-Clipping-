"""
Service: Speaker Turn Density Scorer (Signal 3)
Uses pyannote/segmentation-3.0 to detect speaker switches per 30s window.
Called by pipeline.py — all paths passed explicitly, no hardcoded constants.

HuggingFace setup:
  1. Accept license at hf.co/pyannote/segmentation-3.0
  2. Set HUGGING_FACE_TOKEN in .env
"""

import os
import json
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

WINDOW_SECONDS = 30


def run_diarization(audio_path: str) -> list[dict]:
    """
    Run pyannote segmentation on audio_path.
    Returns sorted speaker segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in .env")

    import torch
    import numpy as np
    import soundfile as sf
    from pyannote.audio import Model, Inference

    print("[diarizer] Loading pyannote/segmentation-3.0 ...")
    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)

    print(f"[diarizer] Loading audio: {audio_path}")
    data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform    = torch.from_numpy(data.T)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    print("[diarizer] Running segmentation inference ...")
    inference  = Inference(seg_model, duration=10.0, step=2.5)
    seg_output = inference(audio_input)

    SILENCE_THRESHOLD = 0.3
    probs = seg_output.data
    if probs.ndim == 3:
        probs = probs.mean(axis=1)   # (n_chunks, n_speakers)
    n_frames    = probs.shape[0]
    sw          = seg_output.sliding_window
    frame_times = sw.start + np.arange(n_frames) * sw.step

    frame_max    = probs.max(axis=1).tolist()
    frame_argmax = probs.argmax(axis=1).tolist()
    frame_labels = [
        f"SPEAKER_{frame_argmax[i]:02d}" if frame_max[i] > SILENCE_THRESHOLD else "SILENCE"
        for i in range(n_frames)
    ]

    segments  = []
    seg_start = float(frame_times[0])
    seg_label = frame_labels[0]
    for i in range(1, n_frames):
        if frame_labels[i] != seg_label:
            if seg_label != "SILENCE":
                segments.append({"start": round(seg_start, 3),
                                  "end":   round(float(frame_times[i]), 3),
                                  "speaker": seg_label})
            seg_start = float(frame_times[i])
            seg_label = frame_labels[i]
    if seg_label != "SILENCE":
        segments.append({"start": round(seg_start, 3),
                          "end":   round(float(frame_times[-1]), 3),
                          "speaker": seg_label})

    segments.sort(key=lambda s: s["start"])
    print(f"[diarizer] {len(segments)} speaker segments found.")
    return segments


def detect_speaker_switches(segments: list[dict]) -> list[dict]:
    switches = []
    for i in range(1, len(segments)):
        prev, curr = segments[i - 1], segments[i]
        if prev["speaker"] != curr["speaker"]:
            switches.append({"time": curr["start"],
                              "from_speaker": prev["speaker"],
                              "to_speaker":   curr["speaker"]})
    return switches


def bucket_switches_into_windows(switches: list[dict], total_duration: float,
                                  window_sec: int = WINDOW_SECONDS) -> list[dict]:
    num_windows = int(total_duration // window_sec) + 1
    counts: dict[int, int] = defaultdict(int)
    for sw in switches:
        counts[int(sw["time"] // window_sec)] += 1

    windows = []
    for idx in range(num_windows):
        start_s = idx * window_sec
        end_s   = min(start_s + window_sec, total_duration)
        windows.append({
            "window_index":   idx,
            "start_seconds":  start_s,
            "end_seconds":    round(end_s, 1),
            "start_time":     _seconds_to_mmss(start_s),
            "end_time":       _seconds_to_mmss(int(end_s)),
            "switch_count":   counts.get(idx, 0),
            "score":          0.0,
        })
    return windows


def normalize_scores(windows: list[dict]) -> list[dict]:
    max_count = max((w["switch_count"] for w in windows), default=0)
    for w in windows:
        w["score"] = round(w["switch_count"] / max_count, 4) if max_count > 0 else 0.0
    return windows


def speaker_stats(segments: list[dict]) -> dict:
    totals: dict[str, float] = defaultdict(float)
    for s in segments:
        totals[s["speaker"]] += s["end"] - s["start"]
    return {k: round(v, 2) for k, v in sorted(totals.items())}


def score_speaker_turn_density(audio_path: str, window_sec: int = WINDOW_SECONDS) -> dict:
    """
    Full pipeline: diarize -> count switches -> bucket -> normalize.
    Returns result dict (does NOT save to disk — call save_results separately).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments = run_diarization(audio_path)
    if not segments:
        raise ValueError("Diarization returned no segments.")

    total_duration = segments[-1]["end"]
    num_speakers   = len({s["speaker"] for s in segments})
    print(f"[scorer]  Duration : {_seconds_to_mmss(int(total_duration))} ({total_duration:.1f}s)")
    print(f"[scorer]  Speakers : {num_speakers}")

    switches = detect_speaker_switches(segments)
    print(f"[scorer]  Switches : {len(switches)} total")

    windows = normalize_scores(bucket_switches_into_windows(switches, total_duration, window_sec))

    top = sorted(windows, key=lambda w: w["score"], reverse=True)[:5]
    print(f"[scorer]  Top window: {top[0]['start_time']}-{top[0]['end_time']} "
          f"({top[0]['switch_count']} switches, score={top[0]['score']})")

    return {
        "windows":       windows,
        "top_windows":   top,
        "switches":      switches,
        "speaker_stats": speaker_stats(segments),
        "metadata": {
            "audio_file":             audio_path,
            "model":                  "pyannote/segmentation-3.0",
            "window_seconds":         window_sec,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_human":   _seconds_to_mmss(int(total_duration)),
            "num_speakers":           num_speakers,
            "total_segments":         len(segments),
            "total_switches":         len(switches),
        },
    }


def save_results(result: dict, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[scorer]  Saved -> {output_file}")


def _seconds_to_mmss(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
