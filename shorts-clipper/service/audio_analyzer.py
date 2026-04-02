"""
Service: Audio Energy Analyzer (Signal 2)
Analyzes audio file for energy peaks per 30-second window.
Called by pipeline.py — all paths passed explicitly, no hardcoded constants.
"""

import json
import warnings
import librosa
import numpy as np

warnings.filterwarnings("ignore")


def analyze_audio(file_path: str, output_file: str) -> dict:
    """
    Analyze audio energy and peaks, save results to output_file.
    Returns the full analysis dict.
    """
    print(f"[audio] Loading '{file_path}' ...")
    y, sr = librosa.load(file_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[audio] Duration: {duration:.1f}s")

    # 0.5-second windows
    win_sec      = 0.5
    frame_length = int(win_sec * sr)
    hop_length   = frame_length

    rms  = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr  = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    # Normalize RMS 0-1
    rms_min, rms_max = np.min(rms), np.max(rms)
    rms_norm = (rms - rms_min) / (rms_max - rms_min) if rms_max > rms_min else rms

    windows_per_30s = int(30.0 / win_sec)
    total_windows   = len(rms_norm)
    aggregated      = []

    for i in range(0, total_windows, windows_per_30s):
        end_idx      = min(i + windows_per_30s, total_windows)
        chunk_e      = rms_norm[i:end_idx]
        chunk_z      = zcr[i:end_idx]
        chunk_c      = cent[i:end_idx]
        mean_energy  = float(np.mean(chunk_e))

        peaks = []
        for j, (e, z, c) in enumerate(zip(chunk_e, chunk_z, chunk_c)):
            t0 = (i + j) * win_sec
            t1 = t0 + win_sec
            is_loud  = e > (mean_energy * 1.5) and e > 0.15
            is_noisy = z > (np.mean(zcr) * 1.2) and c > (np.mean(cent) * 1.2)
            if is_loud and is_noisy:
                peaks.append({"start_sec": t0, "end_sec": t1, "type": "Applause/Laughter",     "energy": float(e)})
            elif is_loud:
                peaks.append({"start_sec": t0, "end_sec": t1, "type": "Emphasis/Raised Voice", "energy": float(e)})

        aggregated.append({
            "window_start":     i * win_sec,
            "window_end":       end_idx * win_sec,
            "mean_energy_30s":  mean_energy,
            "peaks_in_window":  peaks,
        })

    # Normalise 30s mean energies to 0-1 across all windows
    energies = [w["mean_energy_30s"] for w in aggregated]
    lo, hi   = min(energies), max(energies)
    span     = hi - lo or 1.0
    for w in aggregated:
        w["mean_energy_30s"] = (w["mean_energy_30s"] - lo) / span

    result = {
        "audio_file":           file_path,
        "total_duration_sec":   total_windows * win_sec,
        "analysis_30s_windows": aggregated,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"[audio] Saved -> {output_file}")
    return result
