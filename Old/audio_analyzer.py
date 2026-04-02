import librosa
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore') # Suppress librosa warnings for cleaner output

def analyze_audio(file_path="audio_file.wav", output_filename="audio_analysis_output.json"):
    print(f"Loading '{file_path}' with librosa...")
    # Load audio - sr=22050 is default for librosa
    try:
        y, sr = librosa.load(file_path, sr=22050)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please ensure the file is in the directroy.")
        return
        
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio loaded successfully. Total duration: {duration:.2f} seconds")

    # 1. 0.5s Window Processing
    # 0.5 seconds at 22050 Hz = 11025 samples
    duration_sec = 0.5
    frame_length = int(duration_sec * sr)
    hop_length = frame_length # Stride by the same amount so windows don't overlap

    print("Extracting features: RMS Energy, Zero Crossing Rate, Spectral Centroid...")
    
    # Compute RMS energy. 
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Compute Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Compute Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    # 2. Normalize to 0-1 with min-max
    print("Normalizing RMS energy with min-max logic...")
    rms_min = np.min(rms)
    rms_max = np.max(rms)
    
    if rms_max > rms_min:
        rms_normalized = (rms - rms_min) / (rms_max - rms_min)
    else:
        rms_normalized = rms

    # 3. Aggregate per 30s window: mean(energy[window_start:window_end])
    print("Aggregating per 30s windows...")
    windows_per_30s = int(30.0 / duration_sec)
    
    aggregated_30s = []
    total_windows = len(rms_normalized)
    
    # Sliding over the array chunking by 30s
    for i in range(0, total_windows, windows_per_30s):
        end_idx = min(i + windows_per_30s, total_windows)
        
        chunk_energy = rms_normalized[i:end_idx]
        mean_energy_30s = float(np.mean(chunk_energy))
        
        chunk_zcr = zcr[i:end_idx]
        chunk_cent = cent[i:end_idx]
        
        # Internal heuristic: Identify exactly where within these 30s the spikes occur
        peaks = []
        for j, (e, z, c) in enumerate(zip(chunk_energy, chunk_zcr, chunk_cent)):
            time_start = (i + j) * duration_sec
            time_end = time_start + duration_sec
            
            # Simple threshold rules for highlighting
            # "Loud" if current energy is 50% larger than chunk mean and > 0.15 normalized
            is_loud = e > (mean_energy_30s * 1.5) and e > 0.15
            
            # "Noisy" (clapping/audience) if ZCR & Centroid are above the global average
            is_noisy = z > (np.mean(zcr) * 1.2) and c > (np.mean(cent) * 1.2)
            
            if is_loud and is_noisy:
                peaks.append({"start_sec": time_start, "end_sec": time_end, "type": "Applause/Laughter", "energy": float(e)})
            elif is_loud:
                peaks.append({"start_sec": time_start, "end_sec": time_end, "type": "Emphasis/Raised Voice", "energy": float(e)})

        start_time_30s = i * duration_sec
        end_time_30s = end_idx * duration_sec
        
        aggregated_30s.append({
            "window_start": start_time_30s,
            "window_end": end_time_30s,
            "mean_energy_30s": mean_energy_30s,
            "peaks_in_window": peaks
        })

    # Normalize the 30s mean energies across all windows for the combiner
    print("Normalizing 30s aggregated energies to 0-1 scale...")
    if aggregated_30s:
        energies_30s = [w["mean_energy_30s"] for w in aggregated_30s]
        min_e, max_e = min(energies_30s), max(energies_30s)
        if max_e > min_e:
            for w in aggregated_30s:
                w["mean_energy_30s"] = (w["mean_energy_30s"] - min_e) / (max_e - min_e)
        else:
            # Fallback if all windows have identical energy
            for w in aggregated_30s:
                w["mean_energy_30s"] = 0.0

    # Save output to JSON
    output_data = {
        "audio_file": file_path,
        "total_duration_sec": total_windows * duration_sec,
        "analysis_30s_windows": aggregated_30s
    }
    
    print(f"Saving analysis to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print("Audio analysis complete!")

if __name__ == "__main__":
    analyze_audio()
