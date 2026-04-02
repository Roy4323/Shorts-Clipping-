"""Stage 5: Reframe 16:9 clips to 9:16 using subject-aware cropping.

Strategy:
  - podcast_interview / general: sample frames, run MediaPipe face detection,
    use median face-center x to position the crop window.
  - All other content types (sports, music, b-roll): smart center crop.
  - If MediaPipe is not installed: always falls back to center crop.
"""

import json
import statistics
import subprocess
from pathlib import Path

from utils.logger import logger

try:
    import cv2
    import mediapipe as mp

    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False
    logger.warning("mediapipe/opencv not installed — reframer will use center-crop fallback.")

# Content types that benefit from face-aware cropping
_FACE_TYPES = {"podcast_interview", "general"}
# Sample one frame per this many seconds of clip
_SAMPLE_INTERVAL_SEC = 1.0


class ReframerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReframerError(f"ffprobe failed: {result.stderr[:400]}")
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise ReframerError("No video stream found in clip.")


def _detect_vertical_crop(video_path: str) -> tuple[int, int] | None:
    """
    Use FFmpeg's cropdetect to find the active image height and Y-offset,
    stripping away baked-in cinematic black bars.
    Returns (active_h, offset_y) or None if unable to detect.
    """
    import re
    cmd = [
        "ffmpeg", "-y", "-ss", "1", "-i", str(video_path),
        "-vframes", "10", "-vf", "cropdetect=24:16:0", "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse output like: crop=1920:816:0:132
    matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
    if matches:
        # Find the most common detected crop geometry
        most_common = max(set(matches), key=matches.count)
        _, h, _, y = most_common
        return int(h), int(y)
    return None


def _face_center_x_fractions(clip_path: str) -> list[float]:
    """
    Sample frames at _SAMPLE_INTERVAL_SEC intervals and return the relative
    x-center of the most prominent detected face (0.0 = left, 1.0 = right).
    Returns an empty list if no faces are found or MediaPipe is unavailable.
    """
    if not _MEDIAPIPE_OK:
        return []

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_step = max(1, int(fps * _SAMPLE_INTERVAL_SEC))

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    centers: list[float] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_detector.process(rgb)
                if result.detections:
                    box = result.detections[0].location_data.relative_bounding_box
                    centers.append(box.xmin + box.width / 2.0)
            frame_idx += 1
    finally:
        cap.release()
        face_detector.close()

    logger.debug(f"Face centers sampled: {len(centers)} detections over {frame_idx} frames.")
    return centers


def _run_ffmpeg_crop(input_path: str, output_path: str, vf_string: str) -> None:
    """
    Apply a filter chain to ensure the media fills the entire vertical canvas,
    including stripping any baked-in cinematic borders.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf_string,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReframerError(f"FFmpeg crop failed:\n{result.stderr[-600:]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reframe_clip(input_path: str, output_path: Path, content_type: str) -> str:
    """
    Reframe a clip from 16:9 to 9:16.

    Args:
        input_path:   Path to the raw clip (16:9 or any aspect).
        output_path:  Destination path for the reframed clip.
        content_type: ContentType string from the classifier.

    Returns:
        Absolute path to the reframed clip (str).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[REFRAMER] Starting | input={Path(input_path).name} | content_type={content_type} | mediapipe={'available' if _MEDIAPIPE_OK else 'NOT installed'}")

    width, height = _probe_video(input_path)
    
    # NEW: Detect active image bounds to strip cinematic green/black bars
    active_crop = _detect_vertical_crop(input_path)
    if active_crop:
        ah, ay = active_crop
        logger.info(f"[REFRAMER] Detected active frame: {width}x{ah} (offset_y={ay})")
    else:
        ah, ay = height, 0

    crop_w = int(ah * 9 / 16)
    logger.info(f"[REFRAMER] Target crop_w={crop_w} for active height {ah} (9:16)")

    # If the clip is already portrait (crop_w >= width), just copy it through
    if crop_w >= width:
        logger.info(f"[REFRAMER] Already portrait ({width}x{height}) — copying without crop.")
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ReframerError(f"FFmpeg copy failed:\n{result.stderr[-400:]}")
        return str(output_path)

    # Determine crop_x
    crop_x: int
    if content_type in _FACE_TYPES and _MEDIAPIPE_OK:
        logger.info(f"[REFRAMER] Running MediaPipe face detection on sampled frames...")
        centers = _face_center_x_fractions(input_path)
        if centers:
            median_frac = statistics.median(centers)
            crop_x = int(median_frac * width - crop_w / 2)
            logger.info(f"[REFRAMER] Face detection: {len(centers)} samples | median_x={median_frac:.3f} | crop_x={crop_x}")
        else:
            logger.info(f"[REFRAMER] No faces detected — using center crop.")
            crop_x = (width - crop_w) // 2
    else:
        logger.info(f"[REFRAMER] content_type='{content_type}' → center crop (no face detection).")
        crop_x = (width - crop_w) // 2

    # Clamp so we never exceed frame boundaries
    crop_x = max(0, min(crop_x, width - crop_w))
    
    vf_string = f"crop=iw:{ah}:0:{ay},crop={crop_w}:{ah}:{crop_x}:0"
    logger.info(f"[REFRAMER] Final crop chain: {vf_string}")

    _run_ffmpeg_crop(input_path, str(output_path), vf_string)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"[REFRAMER] Done: {output_path.name} ({size_mb:.1f} MB)")
    return str(output_path)
