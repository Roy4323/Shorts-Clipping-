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
from typing import Any

from api.models import LayoutType, Region
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


def _get_layout_vf(regions: list[Region], width: int, height: int) -> tuple[str, LayoutType]:
    """
    Decide on a layout strategy and return the FFmpeg filter string + type.
    """
    if not regions:
        # Default behavior: single center crop
        ah, ay = height, 0  # Simplified for this example
        cw = int(height * 9 / 16)
        cx = (width - cw) // 2
        return f"crop={cw}:ih:{cx}:0,scale=1080:1920", "fill"

    # 1. Look for specialized containers (Screen, Gameplay)
    screens = [r for r in regions if r.label == "screen"]
    gameplay = [r for r in regions if r.label == "gameplay"]
    faces = [r for r in regions if r.label == "face"]

    # 2. Logic: ScreenShare (Face + Screen)
    if screens and faces:
        f, s = faces[0], screens[0]
        # Crop coordinates from 0-1000 to internal pixels
        fx1, fy1, fx2, fy2 = int(f.x1*width/1000), int(f.y1*height/1000), int(f.x2*width/1000), int(f.y2*height/1000)
        sx1, sy1, sx2, sy2 = int(s.x1*width/1000), int(s.y1*height/1000), int(s.x2*width/1000), int(s.y2*height/1000)
        
        # Build 1080x1920 stack
        # [0:v]crop=w:h:x:y,scale=1080:h2[top]; [0:v]crop=sw:sh:sx:sy,scale=1080:sh2[bot]; [top][bot]vstack
        vf = (
            f"[0:v]crop={fx2-fx1}:{fy2-fy1}:{fx1}:{fy1},scale=1080:-1[t]; "
            f"[0:v]crop={sx2-sx1}:{sy2-sy1}:{sx1}:{sy1},scale=1080:-1[b]; "
            f"[t][b]vstack=inputs=2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        )
        return vf, "screenshare"

    # 3. Logic: Gameplay (Face + Gameplay)
    if gameplay and faces:
        f, g = faces[0], gameplay[0]
        fx1, fy1, fx2, fy2 = int(f.x1*width/1000), int(f.y1*height/1000), int(f.x2*width/1000), int(f.y2*height/1000)
        gx1, gy1, gx2, gy2 = int(g.x1*width/1000), int(g.y1*height/1000), int(g.x2*width/1000), int(g.y2*height/1000)
        
        vf = (
            f"[0:v]crop={fx2-fx1}:{fy2-fy1}:{fx1}:{fy1},scale=1080:-1[t]; "
            f"[0:v]crop={gx2-gx1}:{gy2-gy1}:{gx1}:{gy1},scale=1080:-1[b]; "
            f"[t][b]vstack=inputs=2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        )
        return vf, "gameplay"

    # 4. Logic: Multi-Face Split (Interviews)
    if len(faces) >= 2:
        num = min(len(faces), 3)
        clips = []
        for i in range(num):
            f = faces[i]
            x1, y1, x2, y2 = int(f.x1*width/1000), int(f.y1*height/1000), int(f.x2*width/1000), int(f.y2*height/1000)
            clips.append(f"[0:v]crop={x2-x1}:{y2-y1}:{x1}:{y1},scale=1080:-1[v{i}]")
        
        vf = "; ".join(clips) + "; " + "".join(f"[v{i}]" for i in range(num)) + f"vstack=inputs={num},pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        return vf, ("split_two" if num==2 else "split_three")

    # 5. Default Face Centering (Improved)
    if faces:
        f = faces[0]
        # Calculate a nice 9:16 box around the face
        face_x = (f.x1 + f.x2) / 2000.0 * width
        cw = int(height * 9 / 16)
        cx = int(face_x - cw/2)
        cx = max(0, min(cx, width - cw))
        return f"crop={cw}:ih:{cx}:0,scale=1080:1920", "fill"

    # Final fallback
    cw = int(height * 9 / 16)
    cx = (width - cw) // 2
    return f"crop={cw}:ih:{cx}:0,scale=1080:1920", "fill"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reframe_clip(input_path: str, output_path: Path, content_type: str, regions: list[Region] | None = None) -> tuple[str, LayoutType]:
    """
    Reframe a clip from 16:9 to 9:16 using multi-region layout logic.
    Returns (output_path, layout_type).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[REFRAMER] Starting | input={Path(input_path).name} | layout_regions={len(regions or [])}")

    width, height = _probe_video(input_path)
    vf_string, layout_type = _get_layout_vf(regions or [], width, height)
    
    logger.info(f"[REFRAMER] Layout: {layout_type} | Filter: {vf_string[:100]}...")

    _run_ffmpeg_crop(input_path, str(output_path), vf_string)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"[REFRAMER] Done: {output_path.name} ({size_mb:.1f} MB)")
    return str(output_path), layout_type
