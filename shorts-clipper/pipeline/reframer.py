"""Stage 5: Reframe 16:9 clips to 9:16 using dynamic subject tracking.

Strategy:
  1. Sample face positions across the clip using MediaPipe (fast, local).
  2. Build a smooth crop-x trajectory that follows the active speaker.
  3. Use FFmpeg sendcmd to dynamically shift the crop window per-frame.

  Fallback: if MediaPipe unavailable or no faces found, uses the AI-detected
  region from the vision agent as a static center hint.
"""

import json
import statistics
import subprocess
from pathlib import Path

from api.models import LayoutType, Region
from utils.logger import logger

try:
    import cv2
    import mediapipe as mp

    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False
    logger.warning("mediapipe/opencv not installed — reframer will use static crop fallback.")

# Output canvas
_OUT_W = 1080
_OUT_H = 1920

# Face tracking settings
_TRACK_SAMPLE_INTERVAL_SEC = 0.15  # Sample every ~4 frames — near real-time tracking
_MULTI_FACE_SPREAD_PX = 200        # If faces are > 200px apart, go wide to show both
_COLLAPSE_THRESHOLD_PX = 120       # Positions within 120px are "same face" — no cut


class ReframerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Video probe
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> tuple[int, int, float]:
    """Return (width, height, fps) of the first video stream."""
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
            w = int(stream["width"])
            h = int(stream["height"])
            # Parse fps from r_frame_rate (e.g. "30/1" or "24000/1001")
            fps_str = stream.get("r_frame_rate", "30/1")
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps = 30.0
            return w, h, fps
    raise ReframerError("No video stream found in clip.")


# ---------------------------------------------------------------------------
# Dynamic face tracking via MediaPipe
# ---------------------------------------------------------------------------

def _track_face_positions(clip_path: str, src_width: int) -> list[tuple[float, float, float]]:
    """
    Track faces across the clip.
    Returns list of (timestamp_sec, center_x_pixels, spread_px).

    spread_px = distance between leftmost and rightmost face centers.
    If only 1 face, spread = 0.
    When spread > _MULTI_FACE_SPREAD_PX, the crop should widen to show all faces.
    """
    if not _MEDIAPIPE_OK:
        return []

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_step = max(1, int(fps * _TRACK_SAMPLE_INTERVAL_SEC))

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4
    )

    positions: list[tuple[float, float, float]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step == 0:
                timestamp = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_detector.process(rgb)
                if result.detections:
                    # Get all face center X positions
                    face_xs = []
                    for d in result.detections:
                        box = d.location_data.relative_bounding_box
                        cx_px = (box.xmin + box.width / 2.0) * src_width
                        face_xs.append(cx_px)

                    # Center = midpoint of all faces, spread = distance between extremes
                    center_x = (min(face_xs) + max(face_xs)) / 2.0
                    spread = max(face_xs) - min(face_xs)
                    positions.append((timestamp, center_x, spread))
            frame_idx += 1
    finally:
        cap.release()
        face_detector.close()

    logger.info(
        f"[REFRAMER] Face tracking: {len(positions)} detections "
        f"over {frame_idx} frames ({frame_idx / max(fps, 1):.1f}s)"
    )
    return positions


def _smooth_positions(
    positions: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Hard-cut style with multi-face awareness:

    1. If multiple faces are detected frequently (>30% of samples have spread),
       just go WIDE for the entire clip — no cuts at all.
    2. Otherwise, group consecutive same-position samples into holds,
       with instant hard cuts between them.
    """
    if len(positions) < 2:
        return positions

    # Count how many samples have multiple faces (spread > threshold)
    multi_face_count = sum(1 for _, _, spread in positions if spread > _MULTI_FACE_SPREAD_PX)
    multi_face_ratio = multi_face_count / len(positions)

    if multi_face_ratio > 0.3:
        # Multiple faces detected often — go wide for the entire clip.
        # Use the overall center and max spread so all faces are visible.
        all_centers = [x for _, x, _ in positions]
        all_spreads = [s for _, _, s in positions]
        overall_center = sum(all_centers) / len(all_centers)
        overall_spread = max(all_spreads)
        logger.info(
            f"[REFRAMER] Multi-face dominant ({multi_face_ratio:.0%} of frames) "
            f"— using wide shot for entire clip, spread={overall_spread:.0f}px"
        )
        # Single keyframe at t=0 — wide shot, no cuts
        return [(positions[0][0], overall_center, overall_spread)]

    # Single-face dominant — group into holds with hard cuts
    groups: list[list[tuple[float, float, float]]] = [[positions[0]]]
    for ts, x, spread in positions[1:]:
        group_avg_x = sum(p[1] for p in groups[-1]) / len(groups[-1])
        if abs(x - group_avg_x) <= _COLLAPSE_THRESHOLD_PX:
            groups[-1].append((ts, x, spread))
        else:
            groups.append([(ts, x, spread)])

    # Merge short groups (< 0.5s) into neighbors — avoid flickering cuts
    merged: list[list[tuple[float, float, float]]] = [groups[0]]
    for group in groups[1:]:
        duration = group[-1][0] - group[0][0]
        if duration < 0.5 and merged:
            # Too short — absorb into previous group (go wide momentarily)
            merged[-1].extend(group)
        else:
            merged.append(group)

    result: list[tuple[float, float, float]] = []
    for group in merged:
        t_start = group[0][0]
        avg_x = sum(p[1] for p in group) / len(group)
        max_spread = max(p[2] for p in group)
        result.append((t_start, avg_x, max_spread))

    logger.info(
        f"[REFRAMER] Hard-cut collapse: {len(positions)} samples → {len(result)} cuts "
        f"(multi_face={multi_face_ratio:.0%})"
    )
    return result


def _interpolate_crop_x(
    positions: list[tuple[float, float]],
    timestamp: float,
) -> float:
    """Linearly interpolate face X position at a given timestamp."""
    if not positions:
        return -1
    if timestamp <= positions[0][0]:
        return positions[0][1]
    if timestamp >= positions[-1][0]:
        return positions[-1][1]

    # Find surrounding samples
    for i in range(len(positions) - 1):
        t0, x0 = positions[i]
        t1, x1 = positions[i + 1]
        if t0 <= timestamp <= t1:
            frac = (timestamp - t0) / max(t1 - t0, 0.001)
            return x0 + (x1 - x0) * frac
    return positions[-1][1]


# ---------------------------------------------------------------------------
# FFmpeg runner
# ---------------------------------------------------------------------------

def _run_ffmpeg_crop(input_path: str, output_path: str, vf_string: str) -> None:
    """Run FFmpeg with a -vf filter string."""
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


def _run_ffmpeg_dynamic_crop(
    input_path: str,
    output_path: str,
    positions: list[tuple[float, float, float]],
    base_crop_w: int,
    src_width: int,
    src_height: int,
    crop_h: int | None = None,
) -> None:
    """
    Use FFmpeg's crop filter with a time-based expression to dynamically
    move the crop window following tracked face positions.

    When faces are spread apart (multi-face), widens the crop to show both.
    When single face, uses base_crop_w for tighter framing.
    """
    # Convert positions to (timestamp, crop_x, crop_w) — crop_w varies per frame
    keyframes: list[tuple[float, int, int]] = []
    for ts, center_x, spread in positions:
        if spread > _MULTI_FACE_SPREAD_PX:
            # Wide mode: crop must fit all faces + padding
            needed_w = int(spread + 200)  # 100px padding each side
            cw = max(base_crop_w, needed_w)
            # Keep 9:16 ratio: widen but cap at source width
            cw = min(cw, src_width)
            cw = cw - (cw % 2)
        else:
            cw = base_crop_w

        cx = int(center_x - cw / 2)
        cx = max(0, min(cx, src_width - cw))
        keyframes.append((ts, cx, cw))

    # Downsample to max ~30 keyframes for expression length
    if len(keyframes) > 30:
        step = len(keyframes) / 30
        sampled = [keyframes[int(i * step)] for i in range(30)]
        if keyframes[-1] != sampled[-1]:
            sampled.append(keyframes[-1])
        keyframes = sampled

    # Check if crop width varies (multi-face moments)
    widths = set(kf[2] for kf in keyframes)
    dynamic_width = len(widths) > 1

    if dynamic_width:
        # Both x and w change — build expressions for both
        x_expr, w_expr = _build_dual_expr(keyframes)
    else:
        # Only x changes — simpler expression
        fixed_w = keyframes[0][2] if keyframes else base_crop_w
        x_expr = _build_x_expr([(ts, cx) for ts, cx, _ in keyframes])
        w_expr = str(fixed_w)

    ch = crop_h if crop_h else src_height
    cy = max(0, (src_height - ch) // 2)
    vf = f"crop='{w_expr}':{ch}:'{x_expr}':{cy},scale={_OUT_W}:{_OUT_H}"

    logger.info(
        f"[REFRAMER] Dynamic crop v5 | {len(keyframes)} keyframes | "
        f"dynamic_width={dynamic_width} | widths={sorted(widths)}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReframerError(f"FFmpeg dynamic crop failed:\n{result.stderr[-600:]}")


def _build_x_expr(keyframes: list[tuple[float, int]]) -> str:
    """Build FFmpeg step-function expression for crop x — instant jumps, no sliding."""
    if len(keyframes) < 2:
        return str(keyframes[0][1] if keyframes else 0)

    # Step function: hold x0 for the entire interval, then jump to next value
    segments: list[str] = []
    for i in range(len(keyframes) - 1):
        t0, x0 = keyframes[i]
        t1, _ = keyframes[i + 1]
        segments.append(f"if(between(t\\,{t0:.2f}\\,{t1:.2f})\\,{x0}")

    fallback = str(keyframes[-1][1])
    expr = fallback
    for seg in reversed(segments):
        expr = f"{seg}\\,{expr})"
    return expr


def _build_dual_expr(keyframes: list[tuple[float, int, int]]) -> tuple[str, str]:
    """Build FFmpeg step-function expressions for crop x and w — instant jumps."""
    x_segs: list[str] = []
    w_segs: list[str] = []
    for i in range(len(keyframes) - 1):
        t0, x0, w0 = keyframes[i]
        t1, _, _ = keyframes[i + 1]
        x_segs.append(f"if(between(t\\,{t0:.2f}\\,{t1:.2f})\\,{x0}")
        w_segs.append(f"if(between(t\\,{t0:.2f}\\,{t1:.2f})\\,{w0}")

    x_fall = str(keyframes[-1][1])
    w_fall = str(keyframes[-1][2])
    x_expr = x_fall
    w_expr = w_fall
    for seg in reversed(x_segs):
        x_expr = f"{seg}\\,{x_expr})"
    for seg in reversed(w_segs):
        w_expr = f"{seg}\\,{w_expr})"
    return x_expr, w_expr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reframe_clip(
    input_path: str,
    output_path: Path,
    content_type: str,
    regions: list[Region] | None = None,
) -> tuple[str, LayoutType]:
    """
    Reframe a clip from 16:9 to 9:16 with dynamic subject tracking.

    1. Track face positions across the clip with MediaPipe
    2. Smooth the trajectory to avoid jitter
    3. Use FFmpeg sendcmd to move the crop window dynamically

    Falls back to static crop if no faces found or MediaPipe unavailable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[REFRAMER] === REFRAMER v4 (dynamic tracking) === input={Path(input_path).name}")

    width, height, fps = _probe_video(input_path)

    # 9:16 crop dimensions from source — widened by 30% for a zoomed-out feel
    # Pure 9:16 = height * 9/16, but that's too tight. We crop wider and
    # scale to 1080x1920, which keeps more context visible around the subject.
    crop_w = int(height * 9 / 16 * 1.3)
    crop_w = min(crop_w, width)
    crop_w = crop_w - (crop_w % 2)
    # Crop height proportional to maintain 9:16 output after scale
    crop_h = int(crop_w * 16 / 9)
    crop_h = min(crop_h, height)
    crop_h = crop_h - (crop_h % 2)

    # Step 1: Track faces dynamically across the clip
    positions = _track_face_positions(input_path, width)

    if positions and len(positions) >= 3:
        # Step 2: Smooth the trajectory
        smoothed = _smooth_positions(positions)

        logger.info(
            f"[REFRAMER] Dynamic tracking v5 | {len(smoothed)} keyframes | "
            f"crop={crop_w}x{crop_h} src={width}x{height}"
        )

        # Step 3: Dynamic crop
        _run_ffmpeg_dynamic_crop(
            input_path, str(output_path), smoothed,
            crop_w, src_width=width, src_height=height, crop_h=crop_h,
        )
    else:
        # Fallback: static crop using AI region hint or center
        faces = [r for r in (regions or []) if r.label == "face"]
        if faces:
            f = faces[0]
            face_cx = int((f.x1 + f.x2) / 2 * width / 1000)
            frame_cx = width // 2
            target_cx = int(face_cx * 0.6 + frame_cx * 0.4)
            cx = max(0, min(target_cx - crop_w // 2, width - crop_w))
        else:
            cx = (width - crop_w) // 2

        cy = max(0, (height - crop_h) // 2)
        logger.info(
            f"[REFRAMER] Static fallback v5 | crop={crop_w}x{crop_h} at ({cx},{cy}) "
            f"src={width}x{height}"
        )
        vf = f"crop={crop_w}:{crop_h}:{cx}:{cy},scale={_OUT_W}:{_OUT_H}"
        _run_ffmpeg_crop(input_path, str(output_path), vf)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"[REFRAMER] Done: {output_path.name} ({size_mb:.1f} MB)")
    return str(output_path), "fill"
