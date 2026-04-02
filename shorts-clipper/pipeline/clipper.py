"""Stage 4: Cut raw clips from the source video using FFmpeg stream copy."""

import subprocess
from pathlib import Path

from utils.logger import logger


class ClipperError(Exception):
    pass


def cut_clips(video_path: str, windows: list[dict], output_dir: Path, start_offset: int = 0) -> list[str]:
    """
    Cut the source video into individual clips using FFmpeg (re-encode).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[str] = []

    logger.info(f"[CLIPPER] Starting | video={video_path} | {len(windows)} windows | offset={start_offset}")

    for i, window in enumerate(windows, 1):
        idx = start_offset + i
        start = window["start"]
        end = window["end"]
        duration = round(end - start, 3)
        out_path = output_dir / f"clip_{idx:02d}.mp4"

        logger.info(f"[CLIPPER] Clip {idx:02d}: cutting {start:.2f}s → {end:.2f}s ({duration:.1f}s)")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
        logger.debug(f"[CLIPPER] CMD: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[CLIPPER] FFmpeg FAILED for clip {i}:\n{result.stderr[-800:]}")
            raise ClipperError(
                f"FFmpeg failed for clip {i} ({start:.1f}s-{end:.1f}s):\n"
                + result.stderr[-600:]
            )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(f"[CLIPPER] Clip {i:02d} saved: {out_path.name} ({size_mb:.1f} MB)")
        clip_paths.append(str(out_path))

    logger.info(f"[CLIPPER] Done: {len(clip_paths)} clips cut.")
    return clip_paths
