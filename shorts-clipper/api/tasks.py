"""Background task: full 6-stage pipeline.

Stages
------
1. Metadata + Classification               (5 %)
2. Download / Upload + Transcript          (10 → 35 %)
3. Multi-Signal Scoring (real or stub)     (35 → 45 %)
4. Clip Cutting                            (45 → 60 %)
5. Reframing 16:9 → 9:16                  (60 → 80 %)
6. Subtitle Burn                           (80 → 100 %)
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.classifier_agent import classify_content
from api.models import (
    ClassificationResult,
    ClipResult,
    TranscriptResult,
    VideoMetadata,
)
from pipeline.clipper import cut_clips
from pipeline.downloader import download_video, get_metadata
from pipeline.reframer import reframe_clip
from pipeline.scorer import score_windows
from pipeline.subtitles import burn_subtitles
from pipeline.transcript import fetch_transcript
from utils.job_store import job_store
from utils.logger import logger
from utils.storage import get_job_dir, get_job_subdir, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set(job_id: str, **kw) -> None:
    """Thin wrapper: update job store with keyword args."""
    job_store.update(job_id, **kw)


def _extract_audio(video_path: str, audio_path: str) -> None:
    """Extract MP3 audio track from video for UI playback/download."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-codec:a", "libmp3lame", "-q:a", "4",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("[PIPELINE] MP3 audio extraction failed (non-fatal).")


def _extract_wav(video_path: str, wav_path: str) -> bool:
    """Extract mono 16kHz WAV from video for the scoring engine (librosa + pyannote)."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("[PIPELINE] WAV extraction failed — scoring engine will use stub.")
        return False
    logger.info(f"[PIPELINE] WAV extracted: {Path(wav_path).name} ({Path(wav_path).stat().st_size // 1024} KB)")
    return True


def _get_video_duration(video_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("ffprobe failed — duration defaults to 0.")
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video_task(job_id: str, url: str) -> None:
    logger.info(f"🚀 Job {job_id} started.")
    try:
        _run_pipeline(job_id, url)
    except Exception as exc:
        logger.error(f"❌ Job {job_id} failed: {exc}", exc_info=True)
        _set(job_id, stage="failed", progress_pct=0, error_message=str(exc))
        raise


def _run_pipeline(job_id: str, url: str) -> None:
    is_local = url.startswith("local://")
    logger.info(f"[PIPELINE] {'='*60}")
    logger.info(f"[PIPELINE] Job {job_id} | url={url} | is_local={is_local}")

    # ------------------------------------------------------------------
    # Stage 1 — Metadata + Classification
    # ------------------------------------------------------------------
    logger.info(f"[PIPELINE] Stage 1 — Metadata + Classification")
    _set(job_id, stage="downloading", progress_pct=5)

    audio_wav_path: str | None = None

    if is_local:
        payload = job_store.get(job_id)
        metadata = VideoMetadata(**payload["metadata"])
        classification = ClassificationResult(**payload["classification"])
        video_path: str = payload["artifacts"]["video_path"]
        audio_path: str | None = payload["artifacts"].get("audio_path")
        youtube_url: str | None = payload.get("youtube_url")
        logger.info(f"[PIPELINE] Local file: video={video_path} | audio={audio_path} | youtube_url={youtube_url}")
    else:
        logger.info(f"[PIPELINE] Fetching metadata from yt-dlp...")
        metadata = get_metadata(url)
        logger.info(f"[PIPELINE] Metadata: title='{metadata.title}' | duration={metadata.duration}s | content_type TBD")
        classification = classify_content(metadata)
        logger.info(f"[PIPELINE] Classification: {classification.content_type} (source={classification.source}) — {classification.reason}")
        video_path = ""
        audio_path = None
        youtube_url = url
        _set(
            job_id,
            metadata=metadata.model_dump(),
            classification=classification.model_dump(),
        )

    job_dir = get_job_dir(job_id)
    write_json(job_dir / "metadata.json", metadata.model_dump())
    write_json(job_dir / "classification.json", classification.model_dump())
    logger.info(f"[PIPELINE] Stage 1 complete. job_dir={job_dir}")

    # ------------------------------------------------------------------
    # Stage 2 — Download + Transcript
    # ------------------------------------------------------------------
    logger.info(f"[PIPELINE] Stage 2 — Download + Transcript")
    _set(job_id, stage="transcribing", progress_pct=10)

    if is_local:
        if youtube_url:
            logger.info(f"[PIPELINE] Fetching transcript via Supadata for {youtube_url}")
            transcript_result = fetch_transcript(youtube_url)
        else:
            logger.warning("[PIPELINE] No YouTube URL provided — transcript will be empty (no subtitles).")
            transcript_result = TranscriptResult(source="mock", segments=[])
    else:
        logger.info(f"[PIPELINE] Running download + Supadata transcript in parallel...")
        output_dir = get_job_dir(job_id)
        with ThreadPoolExecutor(max_workers=2) as ex:
            dl_future = ex.submit(download_video, url, output_dir)
            tr_future = ex.submit(fetch_transcript, url)
            video_path = str(dl_future.result())
            transcript_result = tr_future.result()
        logger.info(f"[PIPELINE] Download complete: {video_path}")
        # Extract MP3 for UI playback/download
        audio_out = get_job_dir(job_id) / "audio.mp3"
        _extract_audio(video_path, str(audio_out))
        if audio_out.exists():
            audio_path = str(audio_out)
            logger.info(f"[PIPELINE] MP3 extracted: {audio_out.name} ({audio_out.stat().st_size//1024} KB)")
        # Extract WAV for the multi-signal scoring engine (librosa + pyannote)
        wav_out = get_job_dir(job_id) / "audio.wav"
        _extract_wav(video_path, str(wav_out))
        if wav_out.exists():
            audio_wav_path = str(wav_out)

    logger.info(f"[PIPELINE] Transcript: source={transcript_result.source} | segments={len(transcript_result.segments)}")

    transcript_dir = get_job_subdir(job_id, "transcripts")
    transcript_path = transcript_dir / "transcript.json"
    write_json(transcript_path, transcript_result.model_dump())
    logger.info(f"[PIPELINE] Transcript saved to {transcript_path}")

    _set(
        job_id,
        stage="scoring",
        progress_pct=35,
        transcript=transcript_result.model_dump(),
        artifacts={
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript_path": str(transcript_path),
        },
    )
    _run_clipping_pipeline(
        job_id, metadata, classification, transcript_result, video_path,
        audio_wav_path=audio_wav_path,
    )


def regenerate_clips_task(job_id: str) -> None:
    logger.info(f"🚀 Job {job_id} REGENERATION started.")
    try:
        payload = job_store.get(job_id)
        if not payload or not payload.get("transcript"):
            raise ValueError("Job not found or missing transcript. Cannot regenerate.")

        existing_clips = payload.get("clips", [])
        start_offset = len(existing_clips)

        # Reset stage to scoring
        _set(job_id, stage="scoring", progress_pct=35, error_message=None)
        
        metadata = VideoMetadata(**payload["metadata"])
        classification = ClassificationResult(**payload["classification"])
        transcript_result = TranscriptResult(**payload["transcript"])
        video_path = Path(payload["artifacts"]["video_path"])
        # Reuse existing WAV if present (lives beside the video)
        wav_candidate = Path(video_path).parent / "audio.wav"
        regen_wav = str(wav_candidate) if wav_candidate.exists() else None

        _run_clipping_pipeline(
            job_id, metadata, classification, transcript_result, video_path,
            start_offset=start_offset, existing_clips=existing_clips,
            audio_wav_path=regen_wav,
        )
    except Exception as exc:
        logger.error(f"❌ Job {job_id} regeneration failed: {exc}", exc_info=True)
        _set(job_id, stage="failed", progress_pct=0, error_message=f"Regeneration failed: {exc}")
        raise


def _run_clipping_pipeline(
    job_id: str,
    metadata: VideoMetadata,
    classification: ClassificationResult,
    transcript_result: TranscriptResult,
    video_path: Path,
    start_offset: int = 0,
    existing_clips: list[dict] | None = None,
    audio_wav_path: str | None = None,
) -> None:
    """Executes Pipeline Stages 3 through 6. Appends to existing clips if provided."""

    if existing_clips is None:
        existing_clips = []

    # ------------------------------------------------------------------
    # Stage 3 — Multi-Signal Scoring
    # ------------------------------------------------------------------
    logger.info("[PIPELINE] Stage 3 — Multi-Signal Scoring")
    _set(job_id, stage="scoring", progress_pct=40)

    video_duration = _get_video_duration(video_path)
    logger.info(f"[PIPELINE] ffprobe duration: {video_duration:.2f}s")
    if video_duration == 0 and metadata.duration:
        video_duration = float(metadata.duration)
        logger.info(f"[PIPELINE] Using metadata fallback duration: {video_duration:.2f}s")

    windows, scorer_warning = score_windows(
        transcript_result,
        video_duration,
        audio_wav_path=audio_wav_path,
        job_id=job_id,
    )
    if not windows:
        raise RuntimeError(
            f"No scorable windows produced for {video_duration:.1f}s video. "
            "Check that the video has audio/content."
        )

    window_summary = ["{:.0f}-{:.0f}s".format(w["start"], w["end"]) for w in windows]
    logger.info(f"[PIPELINE] Stage 3 complete: {len(windows)} windows {window_summary} | scorer_warning={scorer_warning}")
    _set(job_id, windows=list(windows), scorer_warning=scorer_warning)

    # ------------------------------------------------------------------
    # Stage 4 — Clip Cutting
    # ------------------------------------------------------------------
    logger.info(f"[PIPELINE] Stage 4 — Clip Cutting")
    _set(job_id, stage="clipping", progress_pct=45)

    clips_dir = get_job_subdir(job_id, "clips")
    raw_clips = cut_clips(video_path, windows, clips_dir, start_offset=start_offset)
    logger.info(f"[PIPELINE] Stage 4 complete: {len(raw_clips)} raw clips in {clips_dir}")

    # ------------------------------------------------------------------
    # Stage 5 — Reframing
    # ------------------------------------------------------------------
    logger.info(f"[PIPELINE] Stage 5 — Reframing (16:9 → 9:16)")
    _set(job_id, stage="reframing", progress_pct=60)

    reframed_dir = get_job_subdir(job_id, "reframed")
    reframed_clips: list[str] = []

    for i, clip_path in enumerate(raw_clips, 1):
        idx = start_offset + i
        logger.info(f"[PIPELINE] Reframing clip {idx}/{start_offset + len(raw_clips)}: {Path(clip_path).name}")
        out = reframed_dir / f"reframed_{idx:02d}.mp4"
        reframed_path = reframe_clip(clip_path, out, classification.content_type)
        reframed_clips.append(reframed_path)
        pct = 60 + int(i / len(raw_clips) * 20)
        _set(job_id, progress_pct=pct)

    logger.info(f"[PIPELINE] Stage 5 complete: {len(reframed_clips)} reframed clips in {reframed_dir}")

    # ------------------------------------------------------------------
    # Stage 6 — Subtitle Burn
    # ------------------------------------------------------------------
    logger.info(f"[PIPELINE] Stage 6 — Subtitle Burn")
    _set(job_id, stage="subtitles", progress_pct=80)

    final_dir = get_job_subdir(job_id, "final")
    new_clip_results: list[ClipResult] = []

    for i, (reframed_path, window) in enumerate(zip(reframed_clips, windows), 1):
        idx = start_offset + i
        logger.info(f"[PIPELINE] Burning subtitles for clip {idx}/{start_offset + len(reframed_clips)}: {window['start']:.1f}s-{window['end']:.1f}s")
        out = final_dir / f"short_{idx:02d}.mp4"
        burn_subtitles(
            reframed_path,
            transcript_result,
            window["start"],
            window["end"],
            out,
        )
        new_clip_results.append(
            ClipResult(
                clip_number=idx,
                start_sec=window["start"],
                end_sec=window["end"],
                score=window["score"],
                download_url=f"/api/clip/{job_id}/{idx}",
                hook=window.get("hook", ""),
                engagement_type=window.get("engagement_type", ""),
                reason=window.get("reason", ""),
            )
        )
        pct = 80 + int(i / len(reframed_clips) * 18)
        _set(job_id, progress_pct=pct)

    logger.info(f"[PIPELINE] Stage 6 complete: {len(new_clip_results)} final clips in {final_dir}")

    # Combine with existing
    final_clips = existing_clips + [c.model_dump() for c in new_clip_results]

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _set(
        job_id,
        stage="done",
        progress_pct=100,
        clips=final_clips,
    )
    logger.info(f"[PIPELINE] {'='*60}")
    logger.info(f"[PIPELINE] Job {job_id} COMPLETE — {len(final_clips)} total clips ready.")
