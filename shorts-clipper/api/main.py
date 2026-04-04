import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

# Ensure ffmpeg is in PATH for all subprocess calls (yt-dlp, clipper, reframer, etc.)
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from utils.logger import logger

from agents.classifier_agent import classify_content
from api.models import (
    CTAConfig,
    HealthResponse,
    HookClip,
    HookGenerateRequest,
    HookSuggestRequest,
    JobRequest,
    JobStatus,
    MetadataOnlyResponse,
    ProcessResponse,
)
from api.tasks import process_video_task
from pipeline.downloader import DownloaderError, get_metadata
from utils.job_store import job_store


UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Shorts Auto-Clipping API", version="0.1.0")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Classify only (metadata + content type, no processing)
# ---------------------------------------------------------------------------

@app.post("/api/classify", response_model=MetadataOnlyResponse)
def classify_video(request: JobRequest) -> MetadataOnlyResponse:
    try:
        metadata = get_metadata(str(request.url))
    except DownloaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    classification = classify_content(metadata)
    return MetadataOnlyResponse(metadata=metadata, classification=classification)


# ---------------------------------------------------------------------------
# Process YouTube URL
# ---------------------------------------------------------------------------

@app.post("/api/process", response_model=ProcessResponse)
def process_video(request: JobRequest, background_tasks: BackgroundTasks) -> ProcessResponse:
    job_id = str(uuid4())
    cta_dict = request.cta_config.model_dump() if request.cta_config else None
    job_store.create(
        job_id,
        {
            "url": str(request.url),
            "shorts_count": request.shorts_count,
            "subtitle_preset": request.subtitle_preset,
            "process_hooks": request.process_hooks,
            "cta_config": cta_dict,
            "stage": "queued",
            "progress_pct": 0,
            "clips": [],
            "hook_candidates": [],
            "hook_clips": [],
            "windows": [],
            "artifacts": None,
            "metadata": None,
            "classification": None,
            "transcript": None,
            "error_message": None,
            "scorer_warning": None,
        },
    )
    background_tasks.add_task(process_video_task, job_id, str(request.url))
    return ProcessResponse(job_id=job_id, stage="queued", progress_pct=0)


# ---------------------------------------------------------------------------
# Upload local video (+ optional audio + optional YouTube URL for transcript)
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=ProcessResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(None),
    youtube_url: str = Form(None),
) -> ProcessResponse:
    job_id = str(uuid4())

    video_ext = Path(video.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}_video{video_ext}"

    audio_path: Path | None = None
    if audio and audio.filename:
        audio_ext = Path(audio.filename).suffix or ".mp3"
        audio_path = UPLOAD_DIR / f"{job_id}_audio{audio_ext}"

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        if audio_path:
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save files: {exc}")

    job_store.create(
        job_id,
        {
            "url": f"local://{video_path.name}",
            "youtube_url": youtube_url or None,
            "stage": "queued",
            "progress_pct": 0,
            "clips": [],
            "windows": [],
            "artifacts": {
                "video_path": str(video_path.resolve()),
                "audio_path": str(audio_path.resolve()) if audio_path else None,
                "transcript_path": None,
            },
            "metadata": {
                "video_id": job_id,
                "title": video.filename,
                "description": "Uploaded for manual processing.",
                "duration": 0,
                "uploader": "User",
                "tags": [],
                "categories": [],
                "chapters": [],
                "thumbnail": None,
                "webpage_url": "local://file",
                "auto_caption_available": False,
            },
            "classification": {
                "content_type": "general",
                "source": "heuristic",
                "reason": "Direct user upload.",
            },
            "transcript": None,
            "error_message": None,
        },
    )

    background_tasks.add_task(process_video_task, job_id, f"local://{video_path.name}")
    return ProcessResponse(job_id=job_id, stage="queued", progress_pct=0)


@app.post("/api/generate/{job_id}/regenerate", response_model=ProcessResponse)
def regenerate_job(job_id: str, background_tasks: BackgroundTasks):
    from api.tasks import regenerate_clips_task
    from utils.job_store import job_store

    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(404, "Job not found")
    if not payload.get("transcript"):
        raise HTTPException(400, "Cannot regenerate: original transcript not found")

    background_tasks.add_task(regenerate_clips_task, job_id)
    return ProcessResponse(job_id=job_id, stage="queued", progress_pct=0)


# ---------------------------------------------------------------------------
# Status / Result polling
# ---------------------------------------------------------------------------

@app.get("/api/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str) -> JobStatus:
    payload = job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus.model_validate(payload)


@app.get("/api/result/{job_id}", response_model=JobStatus)
def get_result(job_id: str) -> JobStatus:
    payload = job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if payload["stage"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete yet.")
    return JobStatus.model_validate(payload)


# ---------------------------------------------------------------------------
# Jobs list (all jobs, newest first)
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return job_store.list_all()


# ---------------------------------------------------------------------------
# Clip download + thumbnail
# ---------------------------------------------------------------------------

@app.get("/api/clip/{job_id}/{n}")
def download_clip(job_id: str, n: int) -> FileResponse:
    payload = job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if payload["stage"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete yet.")

    clips: list[dict] = payload.get("clips", [])
    match = next((c for c in clips if c["clip_number"] == n), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Clip {n} not found.")

    # Derive file path: data/jobs/<job_id>/final/short_NN.mp4
    from utils.storage import get_job_subdir
    final_dir = get_job_subdir(job_id, "final")
    clip_file = final_dir / f"short_{n:02d}.mp4"

    if not clip_file.exists():
        raise HTTPException(status_code=404, detail="Clip file not found on disk.")

    return FileResponse(
        path=str(clip_file),
        media_type="video/mp4",
        filename=f"short_{job_id[:8]}_{n:02d}.mp4",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/api/hook/{job_id}/suggest")
def suggest_hook_script(job_id: str, request: HookSuggestRequest) -> dict:
    """
    Ask the AI to suggest a hook script for a specific window.
    Returns the suggestion immediately (synchronous, ~2s).
    """
    from pipeline.hook_generator import generate_hook
    from pipeline.hook_detector import _get_window_text
    from api.models import TranscriptResult

    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found.")

    windows = payload.get("windows", [])
    if request.window_index >= len(windows):
        raise HTTPException(status_code=400, detail=f"Window index {request.window_index} out of range.")

    window = windows[request.window_index]
    transcript_data = payload.get("transcript")
    transcript = TranscriptResult(**transcript_data) if transcript_data else TranscriptResult(source="mock", segments=[])
    spoken_text = _get_window_text(transcript, window["start"], window["end"])

    metadata = payload.get("metadata") or {}
    content_type = request.content_type or (payload.get("classification") or {}).get("content_type", "general")

    try:
        result = generate_hook(
            video_title=metadata.get("title", ""),
            video_description=metadata.get("description", ""),
            weak_transcript=spoken_text,
            signal_reason=window.get("engagement_type") or "high audio energy",
            content_type=content_type,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/hook/{job_id}/generate")
def generate_single_hook_clip(job_id: str, request: HookGenerateRequest) -> dict:
    """
    Generate a single hook clip from a user-provided (or AI-suggested) script.
    Runs TTS → audio replacement → subtitle burn (synchronous, ~10s).
    Returns the HookClip metadata.
    """
    from pipeline.tts_engine import synthesize_hook
    from pipeline.audio_replacer import replace_audio
    from pipeline.hook_subtitles import burn_hook_subtitles
    from utils.storage import get_job_subdir

    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found.")

    windows = payload.get("windows", [])
    if request.window_index >= len(windows):
        raise HTTPException(status_code=400, detail=f"Window index {request.window_index} out of range.")

    window = windows[request.window_index]

    # Raw clip lives at clips/clip_{n:02d}.mp4 (1-based, no regen offset for first pass)
    clips_dir = get_job_subdir(job_id, "clips")
    clip_n = request.window_index + 1
    raw_clip = clips_dir / f"clip_{clip_n:02d}.mp4"
    if not raw_clip.exists():
        raise HTTPException(status_code=404, detail=f"Raw clip not found: {raw_clip.name}")

    hook_dir = get_job_subdir(job_id, "hooks")
    existing_hooks = payload.get("hook_clips", [])
    hook_n = len(existing_hooks) + 1

    try:
        tts_path   = hook_dir / f"hook_audio_{hook_n:02d}.mp3"
        raw_path   = hook_dir / f"hook_raw_{hook_n:02d}.mp4"
        final_path = hook_dir / f"hook_final_{hook_n:02d}.mp4"

        word_timestamps = synthesize_hook(request.hook_text, str(tts_path), voice=request.voice)
        tts_duration = word_timestamps[-1]["end"] if word_timestamps else 12.0

        replace_audio(str(raw_clip), str(tts_path), str(raw_path))
        burn_hook_subtitles(str(raw_path), word_timestamps, str(final_path))

        hook_clip = HookClip(
            clip_number=hook_n,
            hook_text=request.hook_text,
            hook_type="custom",
            start_sec=window["start"],
            end_sec=window["end"],
            duration=round(tts_duration, 2),
            voice=request.voice,
            download_url=f"/api/hook/{job_id}/{hook_n}",
        )

        # Persist to job store so /api/hook/{job_id}/{n} can serve the file
        job_store.update(job_id, hook_clips=existing_hooks + [hook_clip.model_dump()])

        return hook_clip.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/hook/{job_id}/{n}")
def download_hook_clip(job_id: str, n: int) -> FileResponse:
    payload = job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if payload["stage"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete yet.")

    hook_clips: list[dict] = payload.get("hook_clips", [])
    match = next((c for c in hook_clips if c["clip_number"] == n), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Hook clip {n} not found.")

    from utils.storage import get_job_subdir
    hook_dir = get_job_subdir(job_id, "hooks")
    clip_file = hook_dir / f"hook_final_{n:02d}.mp4"

    if not clip_file.exists():
        raise HTTPException(status_code=404, detail="Hook clip file not found on disk.")

    return FileResponse(
        path=str(clip_file),
        media_type="video/mp4",
        filename=f"hook_{job_id[:8]}_{n:02d}.mp4",
    )


@app.get("/api/hook/{job_id}/{n}/thumb")
def hook_clip_thumbnail(job_id: str, n: int) -> FileResponse:
    from utils.storage import get_job_subdir
    hook_dir  = get_job_subdir(job_id, "hooks")
    clip_file = hook_dir / f"hook_final_{n:02d}.mp4"
    if not clip_file.exists():
        raise HTTPException(status_code=404, detail="Hook clip not found.")

    thumb_file = hook_dir / f"hook_thumb_{n:02d}.jpg"
    if not thumb_file.exists():
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", "1", "-i", str(clip_file),
             "-vframes", "1", "-q:v", "3", str(thumb_file)],
            capture_output=True,
        )
        if result.returncode != 0 or not thumb_file.exists():
            raise HTTPException(status_code=500, detail="Hook thumbnail generation failed.")

    return FileResponse(path=str(thumb_file), media_type="image/jpeg")


@app.get("/api/video/{job_id}")
@app.head("/api/video/{job_id}")
def stream_video(job_id: str) -> FileResponse:
    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found.")
    artifacts = payload.get("artifacts") or {}
    vp = artifacts.get("video_path")
    if not vp or not Path(vp).exists():
        raise HTTPException(status_code=404, detail="Video file not ready yet.")
    return FileResponse(path=vp, media_type="video/mp4", filename="original.mp4")


@app.get("/api/audio/{job_id}")
@app.head("/api/audio/{job_id}")
def download_audio(job_id: str) -> FileResponse:
    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found.")
    artifacts = payload.get("artifacts") or {}
    ap = artifacts.get("audio_path")
    # Check stored path first
    if ap and Path(ap).exists():
        return FileResponse(path=ap, media_type="audio/mpeg", filename="audio.mp3")
    # Fallback: look for audio.mp3 in the job directory
    from utils.storage import get_job_dir
    fallback = get_job_dir(job_id) / "audio.mp3"
    if fallback.exists():
        return FileResponse(path=str(fallback), media_type="audio/mpeg", filename="audio.mp3")
    raise HTTPException(status_code=404, detail="Audio not available.")


@app.get("/api/transcript/{job_id}")
def get_transcript(job_id: str) -> dict:
    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found.")
    transcript = payload.get("transcript")
    if not transcript or not transcript.get("segments"):
        raise HTTPException(status_code=404, detail="Transcript not available.")
    return transcript


@app.get("/api/clip/{job_id}/{n}/thumb")
def clip_thumbnail(job_id: str, n: int) -> FileResponse:
    from utils.storage import get_job_subdir
    final_dir = get_job_subdir(job_id, "final")
    clip_file = final_dir / f"short_{n:02d}.mp4"
    if not clip_file.exists():
        raise HTTPException(status_code=404, detail="Clip not found.")

    thumb_file = final_dir / f"thumb_{n:02d}.jpg"
    if not thumb_file.exists():
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", "2", "-i", str(clip_file),
             "-vframes", "1", "-q:v", "3", str(thumb_file)],
            capture_output=True,
        )
        if result.returncode != 0 or not thumb_file.exists():
            raise HTTPException(status_code=500, detail="Thumbnail generation failed.")

    return FileResponse(path=str(thumb_file), media_type="image/jpeg")


@app.get("/api/clip/{job_id}/{n}/thumb-cta")
def clip_thumbnail_cta(job_id: str, n: int) -> FileResponse:
    """Return the CTA thumbnail (frame extracted from near the end of the clip)."""
    from utils.storage import get_job_subdir
    final_dir = get_job_subdir(job_id, "final")
    thumb_cta_file = final_dir / f"thumb_{n:02d}_cta.jpg"
    if not thumb_cta_file.exists():
        raise HTTPException(status_code=404, detail="CTA thumbnail not generated yet.")
    return FileResponse(path=str(thumb_cta_file), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Per-clip CTA bumper — generate + append in-place
# ---------------------------------------------------------------------------

@app.post("/api/clip/{job_id}/{n}/add-cta")
def add_cta_to_clip(job_id: str, n: int, cta: CTAConfig) -> dict:
    """
    Generate a 3-second CTA bumper and append it to clip N for the given job.

    The original clip is backed up as short_NN_orig.mp4 on first call so the
    operation is idempotent — re-applying always appends to the un-modified
    original, never to an already-appended version.

    Returns {"status": "ok", "clip_number": n} on success.
    """
    from utils.storage import get_job_subdir

    payload = job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    final_dir = get_job_subdir(job_id, "final")
    clip_file = final_dir / f"short_{n:02d}.mp4"
    orig_file = final_dir / f"short_{n:02d}_orig.mp4"

    if not clip_file.exists():
        raise HTTPException(status_code=404, detail=f"Clip {n} not found on disk.")

    # Backup the original once so re-applies always start from the clean clip
    if not orig_file.exists():
        shutil.copy2(str(clip_file), str(orig_file))

    # Resolve logo path from logo_id token
    logo_path: Path | None = None
    if cta.logo_id:
        matches = list(CTA_LOGO_DIR.glob(f"{cta.logo_id}.*"))
        if matches:
            logo_path = matches[0]

    tmp_out = final_dir / f"short_{n:02d}_cta_tmp.mp4"
    try:
        # Import is inside try so ImportError (e.g. Pillow missing) gives a clear message
        from pipeline.cta_bumper import generate_cta_bumper, append_cta_to_clip as _concat

        size_before = clip_file.stat().st_size
        logger.info(f"[CTA] clip {n} size BEFORE: {size_before} bytes — {clip_file}")

        bumper_path = generate_cta_bumper(
            channel_name=cta.channel_name,
            subscriber_count=cta.subscriber_count,
            logo_path=str(logo_path) if logo_path else None,
            accent_color=cta.accent_color,
        )
        logger.info(f"[CTA] bumper generated: {bumper_path}")
        _concat(str(orig_file), bumper_path, str(tmp_out))
        logger.info(f"[CTA] concat done — tmp_out size: {tmp_out.stat().st_size} bytes")

        # Replace clip file atomically (os.replace is safe on Windows —
        # no "delete pending" race condition unlike unlink + rename).
        import os as _os
        _os.replace(str(tmp_out), str(clip_file))

        size_after = clip_file.stat().st_size
        logger.info(f"[CTA] clip {n} size AFTER replace: {size_after} bytes — delta: {size_after - size_before}")

        # Invalidate the normal thumbnail and generate a new one from
        # 1 second before the end so it shows the CTA frame.
        thumb_file     = final_dir / f"thumb_{n:02d}.jpg"
        thumb_cta_file = final_dir / f"thumb_{n:02d}_cta.jpg"
        thumb_file.unlink(missing_ok=True)
        thumb_cta_file.unlink(missing_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1.5", "-i", str(clip_file),
             "-vframes", "1", "-q:v", "3", str(thumb_cta_file)],
            capture_output=True,
        )

        # Get the new duration via ffprobe so the frontend can seek to the CTA
        dur_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(clip_file)],
            capture_output=True, text=True,
        )
        import json as _json
        new_duration = 0.0
        try:
            new_duration = float(_json.loads(dur_result.stdout).get("format", {}).get("duration", 0))
        except Exception:
            pass

        return {
            "status": "ok",
            "clip_number": n,
            "size_before": size_before,
            "size_after": size_after,
            "new_duration": round(new_duration, 2),
            "cta_thumb_url": f"/api/clip/{job_id}/{n}/thumb-cta",
        }

    except Exception as exc:
        logger.error(f"[CTA] clip {n} failed: {exc}", exc_info=True)
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Per-hook-clip CTA bumper — generate + append in-place
# ---------------------------------------------------------------------------

@app.post("/api/hook/{job_id}/{n}/add-cta")
def add_cta_to_hook_clip(job_id: str, n: int, cta: CTAConfig) -> dict:
    """
    Same as /api/clip/{job_id}/{n}/add-cta but for Hook Studio clips.
    Appends a CTA bumper to hook_final_NN.mp4 in the hooks/ subdirectory.
    """
    from utils.storage import get_job_subdir

    payload = job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    hooks_dir = get_job_subdir(job_id, "hooks")
    clip_file = hooks_dir / f"hook_final_{n:02d}.mp4"
    orig_file = hooks_dir / f"hook_final_{n:02d}_orig.mp4"

    if not clip_file.exists():
        raise HTTPException(status_code=404, detail=f"Hook clip {n} not found on disk.")

    if not orig_file.exists():
        shutil.copy2(str(clip_file), str(orig_file))

    logo_path: Path | None = None
    if cta.logo_id:
        matches = list(CTA_LOGO_DIR.glob(f"{cta.logo_id}.*"))
        if matches:
            logo_path = matches[0]

    tmp_out = hooks_dir / f"hook_final_{n:02d}_cta_tmp.mp4"
    try:
        from pipeline.cta_bumper import generate_cta_bumper, append_cta_to_clip as _concat

        size_before = clip_file.stat().st_size
        logger.info(f"[CTA] hook clip {n} size BEFORE: {size_before} bytes — {clip_file}")

        bumper_path = generate_cta_bumper(
            channel_name=cta.channel_name,
            subscriber_count=cta.subscriber_count,
            logo_path=str(logo_path) if logo_path else None,
            accent_color=cta.accent_color,
        )
        logger.info(f"[CTA] hook bumper generated: {bumper_path}")
        _concat(str(orig_file), bumper_path, str(tmp_out))
        logger.info(f"[CTA] hook concat done — tmp_out size: {tmp_out.stat().st_size} bytes")

        import os as _os
        _os.replace(str(tmp_out), str(clip_file))

        size_after = clip_file.stat().st_size
        logger.info(f"[CTA] hook clip {n} size AFTER replace: {size_after} bytes — delta: {size_after - size_before}")

        return {"status": "ok", "clip_number": n}

    except Exception as exc:
        logger.error(f"[CTA] hook clip {n} failed: {exc}", exc_info=True)
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# YouTube channel info (for CTA bumper auto-fill)
# ---------------------------------------------------------------------------

@app.get("/api/youtube/channel-info")
def youtube_channel_info(url: str) -> dict:
    """
    Fetch channel name, subscriber count and logo URL from the YouTube Data API.
    Requires YOUTUBE_API_KEY in the environment.
    """
    from pipeline.cta_bumper import fetch_channel_info
    try:
        return fetch_channel_info(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"YouTube API error: {exc}")


# ---------------------------------------------------------------------------
# Logo upload (for CTA bumper)
# ---------------------------------------------------------------------------

CTA_LOGO_DIR = Path("./data/uploads/cta_logos")
CTA_LOGO_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/upload-logo")
async def upload_logo(logo: UploadFile = File(...)) -> dict:
    """
    Save a logo image for use in the CTA bumper.
    Returns a logo_id token that can be passed as CTAConfig.logo_id.
    """
    logo_id   = str(uuid4())
    ext       = Path(logo.filename or "logo.png").suffix or ".png"
    save_path = CTA_LOGO_DIR / f"{logo_id}{ext}"
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(logo.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save logo: {exc}")
    return {"logo_id": logo_id, "ext": ext}


@app.post("/api/fetch-logo-from-url")
async def fetch_logo_from_url(payload: dict) -> dict:
    """
    Download a logo image from an external URL (e.g. YouTube channel thumbnail)
    and save it to the CTA logos directory.  Returns a logo_id token.
    """
    import httpx
    url = (payload or {}).get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            r = await client.get(url)
        r.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download logo: {exc}")

    content_type = r.headers.get("content-type", "image/jpeg")
    ext = ".jpg" if "jpeg" in content_type else ".png" if "png" in content_type else ".jpg"
    logo_id   = str(uuid4())
    save_path = CTA_LOGO_DIR / f"{logo_id}{ext}"
    save_path.write_bytes(r.content)
    return {"logo_id": logo_id, "ext": ext}


# ---------------------------------------------------------------------------
# Favicon — return 204 so browsers don't spam 404s in the log
# ---------------------------------------------------------------------------

@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Static UI (must be last — catches everything else)
# ---------------------------------------------------------------------------

static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
