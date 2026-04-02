import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.classifier_agent import classify_content
from api.models import (
    HealthResponse,
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
    job_store.create(
        job_id,
        {
            "url": str(request.url),
            "shorts_count": request.shorts_count,
            "stage": "queued",
            "progress_pct": 0,
            "clips": [],
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


@app.post("/api/generate/{job_id}/rereframe", response_model=ProcessResponse)
def rereframe_job(job_id: str, background_tasks: BackgroundTasks):
    """Re-run only reframing + subtitles on existing raw clips (Stage 5+6)."""
    from api.tasks import rereframe_clips_task

    payload = job_store.get(job_id)
    if not payload:
        raise HTTPException(404, "Job not found")
    if not payload.get("windows"):
        raise HTTPException(400, "Cannot re-reframe: no clips/windows found")

    background_tasks.add_task(rereframe_clips_task, job_id)
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
    )


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


# ---------------------------------------------------------------------------
# Static UI (must be last — catches everything else)
# ---------------------------------------------------------------------------

static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
