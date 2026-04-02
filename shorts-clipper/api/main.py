from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException

from agents.classifier_agent import classify_content
from api.models import HealthResponse, JobRequest, JobStatus, MetadataOnlyResponse, ProcessResponse
from api.tasks import process_video_task
from pipeline.downloader import DownloaderError, get_metadata
from utils.job_store import job_store


app = FastAPI(
    title="Shorts Auto-Clipping API",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/classify", response_model=MetadataOnlyResponse)
def classify_video(request: JobRequest) -> MetadataOnlyResponse:
    try:
        metadata = get_metadata(str(request.url))
    except DownloaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    classification = classify_content(metadata)
    return MetadataOnlyResponse(
        metadata=metadata,
        classification=classification,
    )


@app.post("/api/process", response_model=ProcessResponse)
def process_video(request: JobRequest, background_tasks: BackgroundTasks) -> ProcessResponse:
    job_id = str(uuid4())
    job_store.create(
        job_id,
        {
            "url": str(request.url),
            "stage": "queued",
            "progress_pct": 0,
            "clips": [],
            "artifacts": None,
            "metadata": None,
            "classification": None,
            "transcript": None,
            "error_message": None,
        },
    )
    background_tasks.add_task(process_video_task, job_id, str(request.url))
    return ProcessResponse(job_id=job_id, stage="queued", progress_pct=0)


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
        raise HTTPException(status_code=409, detail="Job is not complete yet.")
    return JobStatus.model_validate(payload)
