from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.classifier_agent import classify_content
from api.models import JobStatus, TranscriptResult
from pipeline.downloader import download_video, get_metadata
from pipeline.transcript import fetch_transcript
from utils.job_store import job_store
from utils.storage import get_job_subdir, write_json


def _set_status(job_id: str, **updates: object) -> JobStatus:
    payload = job_store.update(job_id, **updates)
    return JobStatus.model_validate(payload)


def process_video_task(job_id: str, url: str) -> None:
    try:
        metadata = get_metadata(url)
        classification = classify_content(metadata).model_dump()

        _set_status(
            job_id,
            stage="queued",
            progress_pct=10,
            metadata=metadata,
            classification=classification,
        )

        job_dir = get_job_subdir(job_id, "")
        raw_dir = get_job_subdir(job_id, "raw")
        transcript_dir = get_job_subdir(job_id, "transcripts")

        write_json(job_dir / "metadata.json", metadata)
        write_json(job_dir / "classification.json", classification)

        _set_status(job_id, stage="downloading", progress_pct=25)

        with ThreadPoolExecutor(max_workers=2) as executor:
            download_future = executor.submit(download_video, url, raw_dir)
            transcript_future = executor.submit(fetch_transcript, url)

            transcript_result: TranscriptResult = transcript_future.result()
            _set_status(
                job_id,
                stage="transcribing",
                progress_pct=65,
                transcript=transcript_result.model_dump(),
            )

            video_path = download_future.result()

        transcript_path = transcript_dir / "transcript.json"
        write_json(transcript_path, {"segments": [s.model_dump() for s in transcript_result.segments]})

        _set_status(
            job_id,
            stage="done",
            progress_pct=100,
            artifacts={
                "video_path": str(Path(video_path).resolve()),
                "transcript_path": str(transcript_path.resolve()),
            },
        )
    except Exception as exc:
        _set_status(job_id, stage="failed", progress_pct=100, error_message=str(exc))
