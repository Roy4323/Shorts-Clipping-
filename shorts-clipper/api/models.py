from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


JobStage = Literal[
    "queued",
    "downloading",
    "transcribing",
    "scoring",
    "clipping",
    "reframing",
    "subtitles",
    "done",
    "failed",
]

ContentType = Literal[
    "podcast_interview",
    "music_lyrics",
    "music_instrumental",
    "sports_action",
    "mute_broll",
    "general",
]


class JobRequest(BaseModel):
    url: HttpUrl


class HealthResponse(BaseModel):
    status: str


class VideoMetadata(BaseModel):
    video_id: str | None = None
    title: str
    description: str = ""
    duration: int | None = None
    uploader: str | None = None
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    chapters: list[dict] = Field(default_factory=list)
    thumbnail: str | None = None
    webpage_url: str
    auto_caption_available: bool = False


class ClassificationResult(BaseModel):
    content_type: ContentType
    source: Literal["openai", "gemini", "heuristic"]
    reason: str


class TranscriptSegment(BaseModel):
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., ge=0)
    text: str = Field(..., min_length=1)


class TranscriptResult(BaseModel):
    source: Literal["supadata", "cache", "mock"]
    segments: list[TranscriptSegment] = Field(default_factory=list)


class StageTwoArtifacts(BaseModel):
    video_path: str | None = None
    audio_path: str | None = None
    transcript_path: str | None = None


class ScoredWindow(BaseModel):
    start: float
    end: float
    score: float


class ClipResult(BaseModel):
    clip_number: int = Field(..., ge=1)
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., gt=0)
    score: float = Field(default=0.0)
    download_url: str
    # Multi-signal scorer enrichment (empty string when stub scoring was used)
    hook: str = Field(default="")
    engagement_type: str = Field(default="")
    reason: str = Field(default="")


class JobStatus(BaseModel):
    job_id: str
    url: str
    stage: JobStage
    progress_pct: int = Field(..., ge=0, le=100)
    created_at: str
    updated_at: str
    metadata: VideoMetadata | None = None
    classification: ClassificationResult | None = None
    transcript: TranscriptResult | None = None
    artifacts: StageTwoArtifacts | None = None
    windows: list[ScoredWindow] = Field(default_factory=list)
    clips: list[ClipResult] = Field(default_factory=list)
    error_message: str | None = None
    scorer_warning: str | None = None


class ProcessResponse(BaseModel):
    job_id: str
    stage: JobStage
    progress_pct: int


class MetadataOnlyResponse(BaseModel):
    metadata: VideoMetadata
    classification: ClassificationResult
