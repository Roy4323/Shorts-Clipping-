from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


JobStage = Literal[
    "queued",
    "downloading",
    "transcribing",
    "scoring",
    "clipping",
    "hook_processing",
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

LayoutType = Literal[
    "fill",
    "screenshare",
    "split_two",
    "split_three",
    "gameplay",
]


class Region(BaseModel):
    label: str  # "face", "screen", "gameplay"
    x1: int     # 0-1000
    y1: int     # 0-1000
    x2: int     # 0-1000
    y2: int     # 0-1000
    score: float = 1.0


class HookCandidate(BaseModel):
    window_index: int          # 0-based index into the scored windows list
    start: float
    end: float
    signal1: float = 0.0      # semantic score (weak by definition for hook candidates)
    signal2: float = 0.0      # audio energy score
    signal3: float = 0.0      # speaker-turn density score
    weak_transcript: str = "" # spoken text from the window (to be replaced)
    engagement_type: str = "" # from Signal 1 (may be empty)


class HookClip(BaseModel):
    clip_number: int           # sequential, starts after regular short clips
    hook_text: str             # generated hook script
    hook_type: str = ""        # question | statement | fact
    start_sec: float
    end_sec: float
    duration: float
    voice: str = ""
    download_url: str


class CTAConfig(BaseModel):
    enabled: bool = False
    channel_name: str = ""
    subscriber_count: str = "0"
    logo_id: str | None = None   # UUID token → data/uploads/cta_logos/{logo_id}.*
    accent_color: str = "#CC0000"


class JobRequest(BaseModel):
    url: HttpUrl
    shorts_count: int = Field(default=3, ge=1, le=10)
    subtitle_preset: str = Field(default="default")
    process_hooks: bool = Field(default=False)
    cta_config: CTAConfig | None = None


class HookSuggestRequest(BaseModel):
    window_index: int = Field(..., ge=0)
    content_type: str = Field(default="general")


class HookGenerateRequest(BaseModel):
    window_index: int = Field(..., ge=0)
    hook_text: str = Field(..., min_length=5)
    voice: str = Field(default="en-US-GuyNeural")


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
    channel_follower_count: int | None = None   # from yt-dlp, no API key needed


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
    signal1: float = 0.0   # semantic score
    signal2: float = 0.0   # audio energy score
    signal3: float = 0.0   # speaker-turn density score
    hook: str = ""
    engagement_type: str = ""
    reason: str = ""


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
    # AI Layouts
    layout: LayoutType = Field(default="fill")
    regions: list[Region] = Field(default_factory=list)


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
    hook_candidates: list[HookCandidate] = Field(default_factory=list)
    hook_clips: list[HookClip] = Field(default_factory=list)
    error_message: str | None = None
    scorer_warning: str | None = None


class ProcessResponse(BaseModel):
    job_id: str
    stage: JobStage
    progress_pct: int


class MetadataOnlyResponse(BaseModel):
    metadata: VideoMetadata
    classification: ClassificationResult
