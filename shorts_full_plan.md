# Shorts Auto-Clipping System
### Complete Technical Architecture
*Hackathon Edition — 2-Member Team*

| ~$0.005 cost/video | ~3–5 min end-to-end (30min video) | 5 signals multi-modal detection |
|---|---|---|

---

## 1. System Overview

This system ingests a YouTube URL, downloads the video intelligently, extracts a transcript, uses a multi-signal AI agent to detect the best highlight windows, reframes the video from 16:9 to 9:16 with subject-aware cropping, and burns subtitles — outputting ready-to-post short-form clips.

> **Core Design Philosophy:** Never use vision models for highlight detection. Treat video as audio + text. Vision models are only invoked for reframing subject detection — and even then, we use free local models (MediaPipe, OpenCV). Total API cost stays under $0.01 per video.

### 1.1 High-Level Flow

The pipeline has 6 stages, two of which run in parallel to minimize user wait time:

- **Stage 1:** URL intake + metadata fetch (no download needed yet)
- **Stage 2:** Parallel — video download + transcript fetch
- **Stage 3:** Multi-signal highlight scoring
- **Stage 4:** Clip selection + windowing
- **Stage 5:** Reframing (16:9 → 9:16) with subject tracking
- **Stage 6:** Subtitle burn + output packaging

### 1.2 Content Classification

Before any processing, the system classifies the video type using yt-dlp metadata (title, description, tags, category). This single Claude Haiku call routes the entire downstream pipeline:

| Content Type | Detection Method | Highlight Signal | Reframe Strategy |
|---|---|---|---|
| Podcast / Interview | Metadata + speaker count | Semantic + speaker turns + energy | MediaPipe face tracking |
| Music with Lyrics | Metadata + Whisper confidence | Chorus detection via librosa | Center crop + beat-sync cuts |
| Music Instrumental | Low transcript confidence | Energy drops + onset density | Motion center crop |
| Sports / Action | Metadata category | Optical flow motion scoring | Motion mass tracking |
| Mute / B-Roll | RMS energy < threshold | PySceneDetect scene scoring | PySceneDetect + warn user |

---

## 2. Stage-by-Stage Technical Spec

### Stage 1 — URL Intake & Metadata

Endpoint: `POST /api/process` — accepts YouTube URL only (no file upload in v1).

Immediately call yt-dlp in metadata-only mode (no download). Extract:
- title, description, tags, categories
- duration (for time estimates + UX progress bar)
- chapters (if available — free highlight hints)
- auto-caption availability flag (for Supadata routing decision)

Send metadata to Claude Haiku for classification. Response is one of 5 content types. This call costs ~$0.0002 and takes under 2 seconds.

```python
ydl_opts = {'skip_download': True, 'quiet': True}
# Returns full info dict in <1 second
```

---

### Stage 2 — Parallel Download + Transcript

These two tasks run concurrently using `asyncio` or `ThreadPoolExecutor`. The transcript call returns in seconds; download is the long pole.

#### 2A — Video Download via yt-dlp

Always download 720p max. 1080p/4K adds download time with zero benefit for short-form output.

```python
'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]'
```

| Video Length | 720p File Size | Download Time* | User UX |
|---|---|---|---|
| 5 minutes | ~150 MB | 12–25 sec | Show spinner |
| 15 minutes | ~450 MB | 35–75 sec | Show progress % |
| 30 minutes | ~900 MB | 70–150 sec | Show stage breakdown |
| 60 minutes | ~1.8 GB | 2.5–5 min | Email/notify on complete |

*On 50–100 Mbps server bandwidth. Cloud instances (AWS/GCP) with fast egress will be faster.*

#### 2B — Transcript via Supadata

For YouTube URLs: Supadata pulls existing captions/auto-captions. Returns timestamped segments in seconds regardless of video length. No transcription compute cost.

For uploaded video (v2 scope): Extract audio with ffmpeg first, then run faster-whisper locally:

```python
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav

model = WhisperModel('small', device='cpu', compute_type='int8')
segments, _ = model.transcribe('audio.wav', word_timestamps=True)
```

faster-whisper `small` model transcription speed: ~0.3x real-time on CPU (30min video → ~9 min). Acceptable for v2.

---

### Stage 3 — Multi-Signal Highlight Scoring

Three signals are computed independently and combined into a final score per 30-second window.

#### Signal 1: Semantic Scoring via Claude Haiku

Send full transcript text to Claude Haiku with timestamps. Prompt asks for the top 5 most engaging 30–60 second windows with reasoning. Returns JSON with `start_time`, `end_time`, `score`, `reason`.

Cost: ~$0.003 for a 1-hour video transcript (~8k tokens). Practically free.

#### Signal 2: Audio Energy via librosa

Detect energy spikes — laughter, applause, raised voice, emphasis. Runs locally, zero cost.

```python
y, sr = librosa.load('audio.mp3')
rms = librosa.feature.rms(y=y, frame_length=sr//2)  # 0.5sec windows
# Normalize 0-1. Windows above 75th percentile = high energy
```

#### Signal 3: Speaker Turn Density via pyannote.audio

Back-and-forth dialogue is inherently more engaging than monologue. Count speaker switches per 30-second window. High switch density = conversational energy = strong clip candidate.

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')
diarization = pipeline('audio.mp3')
```

#### Score Combination

```python
score = (0.40 × semantic) + (0.35 × energy) + (0.25 × dialogue_density)
```

Select top 3–5 windows. Enforce minimum 10-second gap between selected clips to avoid overlapping content.

> **Bonus Signal — Sentence Embeddings (if time allows):** Use `sentence-transformers` to embed all transcript segments. Find segments semantically closest to the video's core topic (extracted once by Claude). Most on-topic + high energy = best clip. Prevents off-topic tangents from being selected.

---

### Stage 4 — Clip Windowing

For each selected highlight window:
- Extend window ±5 seconds for natural cut points
- Snap start/end to sentence boundaries using transcript timestamps
- Enforce min 20 sec, max 90 sec clip duration
- Score and rank final clip list

Cut clips with ffmpeg (stream copy where possible — no re-encode, near instant):

```bash
ffmpeg -ss {start} -i video.mp4 -t {duration} -c copy clip_{n}.mp4
```

---

### Stage 5 — Reframing 16:9 → 9:16

#### Podcast / Interview — MediaPipe Face Tracking

Google's MediaPipe runs locally, free, and detects faces + body pose in real time. For each frame, get face bounding box center. Apply a rolling average over 30 frames (~1 second) to smooth the crop window and prevent jitter.

```python
import mediapipe as mp
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
# Per frame: get bbox → compute crop center → smooth with deque(maxlen=30)
# Output crop: width = original_height * (9/16), centered on smoothed face X
```

#### Sports / Action — Optical Flow Motion Center

```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# Weighted center of high-magnitude regions = crop target
```

#### General Fallback — Smart Center Crop

When no subject is detected, apply a weighted center crop biased slightly upward (subjects are rarely at the very bottom of frame). Safe for most content.

> **Key Smoothing Rule — CRITICAL:** Raw per-frame crop coordinates will produce unwatchable jitter. Always smooth with a rolling average of 24–30 frames before applying the crop. This single step is the difference between professional and amateur output.

---

### Stage 6 — Subtitle Burn

Convert Supadata/Whisper transcript to SRT format. Burn with ffmpeg:

```bash
ffmpeg -i clip.mp4 \
  -vf "subtitles=clip.srt:force_style='FontSize=22,Bold=1,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Alignment=2'" \
  -c:a copy output_final.mp4
```

Style guide for shorts: large font (22–26pt), bold, white with black outline, center-bottom. No karaoke highlighting in v1 — that requires ASS format and adds complexity.

---

## 3. Backend Architecture

### 3.1 Folder Structure

```
shorts-clipper/
├── api/
│   ├── main.py              # FastAPI app, routes
│   ├── models.py            # Pydantic request/response models
│   └── tasks.py             # Celery task definitions
├── pipeline/
│   ├── classifier.py        # Content type classification
│   ├── downloader.py        # yt-dlp wrapper
│   ├── transcript.py        # Supadata + Whisper fallback
│   ├── scorer.py            # Multi-signal highlight scoring
│   ├── clipper.py           # FFmpeg clip cutting
│   ├── reframer.py          # 16:9 → 9:16 with subject tracking
│   └── subtitles.py         # SRT generation + ffmpeg burn
├── agents/
│   ├── highlight_agent.py   # Claude Haiku semantic scoring
│   └── classifier_agent.py  # Claude Haiku content classification
├── utils/
│   ├── audio.py             # librosa helpers
│   ├── ffmpeg_utils.py      # ffmpeg command builders
│   └── storage.py           # output file management
├── config.py                # API keys, constants, model settings
└── requirements.txt
```

### 3.2 API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/process` | POST | API Key | Submit YouTube URL, returns job_id |
| `/api/status/{job_id}` | GET | API Key | Poll job status + progress stage |
| `/api/result/{job_id}` | GET | API Key | Fetch completed clips list + download URLs |
| `/api/clip/{job_id}/{clip_n}` | GET | API Key | Stream individual clip file |
| `/api/classify` | POST | API Key | Classify content type only (preview) |

### 3.3 Job Status Flow

Jobs run async via Celery + Redis. Status stages for frontend progress bar:

```
queued → downloading → transcribing → scoring → clipping → reframing → subtitles → done
```

Each stage update is written to Redis with timestamp. Frontend polls `/api/status` every 3 seconds.

### 3.4 Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| API Framework | FastAPI | Async, fast, auto-docs |
| Task Queue | Celery + Redis | Async video processing jobs |
| Video Download | yt-dlp | Best YouTube support, no alternative |
| Transcript | Supadata API | Instant for YouTube, no compute |
| Transcription (v2) | faster-whisper | 4x faster than openai-whisper, free |
| Audio Analysis | librosa | Energy, tempo, onset detection |
| Speaker Detection | pyannote.audio | Free, local, accurate diarization |
| AI Scoring | Claude Haiku API | Cheapest capable model, <$0.003/video |
| Face Tracking | MediaPipe | Google, free, real-time, local |
| Motion Tracking | OpenCV optical flow | Free, fast, no model needed |
| Video Processing | ffmpeg-python | Industry standard, free |
| Scene Detection | PySceneDetect | Fast local scene boundary detection |

---

## 4. Cost & Performance Targets

| Video Length | Download | Scoring | Reframe/Render | Total Time |
|---|---|---|---|---|
| 5 minutes | ~20 sec | ~15 sec | ~30 sec | ~65 sec |
| 15 minutes | ~55 sec | ~20 sec | ~60 sec | ~2.5 min |
| 30 minutes | ~110 sec | ~30 sec | ~90 sec | ~4 min |
| 60 minutes | ~4 min | ~45 sec | ~3 min | ~8 min |

**API cost per video:** Claude Haiku classify ~$0.0002 + Claude Haiku scoring ~$0.003 = **~$0.003 total.** All other processing is local compute, zero cost. Confirm Supadata free tier limits before hackathon.

---

## 5. requirements.txt

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
celery[redis]>=5.3.0
yt-dlp>=2024.3.10
faster-whisper>=0.10.0
librosa>=0.10.0
pyannote.audio>=3.1.0
mediapipe>=0.10.0
opencv-python>=4.9.0
ffmpeg-python>=0.2.0
scenedetect>=0.6.3
sentence-transformers>=2.6.0
anthropic>=0.20.0
httpx>=0.26.0
pydantic>=2.6.0
redis>=5.0.0
python-multipart>=0.0.9
```

---

## 6. Edge Cases & Graceful Handling

| Scenario | Detection | Handling |
|---|---|---|
| No transcript available | Supadata returns empty | Fall back to faster-whisper on audio |
| Silent/mute video | RMS energy < 0.01 threshold | Visual-only mode, warn user |
| Music only (no lyrics) | Low Whisper word confidence | librosa-only scoring, no semantic |
| No face detected | MediaPipe returns no bbox | Fall back to smart center crop |
| Video too short (<2 min) | Duration check at intake | Return error: min 2 min required |
| Private/age-gated video | yt-dlp error code | Return user-friendly error message |
| Highlight windows overlap | Clip selection logic | Enforce 10-sec minimum gap |

---

*— End of Document —*
