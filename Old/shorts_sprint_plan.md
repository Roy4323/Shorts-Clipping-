# Team Sprint Plan
### Shorts Auto-Clipping System
*24-Hour Hackathon · 2 Members · Build → Review → Ship*

| ⚙ Person 1 — Backend Engineer | 🧠 Person 2 — AI / Pipeline Engineer |
|---|---|
| FastAPI · Celery · Redis · yt-dlp · Storage · API Layer | AI Agent · librosa · MediaPipe · ffmpeg · Scoring |

> **Priority codes:** P0 = must ship · P1 = should ship · P2 = nice to have if time allows

---

## Master Timeline — 24 Hours

| Phase | Time | ⚙ Person 1 (Backend) | 🧠 Person 2 (AI/Pipeline) |
|---|---|---|---|
| Phase 0 | Hr 0–1 | Project setup, env, Redis, FastAPI skeleton | Install libs, test Supadata + Haiku API keys |
| Phase 1 | Hr 1–4 | yt-dlp downloader + metadata fetch + job queue | Content classifier agent + transcript fetcher |
| Phase 2 | Hr 4–8 | Status API + storage layer + Celery task wiring | Multi-signal scorer (Haiku + librosa + pyannote) |
| Phase 3 | Hr 8–12 | API integration + end-to-end smoke test | FFmpeg clipper + reframer (MediaPipe/optical flow) |
| **Sync #1** | **Hr 12** | **Integration checkpoint — wire P1 + P2 together** | **Test full pipeline on 1 real YouTube URL** |
| Phase 4 | Hr 12–16 | Error handling, edge cases, logging | Subtitle burn + output packaging |
| Phase 5 | Hr 16–20 | Performance tuning + parallel download/transcript | Clip quality review + scoring weight tuning |
| **Sync #2** | **Hr 20** | **Final integration + demo video test** | **Polish + edge case fixes** |
| Phase 6 | Hr 20–23 | Deploy + staging test + load check | Demo prep + backup test videos ready |
| Buffer | Hr 23–24 | Bug fixes only | Bug fixes only |

---

## ⚙ Person 1 — Backend Engineer

*FastAPI · Celery · Redis · yt-dlp · Storage · API Layer*

You own everything from the HTTP boundary inward — request handling, job queuing, storage, and the glue that connects the AI pipeline to the outside world. Person 2 hands you functions; you wire them into the job system.

---

### Phase 0 — Setup (Hr 0–1)

#### `[P0]` Project Bootstrap · 45 min
- Create repo, agree on branch naming (main, dev, feat/xxx)
- Set up virtualenv, install FastAPI + uvicorn + celery + redis
- Create `config.py` with all API key slots (Anthropic, Supadata)
- Verify Redis running locally or on cloud
- Create shared `models.py` — JobRequest, JobStatus, ClipResult Pydantic models

---

### Phase 1 — Core Backend (Hr 1–4)

#### `[P0]` yt-dlp Metadata Fetcher · 45 min
- `downloader.py`: `get_metadata(url)` → returns title, duration, chapters, auto_caption_available
- No download yet — `skip_download=True` mode only
- Return clean dict, handle private/unavailable video errors gracefully
- Test with 3 URLs: podcast, music, sports

#### `[P0]` yt-dlp Video Downloader · 60 min
- `download_video(url, output_path, quality='720')` → returns local file path
- Format: `bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]`
- Progress hook → write percentage to Redis key for status polling
- Handle rate limits + retry logic (max 3 attempts)
- Test download times for 5min, 15min, 30min videos — log actual numbers

#### `[P0]` Job Queue (Celery + Redis) · 75 min
- `tasks.py`: `process_video_task(job_id, url)` — main Celery task
- Job stages enum: `QUEUED → DOWNLOADING → TRANSCRIBING → SCORING → CLIPPING → REFRAMING → SUBTITLES → DONE → FAILED`
- `update_job_stage(job_id, stage, progress_pct)` → writes to Redis
- Store job metadata in Redis as JSON (created_at, url, stage, clips list)
- Test: submit job, verify stage transitions in Redis

---

### Phase 2 — API Layer (Hr 4–8)

#### `[P0]` FastAPI Endpoints · 90 min
- `POST /api/process` — validate URL, create job_id (uuid4), enqueue Celery task, return job_id
- `GET /api/status/{job_id}` — return stage + progress_pct + estimated_time_remaining
- `GET /api/result/{job_id}` — return clip list with download URLs (only if DONE)
- `GET /api/clip/{job_id}/{clip_n}` — stream MP4 file with range request support
- `POST /api/classify` — metadata-only classification (no download), returns content_type
- Add CORS middleware for frontend

#### `[P0]` Storage Layer · 45 min
- `storage.py`: `get_job_dir(job_id)` → `/tmp/jobs/{job_id}/`
- Subdirs: `/raw` (downloaded video), `/clips` (cut clips), `/output` (final subtitled clips)
- `cleanup_job(job_id)` → delete after 2 hours (TTL via Celery beat or Redis TTL)
- List job clips → returns `{clip_n, duration, file_size, download_url}`

#### `[P0]` Parallel Execution in Task · 30 min
- In `process_video_task`: run download + transcript fetch concurrently
- Use `asyncio.gather` or `ThreadPoolExecutor` with 2 workers
- Transcript call returns fast — don't block download on it
- Once both done, proceed to scoring stage

---

### Phase 3 — Integration (Hr 8–12)

#### `[P0]` Wire Pipeline Stages · 90 min
Connect Person 2's modules into Celery task in correct order:
```python
classifier.classify(metadata)           → content_type
transcript.fetch(url, content_type)     → timestamped segments
scorer.score(transcript, audio_path, content_type) → top windows
clipper.cut(video_path, windows)        → clip file paths
reframer.reframe(clip_paths, content_type) → reframed clip paths
subtitles.burn(clip_paths, transcript)  → final output paths
```
Update job stage at each step.

#### `[P0]` Smoke Test — End to End · 45 min
- Submit 15-min podcast YouTube URL via `POST /api/process`
- Poll status every 3 sec, verify stage transitions
- Download final clips, verify 9:16 ratio, subtitles visible, audio intact
- Log total wall-clock time — target under 5 min

---

### Phase 4 — Error Handling (Hr 12–16)

#### `[P1]` Edge Case Handling · 60 min
- Private/unavailable video → 400 error with clear message
- Video < 2 min → reject at intake with message
- Supadata returns empty → trigger fallback flag for Person 2's whisper path
- Any pipeline stage fails → mark job FAILED, store error message in Redis
- Duplicate URL submitted → return existing job_id if still processing

---

### Phase 5 — Performance (Hr 16–20)

#### `[P1]` Parallel + Optimization · 60 min
- Profile actual stage times on 30-min video, identify bottleneck
- Confirm download + transcript are truly parallel (check logs)
- If reframing is slow: process clips in parallel (ThreadPoolExecutor)
- Add `estimated_time_remaining` to status response based on video duration

---

## 🧠 Person 2 — AI / Pipeline Engineer

*Claude Haiku Agent · librosa · pyannote · MediaPipe · FFmpeg · Subtitles*

You own the intelligence and media processing — classification, transcript, all scoring signals, video reframing, and subtitle output. You hand Person 1 clean Python functions with clear inputs and outputs.

---

### Phase 0 — Setup (Hr 0–1)

#### `[P0]` Environment & API Validation · 45 min
```bash
pip install anthropic supadata librosa pyannote.audio mediapipe \
  opencv-python ffmpeg-python scenedetect faster-whisper
```
- Test Supadata API: fetch transcript for a known YouTube URL, verify timestamp format
- Test Claude Haiku: send test prompt, verify JSON response parsing works
- Test faster-whisper: transcribe 30-sec audio clip locally
- Verify ffmpeg installed and accessible in PATH

---

### Phase 1 — Classifier + Transcript (Hr 1–4)

#### `[P0]` Content Classifier Agent · 60 min
- `classifier.py`: `classify(metadata_dict)` → content_type string
- Send title + description + tags to Claude Haiku
- Prompt: *return ONLY one of: podcast_interview | music_lyrics | music_instrumental | sports_action | mute_broll*
- Parse response, validate against enum, default to `podcast_interview` on parse fail
- Add fallback: if duration < 120 sec → return `'too_short'` error

#### `[P0]` Transcript Fetcher · 60 min
- `transcript.py`: `fetch(url, content_type, audio_path=None)` → list of `{start, end, text}` dicts
- Primary: Supadata API call with YouTube URL → parse response to standard format
- Fallback: if Supadata empty OR content_type is `music_lyrics` → run faster-whisper
- Audio extraction first: `ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav`
- `WhisperModel('small', device='cpu', compute_type='int8')` with `word_timestamps=True`
- Normalize both outputs to same `{start, end, text}` schema

---

### Phase 2 — Scoring Signals (Hr 4–8)

#### `[P0]` Signal 1: Semantic Scoring via Claude Haiku · 75 min
- `agents/highlight_agent.py`: `score_transcript(segments, content_type)` → list of `{start, end, score, reason}`
- Build windowed transcript: group segments into 30-sec windows with timestamps
- Prompt Haiku: *"Given this timestamped transcript, identify the top 5 most engaging 30–60 second windows. Return JSON array: [{start_sec, end_sec, score (0-1), reason}]"*
- Parse JSON response — strip markdown fences if present before `json.loads()`
- Validate all fields present, scores in 0–1 range

#### `[P0]` Signal 2: Audio Energy via librosa · 60 min
- `utils/audio.py`: `get_energy_scores(audio_path)` → dict of `{second: energy_score}`
```python
y, sr = librosa.load(audio_path, sr=22050)
rms = librosa.feature.rms(y=y, frame_length=sr//2, hop_length=sr//2)
# Normalize to 0-1 range using min-max
# Aggregate per 30-sec window: mean(energy_scores[window_start:window_end])
```
- For music: also compute `onset_strength` for drop detection

#### `[P0]` Signal 3: Speaker Turn Density via pyannote · 60 min
- `utils/audio.py`: `get_speaker_density(audio_path)` → dict of `{window_start: density_score}`
```python
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')
# Count speaker switches per 30-sec window
# Normalize switch count to 0-1 score (max switches in any window = 1.0)
```
> ⚠️ **pyannote needs HuggingFace token — set up BEFORE hackathon**

#### `[P0]` Score Combiner · 30 min
- `scorer.py`: `score(transcript, audio_path, content_type)` → list of top N windows
```python
# Default (podcast/interview):
score = (0.40 × semantic) + (0.35 × energy) + (0.25 × dialogue)

# Music instrumental:
score = (0.0 × semantic) + (0.70 × energy) + (0.30 × onset)

# Sports/action:
score = (0.20 × semantic) + (0.50 × motion) + (0.30 × energy)
```
- Select top 3–5 windows, enforce min 10-sec gap between selections
- Snap window edges to nearest sentence boundary from transcript

---

### Phase 3 — Clipper + Reframer (Hr 8–12)

#### `[P0]` FFmpeg Clipper · 45 min
- `clipper.py`: `cut_clips(video_path, windows, output_dir)` → list of clip paths
- Use stream copy first (no re-encode) for speed:
```bash
ffmpeg -ss {start} -i video -t {dur} -c copy clip.mp4
```
- If stream copy produces AV sync issues (rare), fall back to: `-c:v libx264 -crf 23 -c:a aac`
- Verify each clip plays correctly after cutting
- Name clips: `clip_001.mp4`, `clip_002.mp4` (ordered by score desc)

#### `[P0]` Reframer — MediaPipe (Podcast/Interview) · 90 min
- `reframer.py`: `reframe_clip(clip_path, content_type, output_path)`
```python
import mediapipe as mp
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Per frame:
# 1. Run MediaPipe FaceDetection, get bbox center (x, y)
# 2. If no face: use frame center as fallback
# 3. Smooth crop center: rolling average over last 30 frames (collections.deque)
# 4. Crop: width = frame_height * (9/16), centered on smoothed_x
# 5. Write frames to output with cv2.VideoWriter at same fps

# Merge audio back:
# ffmpeg -i reframed_video.mp4 -i original_clip.mp4 -map 0:v -map 1:a -c:a copy final.mp4
```

#### `[P1]` Reframer — Optical Flow (Sports) · 45 min
- For `sports_action` content_type: use OpenCV optical flow instead of MediaPipe
```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# Weighted centroid of high-magnitude flow vectors = crop target
# Apply same 30-frame smoothing as MediaPipe path
```

---

### Phase 4 — Subtitles (Hr 12–16)

#### `[P0]` SRT Generator + Subtitle Burn · 60 min
- `subtitles.py`: `generate_srt(transcript_segments, clip_start_sec)` → srt_string
- Offset all timestamps by `-clip_start_sec` to make relative to clip
- Format: `00:00:05,000 --> 00:00:08,000`
- Write to temp `.srt` file
```bash
ffmpeg -i clip.mp4 \
  -vf "subtitles=clip.srt:force_style='FontSize=24,Bold=1,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Alignment=2'" \
  -c:a copy output_final.mp4
```
- Verify subtitle timing looks correct on 3 test clips

---

### Phase 5 — Quality Tuning (Hr 16–20)

#### `[P1]` Scoring Weight Tuning · 45 min
- Run full pipeline on 3 different content types with real videos
- Review selected clip windows — do they feel like the best moments?
- Adjust signal weights if semantic or energy is over/under-indexed
- Test edge case: very monotone podcast (energy flat) — does semantic still pick good clips?
- Test edge case: music with no transcript — does energy-only path work?

#### `[P1]` Reframe Quality Check · 30 min
- Watch all output clips — does the crop follow the subject smoothly?
- If jitter visible: increase smoothing window from 30 to 60 frames
- If face goes off-screen: add boundary clamp `crop_x = max(0, min(crop_x, max_x))`
- Check audio sync on reframed clips — should be in sync

---

## Sync Checkpoints

Don't skip these — they prevent 4 hours of debugging integration issues.

### Sync #1 — Hour 12

| | Person 1 brings | Person 2 brings |
|---|---|---|
| ✓ | Working `POST /api/process` + job queue + storage | `classify()` + `fetch_transcript()` + `score()` all callable |
| ✓ | Status polling working in terminal | `cut_clips()` working, `reframe()` working on test clip |

### Sync #2 — Hour 20

| | Person 1 brings | Person 2 brings |
|---|---|---|
| ✓ | Full pipeline integrated, all endpoints working | Subtitles burning correctly, clips watchable |
| ✓ | 2 real test videos successfully processed end-to-end | Demo video ready (backup pre-processed clip available) |

---

## Demo Day Checklist

### Must Have (P0)
- [ ] Full pipeline runs on a 15-min podcast URL end to end
- [ ] At least 3 clips output, all in 9:16 ratio
- [ ] Subtitles visible and correctly timed
- [ ] Status polling shows real progress stages
- [ ] Pre-processed demo backup clips ready (in case of live failure)
- [ ] API responds in under 500ms for /status endpoint

### Should Have (P1)
- [ ] Two content types working (podcast + one more)
- [ ] Error message shown for private video URL
- [ ] Reframing smooth (no jitter) on podcast clip
- [ ] Clip scoring reason visible in result (why this moment was chosen)

### Nice to Have (P2)
- [ ] Music with lyrics content type working
- [ ] Sports/action optical flow reframing
- [ ] Sentence-transformer semantic matching

---

## Interface Contract (P1 ↔ P2)

Agree on these exact signatures at **Hour 0**. Don't change them without telling each other.

```python
# classifier.py
def classify(metadata: dict) -> str:
    # Returns: 'podcast_interview' | 'music_lyrics' | 'music_instrumental' | 'sports_action' | 'mute_broll'

# transcript.py
def fetch(url: str, content_type: str, audio_path: str = None) -> list[dict]:
    # Returns: [{'start': float, 'end': float, 'text': str}, ...]

# scorer.py
def score(segments: list[dict], audio_path: str, content_type: str) -> list[dict]:
    # Returns: [{'start': float, 'end': float, 'score': float, 'reason': str}, ...]

# clipper.py
def cut_clips(video_path: str, windows: list[dict], output_dir: str) -> list[str]:
    # Returns: list of absolute file paths to cut clips

# reframer.py
def reframe_clip(clip_path: str, content_type: str, output_path: str) -> str:
    # Returns: output_path of reframed 9:16 clip

# subtitles.py
def burn_subtitles(clip_path: str, segments: list[dict], clip_start: float, output_path: str) -> str:
    # Returns: output_path of final clip with subtitles burned in
```

---

*— Ship it. —*
