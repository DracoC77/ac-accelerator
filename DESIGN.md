# Audio Chronicle Accelerator — Design Document

## 1. Overview

The Audio Chronicle Accelerator is a native Python service designed to run on Apple Silicon Macs, offloading the two most compute-intensive stages of the audio-chronicle pipeline — speech-to-text transcription and speaker diarization — from a Docker-hosted Linux server to a local Mac with Metal GPU access.

It exposes an **OpenAI-compatible REST API** over the LAN, processes jobs asynchronously with polling-based result retrieval, and uses a content-addressed cache to avoid redundant work.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│         Upstream Service (Docker/Linux)          │
│                                                  │
│  audio-chronicle pipeline                        │
│    ├── detects ACCELERATOR_URL env var           │
│    ├── POST /v1/audio/transcriptions  ──────┐    │
│    ├── POST /v1/diarize               ──────┤    │
│    ├── GET  /v1/jobs/{job_id}  (poll)  ─────┤    │
│    └── fallback: local CPU if unreachable   │    │
└─────────────────────────────────────┬───────┘    │
                                      │ HTTP/LAN   │
┌─────────────────────────────────────▼───────┐    │
│   MacBook Pro — Native Python (port 8765)   │    │
│                                              │    │
│   FastAPI application                        │    │
│   ├── POST /v1/audio/transcriptions          │    │
│   │     → mlx-whisper (Metal GPU)            │    │
│   ├── POST /v1/diarize                       │    │
│   │     → pyannote.audio (PyTorch MPS)       │    │
│   ├── GET  /v1/jobs/{id}                     │    │
│   │     → job status + result                │    │
│   ├── DELETE /v1/jobs/{id}                   │    │
│   │     → cancel / remove job                │    │
│   ├── GET  /v1/jobs                          │    │
│   │     → list recent jobs                   │    │
│   ├── GET  /health                           │    │
│   │     → readiness + version + GPU status   │    │
│   └── GET  /v1/cache/stats                   │    │
│         → cache hit/miss counters            │    │
│                                              │    │
│   SQLite job store (persistent)              │    │
│   Content-addressed cache (SHA256, 7d TTL)   │    │
│                                              │    │
│   macOS Menu Bar App (rumps)                 │    │
│   ├── Start / Pause / Stop service           │    │
│   ├── Active jobs + progress                 │    │
│   ├── Queue depth                            │    │
│   └── Resource monitor (RAM / CPU / Battery) │    │
│                                              │    │
│   Managed by launchd (auto-start, restart)   │    │
└──────────────────────────────────────────────┘
```

---

## 3. Technology Choices

### 3.1 mlx-whisper over faster-whisper

| Factor | mlx-whisper | faster-whisper |
|---|---|---|
| **Backend** | Apple MLX framework (Metal GPU) | CTranslate2 (CPU / CUDA) |
| **Apple Silicon perf** | ~0.10–0.25× RTF on M2 | ~0.5–1.0× RTF (CPU only on Mac) |
| **GPU utilization** | Full Metal acceleration | No Metal support |
| **Model format** | MLX-optimized weights | CTranslate2 INT8/FP16 |
| **API compatibility** | Drop-in OpenAI Whisper interface | Similar but different return schema |

**Decision:** mlx-whisper gives 3–5× speedup over faster-whisper on Apple Silicon because it actually uses the GPU. faster-whisper's CTranslate2 backend has no Metal path and falls back to CPU-only execution on macOS.

### 3.2 pyannote with MPS

PyTorch 2.x supports Apple's Metal Performance Shaders (MPS) backend. Calling `pipeline.to(torch.device("mps"))` moves all neural network inference to the GPU, yielding ~3–5× speedup over CPU-only diarization.

**Key consideration:** pyannote requires a Hugging Face auth token for model downloads. The token is configured once during install and stored in the local HF cache.

### 3.3 Native Python over Docker

Docker Desktop for Mac runs containers inside a Linux VM (HyperKit or Apple Virtualization.framework). The VM layer cannot expose macOS Metal APIs to containerized processes:

```python
# Inside Docker on Mac:
torch.backends.mps.is_available()  # → False
# Native macOS:
torch.backends.mps.is_available()  # → True
```

**Decision:** Native Python with a venv is the only viable path for Metal GPU access on macOS.

### 3.4 launchd over systemd / Docker restart policies

macOS uses `launchd` as its init system. Benefits:
- Auto-start on user login (`RunAtLoad`)
- Automatic restart on crash (`KeepAlive`)
- Resource limits (`SoftResourceLimits`)
- Native macOS integration (no third-party daemon manager)
- Works with the menu bar app for user-visible control

---

## 4. API Contract

### Base URL

```
http://<mac-host>:8765
```

### Authentication

All endpoints except `/health` require a Bearer token:

```
Authorization: Bearer YOUR_TOKEN_HERE
```

### 4.1 POST /v1/audio/transcriptions

Submit an audio file for transcription. Returns immediately with a job ID.

**Request:**
```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
Authorization: Bearer YOUR_TOKEN_HERE

file: <audio file (WAV, OGG, MP3, M4A, FLAC, WEBM)>
model: "mlx-community/whisper-large-v3-turbo" (optional, default)
language: "en" (optional, auto-detect if omitted)
response_format: "verbose_json" (optional, default "verbose_json")
```

**Response (202 Accepted):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "type": "transcription",
  "created_at": "2026-04-16T12:00:00Z",
  "cache_hit": false,
  "poll_url": "/v1/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Cache hit response (200 OK):**
```json
{
  "job_id": "existing-job-id",
  "status": "complete",
  "type": "transcription",
  "cache_hit": true,
  "result": { ... }
}
```

### 4.2 POST /v1/diarize

Submit an audio file for speaker diarization.

**Request:**
```
POST /v1/diarize
Content-Type: multipart/form-data
Authorization: Bearer YOUR_TOKEN_HERE

file: <audio file>
min_speakers: 2 (optional)
max_speakers: 10 (optional)
```

**Response (202 Accepted):**
```json
{
  "job_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "status": "queued",
  "type": "diarization",
  "created_at": "2026-04-16T12:00:00Z",
  "cache_hit": false,
  "poll_url": "/v1/jobs/b2c3d4e5-f6a7-8901-bcde-f12345678901"
}
```

### 4.3 GET /v1/jobs/{job_id}

Poll for job status and results.

**Response (processing):**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "processing",
  "type": "transcription",
  "created_at": "2026-04-16T12:00:00Z",
  "started_at": "2026-04-16T12:00:01Z",
  "progress": 0.45
}
```

**Response (complete — transcription):**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "complete",
  "type": "transcription",
  "created_at": "2026-04-16T12:00:00Z",
  "started_at": "2026-04-16T12:00:01Z",
  "completed_at": "2026-04-16T12:00:15Z",
  "duration_seconds": 14.2,
  "result": {
    "text": "Full transcription text...",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 4.5,
        "text": "Segment text...",
        "avg_logprob": -0.23,
        "no_speech_prob": 0.01
      }
    ],
    "language": "en",
    "duration": 120.5
  }
}
```

**Response (complete — diarization):**
```json
{
  "job_id": "b2c3d4e5-...",
  "status": "complete",
  "type": "diarization",
  "created_at": "2026-04-16T12:00:00Z",
  "completed_at": "2026-04-16T12:00:30Z",
  "duration_seconds": 29.8,
  "result": {
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 3.2
      },
      {
        "speaker": "SPEAKER_01",
        "start": 3.5,
        "end": 7.8
      }
    ],
    "num_speakers": 2,
    "duration": 120.5
  }
}
```

**Response (failed):**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "failed",
  "type": "transcription",
  "error": "Out of memory: audio file too large for available GPU memory",
  "created_at": "2026-04-16T12:00:00Z",
  "failed_at": "2026-04-16T12:00:05Z"
}
```

**Status codes:** `200` (found), `404` (unknown job_id)

### 4.4 GET /v1/jobs

List recent jobs. Supports filtering and pagination.

**Query parameters:**
- `status` — filter by status: `queued`, `processing`, `complete`, `failed`, `cancelled`
- `type` — filter by type: `transcription`, `diarization`
- `limit` — max results (default 20, max 100)
- `offset` — pagination offset (default 0)

**Response (200 OK):**
```json
{
  "jobs": [ ... ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### 4.5 DELETE /v1/jobs/{job_id}

Cancel a queued/processing job, or remove a completed/failed job from history.

**Response (200 OK):**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

### 4.6 GET /health

Health check endpoint. No authentication required.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu": {
    "metal_available": true,
    "mps_available": true,
    "device_name": "Apple M2 Pro"
  },
  "models": {
    "whisper": "mlx-community/whisper-large-v3-turbo",
    "whisper_loaded": true,
    "diarization": "pyannote/speaker-diarization-3.1",
    "diarization_loaded": true
  },
  "queue": {
    "pending": 0,
    "processing": 0
  },
  "uptime_seconds": 3600
}
```

### 4.7 GET /v1/cache/stats

Cache statistics.

**Response (200 OK):**
```json
{
  "total_entries": 156,
  "transcription_entries": 120,
  "diarization_entries": 36,
  "hit_count": 89,
  "miss_count": 156,
  "hit_rate": 0.363,
  "cache_size_bytes": 2048000,
  "oldest_entry": "2026-04-09T12:00:00Z",
  "ttl_days": 7
}
```

---

## 5. Job Lifecycle

```
                    ┌──────────────────────────────────┐
                    │        Client POST request        │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │     Compute SHA256 of file        │
                    └──────────────┬───────────────────┘
                                   │
                          ┌────────┴────────┐
                          │  Cache lookup    │
                          └────────┬────────┘
                         hit /            \ miss
                            ▼              ▼
               ┌────────────────┐  ┌───────────────┐
               │  Return cached │  │   QUEUED       │
               │  result (200)  │  │  (202 + job_id)│
               └────────────────┘  └───────┬───────┘
                                           │
                                           ▼
                                   ┌───────────────┐
                                   │  PROCESSING    │
                                   │  (worker picks │
                                   │   up job)      │
                                   └───────┬───────┘
                                           │
                                  ┌────────┴────────┐
                                  │                 │
                                  ▼                 ▼
                          ┌──────────────┐  ┌──────────────┐
                          │  COMPLETE     │  │  FAILED       │
                          │  (result      │  │  (error msg   │
                          │   cached)     │  │   stored)     │
                          └──────────────┘  └──────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  CACHED       │
                          │  (7-day TTL,  │
                          │   SHA256 key) │
                          └──────────────┘
                                  │
                            after 7 days
                                  ▼
                          ┌──────────────┐
                          │  EVICTED      │
                          └──────────────┘


    Job states: queued → processing → complete | failed
    Side states: cancelled (via DELETE), cached (implicit on complete)
```

### Polling Strategy (client side)

Recommended client polling intervals:
- First 5 seconds: poll every 1s
- 5–30 seconds: poll every 3s
- 30+ seconds: poll every 5s

---

## 6. Content-Addressed Cache

### Design

- **Key:** SHA256 hash of the raw audio file bytes + job type (transcription/diarization) + model parameters
- **Storage:** SQLite table alongside the job store
- **TTL:** 7 days from creation (configurable via `CACHE_TTL_DAYS`)
- **Eviction:** Lazy — checked on lookup; background sweep every hour
- **Scope:** Per job type — same file transcribed and diarized = two cache entries

### Cache Key Computation

```python
import hashlib

def compute_cache_key(file_bytes: bytes, job_type: str, **params) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(job_type.encode())
    for k in sorted(params.keys()):
        h.update(f"{k}={params[k]}".encode())
    return h.hexdigest()
```

Parameters included in the key:
- **Transcription:** model, language (if specified)
- **Diarization:** min_speakers, max_speakers

---

## 7. Deployment Model

### 7.1 install.sh Flow

```
install.sh
    │
    ├── 1. Check prerequisites
    │      ├── macOS version ≥ 13 (Ventura)
    │      ├── Apple Silicon (arm64)
    │      ├── Python ≥ 3.10
    │      ├── Xcode CLT installed
    │      └── ffmpeg installed
    │
    ├── 2. Create Python venv
    │      └── ~/.audio-chronicle-accelerator/venv/
    │
    ├── 3. Install dependencies
    │      ├── pip install -r requirements.txt
    │      ├── mlx-whisper, pyannote.audio, torch, fastapi, uvicorn
    │      └── rumps (menu bar app)
    │
    ├── 4. Download models (first run)
    │      ├── mlx-community/whisper-large-v3-turbo (~3 GB)
    │      └── pyannote/speaker-diarization-3.1 (requires HF token)
    │
    ├── 5. Generate config
    │      └── ~/.audio-chronicle-accelerator/config.env
    │           (ACCELERATOR_TOKEN, PORT, LOG_LEVEL, etc.)
    │
    ├── 6. Install launchd plist
    │      └── ~/Library/LaunchAgents/com.audio-chronicle-accelerator.plist
    │
    └── 7. Load & start service
           └── launchctl load + launchctl start
```

### 7.2 launchd Plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.audio-chronicle-accelerator</string>

    <key>ProgramArguments</key>
    <array>
        <string>~/.audio-chronicle-accelerator/venv/bin/python</string>
        <string>-m</string>
        <string>accelerator.server</string>
    </array>

    <key>WorkingDirectory</key>
    <string>~/.audio-chronicle-accelerator</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>

    <key>StandardOutPath</key>
    <string>~/.audio-chronicle-accelerator/logs/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>~/.audio-chronicle-accelerator/logs/stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>ACCELERATOR_PORT</key>
        <string>8765</string>
    </dict>

    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>4096</integer>
    </dict>
</dict>
</plist>
```

### 7.3 Directory Layout

```
~/.audio-chronicle-accelerator/
├── venv/                      # Python virtual environment
├── config.env                 # Runtime configuration
├── accelerator/               # Application source (or symlink)
│   ├── __init__.py
│   ├── server.py              # FastAPI app + uvicorn entry
│   ├── worker.py              # Background job processor
│   ├── cache.py               # Content-addressed cache logic
│   ├── transcribe.py          # mlx-whisper wrapper
│   ├── diarize.py             # pyannote wrapper
│   ├── models.py              # Pydantic models
│   ├── auth.py                # Bearer token validation
│   └── config.py              # Configuration loader
├── data/
│   ├── jobs.db                # SQLite job + cache store
│   └── uploads/               # Temporary audio file staging
├── logs/
│   ├── stdout.log
│   └── stderr.log
└── menubar/
    └── app.py                 # rumps menu bar application
```

---

## 8. Security Model

### 8.1 Network Exposure

- **Bind address:** `0.0.0.0:8765` (LAN-accessible)
- **Intended scope:** LAN-only — no public exposure, no port forwarding
- The upstream client knows the Mac's LAN IP via `ACCELERATOR_URL`

### 8.2 Authentication

- **Bearer token:** a randomly generated 64-character hex token, created during `install.sh`
- **Storage:** `~/.audio-chronicle-accelerator/config.env` as `ACCELERATOR_TOKEN`
- **Enforcement:** All endpoints except `/health` require `Authorization: Bearer <token>`
- **Constant-time comparison** to prevent timing attacks

### 8.3 Rate Limiting

- **Per-IP rate limit:** 60 requests/minute (configurable)
- **Concurrent job limit:** 2 simultaneous processing jobs (configurable)
- **File size limit:** 500 MB per upload (configurable)
- **Queue depth limit:** 20 pending jobs (configurable)

### 8.4 File Handling

- Uploaded files are written to `data/uploads/` with randomized filenames
- Files are deleted after processing completes (success or failure)
- Cached results store only the output JSON, not the original audio

---

## 9. Performance Targets

### 9.1 Transcription (mlx-whisper, large-v3-turbo)

| Hardware | RTF (Real-Time Factor) | 10 min audio | 1 hr audio |
|---|---|---|---|
| M2 (base, 8 GPU) | ~0.20–0.25× | ~2–2.5 min | ~12–15 min |
| M2 Pro (16 GPU) | ~0.10–0.15× | ~1–1.5 min | ~6–9 min |
| M3 Pro (18 GPU) | ~0.08–0.12× | ~0.8–1.2 min | ~5–7 min |

*RTF < 1.0 means faster than real-time. Lower = faster.*

### 9.2 Diarization (pyannote 3.1, MPS)

| Hardware | RTF | 10 min audio | 1 hr audio |
|---|---|---|---|
| M2 (base) | ~0.30–0.50× | ~3–5 min | ~18–30 min |
| M2 Pro | ~0.15–0.25× | ~1.5–2.5 min | ~9–15 min |
| M3 Pro | ~0.10–0.20× | ~1–2 min | ~6–12 min |

### 9.3 Baseline Comparison (CPU-only on Docker/Linux)

| Task | CPU (Xeon/Docker) | M2 Pro (this service) | Speedup |
|---|---|---|---|
| Transcription (10 min) | ~5–8 min | ~1–1.5 min | **4–6×** |
| Diarization (10 min) | ~8–15 min | ~1.5–2.5 min | **4–6×** |

---

## 10. Fallback Behavior (Upstream Client)

The upstream audio-chronicle service implements a graceful fallback pattern:

```
                 ┌──────────────────┐
                 │  Audio file ready │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
          ┌──────│ ACCELERATOR_URL  │──────┐
          │      │    configured?    │      │
          │      └──────────────────┘      │
         yes                               no
          │                                │
          ▼                                │
   ┌──────────────┐                        │
   │ GET /health  │                        │
   └──────┬───────┘                        │
          │                                │
     ┌────┴────┐                           │
  healthy   unreachable                    │
     │         │                           │
     ▼         ▼                           ▼
  ┌────────┐  ┌──────────────────────────────┐
  │ POST   │  │  Local CPU fallback           │
  │ to     │  │  (slower but always works)    │
  │ accel  │  └──────────────────────────────┘
  └───┬────┘
      │
   ┌──┴──┐
 200   5xx/timeout
   │      │
   ▼      ▼
 Use    Retry once → still failing → CPU fallback
result
```

### Fallback rules:
1. If `ACCELERATOR_URL` is not set → always use local CPU
2. If `/health` returns non-200 → skip accelerator, use local CPU
3. If POST returns 5xx → retry once after 2s → if still failing, use local CPU
4. If job polling shows no progress for 5 minutes → cancel + use local CPU
5. Log all fallback events for debugging

### Reconnection:
- After a fallback, the client retries the accelerator every 5 minutes via `/health`
- Once healthy again, new jobs route to the accelerator
- No automatic re-submission of fallback-processed jobs

---

## 11. Configuration

All configuration is via environment variables, loaded from `~/.audio-chronicle-accelerator/config.env`:

| Variable | Default | Description |
|---|---|---|
| `ACCELERATOR_PORT` | `8765` | Port to bind the server |
| `ACCELERATOR_HOST` | `0.0.0.0` | Bind address |
| `ACCELERATOR_TOKEN` | *(generated)* | Bearer token for API auth |
| `WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | mlx-whisper model ID |
| `WHISPER_DEVICE` | `auto` | Force device (`auto`, `gpu`, `cpu`) |
| `DIARIZE_MODEL` | `pyannote/speaker-diarization-3.1` | pyannote model ID |
| `HF_TOKEN` | *(required)* | Hugging Face token for pyannote download |
| `MAX_CONCURRENT_JOBS` | `2` | Max simultaneous processing jobs |
| `MAX_QUEUE_DEPTH` | `20` | Max pending jobs in queue |
| `MAX_FILE_SIZE_MB` | `500` | Max upload file size (MB) |
| `CACHE_TTL_DAYS` | `7` | Cache entry time-to-live |
| `CACHE_SWEEP_INTERVAL_MIN` | `60` | Minutes between cache eviction sweeps |
| `RATE_LIMIT_PER_MIN` | `60` | Requests per minute per IP |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DB_PATH` | `data/jobs.db` | SQLite database path |
| `UPLOAD_DIR` | `data/uploads` | Temporary upload directory |
| `PRELOAD_MODELS` | `true` | Load models into memory on startup |

---

## 12. SQLite Schema

```sql
-- Job tracking
CREATE TABLE jobs (
    job_id       TEXT PRIMARY KEY,
    job_type     TEXT NOT NULL,        -- 'transcription' | 'diarization'
    status       TEXT NOT NULL,        -- 'queued' | 'processing' | 'complete' | 'failed' | 'cancelled'
    cache_key    TEXT NOT NULL,        -- SHA256-based content key
    params_json  TEXT,                 -- request parameters (JSON)
    result_json  TEXT,                 -- output result (JSON, NULL until complete)
    error        TEXT,                 -- error message (NULL unless failed)
    progress     REAL DEFAULT 0.0,    -- 0.0 to 1.0
    file_path    TEXT,                 -- temp upload file path
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    completed_at TEXT,
    failed_at    TEXT
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_cache_key ON jobs(cache_key);
CREATE INDEX idx_jobs_created ON jobs(created_at);

-- Content-addressed cache
CREATE TABLE cache (
    cache_key    TEXT PRIMARY KEY,
    job_type     TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    hit_count    INTEGER DEFAULT 0
);

CREATE INDEX idx_cache_expires ON cache(expires_at);

-- Cache stats (singleton row, updated atomically)
CREATE TABLE cache_stats (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    total_hits   INTEGER DEFAULT 0,
    total_misses INTEGER DEFAULT 0
);
```

---

## 13. Menu Bar App

### Technology: rumps

[rumps](https://github.com/jaredks/rumps) is a lightweight Python library for macOS status bar (menu bar) apps. It integrates naturally with the Python venv and launchd deployment model.

### Menu Structure

```
📎 Accelerator [●]         ← green dot = healthy, yellow = degraded, red = stopped
├── Status: Running
├── ──────────────
├── Jobs: 1 active, 3 queued
├── Current: transcription (45%)
├── ──────────────
├── Cache: 156 entries (89 hits)
├── ──────────────
├── Resources
│   ├── CPU: 45%
│   ├── RAM: 2.1 GB / 16 GB
│   ├── GPU: Active
│   └── Battery: 78% (plugged in)
├── ──────────────
├── Pause Queue            ← stop accepting new jobs
├── Resume Queue
├── ──────────────
├── Open Logs...           ← opens log file in Console.app
├── Open Config...         ← opens config.env in default editor
├── ──────────────
└── Quit                   ← stop service + unload launchd
```

### Implementation Notes

- **Runs as a separate process** from the FastAPI server
- **Communicates via HTTP** — polls the local `/health`, `/v1/jobs`, and `/v1/cache/stats` endpoints
- **Update interval:** Every 5 seconds
- **Launch:** Started alongside the server via a separate launchd plist, or bundled as a .app

---

## 14. Error Handling

### Server-side

| Scenario | Behavior |
|---|---|
| Model fails to load | `/health` returns `"status": "degraded"`, affected endpoints return 503 |
| Out of memory | Job fails with OOM error, worker restarts for next job |
| Upload too large | 413 Payload Too Large before processing |
| Queue full | 429 Too Many Requests with `Retry-After` header |
| Invalid audio format | Job fails immediately with descriptive error |
| SQLite locked | Retry with exponential backoff (WAL mode minimizes this) |
| Process crash | launchd restarts within 10 seconds, queued jobs resume |

### Client-side (upstream)

| Scenario | Behavior |
|---|---|
| Accelerator unreachable | Immediate CPU fallback |
| Job stuck (no progress 5 min) | Cancel + CPU fallback |
| 5xx from accelerator | Retry once, then CPU fallback |
| 401 Unauthorized | Log auth error, CPU fallback, alert operator |
| Network timeout on polling | Retry poll, not the job (job continues server-side) |

---

## 15. Logging

### Format

```
2026-04-16T12:00:00.123 [INFO] server: Started on 0.0.0.0:8765
2026-04-16T12:00:05.456 [INFO] worker: Job a1b2c3d4 started (transcription, 120.5s audio)
2026-04-16T12:00:19.789 [INFO] worker: Job a1b2c3d4 complete (14.2s, RTF=0.118)
```

### Log Rotation

- Logs written to `~/.audio-chronicle-accelerator/logs/`
- Rotated daily, 7-day retention
- Structured JSON logging available via `LOG_FORMAT=json`

---

## 16. Future Considerations

- **Batched transcription:** multiple short files in one request
- **Streaming results:** WebSocket endpoint for real-time progress
- **Model hot-swap:** change whisper model size without restart
- **Prometheus metrics:** `/metrics` endpoint for monitoring
- **mDNS discovery:** auto-discovery via Bonjour instead of manual IP configuration
