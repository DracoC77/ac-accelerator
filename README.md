# Audio Chronicle Accelerator

Fast speech transcription + diarization server for the Audio Chronicle pipeline.
Supports **macOS** (Apple Silicon, mlx-whisper) and **Windows** (NVIDIA GPU, faster-whisper).
Exposes an OpenAI-compatible REST API over the LAN.

> **Why native Python?** Docker on Mac cannot access macOS Metal APIs, and Docker
> on Windows conflicts with PCIe passthrough in gaming VMs. Native Python gets
> full GPU acceleration on both platforms.

---

## Quick Start

### macOS (Apple Silicon)

**Requirements**

- macOS 13 (Ventura) or later
- Apple Silicon (M1/M2/M3 or later)
- Python 3.10+
- Xcode Command Line Tools (`xcode-select --install`)
- `ffmpeg` (`brew install ffmpeg`)
- ~6 GB disk for models (downloaded on first run)
- Hugging Face account + token (for pyannote diarization)

**One-line install**

```bash
curl -fsSL https://raw.githubusercontent.com/DracoC77/ac-accelerator/main/install.sh | bash
```

Or clone and run manually:

```bash
git clone https://github.com/DracoC77/ac-accelerator.git
cd ac-accelerator
chmod +x install.sh && ./install.sh
```

The installer will:
1. Verify prerequisites (macOS 13+, Apple Silicon, Python 3.10+, Xcode CLT)
2. Create `~/.audio-chronicle-accelerator/` with `data/`, `logs/`, `cache/`
3. Create a Python venv and install all dependencies
4. Prompt for your Hugging Face token
5. Generate a random API auth token and write `config.env`
6. Install a launchd service that starts automatically on login
7. Start the service and verify health at `http://localhost:8765/health`

**Service management**

```bash
launchctl list | grep audio-chronicle-accelerator   # check status
launchctl stop  com.audio-chronicle-accelerator
launchctl start com.audio-chronicle-accelerator
tail -f ~/.audio-chronicle-accelerator/logs/stdout.log
```

**Uninstall**

```bash
./uninstall.sh
```

---

### Windows (NVIDIA GPU)

See **[README_WINDOWS.md](README_WINDOWS.md)** for the full Windows install guide.

**TL;DR** — run `install_windows.ps1` as Administrator in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_windows.ps1
```

Requirements: Windows 10/11 x64, NVIDIA driver ≥ 580, Python 3.12.

---

## Architecture

### Backend abstraction

The server uses an `InferenceBackend` abstract base class
(`accelerator/backends/base.py`) to swap inference engines without changing
server logic. Two backends ship today:

| Backend | Platform | Engine | Speed |
|---|---|---|---|
| `MlxWhisperBackend` | macOS Apple Silicon | mlx-whisper | ~28–30× realtime |
| `FasterWhisperBackend` | Windows/Linux NVIDIA | faster-whisper + CTranslate2 | ~60–90× realtime |

Backend selection order:
1. `INFERENCE_BACKEND` env var (`mlx` or `faster-whisper`)
2. Auto-detect: Darwin/arm64 → `mlx`; CUDA available → `faster-whisper`

For a developer-oriented deep-dive, see **[docs/backend-architecture.md](docs/backend-architecture.md)**.

### Companion app

Both platform companion apps use `companion_client.py` as a shared layer for
HTTP polling and service lifecycle control:

| App | Platform | Toolkit |
|---|---|---|
| `menubar_app.py` | macOS | rumps (menubar) |
| `tray_app.py` | Windows | pystray (system tray) |

The tray/menubar icon changes colour based on server state (green = healthy +
loaded, amber = busy, grey = not loaded, red = unreachable). A **Stop** button
in the menu is the primary way to release VRAM before gaming.

---

## API Reference

All endpoints except `/health` require:

```
Authorization: Bearer YOUR_TOKEN_HERE
```

Your token is in `~/.audio-chronicle-accelerator/config.env` (Mac) or
`%USERPROFILE%\.audio-accelerator\config.env` (Windows) as `ACCELERATOR_TOKEN`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Submit audio for transcription (returns job ID) |
| `POST` | `/v1/diarize` | Submit audio for speaker diarization (returns job ID) |
| `GET` | `/v1/jobs/{job_id}` | Poll job status and retrieve results |
| `GET` | `/v1/jobs` | List recent jobs |
| `DELETE` | `/v1/jobs/{job_id}` | Cancel or remove a job |
| `GET` | `/health` | Health check — no auth required |
| `GET` | `/v1/cache/stats` | Cache hit/miss statistics |

### Example: Transcribe audio

```bash
# macOS
TOKEN=$(grep ACCELERATOR_TOKEN ~/.audio-chronicle-accelerator/config.env | cut -d= -f2)
JOB=$(curl -s -X POST http://localhost:8765/v1/audio/transcriptions \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@recording.ogg" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
curl -s http://localhost:8765/v1/jobs/$JOB -H "Authorization: Bearer $TOKEN"
```

```powershell
# Windows
$token = (Get-Content $env:USERPROFILE\.audio-accelerator\config.env |
          Where-Object { $_ -match "^ACCELERATOR_TOKEN=" }) -replace "ACCELERATOR_TOKEN=",""
curl -X POST http://localhost:8765/v1/audio/transcriptions `
     -H "Authorization: Bearer $token" `
     -F "file=@C:\path\to\audio.wav"
```

---

## Configuration

Key environment variables (set in `config.env`):

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_BACKEND` | *(auto-detect)* | `mlx` or `faster-whisper` |
| `WHISPER_MODEL` | platform-specific | Model name/path |
| `ACCELERATOR_PORT` | `8765` | Server port |
| `ACCELERATOR_HOST` | `0.0.0.0` | Bind address |
| `ACCELERATOR_TOKEN` | *(auto-generated)* | API bearer token |
| `HF_TOKEN` | *(required)* | Hugging Face token for pyannote |
| `MAX_CONCURRENT_JOBS` | `2` | Parallel job limit |
| `CACHE_TTL_DAYS` | `7` | Cache expiry in days |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `FASTER_WHISPER_DEVICE` | `cuda` | Device override (faster-whisper) |
| `FASTER_WHISPER_COMPUTE_TYPE` | `float16` | Compute type override |

See [DESIGN.md](DESIGN.md) for the full configuration reference.

---

## Performance

| Platform | Hardware | Task | Speed |
|---|---|---|---|
| macOS | M2 Pro | Transcription (10 min audio) | ~1–1.5 min |
| macOS | M3 Pro | Transcription (10 min audio) | ~0.8–1.2 min |
| Windows | RTX 5090 | Transcription (10 min audio) | ~8–10 s |

---

## Directory layout

```
~/.audio-chronicle-accelerator/   (macOS)
%USERPROFILE%\.audio-accelerator\ (Windows)
├── app/               # server + accelerator package (Windows)
├── venv/              # Python virtual environment
├── data/
│   ├── jobs.db        # SQLite job + cache store
│   └── uploads/       # Temporary audio upload staging
├── cache/             # Content-addressed result cache
├── logs/
│   ├── stdout.log
│   └── stderr.log
└── config.env         # Runtime configuration (token, model, etc.)
```

---

## Further reading

- **[README_WINDOWS.md](README_WINDOWS.md)** — Windows install, smoke tests, troubleshooting
- **[docs/backend-architecture.md](docs/backend-architecture.md)** — InferenceBackend ABC, adding new backends
- **[DESIGN.md](DESIGN.md)** — Full design doc: API schemas, security, caching, deployment

## License

MIT
