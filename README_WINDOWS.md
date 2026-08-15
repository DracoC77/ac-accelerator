# Audio Chronicle Accelerator — Windows Install Guide

This guide covers installing and running the accelerator natively on Windows
(NVIDIA GPU + CUDA 12.8 + faster-whisper). For the Mac (Apple Silicon) path, see
[README.md](README.md).

---

## Prerequisites

| Requirement | Minimum / Tested |
|---|---|
| Windows 10/11 x64 | 11 24H2 tested |
| NVIDIA driver | **≥ 580** (580.88 tested) |
| GPU | CUDA 12.8 capable — RTX 5090 tested; other modern NVIDIA cards should work |
| Python 3.12 | `py -3.12 --version` must resolve |
| CUDA 12.8 runtime | bundled via PyTorch `cu128` wheel — no separate CUDA SDK install needed |
| cuDNN 9 | `cudnn_ops_infer64_9.dll` on PATH (required for pyannote diarization) |
| Git | any recent |
| HuggingFace token | https://huggingface.co/settings/tokens (required for pyannote) |

> **WSL2 / Docker Desktop are not supported.** They can conflict with PCIe
> GPU passthrough in virtualized/gaming setups (and add overhead). We run
> native Python only.

---

## Install

1. **Clone the repo** somewhere stable:

   ```powershell
   git clone https://github.com/<you>/ac-accelerator C:\src\acc
   cd C:\src\acc
   ```

2. **Run the installer in PowerShell as Administrator** (admin is only needed
   for the firewall step; the rest works as a normal user):

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\install_windows.ps1
   ```

   The script:

   - creates `%USERPROFILE%\.audio-accelerator\{app,venv,data,nssm,tray}`
   - installs Python deps in a venv (`torch` from the `cu128` index first,
     then `faster-whisper`, `pyannote.audio`, `pystray`, `Pillow`, ...)
   - downloads NSSM 2.24 and registers the `AccServer` service
   - opens TCP 8765 in the firewall (Private + Domain profiles only)
   - creates a Startup-folder shortcut to the tray app
   - pre-downloads the Whisper model (`large-v3-turbo`)
   - generates a random `ACCELERATOR_TOKEN` in `config.env`
   - starts the service and confirms `/health` returns 200

   Total time: ~5–15 minutes depending on download speed.

3. **Set your HuggingFace token** (one-time, required for diarization):

   ```powershell
   notepad $env:USERPROFILE\.audio-accelerator\config.env
   # set HF_TOKEN=hf_...
   & "$env:USERPROFILE\.audio-accelerator\nssm\nssm.exe" restart AccServer
   ```

4. The tray app auto-launches at next login. To start it immediately:

   ```powershell
   $tray = "$env:USERPROFILE\.audio-accelerator"
   & "$tray\venv\Scripts\pythonw.exe" "$tray\tray\tray_app.py"
   ```

---

## Tray app

A coloured circle in the system tray shows server state at a glance:

| Glyph | Meaning |
|---|---|
| 🟢 green | server healthy, model loaded, idle |
| 🟡 amber | server is processing N job(s) |
| ⚪ grey | server up but Whisper not loaded yet |
| 🔴 red | service stopped or unreachable |

Right-click the icon for the menu:

```
Status: ● Healthy · faster-whisper
─────
▶ Start
■ Stop          ← THE gaming-stop button
🔄 Restart
─────
🌐 http://localhost:8765
📋 Open logs folder
⚙ Open install folder
─────
Quit tray
```

Quitting the tray does NOT stop the service. Use **Stop** for that.

---

## Stopping for Gaming (VRAM Release)

> **This is important.** The accelerator holds ~3–6 GB of VRAM when Whisper is
> loaded. Free it before launching GPU-heavy games.

The accelerator holds ~3–6 GB of VRAM when Whisper is loaded. To free it:

- **Easiest**: tray icon → **Stop**. VRAM is released in ~2–5 s.
- **PowerShell**:
  ```powershell
  & "$env:USERPROFILE\.audio-accelerator\nssm\nssm.exe" stop AccServer
  ```
- **Last resort** (hard kill — also frees VRAM via OS process cleanup):
  Task Manager → kill `python.exe` (PID is in `data\logs\stdout.log`).

To resume after gaming, use the tray's **Start** or:

```powershell
& "$env:USERPROFILE\.audio-accelerator\nssm\nssm.exe" start AccServer
```

NSSM also restarts the service automatically on crash, so you don't need to
do anything if it dies unexpectedly — just wait ~10 s.

---

## Smoke Test / First Run

Run these after install to verify everything is working end-to-end.

```powershell
# 1. Verify Python and CUDA are visible
python --version
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# 2. Set env and start server manually (first time, to see logs)
$env:INFERENCE_BACKEND = "faster-whisper"
$env:WHISPER_MODEL = "large-v3-turbo"
$env:AUTH_TOKEN = "your-token-here"  # from install_windows.ps1 output
python -m uvicorn server:app --host 0.0.0.0 --port 8765

# 3. From another machine on your LAN, hit the health endpoint
#    (replace <ACCELERATOR_HOST_IP> with the server's LAN IP)
curl http://<ACCELERATOR_HOST_IP>:8765/health
# Expected: {"status": "ok", "backend_name": "faster-whisper", "whisper_loaded": false, ...}

# 4. Trigger a model load (send a short audio file)
# Use the pipeline or curl a test WAV to /v1/audio/transcriptions

# 5. Test NSSM stop releases VRAM (gaming smoke test)
# Start via service: nssm start AccServer
# Check VRAM: nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Stop: nssm stop AccServer
# Check VRAM again: nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Expected: VRAM drops to baseline within 5 seconds
```

**Quick health check (local only):**

```powershell
# Health (no auth required)
curl http://localhost:8765/health

# A transcription (auth required — read token from config.env)
$token = (Get-Content $env:USERPROFILE\.audio-accelerator\config.env |
          Where-Object { $_ -match "^ACCELERATOR_TOKEN=" }) -replace "ACCELERATOR_TOKEN=",""
curl -X POST http://localhost:8765/v1/audio/transcriptions `
     -H "Authorization: Bearer $token" `
     -F "file=@C:\path\to\audio.wav" `
     -F "model=large-v3-turbo"
```

---

## Troubleshooting

### NSSM service won't start

```powershell
# Check service state
& "$env:USERPROFILE\.audio-accelerator\nssm\nssm.exe" status AccServer

# Check stderr for Python tracebacks
Get-Content "$env:USERPROFILE\.audio-accelerator\data\logs\stderr.log" -Tail 50

# Windows Event Viewer: Windows Logs → Application → filter by source "AccServer"
```

Common causes:

- **`torch.cuda.is_available() == False`** — a later pip install pulled a CPU-only
  torch wheel. Fix:
  ```powershell
  & "$env:USERPROFILE\.audio-accelerator\venv\Scripts\python.exe" -m pip `
    install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch
  ```
  Then verify: `python -c "import torch; print(torch.cuda.is_available())"`

- **CUDA not found / driver mismatch**: ensure your NVIDIA driver is ≥ 580.
  Run `nvidia-smi` — if it shows `Driver Version: 5xx.xx` you're fine.
  If `nvidia-smi` fails, reinstall the NVIDIA driver.

- **pyannote import errors** (`ImportError`, `ModuleNotFoundError`): usually
  a wheel incompatibility. Reinstall inside the venv:
  ```powershell
  & "$env:USERPROFILE\.audio-accelerator\venv\Scripts\python.exe" -m pip install `
    --force-reinstall pyannote.audio
  ```
  Also check that `cudnn_ops_infer64_9.dll` (and `cudnn_cnn_infer64_9.dll`)
  are on your PATH — pyannote's onnxruntime provider requires them.

- **HF_TOKEN missing**: pyannote will silently fail to download the diarization
  pipeline. Set it in `config.env` and restart the service.

### VRAM not released after stop

Check `nvidia-smi`. If `python.exe` is still listed:

1. `nssm stop AccServer` may have hit a CUDA hang. NSSM forcibly terminates
   after ~25 s (`AppStopMethodConsole=15000 + AppStopMethodWindow=5000 +
   AppStopMethodThreads=5000`). Wait that long, then check again.
2. If `python.exe` is still holding VRAM, kill it from Task Manager — OS
   process cleanup will reclaim VRAM regardless of Python state.

### Tray icon doesn't appear

- The Startup shortcut needs you to log out and back in once after install.
- Start it manually: `& "$env:USERPROFILE\.audio-accelerator\venv\Scripts\pythonw.exe" "$env:USERPROFILE\.audio-accelerator\tray\tray_app.py"`
- If you get an `ImportError: No module named pystray`, rerun the installer.

### Port 8765 already in use

Pass a different port:

```powershell
.\install_windows.ps1 -Port 18765
```

The installer updates the NSSM service args, firewall rule, and `config.env`.

---

## Uninstall

The installer writes an uninstall script next to itself:

```powershell
powershell -File $env:USERPROFILE\.audio-accelerator\uninstall_windows.ps1
```

Add `-KeepData` to preserve `data\` (cache, uploads, logs):

```powershell
powershell -File $env:USERPROFILE\.audio-accelerator\uninstall_windows.ps1 -KeepData
```

This removes the NSSM service, firewall rule, Startup shortcut, and (by
default) the entire install directory.

---

## File layout reference

```
%USERPROFILE%\.audio-accelerator\
├── app\               # server.py, accelerator/, companion_client.py, tray_app.py
├── venv\              # Python 3.12 + CUDA torch + faster-whisper + pyannote
├── data\
│   ├── logs\          # stdout.log, stderr.log, accelerator.log, tray.log
│   └── uploads\       # transient audio uploads (cleaned per job)
├── nssm\nssm.exe      # service manager
├── tray\tray_app.py   # tray launcher (copy of app\tray_app.py)
├── config.env         # ACCELERATOR_TOKEN, HF_TOKEN, model selection
└── uninstall_windows.ps1
```
