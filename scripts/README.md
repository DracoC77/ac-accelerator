# Test Scripts

Scripts in this directory are intended to be run **manually on the Windows VM**
after `install_windows.ps1` has set up the AccServer service.

---

## `test_vram_release.ps1`

**WI reference:** WI-ACC-27c — "Graceful shutdown / VRAM release" (AC-VR-4)

Tests that GPU VRAM is properly released under all four kill scenarios:

| # | Scenario | How it stops |
|---|----------|-------------|
| 1 | Idle graceful stop | `nssm stop AccServer` while model loaded, no jobs |
| 2 | Mid-job graceful stop | `nssm stop AccServer` while a transcription job is in-flight |
| 3 | Hard kill | `Stop-Process -Force` (Task Manager equivalent) |
| 4 | Leak detection | 5× stop/start cycle — VRAM must return to baseline each time |

Each test measures VRAM via `nvidia-smi` before and after the stop, and
reports elapsed time and pass/fail with colour-coded output.

### Prerequisites

- **AccServer installed and configured** — run `install_windows.ps1` first.
  The service must be registered with NSSM as `AccServer` (or pass
  `-ServiceName YourName`).
- **NVIDIA drivers ≥ 580.88** with `nvidia-smi.exe` on `PATH`.
  (`C:\Program Files\NVIDIA Corporation\NVSMI\` is added automatically by the
  driver installer on most systems.)
- **Python on PATH** — used to generate a silence WAV if no `tests/*.wav`
  fixture is found in the repo.  The venv Python at
  `%USERPROFILE%\.audio-accelerator\venv\Scripts\python.exe` works fine;
  add it to PATH or rely on system Python.
- **PowerShell 5.1 or 7.x** — no external modules required.
- **Admin is not required** unless `nssm.exe` itself requires it (it usually
  does not for stop/start of an already-registered service).

### Usage

```powershell
# Default — uses $env:AUTH_TOKEN if set, open-access mode otherwise
.\scripts\test_vram_release.ps1

# With an explicit auth token
$env:AUTH_TOKEN = "your-token"
.\scripts\test_vram_release.ps1

# Custom service name / base URL (e.g. non-default port)
.\scripts\test_vram_release.ps1 -ServiceName AccServer -BaseUrl http://localhost:8765

# Adjust timeouts and tolerance
.\scripts\test_vram_release.ps1 -GracefulTimeoutSec 15 -ToleranceMB 300 -LeakCycles 10
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-ServiceName` | `AccServer` | NSSM service name |
| `-BaseUrl` | `http://localhost:8765` | Accelerator base URL |
| `-AuthToken` | `$env:AUTH_TOKEN` | Bearer token (optional if open-access) |
| `-ToleranceMB` | `200` | VRAM headroom above baseline considered "released" |
| `-GracefulTimeoutSec` | `10` | Seconds to wait for VRAM drop after graceful stop |
| `-HardKillTimeoutSec` | `10` | Seconds to wait for VRAM drop after hard kill |
| `-MidJobTimeoutSec` | `30` | Seconds to wait for VRAM drop after mid-job stop |
| `-ModelLoadTimeoutSec` | `60` | Seconds to wait for model to load |
| `-LeakCycles` | `5` | Stop/start repetitions in Test 4 |

### Expected output

```
=== Audio Chronicle Accelerator — VRAM Release Test Suite ===
Baseline VRAM: 2048 MB

[TEST 1] Graceful stop (idle — model loaded)
  Model loaded VRAM:  6200 MB (delta: +4152 MB)
  Issuing: nssm stop AccServer ...
  VRAM after 2.5s:    2051 MB ✅ PASS (released in 2.5s)

[TEST 2] Graceful stop (mid-job)
  Submitting transcription job...
  Job submitted (abc123), waiting 2 s then killing service...
  Issuing: nssm stop AccServer ...
  VRAM after 4.1s:    2049 MB ✅ PASS
  Job status after restart: failed ✅

[TEST 3] Hard kill (TerminateProcess — Task Manager equivalent)
  PID: 12345, issuing Stop-Process -Force ...
  VRAM after 0.8s:    2050 MB ✅ PASS (OS cleanup, graceful handler did NOT run)

[TEST 4] Leak detection (5 stop/start cycles)
  Cycle 1: loaded=6198 MB → stopped=2050 MB ✅
  Cycle 2: loaded=6201 MB → stopped=2048 MB ✅
  Cycle 3: loaded=6199 MB → stopped=2051 MB ✅
  Cycle 4: loaded=6200 MB → stopped=2049 MB ✅
  Cycle 5: loaded=6198 MB → stopped=2050 MB ✅
  No VRAM leak detected ✅

=== SUMMARY ===
  Test 1: ✅ PASS
  Test 2: ✅ PASS
  Test 3: ✅ PASS
  Test 4: ✅ PASS
Overall: ✅ ALL PASS
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All tests passed |
| `1` | One or more tests failed (or prerequisites not met) |

### Notes

- The script manages service start/stop itself — do **not** start AccServer
  manually before running it.
- Test 3 (hard kill) will trigger NSSM's automatic restart
  (`AppExit Default Restart`).  The script issues a second `nssm stop` after
  confirming VRAM release so Test 4 starts cleanly.
- If no `tests/*.wav` fixture is found in the repo, a 5-second 16 kHz mono
  silence WAV is generated via Python and cached in `%TEMP%\acc_test_silence.wav`.
- The RTX 5090 typically clears its CUDA context within < 1 s on a hard kill;
  graceful unload (Test 1) should complete within 2–5 s.
