<#
.SYNOPSIS
    Audio Chronicle Accelerator — Windows install script.

.DESCRIPTION
    Installs the accelerator server as a Windows service (via NSSM) and
    registers a per-user Startup shortcut for the system tray app.

    Idempotent: re-running upgrades pip deps and reconfigures NSSM without
    breaking an existing install. Reinstall/uninstall is delegated to the
    generated uninstall_windows.ps1.

.PARAMETER InstallDir
    Root install directory. Default: $env:USERPROFILE\.audio-accelerator

.PARAMETER Port
    TCP port for the accelerator server. Default: 8765

.PARAMETER ServiceName
    NSSM service name. Default: AccServer

.PARAMETER SkipFirewall
    Skip the inbound firewall rule (useful for re-runs that don't need UAC).

.PARAMETER SkipModelPreload
    Skip the warm-cache model download step.

.EXAMPLE
    # Default install (run from the cloned repo root)
    PS> .\install_windows.ps1

.EXAMPLE
    # Custom install location and port
    PS> .\install_windows.ps1 -InstallDir D:\acc -Port 18765

.NOTES
    Prerequisites (the script will check and report):
      * Windows 10/11 x64
      * NVIDIA driver >= 580.88 with an RTX 5090
      * Python 3.12 (py launcher)
      * CUDA 12.8 runtime + cuDNN 9 (cudnn_ops_infer64_9.dll on PATH)

#>

[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE ".audio-accelerator"),
    [string]$Port = "8765",
    [string]$ServiceName = "AccServer",
    [switch]$SkipFirewall,
    [switch]$SkipModelPreload
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step    { param([string]$Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Ok      { param([string]$Msg) Write-Host "    [ok] $Msg" -ForegroundColor Green }
function Write-Warn    { param([string]$Msg) Write-Host "    [warn] $Msg" -ForegroundColor Yellow }
function Write-Err     { param([string]$Msg) Write-Host "    [err] $Msg" -ForegroundColor Red }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p  = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function New-RandomToken {
    param([int]$Length = 32)
    $bytes = New-Object byte[] $Length
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return ([System.BitConverter]::ToString($bytes) -replace "-", "").ToLower()
}

# ---------------------------------------------------------------------------
# 0. Banner
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  Audio Chronicle Accelerator — Windows installer" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  InstallDir : $InstallDir"
Write-Host "  Port       : $Port"
Write-Host "  Service    : $ServiceName"
Write-Host ""

# Capture the source repo (the directory this script lives in).
$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
Write-Host "  Source     : $RepoRoot"
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Prerequisite checks
# ---------------------------------------------------------------------------

Write-Step "1. Checking prerequisites"

# Windows version
try {
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Ok "Windows: $($os.Caption) build $($os.BuildNumber)"
} catch {
    Write-Warn "Could not detect Windows version: $_"
}

# Python 3.12
$pythonExe = $null
try {
    $pyVersion = & py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Python: $pyVersion"
        $pythonExe = (& py -3.12 -c "import sys; print(sys.executable)" 2>&1).Trim()
    } else {
        throw "py -3.12 not available"
    }
} catch {
    Write-Err "Python 3.12 not found. Install from https://www.python.org/downloads/ or 'winget install Python.Python.3.12'"
    exit 1
}

# NVIDIA driver / GPU
try {
    $nvSmi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "NVIDIA: $nvSmi"
    } else {
        Write-Warn "nvidia-smi not available — CUDA build may still work if drivers are installed"
    }
} catch {
    Write-Warn "Could not query NVIDIA driver: $_"
}

# CUDA / cuDNN — soft check only (pip wheels bring CUDA runtime; cuDNN is needed for pyannote)
if ($env:CUDA_PATH) {
    Write-Ok "CUDA_PATH: $env:CUDA_PATH"
} else {
    Write-Warn "CUDA_PATH not set. Torch CUDA wheels include the runtime so this is usually fine,"
    Write-Warn "but pyannote needs cuDNN 9 (cudnn_ops_infer64_9.dll on PATH)."
}

# ---------------------------------------------------------------------------
# 2. Directories
# ---------------------------------------------------------------------------

Write-Step "2. Creating install directories"

$AppDir   = Join-Path $InstallDir "app"
$VenvDir  = Join-Path $InstallDir "venv"
$DataDir  = Join-Path $InstallDir "data"
$LogsDir  = Join-Path $DataDir    "logs"
$UploadDir= Join-Path $DataDir    "uploads"
$NssmDir  = Join-Path $InstallDir "nssm"
$TrayDir  = Join-Path $InstallDir "tray"
$ConfigFile = Join-Path $InstallDir "config.env"

foreach ($d in @($InstallDir, $AppDir, $VenvDir, $DataDir, $LogsDir, $UploadDir, $NssmDir, $TrayDir)) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Ok "created $d"
    } else {
        Write-Ok "exists  $d"
    }
}

# ---------------------------------------------------------------------------
# 3. Copy application files
# ---------------------------------------------------------------------------

Write-Step "3. Copying application files"

$appFiles = @("server.py", "companion_client.py", "tray_app.py", "requirements.txt")
foreach ($f in $appFiles) {
    $src = Join-Path $RepoRoot $f
    if (Test-Path $src) {
        Copy-Item $src -Destination $AppDir -Force
        Write-Ok "copied $f"
    } else {
        Write-Warn "missing in source repo: $f"
    }
}

# Optional accelerator package (backends)
$accPkg = Join-Path $RepoRoot "accelerator"
if (Test-Path $accPkg) {
    Copy-Item $accPkg -Destination $AppDir -Recurse -Force
    Write-Ok "copied accelerator/ package"
}

# Tray launcher script (placed in $TrayDir for the Startup shortcut to invoke)
$trayPyDst = Join-Path $TrayDir "tray_app.py"
$trayPySrc = Join-Path $AppDir  "tray_app.py"
if (Test-Path $trayPySrc) {
    Copy-Item $trayPySrc -Destination $trayPyDst -Force
}
$companionDst = Join-Path $TrayDir "companion_client.py"
$companionSrc = Join-Path $AppDir  "companion_client.py"
if (Test-Path $companionSrc) {
    Copy-Item $companionSrc -Destination $companionDst -Force
}

# ---------------------------------------------------------------------------
# 4. Python venv + pip deps
# ---------------------------------------------------------------------------

Write-Step "4. Creating Python venv"

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
$venvPipw   = Join-Path $VenvDir "Scripts\pip.exe"
$venvPythonw= Join-Path $VenvDir "Scripts\pythonw.exe"

if (-not (Test-Path $venvPython)) {
    & py -3.12 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Err "venv creation failed"; exit 1 }
    Write-Ok "venv created at $VenvDir"
} else {
    Write-Ok "venv already present"
}

Write-Step "5. Upgrading pip + installing PyTorch (CUDA 12.8)"

& $venvPython -m pip install --upgrade pip wheel setuptools | Out-Host

# PyTorch must be installed FIRST from the cu128 index so pyannote doesn't
# later pull in a CPU-only wheel. See the backend design §6.1.
try {
    & $venvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pip install torch (cu128) failed" }

    $cudaOk = & $venvPython -c "import torch; print(torch.cuda.is_available())" 2>&1
    Write-Ok "torch.cuda.is_available() => $cudaOk"
} catch {
    Write-Err "PyTorch CUDA install failed: $_"
    exit 1
}

Write-Step "6. Installing application dependencies"

# Base deps from requirements.txt are mostly Mac (mlx-whisper). Install the
# Windows-relevant subset explicitly, leaving mlx-whisper out.
$winDeps = @(
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "python-multipart>=0.0.18",
    "aiosqlite>=0.20.0",
    "psutil>=5.9.0",
    "requests>=2.31.0",
    "faster-whisper>=1.0.0",
    "pystray>=0.19.0",
    "Pillow>=10.0.0"
)
& $venvPython -m pip install --upgrade-strategy only-if-needed @winDeps | Out-Host
if ($LASTEXITCODE -ne 0) { Write-Err "pip install failed"; exit 1 }
Write-Ok "core deps installed"

# Pyannote with --no-deps to preserve the CUDA torch wheel (see backend design §6.1)
try {
    & $venvPython -m pip install --no-deps "pyannote.audio>=3.1.0,<3.2" | Out-Host
    & $venvPython -m pip install --no-deps pyannote.core pyannote.database pyannote.metrics pyannote.pipeline | Out-Host
    # Other pyannote deps (avoid re-pulling torch)
    & $venvPython -m pip install --upgrade-strategy only-if-needed `
        asteroid-filterbanks einops huggingface_hub lightning omegaconf `
        sortedcontainers soundfile speechbrain tensorboardx `
        torch_audiomentations torchaudio torchmetrics | Out-Host

    $cudaStillOk = & $venvPython -c "import torch; print(torch.cuda.is_available())" 2>&1
    Write-Ok "pyannote installed; torch.cuda.is_available() => $cudaStillOk"
    if ("$cudaStillOk".Trim() -ne "True") {
        Write-Warn "CUDA torch wheel may have been overwritten. Reinstalling cu128 torch."
        & $venvPython -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch | Out-Host
    }
} catch {
    Write-Warn "pyannote install hit an issue: $_  (server will start; diarization may fail)"
}

# ---------------------------------------------------------------------------
# 7. NSSM download
# ---------------------------------------------------------------------------

Write-Step "7. Setting up NSSM"

$nssmExe = Join-Path $NssmDir "nssm.exe"
if (-not (Test-Path $nssmExe)) {
    $nssmZipUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $nssmZip    = Join-Path $env:TEMP "nssm-2.24.zip"
    try {
        Write-Ok "downloading nssm-2.24.zip ..."
        Invoke-WebRequest -Uri $nssmZipUrl -OutFile $nssmZip -UseBasicParsing
        $extractDir = Join-Path $env:TEMP "nssm-2.24-extract"
        if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
        Expand-Archive -Path $nssmZip -DestinationPath $extractDir -Force
        # Pick the 64-bit binary
        $nssmSrc = Get-ChildItem -Path $extractDir -Recurse -Filter "nssm.exe" |
                   Where-Object { $_.FullName -like "*win64*" } |
                   Select-Object -First 1
        if (-not $nssmSrc) {
            Write-Err "Could not locate win64\nssm.exe in $extractDir"
            exit 1
        }
        Copy-Item $nssmSrc.FullName -Destination $nssmExe -Force
        Write-Ok "nssm.exe installed at $nssmExe"
    } catch {
        Write-Err "NSSM download/extract failed: $_"
        exit 1
    }
} else {
    Write-Ok "nssm.exe already present"
}

# ---------------------------------------------------------------------------
# 8. config.env — interactive token setup (mirrors Mac install.sh prompts)
# ---------------------------------------------------------------------------

Write-Step "8. Configuring tokens and writing config.env"

$accToken = ""
$hfToken = ""

if (Test-Path $ConfigFile) {
    # ── Reinstall: show current tokens, offer to update ──────────────────
    Write-Host ""
    Write-Host "  config.env already exists." -ForegroundColor Cyan

    $tokenLine = Select-String -Path $ConfigFile -Pattern "^ACCELERATOR_TOKEN=" | Select-Object -First 1
    $accToken = if ($tokenLine) { ($tokenLine.Line -split "=", 2)[1].Trim() } else { "" }

    $hfLine = Select-String -Path $ConfigFile -Pattern "^HF_TOKEN=" | Select-Object -First 1
    $hfToken = if ($hfLine) { ($hfLine.Line -split "=", 2)[1].Trim() } else { "" }

    # Auth token
    Write-Host "  Current auth token: $accToken" -ForegroundColor Yellow
    Write-Host "    1) Keep existing token"
    Write-Host "    2) Enter a new token"
    Write-Host "    3) Auto-generate a new token"
    $tokenChoice = Read-Host "  Choice [1]"
    if (-not $tokenChoice) { $tokenChoice = "1" }
    switch ($tokenChoice) {
        "2" {
            $newToken = Read-Host "  Enter new auth token"
            if ($newToken) {
                $accToken = $newToken
                (Get-Content $ConfigFile) -replace "^ACCELERATOR_TOKEN=.*", "ACCELERATOR_TOKEN=$accToken" |
                    Set-Content $ConfigFile -Encoding UTF8
                Write-Ok "Auth token updated."
            } else {
                Write-Host "  No token entered — keeping existing." -ForegroundColor Yellow
            }
        }
        "3" {
            $accToken = New-RandomToken -Length 32
            (Get-Content $ConfigFile) -replace "^ACCELERATOR_TOKEN=.*", "ACCELERATOR_TOKEN=$accToken" |
                Set-Content $ConfigFile -Encoding UTF8
            Write-Ok "New auth token generated."
        }
        default { Write-Ok "Keeping existing auth token." }
    }

    # HF token
    if (-not $hfToken) {
        Write-Host ""
        Write-Host "  HuggingFace token is not set (diarization will not work)." -ForegroundColor Yellow
        Write-Host "  Get your token at: https://huggingface.co/settings/tokens"
        $newHf = Read-Host "  Enter HuggingFace token (or press Enter to skip)"
        if ($newHf) {
            $hfToken = $newHf
            (Get-Content $ConfigFile) -replace "^HF_TOKEN=.*", "HF_TOKEN=$hfToken" |
                Set-Content $ConfigFile -Encoding UTF8
            Write-Ok "HuggingFace token saved."
        } else {
            Write-Host "  Skipped — diarization will not work until HF_TOKEN is set in $ConfigFile" -ForegroundColor Yellow
        }
    } else {
        Write-Ok "HuggingFace token already set."
    }

} else {
    # ── Fresh install: prompt for both tokens ─────────────────────────────
    Write-Host ""
    Write-Host "  HuggingFace token is required for speaker diarization (pyannote models)."
    Write-Host "  Get yours at: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
    $hfToken = Read-Host "  Enter HuggingFace token (or press Enter to skip — diarization will not work)"
    if (-not $hfToken) {
        Write-Host "  Skipped — you can add HF_TOKEN later in $ConfigFile" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  Enter an auth token for API authentication (used by the pipeline to call this server)."
    $authInput = Read-Host "  Auth token (press Enter to auto-generate)"
    if ($authInput) {
        $accToken = $authInput
        Write-Ok "Using provided auth token."
    } else {
        $accToken = New-RandomToken -Length 32
        Write-Ok "Auto-generated auth token."
    }

    $cfg = @"
# Audio Chronicle Accelerator — Windows config
# Generated by install_windows.ps1
ACCELERATOR_URL=http://localhost:$Port
ACCELERATOR_PORT=$Port
ACCELERATOR_TOKEN=$accToken
INFERENCE_BACKEND=faster_whisper
WHISPER_MODEL=large-v3-turbo
DATA_DIR=$DataDir
LOG_LEVEL=INFO
HF_TOKEN=$hfToken

# Cache
CACHE_TTL_DAYS=30
CACHE_SWEEP_INTERVAL_MIN=60

# Job Queue
MAX_CONCURRENT_JOBS=3
MAX_QUEUE_DEPTH=20
"@
    Set-Content -Path $ConfigFile -Value $cfg -Encoding UTF8
    Write-Ok "config.env written."
}

# ---------------------------------------------------------------------------
# 9. NSSM service registration
# ---------------------------------------------------------------------------

Write-Step "9. Registering NSSM service '$ServiceName'"

# Detect existing service
$existing = & $nssmExe status $ServiceName 2>&1
if ($LASTEXITCODE -eq 0 -and $existing -notmatch "SERVICE_NOT_FOUND") {
    Write-Ok "service exists ($($existing.Trim())) — reconfiguring"
    try { & $nssmExe stop $ServiceName confirm 2>&1 | Out-Null } catch {}
} else {
    & $nssmExe install $ServiceName $venvPython "-m" "uvicorn" "server:app" "--host" "0.0.0.0" "--port" $Port 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Err "nssm install failed"; exit 1 }
    Write-Ok "service installed"
}

# Configure (always — keeps NSSM in sync with this script's params)
& $nssmExe set $ServiceName Application       $venvPython 2>&1 | Out-Null
& $nssmExe set $ServiceName AppParameters     "-m uvicorn server:app --host 0.0.0.0 --port $Port" 2>&1 | Out-Null
& $nssmExe set $ServiceName AppDirectory      $AppDir 2>&1 | Out-Null

# Environment block — NSSM AppEnvironmentExtra takes a NUL-delimited list, but
# the CLI accepts repeated KEY=VAL args. PowerShell handles the NUL conversion.
$envExtra = @(
    "DATA_DIR=$DataDir",
    "ACCELERATOR_TOKEN=$accToken",
    "HF_TOKEN=$hfToken",
    "WHISPER_MODEL=large-v3-turbo",
    "INFERENCE_BACKEND=faster_whisper",
    "LOG_LEVEL=INFO",
    "USERPROFILE=$env:USERPROFILE",
    "HOME=$env:USERPROFILE"
)
& $nssmExe set $ServiceName AppEnvironmentExtra @envExtra 2>&1 | Out-Null

# Log rotation via NSSM
$stdoutLog = Join-Path $LogsDir "stdout.log"
$stderrLog = Join-Path $LogsDir "stderr.log"
& $nssmExe set $ServiceName AppStdout $stdoutLog 2>&1 | Out-Null
& $nssmExe set $ServiceName AppStderr $stderrLog 2>&1 | Out-Null
& $nssmExe set $ServiceName AppRotateFiles 1 2>&1 | Out-Null
& $nssmExe set $ServiceName AppRotateBytes 10485760 2>&1 | Out-Null

# Graceful shutdown timings (see backend design §3.2)
& $nssmExe set $ServiceName AppStopMethodSkip    0     2>&1 | Out-Null
& $nssmExe set $ServiceName AppStopMethodConsole 15000 2>&1 | Out-Null
& $nssmExe set $ServiceName AppStopMethodWindow  5000  2>&1 | Out-Null
& $nssmExe set $ServiceName AppStopMethodThreads 5000  2>&1 | Out-Null

# Restart-on-crash
& $nssmExe set $ServiceName Start          SERVICE_AUTO_START 2>&1 | Out-Null
& $nssmExe set $ServiceName AppExit Default Restart           2>&1 | Out-Null
& $nssmExe set $ServiceName AppThrottle    10000              2>&1 | Out-Null

Write-Ok "NSSM service configured"

# ---------------------------------------------------------------------------
# 10. Firewall rule (requires admin)
# ---------------------------------------------------------------------------

if (-not $SkipFirewall) {
    Write-Step "10. Adding inbound firewall rule for TCP $Port"
    if (Test-Admin) {
        $ruleName = "Audio Accelerator (TCP $Port)"
        $existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
        if (-not $existingRule) {
            try {
                New-NetFirewallRule -DisplayName $ruleName `
                    -Direction Inbound -Protocol TCP -LocalPort $Port `
                    -Action Allow -Profile Private,Domain | Out-Null
                Write-Ok "firewall rule added"
            } catch {
                Write-Warn "firewall rule failed: $_"
            }
        } else {
            Write-Ok "firewall rule already present"
        }
    } else {
        Write-Warn "Not running as admin — skipping firewall rule. Re-run as Administrator,"
        Write-Warn "or open PowerShell as admin and run:"
        Write-Warn "  New-NetFirewallRule -DisplayName 'Audio Accelerator (TCP $Port)' \\"
        Write-Warn "    -Direction Inbound -Protocol TCP -LocalPort $Port \\"
        Write-Warn "    -Action Allow -Profile Private,Domain"
    }
} else {
    Write-Ok "skipping firewall rule (-SkipFirewall)"
}

# ---------------------------------------------------------------------------
# 11. Tray Startup shortcut
# ---------------------------------------------------------------------------

Write-Step "11. Creating tray Startup shortcut"

$startupDir = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir "AcceleratorTray.lnk"
try {
    $WshShell = New-Object -ComObject WScript.Shell
    $shortcut = $WshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $venvPythonw
    $shortcut.Arguments  = "`"$trayPyDst`""
    $shortcut.WorkingDirectory = $TrayDir
    $shortcut.IconLocation = "$venvPythonw,0"
    $shortcut.Description = "Audio Chronicle Accelerator tray icon"
    $shortcut.Save()
    Write-Ok "shortcut: $shortcutPath"
} catch {
    Write-Warn "could not create Startup shortcut: $_"
}

# ---------------------------------------------------------------------------
# 12. Model pre-download
# ---------------------------------------------------------------------------

if (-not $SkipModelPreload) {
    Write-Step "12. Pre-downloading Whisper model (large-v3-turbo)"
    try {
        & $venvPython -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')" | Out-Host
        Write-Ok "whisper model cached"
    } catch {
        Write-Warn "model preload failed (you can rerun later): $_"
    }
} else {
    Write-Ok "skipping model preload (-SkipModelPreload)"
}

# ---------------------------------------------------------------------------
# 13. Write uninstall script
# ---------------------------------------------------------------------------

Write-Step "13. Writing uninstall_windows.ps1"

$uninstallPath = Join-Path $InstallDir "uninstall_windows.ps1"
$uninstallContent = @"
# Auto-generated by install_windows.ps1
param([switch]`$KeepData)
`$ErrorActionPreference = "SilentlyContinue"
Write-Host "Stopping service '$ServiceName' ..."
& "$nssmExe" stop $ServiceName confirm | Out-Null
Write-Host "Removing service '$ServiceName' ..."
& "$nssmExe" remove $ServiceName confirm | Out-Null

Write-Host "Removing firewall rule ..."
Remove-NetFirewallRule -DisplayName "Audio Accelerator (TCP $Port)" -ErrorAction SilentlyContinue

Write-Host "Removing Startup shortcut ..."
Remove-Item -Path "$shortcutPath" -Force -ErrorAction SilentlyContinue

if (-not `$KeepData) {
    Write-Host "Removing install directory $InstallDir ..."
    Remove-Item -Recurse -Force "$InstallDir"
} else {
    Write-Host "Keeping data directory: $DataDir"
}
Write-Host "Uninstall complete."
"@
Set-Content -Path $uninstallPath -Value $uninstallContent -Encoding UTF8
Write-Ok "uninstall script: $uninstallPath"

# ---------------------------------------------------------------------------
# 14. Smoke test
# ---------------------------------------------------------------------------

Write-Step "14. Starting service + smoke test"

& $nssmExe start $ServiceName 2>&1 | Out-Null
$healthUrl = "http://localhost:$Port/health"
$ok = $false
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $ok = $true; break }
    } catch {
        Write-Host "    ... waiting for /health ($i/15)" -ForegroundColor DarkGray
    }
}
if ($ok) {
    Write-Ok "service responding at $healthUrl"
} else {
    Write-Warn "service did not respond within 30s — check $stderrLog"
}

# ---------------------------------------------------------------------------
# 15. Post-install banner
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Install complete." -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  URL          : http://localhost:$Port"
Write-Host "  Token        : (see $ConfigFile)"
Write-Host "  Logs         : $LogsDir"
Write-Host "  Uninstall    : powershell -File $uninstallPath"
Write-Host ""
Write-Host "  To stop the accelerator (frees VRAM for gaming):"
Write-Host "    - Tray icon → Stop  (recommended)"
Write-Host "    - PowerShell:  & '$nssmExe' stop $ServiceName"
Write-Host ""
Write-Host "  To start it again:"
Write-Host "    - Tray icon → Start"
Write-Host "    - PowerShell:  & '$nssmExe' start $ServiceName"
Write-Host ""
Write-Host "  Tray app will auto-launch at next login. To start it now:"
Write-Host "    & '$venvPythonw' '$trayPyDst'"
Write-Host ""
