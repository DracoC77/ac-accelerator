#!/usr/bin/env bash
# Audio Chronicle Accelerator — Installer
# Supports: macOS 13+ (Ventura or later), Apple Silicon only
set -euo pipefail

# ─── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }
step()    { echo -e "\n${BOLD}==> $*${NC}"; }

# ─── Paths ─────────────────────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.audio-chronicle-accelerator"
VENV_DIR="${INSTALL_DIR}/venv"
CONFIG_FILE="${INSTALL_DIR}/config.env"
LOG_DIR="${INSTALL_DIR}/logs"
DATA_DIR="${INSTALL_DIR}/data"
UPLOAD_DIR="${DATA_DIR}/uploads"
CACHE_DIR="${INSTALL_DIR}/cache"
PLIST_LABEL="com.audio-chronicle-accelerator"
PLIST_PATH="${HOME}/Library/LaunchAgents/${PLIST_LABEL}.plist"
PORT="${ACCELERATOR_PORT:-8765}"

# ─── Step 1: Prerequisites ─────────────────────────────────────────────────
step "Checking prerequisites"

# macOS only
if [[ "$(uname -s)" != "Darwin" ]]; then
    die "This installer requires macOS. Current OS: $(uname -s)"
fi

# Apple Silicon (arm64)
ARCH="$(uname -m)"
if [[ "${ARCH}" != "arm64" ]]; then
    die "Apple Silicon (arm64) required. Detected architecture: ${ARCH}"
fi
success "Apple Silicon confirmed"

# macOS version ≥ 13
OS_VER="$(sw_vers -productVersion)"
OS_MAJOR="$(echo "${OS_VER}" | cut -d. -f1)"
if [[ "${OS_MAJOR}" -lt 13 ]]; then
    die "macOS 13 (Ventura) or later required. Detected: ${OS_VER}"
fi
success "macOS ${OS_VER}"

# Python 3.10+
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" &>/dev/null; then
        PY_VER="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        PY_MAJOR="${PY_VER%%.*}"
        PY_MINOR="${PY_VER#*.}"
        if [[ "${PY_MAJOR}" -gt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -ge 10 ]]; }; then
            PYTHON="${candidate}"
            break
        fi
    fi
done

if [[ -z "${PYTHON}" ]]; then
    die "Python 3.10 or later is required but not found.\nInstall via: brew install python@3.12\nOr download from: https://www.python.org/downloads/"
fi
success "Python ${PY_VER} (${PYTHON})"

# Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    die "Xcode Command Line Tools are required.\nInstall with: xcode-select --install"
fi
success "Xcode CLT installed"

# ffmpeg (required for audio format conversion)
if ! command -v ffmpeg &>/dev/null; then
    die "ffmpeg is required but not found.\nInstall with: brew install ffmpeg\nThen re-run this installer."
fi
FFMPEG_VER="$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
success "ffmpeg ${FFMPEG_VER}"

# Check if we're running from a git repo (for server.py / requirements.txt)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAS_REPO_FILES=false
if [[ -f "${SCRIPT_DIR}/server.py" ]] && [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    HAS_REPO_FILES=true
fi

# ─── Step 2: Create directory structure ────────────────────────────────────
step "Creating directory structure"

mkdir -p "${INSTALL_DIR}" "${VENV_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${UPLOAD_DIR}" "${CACHE_DIR}"
success "Created ${INSTALL_DIR}/"

# ─── Step 3: Python virtual environment ────────────────────────────────────
step "Setting up Python virtual environment"

if [[ -d "${VENV_DIR}/bin" ]]; then
    info "Existing venv found at ${VENV_DIR} — reusing"
else
    "${PYTHON}" -m venv "${VENV_DIR}"
    success "Created venv at ${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# Upgrade pip silently
"${VENV_PYTHON}" -m pip install --upgrade pip --quiet

# ─── Step 4: Install dependencies ──────────────────────────────────────────
step "Installing dependencies"

if [[ "${HAS_REPO_FILES}" == true ]]; then
    REQ_FILE="${SCRIPT_DIR}/requirements.txt"
elif [[ -f "${INSTALL_DIR}/requirements.txt" ]]; then
    REQ_FILE="${INSTALL_DIR}/requirements.txt"
else
    die "requirements.txt not found. Please run this installer from the cloned repository directory."
fi

info "Installing from ${REQ_FILE} (this may take several minutes)..."
"${VENV_PIP}" install -r "${REQ_FILE}" --quiet
success "Dependencies installed"

# ─── Step 5: Copy server.py ────────────────────────────────────────────────
step "Installing application files"

if [[ "${HAS_REPO_FILES}" == true ]]; then
    # Always overwrite server.py — it is managed application code, not user-editable.
    # The "skip if exists" guard must NOT apply here; without overwriting, bug fixes
    # and upgrades in server.py never reach an existing install.
    cp "${SCRIPT_DIR}/server.py" "${INSTALL_DIR}/server.py"
    success "Copied server.py to ${INSTALL_DIR}/"
    # Copy the accelerator/ package alongside server.py.
    # server.py imports from accelerator.backends at module level; without this
    # the service fails with ModuleNotFoundError: No module named 'accelerator'.
    # rm -rf first ("managed code, fully overwritten" philosophy):
    # `cp -r src dst` nests src inside dst when dst already exists, so a re-run
    # would produce accelerator/accelerator/ and silently load stale code instead.
    if [[ -d "${SCRIPT_DIR}/accelerator" ]]; then
        rm -rf "${INSTALL_DIR}/accelerator"
        cp -r "${SCRIPT_DIR}/accelerator" "${INSTALL_DIR}/accelerator"
        success "Copied accelerator/ package to ${INSTALL_DIR}/"
    fi
elif [[ -f "${INSTALL_DIR}/server.py" ]]; then
    # Running from outside the repo (no server.py next to install.sh).
    # An existing copy is present — nothing to do, but warn that the user
    # should re-run from the cloned repo to pick up any updates.
    warn "Running outside the repo: server.py not updated (re-run install.sh from the cloned repo to update)"
else
    die "server.py not found. Please run this installer from the cloned repository directory."
fi

# ─── Step 6: Generate configuration (.env) ─────────────────────────────────
step "Configuring environment"

if [[ -f "${CONFIG_FILE}" ]]; then
    # ── Reinstall: show current token and offer to update ──────────────────
    echo ""
    echo -e "${BOLD}Existing config found at ${CONFIG_FILE}${NC}"
    echo ""
    CURRENT_TOKEN="$(grep '^ACCELERATOR_TOKEN=' "${CONFIG_FILE}" | cut -d= -f2)"
    echo -e "Current auth token: ${CURRENT_TOKEN}"
    echo ""
    echo "Options:"
    echo "  1) Keep existing token"
    echo "  2) Enter a new token"
    echo "  3) Auto-generate a new token"
    echo ""
    read -rp "Choice [1]: " TOKEN_CHOICE
    TOKEN_CHOICE="${TOKEN_CHOICE:-1}"

    case "${TOKEN_CHOICE}" in
        2)
            echo ""
            read -rp "Enter new auth token: " NEW_TOKEN
            if [[ -z "${NEW_TOKEN}" ]]; then
                warn "No token entered — keeping existing token."
            else
                sed -i '' "s|^ACCELERATOR_TOKEN=.*|ACCELERATOR_TOKEN=${NEW_TOKEN}|" "${CONFIG_FILE}"
                success "Auth token updated."
            fi
            ;;
        3)
            NEW_TOKEN="$(${VENV_PYTHON} -c "import secrets; print(secrets.token_hex(32))")"
            sed -i '' "s|^ACCELERATOR_TOKEN=.*|ACCELERATOR_TOKEN=${NEW_TOKEN}|" "${CONFIG_FILE}"
            success "New auth token generated and saved."
            ;;
        *)
            info "Keeping existing token."
            ;;
    esac
else
    # ── Fresh install: prompt for token or auto-generate ───────────────────
    echo ""
    echo -e "${BOLD}Hugging Face Token (required for pyannote diarization model)${NC}"
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo "You must also accept the pyannote model terms at:"
    echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo ""
    read -rp "Enter your Hugging Face token (or press Enter to skip — diarization will not work): " HF_TOKEN
    HF_TOKEN="${HF_TOKEN:-}"

    echo ""
    echo -e "${BOLD}Auth Token Setup${NC}"
    echo "────────────────"
    read -rp "Enter an auth token for API authentication (press Enter to auto-generate): " AUTH_INPUT
    if [[ -n "${AUTH_INPUT}" ]]; then
        ACCELERATOR_TOKEN="${AUTH_INPUT}"
    else
        ACCELERATOR_TOKEN="$(${VENV_PYTHON} -c "import secrets; print(secrets.token_hex(32))")"
        info "Auto-generated auth token."
    fi

    cat > "${CONFIG_FILE}" << EOF
# Audio Chronicle Accelerator — Runtime Configuration
# Generated by install.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ── Server ──────────────────────────────────────────────────────────────────
ACCELERATOR_PORT=${PORT}
ACCELERATOR_HOST=0.0.0.0

# ── Authentication ──────────────────────────────────────────────────────────
ACCELERATOR_TOKEN=${ACCELERATOR_TOKEN}

# ── Models ──────────────────────────────────────────────────────────────────
WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
WHISPER_DEVICE=auto
DIARIZE_MODEL=pyannote/speaker-diarization-3.1
HF_TOKEN=${HF_TOKEN}

# ── Job Queue ───────────────────────────────────────────────────────────────
MAX_CONCURRENT_JOBS=2
MAX_QUEUE_DEPTH=20
# Match server.py default (2048). Previously hardcoded 500 here,
# which silently re-clobbered the large-file cap on every reinstall.
MAX_FILE_SIZE_MB=2048

# ── Cache ───────────────────────────────────────────────────────────────────
CACHE_TTL_DAYS=30
CACHE_SWEEP_INTERVAL_MIN=60

# ── Limits ──────────────────────────────────────────────────────────────────
RATE_LIMIT_PER_MIN=60

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FORMAT=text

# ── Storage ─────────────────────────────────────────────────────────────────
DB_PATH=data/jobs.db
UPLOAD_DIR=data/uploads

# ── Startup ─────────────────────────────────────────────────────────────────
PRELOAD_MODELS=true
EOF

    chmod 600 "${CONFIG_FILE}"
    success "Configuration written to ${CONFIG_FILE}"

    if [[ -z "${HF_TOKEN}" ]]; then
        warn "No Hugging Face token provided. Speaker diarization will not work until HF_TOKEN is set in ${CONFIG_FILE}"
    fi
fi

# ─── Step 7: Install start.sh wrapper ──────────────────────────────────────
step "Installing service wrapper scripts"

START_SCRIPT="${INSTALL_DIR}/start.sh"
cat > "${START_SCRIPT}" << EOF
#!/usr/bin/env bash
# Audio Chronicle Accelerator — Service start wrapper
# Sourced by launchd. Loads config.env before starting the server.
set -euo pipefail

# Ensure Homebrew binaries (including ffmpeg) are in PATH for launchd
# launchd starts with a minimal PATH that excludes /opt/homebrew/bin
export PATH="/opt/homebrew/bin:/usr/local/bin:\${PATH}"

INSTALL_DIR="\${HOME}/.audio-chronicle-accelerator"

# Load configuration
if [[ -f "\${INSTALL_DIR}/config.env" ]]; then
    set -a
    source "\${INSTALL_DIR}/config.env"
    set +a
fi

exec "\${INSTALL_DIR}/venv/bin/python" -m uvicorn server:app \\
    --host "\${ACCELERATOR_HOST:-0.0.0.0}" \\
    --port "\${ACCELERATOR_PORT:-${PORT}}" \\
    --log-level "\$(echo "\${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')"
EOF
chmod +x "${START_SCRIPT}"
success "Installed start.sh"

# ─── Step 8: Install launchd plist ─────────────────────────────────────────
step "Installing launchd service"

mkdir -p "${HOME}/Library/LaunchAgents"

cat > "${PLIST_PATH}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${INSTALL_DIR}/start.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>

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
    <string>${LOG_DIR}/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>

    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>4096</integer>
    </dict>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

success "Installed plist to ${PLIST_PATH}"

# ─── Step 9: Load and start the service ────────────────────────────────────
step "Starting service"

# Unload if already loaded (update case)
if launchctl list | grep -q "${PLIST_LABEL}" 2>/dev/null; then
    info "Service already loaded — reloading..."
    launchctl unload "${PLIST_PATH}" 2>/dev/null || true
fi

launchctl load "${PLIST_PATH}"
success "Service loaded"

# Wait up to 15 seconds for /health to return 200
info "Waiting for service to become healthy (up to 15s)..."
HEALTH_OK=false
for i in $(seq 1 15); do
    if curl -sf --max-time 2 "http://localhost:${PORT}/health" &>/dev/null; then
        HEALTH_OK=true
        break
    fi
    sleep 1
done

if [[ "${HEALTH_OK}" == true ]]; then
    HEALTH_JSON="$(curl -sf --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo '{}')"
    GPU_STATUS="$(echo "${HEALTH_JSON}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Metal GPU: ' + str(d.get('gpu',{}).get('metal_available','?')))" 2>/dev/null || echo "")"
    success "Service is healthy! ${GPU_STATUS}"
else
    warn "Service did not become healthy within 15 seconds."
    warn "Check logs: tail -f ${LOG_DIR}/stderr.log"
fi

# ─── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Audio Chronicle Accelerator — Installation Complete${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo ""
if [[ -f "${CONFIG_FILE}" ]]; then
    TOKEN_VAL="$(grep '^ACCELERATOR_TOKEN=' "${CONFIG_FILE}" | cut -d= -f2)"
    echo -e "${BOLD}Your API auth token (copy this to ACCELERATOR_TOKEN in your client's config):${NC}"
    echo "  ┌─────────────────────────────────────────────────────────┐"
    printf  "  │  %-55s│\n" "${TOKEN_VAL}"
    echo "  └─────────────────────────────────────────────────────────┘"
    echo ""
fi
echo -e "${BOLD}Service management:${NC}"
echo "  Stop:            launchctl stop ${PLIST_LABEL}"
echo "  Start:           launchctl start ${PLIST_LABEL}"
echo "  Disable autostart: launchctl unload ${PLIST_PATH}"
echo "  Re-enable:       launchctl load ${PLIST_PATH}"
echo ""
echo -e "${BOLD}Logs:${NC}"
echo "  tail -f ${LOG_DIR}/stdout.log"
echo "  tail -f ${LOG_DIR}/stderr.log"
echo ""
echo -e "${BOLD}Health check:${NC}"
echo "  curl http://localhost:${PORT}/health"
echo ""
echo -e "${BOLD}Config:${NC}"
echo "  ${CONFIG_FILE}"
echo ""
if [[ -f "${CONFIG_FILE}" ]] && grep -q '^HF_TOKEN=$' "${CONFIG_FILE}" 2>/dev/null; then
    echo -e "${YELLOW}⚠  Reminder: Set HF_TOKEN in ${CONFIG_FILE} to enable diarization${NC}"
    echo ""
fi
echo "To uninstall, run: ./uninstall.sh"
echo ""

