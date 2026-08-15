#!/usr/bin/env bash
# Audio Chronicle Accelerator — Service start wrapper
#
# Called by launchd. Sources config.env to load all environment variables
# (HF_TOKEN, AUTH_TOKEN, ACCELERATOR_PORT, etc.) before starting the server.
#
# launchd cannot source shell files or expand ~ in EnvironmentVariables,
# so this wrapper handles both concerns.
set -euo pipefail

# Ensure Homebrew binaries (including ffmpeg) are in PATH for launchd
# launchd starts with a minimal PATH that excludes /opt/homebrew/bin
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

INSTALL_DIR="${HOME}/.audio-chronicle-accelerator"

# Load runtime configuration (all key=value pairs exported to environment)
if [[ -f "${INSTALL_DIR}/config.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${INSTALL_DIR}/config.env"
    set +a
fi

# Exec uvicorn (replaces this shell — launchd tracks the server PID directly)
exec "${INSTALL_DIR}/venv/bin/python" -m uvicorn server:app \
    --host "${ACCELERATOR_HOST:-0.0.0.0}" \
    --port "${ACCELERATOR_PORT:-8765}" \
    --log-level "$(echo "${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')"
