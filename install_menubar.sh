#!/usr/bin/env bash
# Audio Chronicle Accelerator — Menu Bar App Installer
# Installs menubar_app.py and registers it as a login item via launchd.
# Run after install.sh has set up the main accelerator service.
set -euo pipefail

# ─── Colors ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}==> $*${NC}"; }

# ─── Paths ─────────────────────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.audio-chronicle-accelerator"
VENV_DIR="${INSTALL_DIR}/venv"
LOG_DIR="${INSTALL_DIR}/logs"
MENUBAR_PLIST_LABEL="com.audio-chronicle-accelerator-menubar"
MENUBAR_PLIST_PATH="${HOME}/Library/LaunchAgents/${MENUBAR_PLIST_LABEL}.plist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Step 1: Check prerequisites ───────────────────────────────────────────
step "Checking prerequisites"

if [[ ! -d "${VENV_DIR}" ]]; then
    die "Accelerator venv not found at ${VENV_DIR}. Run install.sh first."
fi
success "Accelerator venv found"

if [[ ! -f "${SCRIPT_DIR}/menubar_app.py" ]]; then
    die "menubar_app.py not found in ${SCRIPT_DIR}. Run from the repo directory."
fi

# ─── Step 2: Install menubar dependencies ──────────────────────────────────
step "Installing menu bar dependencies (rumps, psutil, requests)"

"${VENV_DIR}/bin/pip" install rumps psutil requests --quiet
success "Dependencies installed"

# ─── Step 3: Copy menubar_app.py ───────────────────────────────────────────
step "Copying menubar_app.py"

cp "${SCRIPT_DIR}/menubar_app.py" "${INSTALL_DIR}/menubar_app.py"
success "Copied to ${INSTALL_DIR}/menubar_app.py"

# ─── Step 4: Install launchd plist ─────────────────────────────────────────
step "Installing launchd plist (login item)"

mkdir -p "${HOME}/Library/LaunchAgents" "${LOG_DIR}"

cat > "${MENUBAR_PLIST_PATH}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${MENUBAR_PLIST_LABEL}</string>

    <key>LimitLoadToSessionType</key>
    <string>Aqua</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>exec "${VENV_DIR}/bin/python" "${INSTALL_DIR}/menubar_app.py"</string>
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
    <string>${LOG_DIR}/menubar_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/menubar_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

success "Installed plist to ${MENUBAR_PLIST_PATH}"

# ─── Step 5: Load the login item ───────────────────────────────────────────
step "Registering menu bar app as login item"

# Unload first if already registered (update case)
if launchctl list | grep -q "${MENUBAR_PLIST_LABEL}" 2>/dev/null; then
    info "Already loaded — reloading..."
    launchctl unload "${MENUBAR_PLIST_PATH}" 2>/dev/null || true
fi

launchctl load "${MENUBAR_PLIST_PATH}"
success "Menu bar app registered"

# ─── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Menu Bar App — Installation Complete${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo "The ⚡ icon will appear in your menu bar on next login,"
echo "or launch it now with:"
echo ""
echo "  ${VENV_DIR}/bin/python ${INSTALL_DIR}/menubar_app.py &"
echo ""
echo -e "${BOLD}To uninstall the menu bar app:${NC}"
echo "  launchctl unload ${MENUBAR_PLIST_PATH}"
echo "  rm ${MENUBAR_PLIST_PATH}"
echo ""
