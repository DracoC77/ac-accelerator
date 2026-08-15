#!/usr/bin/env bash
# Audio Chronicle Accelerator — Uninstaller
set -euo pipefail

# ─── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "\033[0;34m[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

INSTALL_DIR="${HOME}/.audio-chronicle-accelerator"
PLIST_LABEL="com.audio-chronicle-accelerator"
PLIST_PATH="${HOME}/Library/LaunchAgents/${PLIST_LABEL}.plist"

echo ""
echo -e "${BOLD}Audio Chronicle Accelerator — Uninstaller${NC}"
echo ""

# ─── Step 1: Stop and unload launchd service ───────────────────────────────
info "Stopping service..."

if launchctl list | grep -q "${PLIST_LABEL}" 2>/dev/null; then
    launchctl stop "${PLIST_LABEL}" 2>/dev/null || true
    launchctl unload "${PLIST_PATH}" 2>/dev/null || true
    success "Service stopped and unloaded"
else
    info "Service was not running"
fi

# ─── Step 2: Remove plist ──────────────────────────────────────────────────
if [[ -f "${PLIST_PATH}" ]]; then
    rm -f "${PLIST_PATH}"
    success "Removed ${PLIST_PATH}"
else
    info "Plist not found at ${PLIST_PATH} — skipping"
fi

# ─── Step 3: Remove application data ──────────────────────────────────────
if [[ -d "${INSTALL_DIR}" ]]; then
    echo ""
    echo -e "${YELLOW}${BOLD}⚠  WARNING: Data Removal${NC}"
    echo "The following directory contains your configuration, cached results, and logs:"
    echo "  ${INSTALL_DIR}"
    echo ""
    echo "Contents:"
    du -sh "${INSTALL_DIR}"/* 2>/dev/null | head -20 || true
    echo ""

    read -rp "Delete ${INSTALL_DIR} and all its contents? [y/N] " CONFIRM
    if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
        rm -rf "${INSTALL_DIR}"
        success "Removed ${INSTALL_DIR}"
    else
        info "Skipped — ${INSTALL_DIR} was NOT removed"
        echo ""
        echo "Your configuration and cached data are preserved at:"
        echo "  ${INSTALL_DIR}"
        echo ""
        echo "To reinstall later, run install.sh from the repository."
    fi
else
    info "${INSTALL_DIR} not found — nothing to remove"
fi

echo ""
echo -e "${GREEN}${BOLD}Uninstall complete.${NC}"
echo ""
