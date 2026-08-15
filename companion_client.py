"""
companion_client.py — shared server polling and control layer for menubar/tray apps.

Used by:
  - menubar_app.py  (macOS, rumps)        — not yet refactored to use this module
  - tray_app.py     (Windows, pystray)    — uses this module directly

This module abstracts:
  - HTTP polling of the accelerator service (``/health``, ``/memory``, ``/logs``)
  - Service lifecycle control (start/stop/restart) dispatched per platform:
      * macOS  → ``launchctl`` against the launchd plist label
      * Windows → ``nssm.exe`` against the NSSM-managed service name

Design notes
------------
- Pure stdlib + ``requests`` only. No GUI / tray-toolkit dependencies, so the
  module can be imported and unit-tested without a display.
- All network and subprocess errors are caught and converted to a structured
  ``AcceleratorStatus(reachable=False, ...)`` or a boolean False. Callers
  should never see exceptions during normal poll loops.
- The Mac ``menubar_app.py`` already contains the same control logic inline.
  Extracting it here is purely additive for this WI; the Mac app keeps its
  current behavior. A future WI can refactor ``menubar_app.py`` to import
  these helpers.

WI: WI-ACC-27d (Windows packaging)
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger("accelerator.companion_client")

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8765"
DEFAULT_TIMEOUT = 2.0  # seconds, for /health polling

# Service identifiers per platform.
# These match install_windows.ps1 (NSSM service name) and
# com.audio-chronicle-accelerator.plist (launchd label) respectively.
NSSM_SERVICE_NAME = os.environ.get("ACCELERATOR_SERVICE_NAME", "AccServer")
LAUNCHD_PLIST_LABEL = os.environ.get(
    "ACCELERATOR_PLIST_LABEL", "com.audio-chronicle-accelerator"
)


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------


@dataclass
class AcceleratorStatus:
    """Snapshot of the accelerator server's state at a single poll moment.

    All fields are best-effort: if a field could not be determined (e.g. the
    server was unreachable, or the endpoint returned an unexpected shape),
    a sensible default is used. The ``reachable`` flag is the canonical signal
    for "is the server alive at all".
    """

    reachable: bool = False
    healthy: bool = False
    processing_count: int = 0
    pending_count: int = 0
    whisper_loaded: bool = False
    pyannote_loaded: bool = False
    uptime_seconds: float = 0.0
    rss_mb: float = 0.0
    backend_name: str = ""
    local_url: str = ""
    raw_health: dict = field(default_factory=dict)
    raw_memory: dict = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# HTTP polling
# ---------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    """Read the accelerator bearer token from env, if present."""
    token = os.environ.get("ACCELERATOR_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def get_status(
    base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT
) -> AcceleratorStatus:
    """Poll ``/health`` and ``/memory`` and return an ``AcceleratorStatus``.

    Never raises. On any error (connection refused, timeout, malformed JSON)
    the returned status has ``reachable=False`` and ``error`` populated.

    The ``/memory`` endpoint requires the bearer token; if no token is set we
    still get a useful status from ``/health`` (which is unauthenticated).
    """
    base_url = base_url.rstrip("/")
    status = AcceleratorStatus(local_url=base_url)

    # --- /health (unauthenticated) ---------------------------------------
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        status.error = f"connection refused: {exc}"
        return status
    except requests.exceptions.Timeout as exc:
        status.error = f"timeout: {exc}"
        return status
    except Exception as exc:  # noqa: BLE001 — companion polling must never raise
        status.error = f"unexpected error: {exc}"
        return status

    if resp.status_code != 200:
        status.error = f"/health returned HTTP {resp.status_code}"
        return status

    try:
        payload = resp.json()
    except ValueError as exc:
        status.error = f"invalid health JSON: {exc}"
        return status

    status.reachable = True
    status.raw_health = payload
    status.healthy = str(payload.get("status", "")).lower() in ("ok", "healthy")

    # Queue counts: server uses {"queue": {"pending": N, "processing": N}};
    # earlier drafts used flat keys. Accept both for robustness.
    queue = payload.get("queue") or {}
    status.processing_count = int(
        queue.get("processing", payload.get("processing", 0)) or 0
    )
    status.pending_count = int(queue.get("pending", payload.get("pending", 0)) or 0)

    # Model load state: nested under "models" in current server.py, but the
    # WI-ACC-27d spec example uses top-level keys. Accept both.
    models = payload.get("models") or {}
    status.whisper_loaded = bool(
        models.get("whisper_loaded", payload.get("whisper_loaded", False))
    )
    status.pyannote_loaded = bool(
        models.get("diarization_loaded", payload.get("pyannote_loaded", False))
    )

    status.uptime_seconds = float(payload.get("uptime_seconds", 0) or 0)
    status.backend_name = str(
        payload.get("backend_name") or models.get("backend_name") or ""
    )

    # --- /memory (authenticated) -----------------------------------------
    try:
        mem_resp = requests.get(
            f"{base_url}/memory", timeout=timeout, headers=_auth_headers()
        )
        if mem_resp.status_code == 200:
            mem = mem_resp.json()
            status.raw_memory = mem
            status.rss_mb = float(mem.get("rss_mb", 0) or 0)
    except Exception as exc:  # noqa: BLE001 — memory is optional context
        logger.debug("companion_client: /memory fetch failed: %s", exc)

    return status


def get_logs(
    base_url: str = DEFAULT_BASE_URL,
    n: int = 200,
    source: str = "companion",
    timeout: float = 5.0,
) -> str:
    """GET ``/logs?source={source}&lines={n}&format=text`` and return the text.

    ``source`` defaults to ``"companion"`` — the cross-platform name for the
    menubar/tray app log. The server accepts ``server``, ``menubar``,
    ``companion``, or ``all`` (per the WI-ACC-27 §2.4 alias).

    Returns the response body verbatim on 2xx, or a short ``"(error: ...)"``
    string on failure. Never raises.
    """
    base_url = base_url.rstrip("/")
    try:
        resp = requests.get(
            f"{base_url}/logs",
            params={"source": source, "lines": n, "format": "text"},
            headers=_auth_headers(),
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.text
        return f"(error: /logs returned HTTP {resp.status_code})\n{resp.text[:500]}"
    except Exception as exc:  # noqa: BLE001 — log retrieval must never raise
        return f"(error fetching logs: {exc})"


# ---------------------------------------------------------------------------
# Service control — platform dispatched
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 30.0) -> bool:
    """Run a subprocess command, return True on exit code 0."""
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if completed.returncode != 0:
            logger.warning(
                "companion_client: %s exited %d; stderr=%s",
                cmd[0], completed.returncode, completed.stderr.strip()[:300],
            )
            return False
        return True
    except FileNotFoundError as exc:
        logger.warning("companion_client: command not found: %s (%s)", cmd[0], exc)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("companion_client: command timed out: %s", cmd)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("companion_client: command failed: %s (%s)", cmd, exc)
        return False


def _system() -> str:
    """Return the platform name. Wrapped so tests can monkeypatch."""
    return platform.system()


# ---- macOS: launchctl -------------------------------------------------------


def _mac_kickstart(restart: bool = False) -> bool:
    """``launchctl kickstart [-k] gui/<uid>/<label>`` for the user agent."""
    label = LAUNCHD_PLIST_LABEL
    uid = os.getuid() if hasattr(os, "getuid") else 0
    cmd = ["launchctl", "kickstart"]
    if restart:
        cmd.append("-k")
    cmd.append(f"gui/{uid}/{label}")
    return _run(cmd)


def _mac_stop() -> bool:
    """``launchctl stop <label>`` — launchd will not auto-restart it."""
    return _run(["launchctl", "stop", LAUNCHD_PLIST_LABEL])


# ---- Windows: nssm ----------------------------------------------------------


def _nssm_path() -> str:
    """Resolve the nssm executable. ``ACCELERATOR_NSSM_PATH`` env var wins."""
    return os.environ.get("ACCELERATOR_NSSM_PATH", "nssm.exe")


def _win_nssm(action: str) -> bool:
    """Run ``nssm.exe <action> <service>``. Action ∈ {start, stop, restart}."""
    return _run([_nssm_path(), action, NSSM_SERVICE_NAME])


# ---- Public API -------------------------------------------------------------


def start_service() -> bool:
    """Start the accelerator service. Returns True on success."""
    system = _system()
    if system == "Darwin":
        return _mac_kickstart(restart=False)
    if system == "Windows":
        return _win_nssm("start")
    logger.warning("companion_client.start_service(): unsupported platform %s", system)
    return False


def stop_service() -> bool:
    """Stop the accelerator service. Returns True on success."""
    system = _system()
    if system == "Darwin":
        return _mac_stop()
    if system == "Windows":
        return _win_nssm("stop")
    logger.warning("companion_client.stop_service(): unsupported platform %s", system)
    return False


def restart_service() -> bool:
    """Restart the accelerator service atomically. Returns True on success."""
    system = _system()
    if system == "Darwin":
        # ``launchctl kickstart -k`` is an atomic kill+restart, matching the
        # behavior used by menubar_app.py._on_restart.
        return _mac_kickstart(restart=True)
    if system == "Windows":
        return _win_nssm("restart")
    logger.warning(
        "companion_client.restart_service(): unsupported platform %s", system
    )
    return False


__all__ = [
    "AcceleratorStatus",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "LAUNCHD_PLIST_LABEL",
    "NSSM_SERVICE_NAME",
    "get_status",
    "get_logs",
    "start_service",
    "stop_service",
    "restart_service",
]
