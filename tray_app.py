"""
tray_app.py — Windows system tray app for the Audio Chronicle Accelerator.

Uses ``pystray`` (cross-platform tray library) + ``Pillow`` for icon rendering.
Runs as a separate user-session process (NOT a Windows service). The NSSM
service ``AccServer`` owns the uvicorn lifecycle; this tray app only:

  - polls ``GET /health`` every 5 s and updates the tray icon glyph + tooltip
  - shells out to ``nssm.exe start|stop|restart AccServer`` for control
  - opens the logs folder in Explorer

It is started by a per-user Startup folder shortcut written by
``install_windows.ps1`` (step 10 in §4.2 of the WI spec). The tray app is
intentionally minimal — see WI-ACC-27 §3.4 for the scope decisions.

The ``pystray`` / ``PIL`` imports are guarded so this module remains
importable on CI machines (Linux, no display) for at least a syntax check
and the companion_client integration test.

WI: WI-ACC-27d (Windows packaging)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import companion_client

# ---------------------------------------------------------------------------
# Optional GUI imports
# ---------------------------------------------------------------------------
#
# pystray + PIL require a display (Quartz on macOS, GDI on Windows, X11 on
# Linux). Guard the import so unit tests / `python -c "import tray_app"` on
# headless CI hosts still work — only AcceleratorTray().run() will fail
# without these libraries available, and the failure message is explicit.

try:
    import pystray  # type: ignore
    from PIL import Image, ImageDraw  # type: ignore

    _GUI_AVAILABLE = True
    _GUI_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 — we want to capture *any* import failure
    pystray = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    _GUI_AVAILABLE = False
    _GUI_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("ACCELERATOR_URL", "http://localhost:8765").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.environ.get("ACCELERATOR_POLL_INTERVAL", "5"))

# Where the server writes its rotating log files on Windows. install_windows.ps1
# sets DATA_DIR to %USERPROFILE%\.audio-accelerator\data and the server writes
# logs under data\logs\.
DATA_DIR = Path(
    os.environ.get("DATA_DIR", os.path.expanduser("~/.audio-accelerator/data"))
)
LOGS_DIR = DATA_DIR / "logs"

# Where this tray app writes its own log. Kept separate from server logs so
# the tray's polling chatter doesn't interleave with the inference output.
TRAY_LOG_FILE = LOGS_DIR / "tray.log"

# Icon colours for each state. Tuples are (R, G, B). Alpha is added later.
ICON_COLORS = {
    "healthy": (76, 175, 80),     # green
    "processing": (255, 193, 7),  # amber
    "idle": (158, 158, 158),      # grey  (server up but model not loaded)
    "offline": (244, 67, 54),     # red
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("accelerator.tray")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        import logging.handlers as _lh
        fh = _lh.RotatingFileHandler(
            TRAY_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        log.addHandler(fh)
    except Exception:  # noqa: BLE001 — never let logging setup crash the tray
        pass
    return log


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Icon rendering
# ---------------------------------------------------------------------------


def _make_icon(state: str):
    """Create a 64x64 filled-circle icon for the given state.

    Returns a PIL Image (when pystray is available), or ``None`` otherwise
    so the caller can fail gracefully on headless hosts.
    """
    if not _GUI_AVAILABLE:
        return None
    color = ICON_COLORS.get(state, ICON_COLORS["offline"])
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(*color, 255))
    return img


def _classify(status: companion_client.AcceleratorStatus) -> tuple[str, str]:
    """Map an AcceleratorStatus to (icon_state, tooltip)."""
    if not status.reachable:
        return "offline", "Accelerator: Offline (service stopped or unreachable)"
    if status.processing_count > 0:
        return (
            "processing",
            f"Accelerator: Processing ({status.processing_count} job"
            f"{'s' if status.processing_count != 1 else ''})",
        )
    if status.whisper_loaded:
        uptime_min = int(status.uptime_seconds // 60)
        backend = f" · {status.backend_name}" if status.backend_name else ""
        return "healthy", f"Accelerator: Ready · up {uptime_min}m{backend}"
    return "idle", "Accelerator: Idle (model not loaded)"


# ---------------------------------------------------------------------------
# Tray app
# ---------------------------------------------------------------------------


class AcceleratorTray:
    """Minimal Windows tray icon for controlling the accelerator service.

    Architecture:
      - One background thread polls /health every ``POLL_INTERVAL_SECONDS``.
      - The icon glyph + tooltip are refreshed in-place on each poll.
      - Menu callbacks shell out via ``companion_client`` (NSSM on Windows).
      - The menu items are *labels* generated from the latest status snapshot,
        so the user always sees up-to-date state when they open the menu.
    """

    def __init__(self) -> None:
        self._status: Optional[companion_client.AcceleratorStatus] = None
        self._icon = None  # type: ignore[assignment]
        self._running = True
        self._poll_thread: Optional[threading.Thread] = None

    # ── Polling ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: refresh status every POLL_INTERVAL_SECONDS."""
        while self._running:
            try:
                self._status = companion_client.get_status(BASE_URL)
                self._refresh_icon()
            except Exception as exc:  # noqa: BLE001
                logger.warning("poll loop: %s", exc)
            # Sleep in small slices so quit is responsive.
            for _ in range(int(max(POLL_INTERVAL_SECONDS, 0.1) * 10)):
                if not self._running:
                    return
                time.sleep(0.1)

    def _refresh_icon(self) -> None:
        if self._icon is None or self._status is None:
            return
        state, tooltip = _classify(self._status)
        new_icon = _make_icon(state)
        if new_icon is not None:
            self._icon.icon = new_icon
        self._icon.title = tooltip

    # ── Menu ────────────────────────────────────────────────────────────────

    def _status_label(self, _item=None) -> str:
        """Top-of-menu read-only status row."""
        if self._status is None or not self._status.reachable:
            return "Status: ● Offline"
        if self._status.processing_count > 0:
            return f"Status: ● Processing ({self._status.processing_count})"
        if self._status.whisper_loaded:
            return f"Status: ● Healthy · {self._status.backend_name or 'ready'}"
        return "Status: ● Idle (model not loaded)"

    def _build_menu(self):
        """Build the pystray Menu. Lambdas capture `self` for callbacks."""
        if not _GUI_AVAILABLE:
            return None
        return pystray.Menu(
            pystray.MenuItem(self._status_label, action=None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("▶ Start", self._on_start),
            pystray.MenuItem("■ Stop", self._on_stop),
            pystray.MenuItem("🔄 Restart", self._on_restart),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"🌐 {BASE_URL}", self._on_copy_url),
            pystray.MenuItem("📋 Open logs folder", self._on_open_logs),
            pystray.MenuItem("⚙ Open install folder", self._on_open_install),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit tray", self._on_quit),
        )

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_start(self, _icon=None, _item=None) -> None:
        logger.info("user: start service")
        companion_client.start_service()

    def _on_stop(self, _icon=None, _item=None) -> None:
        logger.info("user: stop service")
        companion_client.stop_service()

    def _on_restart(self, _icon=None, _item=None) -> None:
        logger.info("user: restart service")
        companion_client.restart_service()

    def _on_copy_url(self, _icon=None, _item=None) -> None:
        """Copy BASE_URL to the Windows clipboard."""
        try:
            # ``clip`` is a built-in Windows utility on Vista+.
            subprocess.run(
                ["cmd", "/c", "echo", BASE_URL, "|", "clip"],
                shell=False,
                capture_output=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("copy URL failed: %s", exc)

    def _on_open_logs(self, _icon=None, _item=None) -> None:
        """Open the logs folder in Explorer (or Finder on macOS for dev)."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                # Explorer expects backslashed Windows paths.
                subprocess.Popen(["explorer", str(LOGS_DIR)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(LOGS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(LOGS_DIR)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("open logs failed: %s", exc)

    def _on_open_install(self, _icon=None, _item=None) -> None:
        """Open the install root (parent of DATA_DIR's data folder)."""
        try:
            install_dir = DATA_DIR.parent
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(install_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(install_dir)])
            else:
                subprocess.Popen(["xdg-open", str(install_dir)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("open install folder failed: %s", exc)

    def _on_quit(self, _icon=None, _item=None) -> None:
        logger.info("user: quit tray")
        self._running = False
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:  # noqa: BLE001
                pass

    # ── Entry point ─────────────────────────────────────────────────────────

    def run(self) -> int:
        """Block until the user quits the tray icon. Returns process exit code."""
        if not _GUI_AVAILABLE:
            sys.stderr.write(
                "tray_app: GUI libraries unavailable — pystray/PIL not installed "
                f"({_GUI_IMPORT_ERROR}).\n"
                "Install with: pip install pystray Pillow\n"
            )
            return 2

        # Start polling thread *before* the icon, so the first menu render
        # already shows a real status.
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="acc-tray-poll"
        )
        self._poll_thread.start()

        # Give the poll thread a brief moment to populate self._status so the
        # initial icon glyph isn't always "offline".
        time.sleep(0.2)

        initial_icon = _make_icon("offline") if self._status is None else _make_icon(
            _classify(self._status)[0]
        )
        self._icon = pystray.Icon(
            "AccServer",
            initial_icon,
            "Accelerator: Starting…",
            menu=self._build_menu(),
        )
        logger.info("tray app starting (BASE_URL=%s)", BASE_URL)
        try:
            self._icon.run()
        finally:
            self._running = False
        return 0


def main() -> int:
    return AcceleratorTray().run()


if __name__ == "__main__":
    sys.exit(main())
