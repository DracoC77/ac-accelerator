"""
Audio Chronicle Accelerator — macOS Menu Bar App (ACC-10 redesign)

Provides a compact, state-driven menu bar UI for monitoring and controlling
the accelerator service.

Menu bar icon (single emoji, state-driven):
  🎙  Idle/healthy  (processing=0, /health ok)
  🟢  Processing    (processing > 0)
  ⚠️  Warning       (whisper_loaded=false OR diarization_loaded=false)
  ❌  Offline/error (/health unreachable or non-200)
  ⏸  Paused        (launchd job stopped)

Menu layout:
  🎙 Healthy · up 2h 14m        ← status + uptime from /health uptime_seconds
  ─────
  🟢 Processing: transcription   ← only when processing > 0
     ⏱ 1m 23s elapsed            ← elapsed since job started locally
     📁 Last: 3m ago · 8 (12h)   ← last completion + 12h count
  ─────
  🌐 http://<local-ip>:<port>    ← ACC-19: local IP + port, click to copy
  ─────
  ☑ Whisper  ☑ Pyannote          ← ☒ when not loaded
  🧠 1.2 GB                      ← from /memory rss_mb
  ─────
  ⏸ Pause                        ← launchctl stop PLIST_LABEL
  🔄 Restart                     ← launchctl kickstart -k gui/<uid>/<PLIST_LABEL>
  ─────
  📋 Logs                        ← tail last 200 lines → temp .txt → open
  ⚙️  Config                     ← open -e config.env
  ─────
  Quit

Usage:
    ~/.audio-chronicle-accelerator/venv/bin/python menubar_app.py
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import psutil
import requests
import rumps
import socket

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INSTALL_DIR = Path.home() / ".audio-chronicle-accelerator"
CONFIG_FILE = INSTALL_DIR / "config.env"
LOG_FILE = INSTALL_DIR / "logs" / "accelerator.log"
PLIST_LABEL = "com.audio-chronicle-accelerator"

DEFAULT_ACCELERATOR_URL = "http://localhost:8765"
POLL_INTERVAL_SECONDS = 5

# How many lines to tail for the Logs viewer
LOG_TAIL_LINES = 200

# How far back to count completed jobs for the "N jobs (12h)" display
JOBS_WINDOW_HOURS = 12


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

MENUBAR_LOG_FILE = INSTALL_DIR / "logs" / "menubar.log"
# launchd captures the process's stdio streams to these files.
STDERR_LOG_FILE = INSTALL_DIR / "logs" / "stderr.log"

# LOG_LEVEL_FILE controls the rotating file handler level (default INFO).
# LOG_LEVEL is unused here but reserved for future console-level control.
_LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "INFO").upper()


def _setup_logging() -> logging.Logger:
    """Configure a rotating file logger for the menu bar app.

    The handler is attached directly to this logger; propagation is left
    on (Python default) so that log records also reach the root logger
    (e.g. the console handler configured by uvicorn or during tests).

    The log level is controlled by the LOG_LEVEL_FILE env var (default INFO).
    Set LOG_LEVEL_FILE=DEBUG to get verbose menubar diagnostics without
    changing the server's console log level.
    """
    log = logging.getLogger("accelerator.menubar")
    if log.handlers:
        return log

    file_level = getattr(logging, _LOG_LEVEL_FILE, logging.INFO)
    # Set the logger's own level so records aren't filtered before reaching
    # the handler; propagation is left on so console output still flows to
    # the root logger when one is configured.
    log.setLevel(file_level)

    MENUBAR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.handlers.RotatingFileHandler(
        MENUBAR_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(file_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


logger = _setup_logging()


def _get_local_ip() -> str:
    """Return the primary non-loopback IPv4 address using the UDP socket trick."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _load_config() -> dict[str, str]:
    """Read key=value pairs from config.env, ignoring comments and blanks."""
    config: dict[str, str] = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    config[key.strip()] = val.strip()
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    """Format uptime_seconds into a human-readable string like '2h 14m' or '45s'."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds) // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def _format_ago(ts: float) -> str:
    """Format a monotonic/epoch timestamp as 'Ns ago', 'Nm ago', or 'Nh ago'."""
    elapsed = time.time() - ts
    if elapsed < 60:
        return f"{int(elapsed)}s ago"
    if elapsed < 3600:
        return f"{int(elapsed) // 60}m ago"
    return f"{int(elapsed) // 3600}h ago"


def _format_elapsed(ts: float) -> str:
    """Format elapsed seconds since ts as 'Xm Ys' or 'Xs'."""
    elapsed = time.time() - ts
    if elapsed < 60:
        return f"{int(elapsed)}s"
    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    return f"{mins}m {secs}s"


# ---------------------------------------------------------------------------
# Menu Bar App
# ---------------------------------------------------------------------------


class AcceleratorMenuBar(rumps.App):
    """Menu bar app for the Audio Chronicle Accelerator service (ACC-10)."""

    def __init__(self) -> None:
        super().__init__("🎙", quit_button=None)

        # Load config
        cfg = _load_config()
        self._base_url: str = cfg.get("ACCELERATOR_URL", DEFAULT_ACCELERATOR_URL).rstrip("/")
        token = cfg.get("ACCELERATOR_TOKEN", "")
        self._headers: dict[str, str] = (
            {"Authorization": f"Bearer {token}"} if token else {}
        )

        # ACC-19: resolve local URL once at startup
        port = cfg.get("ACCELERATOR_PORT", "8765")
        local_ip = _get_local_ip()
        self._local_url: str = f"http://{local_ip}:{port}"

        # Internal state
        self._service_paused: bool = False
        self._lock = threading.Lock()

        # ACC-10: local job state tracking
        # These are updated by _update_menu() to detect processing transitions.
        self._prev_processing: int = 0        # previous poll's queue.processing count
        self._last_job_completed_at: Optional[float] = None   # epoch when processing → 0
        self._jobs_completed_12h: list[float] = []            # epoch timestamps of recent completions
        self._job_started_at: Optional[float] = None          # epoch when processing went 0 → 1

        # ── Status header (non-clickable) ───────────────────────────────────
        self._status_item = rumps.MenuItem("🎙 Checking…")
        self._status_item.set_callback(None)

        # ── Separator after status ───────────────────────────────────────────
        # ── Active job section (hidden when idle) ────────────────────────────
        self._processing_item = rumps.MenuItem("")
        self._processing_item.set_callback(None)
        self._processing_item.hidden = True

        self._elapsed_item = rumps.MenuItem("")
        self._elapsed_item.set_callback(None)
        self._elapsed_item.hidden = True

        self._last_job_item = rumps.MenuItem("📁 Last: never")
        self._last_job_item.set_callback(None)

        # ── ACC-19: Local URL (display-only, WI-ACC-BUG-3) ──────────────────────
        self._local_url_item = rumps.MenuItem(f"🌐 {self._local_url}")
        # WI-ACC-BUG-3: display-only (no callback = greyed out, non-clickable)
        self._local_url_item.set_callback(None)

        # ── Model / memory status ────────────────────────────────────────────
        self._whisper_item = rumps.MenuItem("☑ Whisper  ☑ Pyannote")
        self._whisper_item.set_callback(None)

        self._ram_item = rumps.MenuItem("🧠 — GB")
        self._ram_item.set_callback(None)

        # ── Battery (MacBook only — hidden on Mac Mini / desktop) ───────────
        _has_battery = psutil.sensors_battery() is not None
        if _has_battery:
            self._battery_item: Optional[rumps.MenuItem] = rumps.MenuItem("🔋 —")
            self._battery_item.set_callback(None)
        else:
            self._battery_item = None

        # ── Controls ─────────────────────────────────────────────────────────
        self._pause_item = rumps.MenuItem("⏸ Pause", callback=self._on_pause_resume)
        self._restart_item = rumps.MenuItem("🔄 Restart", callback=self._on_restart)

        # ── Cache management ──────────────────────────────────────────────────
        self._clear_cache_item = rumps.MenuItem("🗑️ Clear Cache", callback=self._on_clear_cache)

        # ── Utility ──────────────────────────────────────────────────────────
        self._logs_item = rumps.MenuItem("📋 Logs", callback=self._on_view_logs)
        self._config_item = rumps.MenuItem("⚙️  Config", callback=self._on_open_config)

        quit_item = rumps.MenuItem("Quit", callback=rumps.quit_application)

        self.menu = [
            self._status_item,
            rumps.separator,
            self._processing_item,
            self._elapsed_item,
            self._last_job_item,
            rumps.separator,
            self._local_url_item,
            rumps.separator,
            self._whisper_item,
            self._ram_item,
            rumps.separator,
            self._pause_item,
            self._restart_item,
            rumps.separator,
            self._clear_cache_item,
            rumps.separator,
            self._logs_item,
            self._config_item,
            rumps.separator,
            quit_item,
        ]

        # Insert battery item into menu after RAM item if present
        if self._battery_item is not None:
            self.menu.insert_after(self._ram_item.title, self._battery_item)

        # Start polling timer
        self._poll_timer = rumps.Timer(self._poll, POLL_INTERVAL_SECONDS)
        self._poll_timer.start()

        # Initial poll
        self._poll(None)

    # ── Polling ─────────────────────────────────────────────────────────────

    def _poll(self, _sender) -> None:
        """Fetch service health + memory and update all menu items."""
        health = self._fetch_health()
        memory = self._fetch_memory()
        self._update_menu(health, memory)

    def _fetch_health(self) -> Optional[dict]:
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=3)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Health endpoint returned %s", resp.status_code)
        except requests.exceptions.ConnectionError:
            logger.debug("Health check: service unreachable at %s", self._base_url)
        except requests.exceptions.Timeout:
            logger.warning("Health check timed out (base_url=%s)", self._base_url)
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
        return None

    def _fetch_memory(self) -> Optional[dict]:
        try:
            resp = requests.get(
                f"{self._base_url}/memory",
                timeout=3,
                headers=self._headers,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    # ── Menu updates ────────────────────────────────────────────────────────

    def _update_menu(self, health: Optional[dict], memory: Optional[dict]) -> None:
        """Refresh all dynamic menu items from fetched data."""
        with self._lock:
            # ── Offline / error ────────────────────────────────────────────
            if health is None:
                if not self._service_paused:
                    self.title = "❌"
                    self._status_item.title = "❌ Offline"
                self._processing_item.hidden = True
                self._elapsed_item.hidden = True
                # Keep last job row visible even when offline
                if self._last_job_completed_at is not None:
                    n = len(self._jobs_completed_12h)
                    self._last_job_item.title = (
                        f"📁 Last: {_format_ago(self._last_job_completed_at)} · {n} jobs (12h)"
                    )
                else:
                    self._last_job_item.title = "📁 Last: never"
                self._whisper_item.title = "☒ Whisper  ☒ Pyannote"
                self._ram_item.title = "🧠 — GB"
                self._pause_item.title = "⏸ Pause"
                self._pause_item.set_callback(None)
                self._restart_item.set_callback(self._on_restart)
                return

            # Re-enable pause control
            self._pause_item.set_callback(self._on_pause_resume)

            queue_info = health.get("queue", {})
            processing = int(queue_info.get("processing", 0))
            uptime_s = float(health.get("uptime_seconds", 0))

            models = health.get("models", {})
            whisper_loaded = models.get("whisper_loaded", False)
            diarize_loaded = models.get("diarization_loaded", False)

            # ── Detect processing transitions ──────────────────────────────
            now = time.time()

            if self._prev_processing == 0 and processing > 0:
                # Transition: idle → processing
                self._job_started_at = now

            if self._prev_processing > 0 and processing == 0:
                # Transition: processing → idle
                self._last_job_completed_at = now
                self._jobs_completed_12h.append(now)
                self._job_started_at = None

            # Prune completions older than 12h
            cutoff = now - JOBS_WINDOW_HOURS * 3600
            self._jobs_completed_12h = [
                t for t in self._jobs_completed_12h if t >= cutoff
            ]

            self._prev_processing = processing

            # ── Icon + status header ───────────────────────────────────────
            if self._service_paused:
                self.title = "⏸"
                self._status_item.title = "⏸ Paused"
            elif processing > 0:
                self.title = "🟢"
                self._status_item.title = f"🟢 Processing · up {_format_uptime(uptime_s)}"
            elif not whisper_loaded or not diarize_loaded:
                self.title = "⚠️"
                self._status_item.title = f"⚠️ Warning · up {_format_uptime(uptime_s)}"
            else:
                self.title = "🎙"
                self._status_item.title = f"🎙 Healthy · up {_format_uptime(uptime_s)}"

            # ── Active processing rows ─────────────────────────────────────
            if processing > 0:
                # Determine active job type from /health (not a separate jobs call)
                # /health doesn't include job type — use generic label
                self._processing_item.title = f"🟢 Processing: job running"
                self._processing_item.hidden = False

                if self._job_started_at is not None:
                    self._elapsed_item.title = f"   ⏱ {_format_elapsed(self._job_started_at)} elapsed"
                    self._elapsed_item.hidden = False
                else:
                    self._elapsed_item.hidden = True

                # Always show last job row (indented during active processing)
                if self._last_job_completed_at is not None:
                    n = len(self._jobs_completed_12h)
                    self._last_job_item.title = (
                        f"   📁 Last: {_format_ago(self._last_job_completed_at)} · {n} jobs (12h)"
                    )
                else:
                    self._last_job_item.title = "   📁 Last: never"
            else:
                self._processing_item.hidden = True
                self._elapsed_item.hidden = True

                # Always show last job row — "never" until first job completes
                if self._last_job_completed_at is not None:
                    n = len(self._jobs_completed_12h)
                    self._last_job_item.title = (
                        f"📁 Last: {_format_ago(self._last_job_completed_at)} · {n} jobs (12h)"
                    )
                else:
                    self._last_job_item.title = "📁 Last: never"

            # ── Model status ───────────────────────────────────────────────
            w_icon = "☑" if whisper_loaded else "☒"
            d_icon = "☑" if diarize_loaded else "☒"
            self._whisper_item.title = f"{w_icon} Whisper  {d_icon} Pyannote"

            # ── RAM from /memory endpoint ──────────────────────────────────
            if memory is not None:
                rss_mb = float(memory.get("rss_mb", 0.0))
                ram_gb = rss_mb / 1024.0
                self._ram_item.title = f"🧠 {ram_gb:.1f} GB"
            else:
                self._ram_item.title = "🧠 — GB"

            # ── Battery (MacBook only) ─────────────────────────────────────
            if self._battery_item is not None:
                try:
                    bat = psutil.sensors_battery()
                    if bat is not None:
                        pct = bat.percent
                        status = " (charging)" if bat.power_plugged else ""
                        self._battery_item.title = f"🔋 {pct:.0f}%{status}"
                    else:
                        self._battery_item.title = "🔋 —"
                except Exception:
                    self._battery_item.title = "🔋 —"

            # ── Pause/Resume toggle ────────────────────────────────────────
            if self._service_paused:
                self._pause_item.title = "▶ Resume"
            else:
                self._pause_item.title = "⏸ Pause"

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_pause_resume(self, _sender) -> None:
        """Toggle between pausing (stop launchd service) and resuming."""
        with self._lock:
            currently_paused = self._service_paused

        if currently_paused:
            subprocess.run(["launchctl", "kickstart", f"gui/{os.getuid()}/{PLIST_LABEL}"], capture_output=True)
            with self._lock:
                self._service_paused = False
            self._pause_item.title = "⏸ Pause"
        else:
            subprocess.run(["launchctl", "stop", PLIST_LABEL], capture_output=True)
            with self._lock:
                self._service_paused = True
            self._pause_item.title = "▶ Resume"
            self.title = "⏸"
            self._status_item.title = "⏸ Paused"

    def _on_restart(self, _sender) -> None:
        """Restart the launchd service atomically (kill + restart in one step)."""
        # launchctl kickstart -k does an atomic kill+restart without needing
        # two separate stop/start commands. Using gui/<uid>/<label> for the
        # user launchd domain (LaunchAgents).
        uid = os.getuid()
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{PLIST_LABEL}"],
            capture_output=True,
        )
        with self._lock:
            self._service_paused = False

    def _on_view_logs(self, _sender) -> None:
        """Tail last N lines of log files and open them in a temp .txt.

        Shows up to three sections in order:
          1. accelerator.log  — rotating file log written by the server
          2. menubar.log      — rotating file log written by this menubar app
          3. stderr.log       — launchd stdio capture (fallback / always shown
                               when accelerator.log is empty or missing, so
                               there is always useful diagnostic output even
                               before the server has produced any file logs)

        Console.app requires files to be indexed, which is unreliable for our
        log paths.  Writing a .txt and opening with `open` uses the system
        default text viewer (TextEdit/Preview) and always works.
        """
        try:
            sections: list[str] = []

            # ── accelerator.log (server rotating file handler) ───────────
            acc_log_has_content = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0
            sections.append("=== accelerator.log ===")
            if acc_log_has_content:
                lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                sections.append("\n".join(lines[-LOG_TAIL_LINES:]))
            else:
                sections.append(
                    f"(empty or not found: {LOG_FILE})\n"
                    "Showing stderr.log fallback below."
                )

            # ── menubar.log ───────────────────────────────────────────────
            sections.append("=== menubar.log ===")
            if MENUBAR_LOG_FILE.exists() and MENUBAR_LOG_FILE.stat().st_size > 0:
                lines = MENUBAR_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                sections.append("\n".join(lines[-LOG_TAIL_LINES:]))
            else:
                sections.append(f"(empty or not found: {MENUBAR_LOG_FILE})")

            # ── stderr.log fallback ───────────────────────────────────────
            # Always include stderr.log when accelerator.log is empty/missing
            # so that launchd stdio output is accessible from the menu.
            if not acc_log_has_content:
                sections.append("=== stderr.log (launchd fallback) ===")
                if STDERR_LOG_FILE.exists() and STDERR_LOG_FILE.stat().st_size > 0:
                    lines = STDERR_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                    sections.append("\n".join(lines[-LOG_TAIL_LINES:]))
                else:
                    sections.append(f"(empty or not found: {STDERR_LOG_FILE})")

            content = "\n\n".join(sections)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix="accelerator_log_",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            subprocess.run(["open", tmp_path], capture_output=True)
        except Exception as exc:
            logger.warning("Failed to open logs: %s", exc)

    def _on_open_config(self, _sender) -> None:
        """Open config.env in TextEdit (forced via -e flag)."""
        # `open config_path` would use whatever default app is registered
        # for .env files (often nothing, or an IDE). The -e flag forces
        # TextEdit so users always get a plain-text editor (ACC-10).
        if not CONFIG_FILE.exists():
            INSTALL_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.touch()
        subprocess.run(["open", "-e", str(CONFIG_FILE)], capture_output=True)

    def _on_clear_cache(self, _sender) -> None:
        """Clear all accelerator cache entries and show a notification (WI-ACC-23)."""
        try:
            resp = requests.delete(
                f"{self._base_url}/cache",
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                deleted = data.get("deleted", 0)
                rumps.notification(
                    title="Audio Chronicle Accelerator",
                    subtitle="Cache Cleared",
                    message=f"Cache cleared ({deleted} {'entry' if deleted == 1 else 'entries'} deleted)",
                )
                logger.info("Cache cleared via menubar (%d entries deleted)", deleted)
            else:
                error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                rumps.notification(
                    title="Audio Chronicle Accelerator",
                    subtitle="Cache Clear Failed",
                    message=f"Cache clear failed: {resp.status_code} {error_detail}",
                )
                logger.warning("Cache clear failed (status=%d, detail=%s)", resp.status_code, error_detail)
        except requests.exceptions.ConnectionError:
            rumps.notification(
                title="Audio Chronicle Accelerator",
                subtitle="Cache Clear Failed",
                message="Cache clear failed: accelerator not reachable",
            )
            logger.warning("Cache clear failed: connection error (base_url=%s)", self._base_url)
        except Exception as exc:
            rumps.notification(
                title="Audio Chronicle Accelerator",
                subtitle="Cache Clear Failed",
                message=f"Cache clear failed: {exc}",
            )
            logger.warning("Cache clear failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # WI-ACC-22: suppress Dock icon. NSApplication.sharedApplication() initialises the
    # NSApp singleton (idempotent — rumps calls it again in app.run()). The policy must
    # be set before the run loop starts; calling sharedApplication() here is the only
    # safe window that works across rumps versions.
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory  # noqa: PLC0415
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app = AcceleratorMenuBar()
    app.run()


if __name__ == "__main__":
    main()

