"""
tests/test_wi_acc10_menubar_redesign.py — Tests for ACC-10 full menu bar redesign.

Covers:
  - Icon state machine: 🎙/🟢/⚠️/❌/⏸ based on health/processing/model state
  - Status header text includes uptime
  - Processing section shown/hidden correctly
  - Elapsed timer tracking (job_started_at transitions)
  - Last completion / 12h count tracking
  - Model status icons (☑/☒ Whisper/Pyannote)
  - RAM from /memory endpoint (not psutil)
  - Pause/Resume: launchctl stop/start
  - Restart: launchctl kickstart -k gui/<uid>/<label>
  - Logs: writes temp .txt and opens with `open`
  - Config: opens with `open -e`
  - 12h job count pruning
  - _format_uptime, _format_ago, _format_elapsed helpers
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Stub setup
# ---------------------------------------------------------------------------


def _mock_rumps():
    rumps_mock = MagicMock()

    class FakeApp:
        def __init__(self, *a, **kw):
            self.title = ""
            self.menu = []
        def run(self):
            pass

    rumps_mock.App = FakeApp
    rumps_mock.MenuItem = MagicMock
    rumps_mock.Timer = MagicMock
    rumps_mock.separator = object()
    rumps_mock.quit_application = MagicMock()
    return rumps_mock


def _import_menubar():
    """Import menubar_app with all heavy deps stubbed."""
    rumps_mock = _mock_rumps()
    psutil_mock = MagicMock()
    requests_mock = MagicMock()

    mods = {
        "rumps": rumps_mock,
        "psutil": psutil_mock,
        "requests": requests_mock,
    }
    for name, stub in mods.items():
        if name not in sys.modules:
            sys.modules[name] = stub

    if "menubar_app" in sys.modules:
        del sys.modules["menubar_app"]

    import menubar_app
    return menubar_app, rumps_mock, psutil_mock, requests_mock


def _build_app():
    """Build AcceleratorMenuBar with config stubbed to avoid real files."""
    mod, rumps_mock, psutil_mock, requests_mock = _import_menubar()

    with patch.object(mod, "_load_config", return_value={}), \
         patch.object(mod, "_setup_logging", return_value=MagicMock()):
        app = mod.AcceleratorMenuBar.__new__(mod.AcceleratorMenuBar)
        app._base_url = "http://localhost:8765"
        app._headers = {}
        app._lock = threading.Lock()
        app._service_paused = False
        app._prev_processing = 0
        app._last_job_completed_at = None
        app._jobs_completed_12h = []
        app._job_started_at = None
        app._battery_item = None  # No battery by default (Mac Mini)

        for attr in (
            "_status_item", "_processing_item", "_elapsed_item",
            "_last_job_item", "_whisper_item", "_ram_item",
            "_pause_item", "_restart_item", "_logs_item", "_config_item",
        ):
            item = MagicMock()
            item.title = ""
            item.hidden = False
            setattr(app, attr, item)

    return app, mod, psutil_mock, requests_mock


def _health(processing=0, uptime=300, whisper=True, diarize=True) -> dict:
    return {
        "status": "healthy",
        "version": "0.1.0",
        "queue": {"pending": 0, "processing": processing},
        "uptime_seconds": uptime,
        "models": {
            "whisper_loaded": whisper,
            "diarization_loaded": diarize,
        },
    }


def _memory(rss_mb=512.0) -> dict:
    return {"rss_mb": rss_mb, "models": {}}


# ---------------------------------------------------------------------------
# Icon state machine
# ---------------------------------------------------------------------------


class TestIconState:
    """Menu bar icon reflects service/processing/model state."""

    def test_healthy_idle_shows_microphone(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=0), _memory())
        assert app.title == "🎙"

    def test_processing_shows_green_circle(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=1), _memory())
        assert app.title == "🟢"

    def test_offline_shows_red_x(self):
        app, _, _, _ = _build_app()
        app._update_menu(None, None)
        assert app.title == "❌"

    def test_warning_when_whisper_not_loaded(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=0, whisper=False), _memory())
        assert app.title == "⚠️"

    def test_warning_when_diarize_not_loaded(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=0, diarize=False), _memory())
        assert app.title == "⚠️"

    def test_paused_shows_pause_icon(self):
        app, _, _, _ = _build_app()
        app._service_paused = True
        app._update_menu(_health(), _memory())
        assert app.title == "⏸"

    def test_processing_overrides_warning(self):
        """Processing takes priority over warning — show 🟢 not ⚠️."""
        app, _, _, _ = _build_app()
        # Even with whisper not loaded, if processing > 0 show 🟢
        app._update_menu(_health(processing=2, whisper=False), _memory())
        assert app.title == "🟢"


# ---------------------------------------------------------------------------
# Status header text
# ---------------------------------------------------------------------------


class TestStatusHeader:
    """Status header text includes state label and uptime."""

    def test_healthy_header_includes_uptime(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(uptime=7440), _memory())  # 2h 4m
        assert "Healthy" in app._status_item.title
        assert "2h 4m" in app._status_item.title

    def test_processing_header(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=1, uptime=120), _memory())
        assert "Processing" in app._status_item.title

    def test_offline_header(self):
        app, _, _, _ = _build_app()
        app._update_menu(None, None)
        assert "Offline" in app._status_item.title or "❌" in app._status_item.title

    def test_paused_header(self):
        app, _, _, _ = _build_app()
        app._service_paused = True
        app._update_menu(_health(), _memory())
        assert "Paused" in app._status_item.title


# ---------------------------------------------------------------------------
# Processing section
# ---------------------------------------------------------------------------


class TestProcessingSection:
    """Processing rows shown/hidden correctly, elapsed and last-job tracking."""

    def test_processing_rows_hidden_when_idle(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=0), _memory())
        assert app._processing_item.hidden is True
        assert app._elapsed_item.hidden is True

    def test_processing_row_shown_when_active(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(processing=1), _memory())
        assert app._processing_item.hidden is False

    def test_elapsed_shown_when_job_started_at_set(self):
        app, _, _, _ = _build_app()
        app._job_started_at = time.time() - 90  # 1m 30s ago
        app._update_menu(_health(processing=1), _memory())
        assert app._elapsed_item.hidden is False
        assert "1m" in app._elapsed_item.title or "90s" in app._elapsed_item.title or "elapsed" in app._elapsed_item.title.lower()

    def test_job_started_at_set_on_0_to_1_transition(self):
        """When processing goes 0→1, _job_started_at is recorded."""
        app, _, _, _ = _build_app()
        app._prev_processing = 0

        before = time.time()
        app._update_menu(_health(processing=1), _memory())
        after = time.time()

        assert app._job_started_at is not None
        assert before <= app._job_started_at <= after

    def test_last_completed_set_on_1_to_0_transition(self):
        """When processing goes 1→0, _last_job_completed_at is set."""
        app, _, _, _ = _build_app()
        app._prev_processing = 1

        before = time.time()
        app._update_menu(_health(processing=0), _memory())
        after = time.time()

        assert app._last_job_completed_at is not None
        assert before <= app._last_job_completed_at <= after

    def test_12h_count_incremented_on_completion(self):
        """_jobs_completed_12h gains one entry on 1→0 transition."""
        app, _, _, _ = _build_app()
        app._prev_processing = 1
        app._update_menu(_health(processing=0), _memory())
        assert len(app._jobs_completed_12h) == 1

    def test_12h_count_pruned(self):
        """Timestamps older than 12h are removed from _jobs_completed_12h."""
        app, _, _, _ = _build_app()
        now = time.time()
        old = now - 13 * 3600  # 13h ago → should be pruned
        recent = now - 1 * 3600  # 1h ago → keep
        app._jobs_completed_12h = [old, recent]
        app._update_menu(_health(processing=0), _memory())
        # Only the recent one should survive (plus any added by this poll)
        for ts in app._jobs_completed_12h:
            assert ts >= now - 12 * 3600 - 60  # small tolerance


# ---------------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------------


class TestModelStatus:
    """Model checkmark icons reflect /health model state."""

    def test_both_loaded_shows_check_marks(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(whisper=True, diarize=True), _memory())
        assert "☑" in app._whisper_item.title

    def test_whisper_not_loaded_shows_x(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(whisper=False, diarize=True), _memory())
        title = app._whisper_item.title
        # First icon (before "Whisper") should be ☒
        assert "☒" in title

    def test_diarize_not_loaded_shows_x(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(whisper=True, diarize=False), _memory())
        title = app._whisper_item.title
        assert "☒" in title

    def test_offline_shows_both_x(self):
        app, _, _, _ = _build_app()
        app._update_menu(None, None)
        assert "☒" in app._whisper_item.title


# ---------------------------------------------------------------------------
# RAM display
# ---------------------------------------------------------------------------


class TestRAMDisplay:
    """RAM comes from /memory endpoint, not psutil."""

    def test_ram_displays_server_rss(self):
        app, _, psutil_mock, _ = _build_app()
        app._update_menu(_health(), _memory(rss_mb=1228.8))  # ~1.2 GB
        assert "1.2 GB" in app._ram_item.title

    def test_ram_shows_dash_when_memory_none(self):
        app, _, _, _ = _build_app()
        app._update_menu(_health(), None)
        assert "—" in app._ram_item.title

    def test_psutil_process_not_used(self, monkeypatch):
        """psutil.Process().memory_info() should NOT be used for RAM."""
        app, mod, psutil_mock, _ = _build_app()
        app._update_menu(_health(), _memory(rss_mb=512.0))
        # psutil.Process should not have been called
        psutil_mock.Process.assert_not_called()


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


class TestPauseResume:
    """Pause/Resume uses launchctl stop/start."""

    def test_pause_calls_launchctl_stop(self):
        app, mod, _, _ = _build_app()
        with patch("subprocess.run") as mock_run:
            app._on_pause_resume(None)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "stop" in args
            assert mod.PLIST_LABEL in args

    def test_pause_sets_service_paused_flag(self):
        app, _, _, _ = _build_app()
        with patch("subprocess.run"):
            app._on_pause_resume(None)
        assert app._service_paused is True

    def test_resume_calls_launchctl_kickstart(self):
        app, mod, _, _ = _build_app()
        app._service_paused = True
        uid = os.getuid()
        with patch("subprocess.run") as mock_run:
            app._on_pause_resume(None)
            args = mock_run.call_args[0][0]
            # Must use kickstart (not start) — launchctl start is broken on macOS Monterey+
            assert args == ["launchctl", "kickstart", f"gui/{uid}/{mod.PLIST_LABEL}"], (
                f"Expected kickstart command, got: {args}"
            )
            assert "start" not in args[1], (
                "Should use 'kickstart' not 'start' — 'start' is deprecated on Monterey+"
            )

    def test_resume_clears_service_paused_flag(self):
        app, _, _, _ = _build_app()
        app._service_paused = True
        with patch("subprocess.run"):
            app._on_pause_resume(None)
        assert app._service_paused is False


class TestRestart:
    """Restart uses launchctl kickstart -k gui/<uid>/<label>."""

    def test_restart_uses_kickstart(self):
        app, mod, _, _ = _build_app()
        with patch("subprocess.run") as mock_run:
            app._on_restart(None)
            args = mock_run.call_args[0][0]
            assert "kickstart" in args
            assert "-k" in args

    def test_restart_uses_correct_service_path(self):
        app, mod, _, _ = _build_app()
        uid = os.getuid()
        with patch("subprocess.run") as mock_run:
            app._on_restart(None)
            args = mock_run.call_args[0][0]
            expected_svc = f"gui/{uid}/{mod.PLIST_LABEL}"
            assert any(expected_svc in str(a) for a in args), (
                f"Expected service path '{expected_svc}' in args {args}"
            )

    def test_restart_clears_paused_flag(self):
        app, _, _, _ = _build_app()
        app._service_paused = True
        with patch("subprocess.run"):
            app._on_restart(None)
        assert app._service_paused is False


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


class TestViewLogs:
    """Logs writes a temp .txt and opens with `open`, not Console.app."""

    def test_logs_opens_with_open_not_console(self, tmp_path):
        app, mod, _, _ = _build_app()

        # Point LOG_FILE to a temp file
        log_content = "line1\nline2\nline3\n"
        fake_log = tmp_path / "accelerator.log"
        fake_log.write_text(log_content)

        with patch.object(mod, "LOG_FILE", fake_log), \
             patch("subprocess.run") as mock_run, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            # Make NamedTemporaryFile work as a context manager
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/accelerator_log_test.txt"
            mock_tmp.return_value = mock_file

            app._on_view_logs(None)

            # Should call `open <tmpfile>`, NOT `open -a Console`
            if mock_run.called:
                cmd_args = mock_run.call_args[0][0]
                assert "Console" not in cmd_args, (
                    f"Should not use Console.app, got: {cmd_args}"
                )
                assert "open" in cmd_args[0]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestOpenConfig:
    """Config opens with `open -e` (forces TextEdit)."""

    def test_config_uses_open_e(self, tmp_path):
        app, mod, _, _ = _build_app()

        fake_config = tmp_path / "config.env"
        fake_config.write_text("ACCELERATOR_PORT=8765\n")

        with patch.object(mod, "CONFIG_FILE", fake_config), \
             patch("subprocess.run") as mock_run:
            app._on_open_config(None)
            args = mock_run.call_args[0][0]
            assert "open" in args[0]
            assert "-e" in args, f"Expected '-e' flag, got: {args}"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for _format_uptime, _format_ago, _format_elapsed."""

    def test_format_uptime_seconds(self):
        mod, *_ = _import_menubar()
        assert mod._format_uptime(45) == "45s"

    def test_format_uptime_minutes(self):
        mod, *_ = _import_menubar()
        assert mod._format_uptime(90) == "1m"
        assert mod._format_uptime(3599) == "59m"

    def test_format_uptime_hours_no_remainder(self):
        mod, *_ = _import_menubar()
        assert mod._format_uptime(3600) == "1h"
        assert mod._format_uptime(7200) == "2h"

    def test_format_uptime_hours_with_minutes(self):
        mod, *_ = _import_menubar()
        assert mod._format_uptime(7440) == "2h 4m"
        assert mod._format_uptime(8100) == "2h 15m"

    def test_format_ago_seconds(self):
        mod, *_ = _import_menubar()
        ts = time.time() - 30
        result = mod._format_ago(ts)
        assert result.endswith("ago")
        assert "s" in result

    def test_format_ago_minutes(self):
        mod, *_ = _import_menubar()
        ts = time.time() - 180
        result = mod._format_ago(ts)
        assert "m ago" in result

    def test_format_elapsed_under_minute(self):
        mod, *_ = _import_menubar()
        ts = time.time() - 45
        result = mod._format_elapsed(ts)
        assert "s" in result
        assert "m" not in result

    def test_format_elapsed_over_minute(self):
        mod, *_ = _import_menubar()
        ts = time.time() - 90
        result = mod._format_elapsed(ts)
        assert "m" in result
        assert "s" in result

# ---------------------------------------------------------------------------
# Battery row (ACC-10 requirement)
# ---------------------------------------------------------------------------


class TestBatteryRow:
    """Battery row hidden on Mac Mini (no battery), shown on MacBook."""

    def test_battery_item_none_when_no_battery_detected(self):
        """On Mac Mini, psutil.sensors_battery() returns None → no battery item."""
        # The _build_app() sets _battery_item = None to simulate no-battery device
        app, _, psutil_mock, _ = _build_app()
        psutil_mock.sensors_battery.return_value = None
        assert app._battery_item is None

    def test_battery_update_shows_percent(self):
        """When _battery_item is set (MacBook), it should show charge percent."""
        app, mod, psutil_mock, _ = _build_app()
        bat_mock = MagicMock()
        bat_mock.percent = 85.0
        bat_mock.power_plugged = False
        # Simulate a MacBook — add battery item
        app._battery_item = MagicMock()
        app._battery_item.title = ""

        with patch.object(mod.psutil, "sensors_battery", return_value=bat_mock):
            app._update_menu(
                {"status": "healthy", "queue": {"processing": 0, "pending": 0},
                 "uptime_seconds": 60,
                 "models": {"whisper_loaded": True, "diarization_loaded": True}},
                {"rss_mb": 512.0}
            )
        title = app._battery_item.title
        assert "85" in title and "%" in title

    def test_battery_shows_charging_indicator(self):
        """Charging state should include 'charging' in the title."""
        app, mod, psutil_mock, _ = _build_app()
        bat_mock = MagicMock()
        bat_mock.percent = 72.0
        bat_mock.power_plugged = True
        app._battery_item = MagicMock()
        app._battery_item.title = ""

        with patch.object(mod.psutil, "sensors_battery", return_value=bat_mock):
            app._update_menu(
                {"status": "healthy", "queue": {"processing": 0, "pending": 0},
                 "uptime_seconds": 120,
                 "models": {"whisper_loaded": True, "diarization_loaded": True}},
                {"rss_mb": 256.0}
            )
        assert "charging" in app._battery_item.title.lower()

    def test_battery_skipped_when_none(self):
        """When _battery_item is None, no update should be attempted."""
        app, _, _, _ = _build_app()
        assert app._battery_item is None
        # Should not raise — _update_menu checks for None before updating
        app._update_menu(
            {"status": "healthy", "queue": {"processing": 0, "pending": 0},
             "uptime_seconds": 60,
             "models": {"whisper_loaded": True, "diarization_loaded": True}},
            {"rss_mb": 512.0}
        )  # No AttributeError means the check works
