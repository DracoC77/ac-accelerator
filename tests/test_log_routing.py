"""
Tests for fixed log routing in server.py and menubar_app.py.

Verifies:
 - _setup_file_logging() attaches a RotatingFileHandler to the root logger
 - The handler writes to DATA_DIR/logs/accelerator.log
 - LOG_LEVEL_FILE env var controls the file handler's level (default INFO)
 - Duplicate handler guard: calling _setup_file_logging() twice adds only one handler
 - menubar_app._setup_logging() creates logs/menubar.log and writes to it
 - menubar_app LOG_LEVEL_FILE controls file handler level
 - _on_view_logs: shows stderr.log fallback section when accelerator.log is empty/missing
 - _on_view_logs: does NOT show stderr.log fallback when accelerator.log has content
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# server.py — _setup_file_logging
# ---------------------------------------------------------------------------

class TestSetupFileLogging:
    """Tests for server._setup_file_logging()."""

    def test_attaches_rotating_handler_to_root(self, tmp_path, monkeypatch):
        """_setup_file_logging() must attach exactly one RotatingFileHandler to root."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("LOG_LEVEL_FILE", "INFO")

        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")

        root = logging.getLogger()
        before = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]

        server._setup_file_logging()

        after = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        new_handlers = [h for h in after if h not in before]
        assert len(new_handlers) == 1, f"Expected 1 new RotatingFileHandler, got {len(new_handlers)}"

        # Clean up
        for h in new_handlers:
            root.removeHandler(h)
            h.close()

    def test_log_file_path_under_data_dir(self, tmp_path, monkeypatch):
        """The handler should write to DATA_DIR/logs/accelerator.log."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("LOG_LEVEL_FILE", "INFO")

        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")

        root = logging.getLogger()
        before = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}

        server._setup_file_logging()

        after = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}
        new_handlers = after - before
        assert new_handlers
        handler = next(iter(new_handlers))

        expected = str(tmp_path / "data" / "logs" / "accelerator.log")
        assert handler.baseFilename == expected

        # Clean up
        root.removeHandler(handler)
        handler.close()

    def test_log_level_file_default_info(self, tmp_path, monkeypatch):
        """Default LOG_LEVEL_FILE should set handler level to INFO."""
        monkeypatch.delenv("LOG_LEVEL_FILE", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(server, "LOG_LEVEL_FILE", "INFO")

        root = logging.getLogger()
        before = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}

        server._setup_file_logging()

        after = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}
        new_handlers = after - before
        assert new_handlers
        handler = next(iter(new_handlers))

        assert handler.level == logging.INFO

        root.removeHandler(handler)
        handler.close()

    def test_log_level_file_debug(self, tmp_path, monkeypatch):
        """LOG_LEVEL_FILE=DEBUG should set handler level to DEBUG."""
        monkeypatch.setenv("LOG_LEVEL_FILE", "DEBUG")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(server, "LOG_LEVEL_FILE", "DEBUG")

        root = logging.getLogger()
        before = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}

        server._setup_file_logging()

        after = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}
        new_handlers = after - before
        assert new_handlers
        handler = next(iter(new_handlers))

        assert handler.level == logging.DEBUG

        root.removeHandler(handler)
        handler.close()

    def test_duplicate_guard(self, tmp_path, monkeypatch):
        """Calling _setup_file_logging() twice must not add a second handler."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("LOG_LEVEL_FILE", "INFO")

        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")

        root = logging.getLogger()
        before_count = len([h for h in root.handlers
                            if isinstance(h, logging.handlers.RotatingFileHandler)])

        server._setup_file_logging()
        count_after_first = len([h for h in root.handlers
                                  if isinstance(h, logging.handlers.RotatingFileHandler)])
        server._setup_file_logging()
        count_after_second = len([h for h in root.handlers
                                   if isinstance(h, logging.handlers.RotatingFileHandler)])

        assert count_after_second == count_after_first, (
            "Second call added an extra handler"
        )

        # Clean up
        for h in [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]:
            root.removeHandler(h)
            h.close()

    def test_log_dir_created_if_missing(self, tmp_path, monkeypatch):
        """_setup_file_logging() should create the logs/ directory if needed."""
        data_dir = tmp_path / "fresh_data"
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("LOG_LEVEL_FILE", "INFO")

        import server
        monkeypatch.setattr(server, "DATA_DIR", data_dir)

        logs_dir = data_dir / "logs"
        assert not logs_dir.exists()

        root = logging.getLogger()
        before = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}

        server._setup_file_logging()

        assert logs_dir.exists()

        after = {h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)}
        for h in after - before:
            root.removeHandler(h)
            h.close()


# ---------------------------------------------------------------------------
# menubar_app.py — stub helpers (rumps / psutil / requests are macOS-only)
# ---------------------------------------------------------------------------

def _mock_rumps_module():
    """Build a minimal rumps stub so menubar_app can be imported on Linux/CI."""
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


def _import_menubar_fresh():
    """Import (or re-import) menubar_app with macOS deps fully stubbed."""
    for mod_name in ("menubar_app", "rumps", "psutil", "requests"):
        sys.modules.pop(mod_name, None)

    rumps_mock = _mock_rumps_module()
    psutil_mock = MagicMock()
    requests_mock = MagicMock()

    sys.modules["rumps"] = rumps_mock
    sys.modules["psutil"] = psutil_mock
    sys.modules["requests"] = requests_mock

    import menubar_app
    return menubar_app


# ---------------------------------------------------------------------------
# menubar_app.py — _setup_logging
# ---------------------------------------------------------------------------

class TestMenubarLogging:
    """Tests for menubar_app._setup_logging()."""

    def _fresh_logger(self):
        """Remove all handlers from accelerator.menubar and return it."""
        log = logging.getLogger("accelerator.menubar")
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()
        return log

    def test_creates_rotating_handler(self, tmp_path):
        """_setup_logging() must attach a RotatingFileHandler to the menubar logger."""
        mod = _import_menubar_fresh()
        mod.MENUBAR_LOG_FILE = tmp_path / "menubar.log"
        mod._LOG_LEVEL_FILE = "INFO"

        log = self._fresh_logger()
        result_log = mod._setup_logging()

        handlers = [
            h for h in result_log.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(handlers) == 1

        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()

    def test_writes_to_menubar_log(self, tmp_path):
        """After setup, a log record must appear in the menubar.log file."""
        mod = _import_menubar_fresh()
        menubar_log = tmp_path / "menubar.log"
        mod.MENUBAR_LOG_FILE = menubar_log
        mod._LOG_LEVEL_FILE = "INFO"

        log = self._fresh_logger()
        mod._setup_logging()
        log.info("Test menubar log entry")
        for h in log.handlers:
            h.flush()

        assert menubar_log.exists()
        assert "Test menubar log entry" in menubar_log.read_text()

        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()

    def test_log_level_file_controls_handler_level(self, tmp_path):
        """LOG_LEVEL_FILE must control the file handler level."""
        mod = _import_menubar_fresh()
        mod.MENUBAR_LOG_FILE = tmp_path / "menubar.log"
        mod._LOG_LEVEL_FILE = "WARNING"

        log = self._fresh_logger()
        mod._setup_logging()

        handlers = [
            h for h in log.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert handlers
        assert handlers[0].level == logging.WARNING

        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()

    def test_idempotent_setup(self, tmp_path):
        """Calling _setup_logging() twice must not add a second handler."""
        mod = _import_menubar_fresh()
        mod.MENUBAR_LOG_FILE = tmp_path / "menubar.log"
        mod._LOG_LEVEL_FILE = "INFO"

        log = self._fresh_logger()
        mod._setup_logging()
        count_first = len(log.handlers)
        mod._setup_logging()
        count_second = len(log.handlers)

        assert count_second == count_first

        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()


# ---------------------------------------------------------------------------
# menubar_app.py — _on_view_logs
# ---------------------------------------------------------------------------

def _build_view_logs_app(mod):
    """Create a bare AcceleratorMenuBar instance without running __init__."""
    app = mod.AcceleratorMenuBar.__new__(mod.AcceleratorMenuBar)
    return app


class TestOnViewLogs:
    """Tests for AcceleratorMenuBar._on_view_logs log viewer behaviour."""

    def test_shows_stderr_fallback_when_acc_log_missing(self, tmp_path):
        """When accelerator.log is absent, stderr.log section must appear."""
        mod = _import_menubar_fresh()

        acc_log = tmp_path / "accelerator.log"     # does NOT exist
        menubar_log = tmp_path / "menubar.log"
        stderr_log = tmp_path / "stderr.log"
        stderr_log.write_text("launchd stderr line\n")

        mod.LOG_FILE = acc_log
        mod.MENUBAR_LOG_FILE = menubar_log
        mod.STDERR_LOG_FILE = stderr_log

        opened_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "open":
                opened_paths.append(cmd[1])

        mod.subprocess.run = fake_run

        app = _build_view_logs_app(mod)
        app._on_view_logs(None)

        assert opened_paths, "open() was never called"
        content = Path(opened_paths[0]).read_text()
        assert "stderr.log" in content
        assert "launchd stderr line" in content

    def test_shows_stderr_fallback_when_acc_log_empty(self, tmp_path):
        """When accelerator.log is 0 bytes, stderr.log section must appear."""
        mod = _import_menubar_fresh()

        acc_log = tmp_path / "accelerator.log"
        acc_log.write_text("")                     # exists but empty
        menubar_log = tmp_path / "menubar.log"
        stderr_log = tmp_path / "stderr.log"
        stderr_log.write_text("stderr output\n")

        mod.LOG_FILE = acc_log
        mod.MENUBAR_LOG_FILE = menubar_log
        mod.STDERR_LOG_FILE = stderr_log

        opened_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "open":
                opened_paths.append(cmd[1])

        mod.subprocess.run = fake_run

        app = _build_view_logs_app(mod)
        app._on_view_logs(None)

        assert opened_paths
        content = Path(opened_paths[0]).read_text()
        assert "stderr.log" in content
        assert "stderr output" in content

    def test_no_stderr_fallback_when_acc_log_has_content(self, tmp_path):
        """When accelerator.log has content, the stderr.log section must NOT appear."""
        mod = _import_menubar_fresh()

        acc_log = tmp_path / "accelerator.log"
        acc_log.write_text("INFO server started\n")  # non-empty
        menubar_log = tmp_path / "menubar.log"
        stderr_log = tmp_path / "stderr.log"
        stderr_log.write_text("launchd stderr line\n")

        mod.LOG_FILE = acc_log
        mod.MENUBAR_LOG_FILE = menubar_log
        mod.STDERR_LOG_FILE = stderr_log

        opened_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "open":
                opened_paths.append(cmd[1])

        mod.subprocess.run = fake_run

        app = _build_view_logs_app(mod)
        app._on_view_logs(None)

        assert opened_paths
        content = Path(opened_paths[0]).read_text()
        # accelerator.log content should be present
        assert "INFO server started" in content
        # stderr.log fallback section must NOT be present
        assert "launchd fallback" not in content
        assert "launchd stderr line" not in content

    def test_acc_log_content_shown(self, tmp_path):
        """When accelerator.log has content, it must appear in the output."""
        mod = _import_menubar_fresh()

        acc_log = tmp_path / "accelerator.log"
        acc_log.write_text("2024-01-01T00:00:00 [INFO] accelerator: started\n")
        menubar_log = tmp_path / "menubar.log"
        stderr_log = tmp_path / "stderr.log"

        mod.LOG_FILE = acc_log
        mod.MENUBAR_LOG_FILE = menubar_log
        mod.STDERR_LOG_FILE = stderr_log

        opened_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "open":
                opened_paths.append(cmd[1])

        mod.subprocess.run = fake_run

        app = _build_view_logs_app(mod)
        app._on_view_logs(None)

        assert opened_paths
        content = Path(opened_paths[0]).read_text()
        assert "accelerator: started" in content

    def test_menubar_log_shown_in_output(self, tmp_path):
        """menubar.log content must always appear in the viewer output."""
        mod = _import_menubar_fresh()

        acc_log = tmp_path / "accelerator.log"
        acc_log.write_text("server line\n")
        menubar_log = tmp_path / "menubar.log"
        menubar_log.write_text("menubar started\n")
        stderr_log = tmp_path / "stderr.log"

        mod.LOG_FILE = acc_log
        mod.MENUBAR_LOG_FILE = menubar_log
        mod.STDERR_LOG_FILE = stderr_log

        opened_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "open":
                opened_paths.append(cmd[1])

        mod.subprocess.run = fake_run

        app = _build_view_logs_app(mod)
        app._on_view_logs(None)

        assert opened_paths
        content = Path(opened_paths[0]).read_text()
        assert "menubar started" in content
