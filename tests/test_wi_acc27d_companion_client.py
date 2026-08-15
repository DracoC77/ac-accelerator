"""WI-ACC-27d: companion_client.py tests.

These tests cover the cross-platform server-polling and service-control
layer used by both ``menubar_app.py`` (Mac) and ``tray_app.py`` (Windows).

No display / GUI deps required: companion_client.py is pure stdlib +
``requests`` so this whole module is unit-testable in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

import companion_client


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


def test_get_status_unreachable_on_connection_error():
    """A ConnectionError must surface as reachable=False with a populated error."""
    with patch(
        "companion_client.requests.get",
        side_effect=requests.exceptions.ConnectionError("conn refused"),
    ):
        status = companion_client.get_status("http://localhost:8765")
    assert not status.reachable
    assert not status.healthy
    assert status.error is not None
    assert "conn refused" in status.error


def test_get_status_unreachable_on_timeout():
    """A Timeout must also surface as reachable=False."""
    with patch(
        "companion_client.requests.get",
        side_effect=requests.exceptions.Timeout("timed out"),
    ):
        status = companion_client.get_status("http://localhost:8765")
    assert not status.reachable
    assert status.error is not None
    assert "timeout" in status.error.lower()


def test_get_status_unreachable_on_500():
    """Non-200 responses are not reachable."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    with patch("companion_client.requests.get", return_value=mock_resp):
        status = companion_client.get_status("http://localhost:8765")
    assert not status.reachable
    assert "500" in (status.error or "")


def test_get_status_parses_current_server_shape():
    """Parse the actual /health shape from server.py (nested queue/models)."""
    health_payload = {
        "status": "healthy",
        "version": "0.1",
        "backend_name": "faster-whisper",
        "models": {
            "whisper": "large-v3-turbo",
            "whisper_loaded": True,
            "backend_name": "faster-whisper",
            "diarization": "pyannote/3.1",
            "diarization_loaded": True,
        },
        "queue": {"pending": 2, "processing": 1},
        "uptime_seconds": 3600.0,
    }
    memory_payload = {
        "rss_mb": 1234.5,
        "models": {"whisper_loaded": True, "diarization_loaded": True},
    }

    def fake_get(url, *args, **kwargs):
        m = MagicMock()
        m.status_code = 200
        if url.endswith("/health"):
            m.json.return_value = health_payload
        else:
            m.json.return_value = memory_payload
        return m

    with patch("companion_client.requests.get", side_effect=fake_get):
        status = companion_client.get_status("http://localhost:8765")

    assert status.reachable
    assert status.healthy
    assert status.processing_count == 1
    assert status.pending_count == 2
    assert status.whisper_loaded
    assert status.pyannote_loaded
    assert status.uptime_seconds == 3600.0
    assert status.backend_name == "faster-whisper"
    assert status.rss_mb == 1234.5


def test_get_status_parses_flat_health_shape():
    """Parse the simpler flat /health shape from the WI-ACC-27d spec example."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "processing": 0,
        "whisper_loaded": True,
        "pyannote_loaded": False,
        "uptime_seconds": 3600,
        "backend_name": "faster-whisper",
    }
    with patch("companion_client.requests.get", return_value=mock_resp):
        status = companion_client.get_status("http://localhost:8765")
    assert status.reachable
    assert status.healthy
    assert status.whisper_loaded
    assert not status.pyannote_loaded
    assert status.backend_name == "faster-whisper"
    assert status.uptime_seconds == 3600


def test_get_status_memory_failure_does_not_break_status():
    """If /memory fails but /health succeeds, status is still reachable."""
    def fake_get(url, *args, **kwargs):
        if url.endswith("/memory"):
            raise requests.exceptions.ConnectionError("nope")
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {
            "status": "healthy",
            "queue": {"pending": 0, "processing": 0},
            "models": {"whisper_loaded": True, "diarization_loaded": True},
            "uptime_seconds": 10,
            "backend_name": "test",
        }
        return m

    with patch("companion_client.requests.get", side_effect=fake_get):
        status = companion_client.get_status("http://localhost:8765")
    assert status.reachable
    assert status.rss_mb == 0.0


# ---------------------------------------------------------------------------
# get_logs
# ---------------------------------------------------------------------------


def test_get_logs_returns_text_on_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "log line 1\nlog line 2"
    with patch("companion_client.requests.get", return_value=mock_resp):
        logs = companion_client.get_logs("http://localhost:8765")
    assert "log line 1" in logs
    assert "log line 2" in logs


def test_get_logs_returns_error_string_on_failure():
    with patch(
        "companion_client.requests.get",
        side_effect=requests.exceptions.ConnectionError("nope"),
    ):
        logs = companion_client.get_logs("http://localhost:8765")
    assert logs.startswith("(error")


def test_get_logs_passes_source_param():
    """The ``source`` kwarg is forwarded as a query param to /logs."""
    captured = {}

    def fake_get(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        m = MagicMock()
        m.status_code = 200
        m.text = "ok"
        return m

    with patch("companion_client.requests.get", side_effect=fake_get):
        companion_client.get_logs("http://localhost:8765", n=50, source="server")

    assert captured["params"]["source"] == "server"
    assert captured["params"]["lines"] == 50


# ---------------------------------------------------------------------------
# Service control — platform dispatch
# ---------------------------------------------------------------------------


def _ok_subprocess_run():
    mock = MagicMock()
    mock.returncode = 0
    mock.stderr = ""
    return mock


def test_start_service_on_windows_calls_nssm_start():
    with patch("companion_client._system", return_value="Windows"), \
         patch("companion_client.subprocess.run", return_value=_ok_subprocess_run()) as run:
        assert companion_client.start_service() is True
    args = run.call_args[0][0]
    assert args[1] == "start"
    assert args[2] == companion_client.NSSM_SERVICE_NAME


def test_stop_service_on_windows_calls_nssm_stop():
    with patch("companion_client._system", return_value="Windows"), \
         patch("companion_client.subprocess.run", return_value=_ok_subprocess_run()) as run:
        assert companion_client.stop_service() is True
    args = run.call_args[0][0]
    assert args[1] == "stop"
    assert args[2] == companion_client.NSSM_SERVICE_NAME


def test_restart_service_on_windows_calls_nssm_restart():
    with patch("companion_client._system", return_value="Windows"), \
         patch("companion_client.subprocess.run", return_value=_ok_subprocess_run()) as run:
        assert companion_client.restart_service() is True
    args = run.call_args[0][0]
    assert args[1] == "restart"


def test_restart_service_on_darwin_uses_launchctl_kickstart_k():
    """Mac restart uses ``launchctl kickstart -k`` for atomic stop+start."""
    with patch("companion_client._system", return_value="Darwin"), \
         patch("companion_client.subprocess.run", return_value=_ok_subprocess_run()) as run:
        assert companion_client.restart_service() is True
    args = run.call_args[0][0]
    assert args[0] == "launchctl"
    assert args[1] == "kickstart"
    assert "-k" in args
    assert any(companion_client.LAUNCHD_PLIST_LABEL in a for a in args)


def test_stop_service_on_darwin_uses_launchctl_stop():
    with patch("companion_client._system", return_value="Darwin"), \
         patch("companion_client.subprocess.run", return_value=_ok_subprocess_run()) as run:
        assert companion_client.stop_service() is True
    args = run.call_args[0][0]
    assert args == ["launchctl", "stop", companion_client.LAUNCHD_PLIST_LABEL]


def test_service_returns_false_on_subprocess_failure():
    fail = MagicMock(returncode=1, stderr="boom")
    with patch("companion_client._system", return_value="Windows"), \
         patch("companion_client.subprocess.run", return_value=fail):
        assert companion_client.start_service() is False


def test_service_returns_false_on_unknown_platform():
    with patch("companion_client._system", return_value="OS/2"):
        assert companion_client.start_service() is False
        assert companion_client.stop_service() is False
        assert companion_client.restart_service() is False


def test_service_returns_false_when_command_missing():
    """FileNotFoundError (e.g. nssm.exe not on PATH) returns False, not raise."""
    with patch("companion_client._system", return_value="Windows"), \
         patch(
             "companion_client.subprocess.run",
             side_effect=FileNotFoundError("nssm.exe not found"),
         ):
        assert companion_client.start_service() is False


# ---------------------------------------------------------------------------
# tray_app import guard
# ---------------------------------------------------------------------------


def test_tray_app_importable_without_pystray():
    """tray_app.py must import on CI hosts where pystray is not installed."""
    import importlib
    import tray_app  # noqa: F401  — the import itself is the assertion
    importlib.reload(tray_app)
    # If pystray is not installed in this venv, _GUI_AVAILABLE will be False
    # and run() will refuse cleanly — both are acceptable here.
    assert hasattr(tray_app, "AcceleratorTray")
    assert hasattr(tray_app, "main")


def test_tray_app_classify_handles_all_states():
    """Smoke-test the icon-state classifier without needing pystray."""
    import tray_app
    # Offline
    s = companion_client.AcceleratorStatus(reachable=False)
    state, tip = tray_app._classify(s)
    assert state == "offline"
    assert "Offline" in tip
    # Processing
    s = companion_client.AcceleratorStatus(
        reachable=True, healthy=True, processing_count=2, whisper_loaded=True
    )
    state, _ = tray_app._classify(s)
    assert state == "processing"
    # Healthy
    s = companion_client.AcceleratorStatus(
        reachable=True, healthy=True, whisper_loaded=True, backend_name="fw"
    )
    state, _ = tray_app._classify(s)
    assert state == "healthy"
    # Idle (server up but model not loaded)
    s = companion_client.AcceleratorStatus(
        reachable=True, healthy=True, whisper_loaded=False
    )
    state, _ = tray_app._classify(s)
    assert state == "idle"
