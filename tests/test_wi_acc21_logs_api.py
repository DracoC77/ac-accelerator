"""
WI-ACC-21: Unit tests for the GET /logs API endpoint.

Verifies:
  - Auth is enforced (401 without token when ACCELERATOR_TOKEN is set)
  - GET /logs returns last N lines (default 200) with source=all (default)
  - source=server returns server log (accelerator.log or stderr.log fallback)
  - source=menubar returns menubar log
  - source=all returns both sources in one response
  - format=text returns plain text with section headers (curl-friendly)
  - Missing log files reported as exists=false, empty content (not a 404)
  - lines capped at 1000; returns 400 if over limit
  - Invalid source / format params return 400
  - _resolve_server_log fallback: uses stderr.log when accelerator.log missing/empty
  - _tail_file: returns correct tail, handles empty file, handles missing file
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# WI-ACC-28: _reset_rate_limits fixture removed — per-IP rate limiter deleted.


@pytest.fixture()
def client_no_auth(tmp_path, monkeypatch):
    """TestClient with no auth token, isolated temp data dir."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    server._init_db()

    from fastapi.testclient import TestClient as TC
    with TC(server.app) as c:
        yield c, tmp_path / "data"


@pytest.fixture()
def client_with_auth(tmp_path, monkeypatch):
    """TestClient with a bearer token configured."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-secret")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    server._init_db()

    from fastapi.testclient import TestClient as TC
    with TC(server.app) as c:
        yield c, tmp_path / "data"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_log(data_dir: Path, name: str, content: str) -> Path:
    log_path = data_dir / "logs" / name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")
    return log_path


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------

class TestLogsAuth:
    def test_401_without_token_when_auth_configured(self, client_with_auth):
        client, _ = client_with_auth
        resp = client.get("/logs")
        assert resp.status_code == 401

    def test_401_bad_token(self, client_with_auth):
        client, _ = client_with_auth
        resp = client.get("/logs", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401

    def test_200_with_correct_token(self, client_with_auth):
        client, _ = client_with_auth
        resp = client.get("/logs", headers={"Authorization": "Bearer test-secret"})
        assert resp.status_code == 200

    def test_200_no_auth_when_token_not_set(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# JSON format — default behaviour
# ---------------------------------------------------------------------------

class TestLogsJsonFormat:
    def test_default_response_shape(self, client_no_auth):
        """Default call returns sources dict with server and menubar keys."""
        client, data_dir = client_no_auth
        resp = client.get("/logs")
        assert resp.status_code == 200
        body = resp.json()
        assert "sources" in body
        assert "timestamp" in body
        # default source=all → both keys present
        assert "server" in body["sources"]
        assert "menubar" in body["sources"]

    def test_source_keys_have_expected_fields(self, client_no_auth):
        client, data_dir = client_no_auth
        resp = client.get("/logs")
        body = resp.json()
        for src in ("server", "menubar"):
            s = body["sources"][src]
            assert "path" in s
            assert "lines_returned" in s
            assert "content" in s
            assert "size_bytes" in s
            assert "exists" in s

    def test_missing_logs_reported_as_exists_false(self, client_no_auth):
        """When log files don't exist, exists=false and content=[] — not a 404.

        Note: the TestClient lifespan calls _setup_file_logging() which creates
        accelerator.log; we only test menubar.log here (never auto-created).
        """
        client, data_dir = client_no_auth
        resp = client.get("/logs?source=menubar")
        assert resp.status_code == 200
        body = resp.json()
        # menubar.log is never auto-created by the lifespan
        m = body["sources"]["menubar"]
        assert m["exists"] is False
        assert m["content"] == []
        assert m["lines_returned"] == 0

    def test_source_server_returns_server_log_content(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "line1\nline2\nline3\n")
        resp = client.get("/logs?source=server")
        body = resp.json()
        assert "server" in body["sources"]
        assert "menubar" not in body["sources"]
        s = body["sources"]["server"]
        assert s["exists"] is True
        assert s["content"] == ["line1", "line2", "line3"]
        assert s["lines_returned"] == 3

    def test_source_menubar_returns_menubar_log_content(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "menubar.log", "menubar line A\nmenubar line B\n")
        resp = client.get("/logs?source=menubar")
        body = resp.json()
        assert "menubar" in body["sources"]
        assert "server" not in body["sources"]
        m = body["sources"]["menubar"]
        assert m["exists"] is True
        assert m["content"] == ["menubar line A", "menubar line B"]

    def test_source_all_returns_both(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "server line\n")
        _write_log(data_dir, "menubar.log", "menubar line\n")
        resp = client.get("/logs?source=all")
        body = resp.json()
        assert "server" in body["sources"]
        assert "menubar" in body["sources"]
        assert body["sources"]["server"]["content"] == ["server line"]
        assert body["sources"]["menubar"]["content"] == ["menubar line"]

    def test_lines_param_limits_output(self, client_no_auth):
        client, data_dir = client_no_auth
        content = "\n".join(f"line{i}" for i in range(50))
        _write_log(data_dir, "accelerator.log", content + "\n")
        resp = client.get("/logs?source=server&lines=10")
        body = resp.json()
        s = body["sources"]["server"]
        assert s["lines_returned"] == 10
        # Should be the last 10 lines
        assert s["content"] == [f"line{i}" for i in range(40, 50)]

    def test_size_bytes_reported(self, client_no_auth):
        """size_bytes reflects the actual file size at query time."""
        client, data_dir = client_no_auth
        # Use menubar.log to avoid the auto-created accelerator.log
        log_content = "hello\nworld\n"
        log_path = _write_log(data_dir, "menubar.log", log_content)
        resp = client.get("/logs?source=menubar")
        body = resp.json()
        expected_size = log_path.stat().st_size
        assert body["sources"]["menubar"]["size_bytes"] == expected_size


# ---------------------------------------------------------------------------
# Server log fallback: accelerator.log → stderr.log
# ---------------------------------------------------------------------------

class TestServerLogFallback:
    def test_uses_accelerator_log_when_present_and_nonempty(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "from accelerator\n")
        _write_log(data_dir, "stderr.log", "from stderr\n")
        resp = client.get("/logs?source=server")
        body = resp.json()
        s = body["sources"]["server"]
        assert "from accelerator" in s["content"]
        assert "from stderr" not in s["content"]
        assert "accelerator.log" in s["path"]

    def test_falls_back_to_stderr_when_accelerator_log_missing(self, tmp_path, monkeypatch):
        """Fallback to stderr.log when accelerator.log is absent.

        We test this via the helper directly rather than through TestClient, because
        the TestClient lifespan always creates and writes accelerator.log.
        """
        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        logs_dir = tmp_path / "data" / "logs"
        logs_dir.mkdir(parents=True)
        # accelerator.log does NOT exist; only stderr.log
        stderr_log = logs_dir / "stderr.log"
        stderr_log.write_text("stderr fallback line\n")

        result = server._resolve_server_log()
        assert result.name == "stderr.log"

        info = server._log_source_info(result, 200)
        assert info["exists"] is True
        assert "stderr fallback line" in info["content"]
        assert "stderr.log" in info["path"]

    def test_falls_back_to_stderr_when_accelerator_log_empty(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "")  # 0 bytes
        _write_log(data_dir, "stderr.log", "stderr fallback line\n")
        resp = client.get("/logs?source=server")
        body = resp.json()
        s = body["sources"]["server"]
        assert "stderr.log" in s["path"]
        assert "stderr fallback line" in s["content"]

    def test_fallback_path_reported_in_exists_false_when_no_stderr_either(self, tmp_path, monkeypatch):
        """When both accelerator.log and stderr.log are absent, exists=false.

        We test via helpers directly — TestClient lifespan creates accelerator.log.
        """
        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data" / "logs").mkdir(parents=True)
        # Neither accelerator.log nor stderr.log exist
        result = server._resolve_server_log()
        assert result.name == "stderr.log"
        info = server._log_source_info(result, 200)
        assert info["exists"] is False
        assert "stderr.log" in info["path"]


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------

class TestLogsTextFormat:
    def test_text_format_returns_plain_text(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs?format=text")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_text_format_includes_section_headers(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "server line\n")
        _write_log(data_dir, "menubar.log", "menubar line\n")
        resp = client.get("/logs?format=text&source=all")
        body = resp.text
        assert "===" in body
        assert "accelerator.log" in body or "stderr.log" in body
        assert "menubar.log" in body

    def test_text_format_server_only(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "srv line1\nsrv line2\n")
        resp = client.get("/logs?format=text&source=server")
        body = resp.text
        assert "srv line1" in body
        assert "srv line2" in body
        assert "menubar" not in body

    def test_text_format_menubar_only(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "menubar.log", "mb line\n")
        resp = client.get("/logs?format=text&source=menubar")
        body = resp.text
        assert "mb line" in body
        assert "accelerator" not in body

    def test_text_format_missing_file_shows_not_found_marker(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs?format=text&source=all")
        body = resp.text
        assert "file not found" in body

    def test_text_format_content_when_source_all(self, client_no_auth):
        client, data_dir = client_no_auth
        _write_log(data_dir, "accelerator.log", "server content\n")
        _write_log(data_dir, "menubar.log", "menubar content\n")
        resp = client.get("/logs?format=text&source=all")
        body = resp.text
        assert "server content" in body
        assert "menubar content" in body


# ---------------------------------------------------------------------------
# lines cap and validation
# ---------------------------------------------------------------------------

class TestLogsValidation:
    def test_lines_over_1000_returns_400(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs?lines=1001")
        assert resp.status_code == 400
        assert "1000" in resp.json()["detail"]

    def test_lines_exactly_1000_is_ok(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs?lines=1000")
        assert resp.status_code == 200

    def test_lines_default_is_200(self, client_no_auth):
        client, data_dir = client_no_auth
        # Write 300 lines
        content = "\n".join(f"line{i}" for i in range(300))
        _write_log(data_dir, "accelerator.log", content + "\n")
        resp = client.get("/logs?source=server")
        body = resp.json()
        s = body["sources"]["server"]
        assert s["lines_returned"] == 200

    def test_invalid_source_returns_400(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs?source=badvalue")
        assert resp.status_code == 400

    def test_invalid_format_returns_400(self, client_no_auth):
        client, _ = client_no_auth
        resp = client.get("/logs?format=xml")
        assert resp.status_code == 400

    def test_lines_zero_or_negative_returns_422(self, client_no_auth):
        """FastAPI Query(ge=1) should enforce min value."""
        client, _ = client_no_auth
        resp = client.get("/logs?lines=0")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestTailFile:
    def test_returns_last_n_lines(self, tmp_path):
        import server
        log = tmp_path / "test.log"
        log.write_text("\n".join(f"line{i}" for i in range(20)) + "\n")
        result = server._tail_file(log, 5)
        assert result == [f"line{i}" for i in range(15, 20)]

    def test_returns_all_lines_when_fewer_than_n(self, tmp_path):
        import server
        log = tmp_path / "test.log"
        log.write_text("a\nb\nc\n")
        result = server._tail_file(log, 100)
        assert result == ["a", "b", "c"]

    def test_returns_empty_list_for_missing_file(self, tmp_path):
        import server
        log = tmp_path / "nonexistent.log"
        result = server._tail_file(log, 10)
        assert result == []

    def test_returns_empty_list_for_empty_file(self, tmp_path):
        import server
        log = tmp_path / "empty.log"
        log.write_text("")
        result = server._tail_file(log, 10)
        assert result == []

    def test_strips_trailing_newlines_from_lines(self, tmp_path):
        import server
        log = tmp_path / "test.log"
        log.write_text("hello\nworld\n")
        result = server._tail_file(log, 10)
        assert result == ["hello", "world"]


class TestResolveServerLog:
    def test_returns_accelerator_log_when_exists_and_nonempty(self, tmp_path, monkeypatch):
        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data" / "logs").mkdir(parents=True)
        acc_log = tmp_path / "data" / "logs" / "accelerator.log"
        acc_log.write_text("some content")
        result = server._resolve_server_log()
        assert result == acc_log

    def test_falls_back_to_stderr_when_accelerator_missing(self, tmp_path, monkeypatch):
        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data" / "logs").mkdir(parents=True)
        # accelerator.log does NOT exist
        result = server._resolve_server_log()
        assert result.name == "stderr.log"

    def test_falls_back_to_stderr_when_accelerator_empty(self, tmp_path, monkeypatch):
        import server
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data" / "logs").mkdir(parents=True)
        acc_log = tmp_path / "data" / "logs" / "accelerator.log"
        acc_log.write_text("")  # 0 bytes
        result = server._resolve_server_log()
        assert result.name == "stderr.log"
