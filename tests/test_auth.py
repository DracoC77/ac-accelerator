"""
Tests for bearer token auth, rate limiting, file size enforcement,
and security headers.

These tests use FastAPI's TestClient and monkeypatching — no real ML models
are loaded, no real database is written (all DB calls are skipped via mocks
where a real SQLite in-process DB would be used for job creation tests).
"""

from __future__ import annotations

import io
import os
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio_bytes(size_bytes: int = 100) -> bytes:
    """Return a minimal WAV-like byte sequence of the requested size."""
    return b"\x00" * size_bytes


def _audio_file(size_bytes: int = 100, name: str = "test.wav"):
    return ("file", (name, io.BytesIO(_make_audio_bytes(size_bytes)), "audio/wav"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# reset_rate_limits fixture removed — per-IP rate limiter deleted.
# Queue-depth 503 admission control is the only gate; no shared state to reset.


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Return a TestClient with no token configured."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server  # import after env vars are set
    # Reinitialise paths from env
    server.DATA_DIR.mkdir(parents=True, exist_ok=True)
    server.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server._init_db()

    from fastapi.testclient import TestClient as TC
    with TC(server.app) as c:
        yield c


@pytest.fixture()
def authed_client(tmp_path, monkeypatch):
    """Return a TestClient with a known token configured."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-secret-token")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server
    server.DATA_DIR.mkdir(parents=True, exist_ok=True)
    server.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server._init_db()

    from fastapi.testclient import TestClient as TC
    with TC(server.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared mock for job submission endpoints (avoids touching real ML pipelines)
# ---------------------------------------------------------------------------

def _mock_job_create(job_id, job_type, cache_key, file_path, params):
    import server
    return {
        "job_id": job_id,
        "status": "queued",
        "type": job_type,
        "created_at": server._utcnow(),
        "cache_hit": False,
        "poll_url": f"/v1/jobs/{job_id}",
    }


# ---------------------------------------------------------------------------
# Test: auth not required when ACCELERATOR_TOKEN is unset
# ---------------------------------------------------------------------------

def test_no_auth_required_when_token_not_set(client, monkeypatch):
    """When ACCELERATOR_TOKEN is empty, all endpoints are open."""
    import server
    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
    # /health is always open — use it as a sanity check
    resp = client.get("/health")
    assert resp.status_code == 200

    # Protected endpoints should also work with no token
    resp = client.get("/v1/jobs")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test: auth required when ACCELERATOR_TOKEN is set
# ---------------------------------------------------------------------------

def test_auth_required_when_token_set(authed_client, monkeypatch):
    """When ACCELERATOR_TOKEN is set, protected endpoints require a token."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-secret-token")
    resp = authed_client.get("/v1/jobs")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: /health is exempt from auth even when token is set
# ---------------------------------------------------------------------------

def test_health_exempt_from_auth(authed_client, monkeypatch):
    """GET /health must always respond 200 regardless of token presence."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-secret-token")
    # No Authorization header
    resp = authed_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Test: invalid token returns 401
# ---------------------------------------------------------------------------

def test_invalid_token_returns_401(authed_client, monkeypatch):
    """A wrong token must be rejected with 401."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-secret-token")
    resp = authed_client.get(
        "/v1/jobs", headers={"Authorization": "Bearer wrong-token"}
    )
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"] or "missing" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test: valid token returns 200
# ---------------------------------------------------------------------------

def test_valid_token_returns_200(authed_client, monkeypatch):
    """A correct token must be accepted."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-secret-token")
    resp = authed_client.get(
        "/v1/jobs", headers={"Authorization": "Bearer test-secret-token"}
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# test_rate_limit_enforced removed
# ---------------------------------------------------------------------------
# The per-IP sliding-window rate limiter was deleted.
# Queue-depth 503 tests now live in test_server.py (test_transcribe_queue_full_returns_503,
# test_diarize_queue_full_returns_503).


# ---------------------------------------------------------------------------
# Test: file too large returns 413
# ---------------------------------------------------------------------------

def test_file_too_large_returns_413(client, monkeypatch):
    """Files exceeding MAX_FILE_SIZE_MB must be rejected with 413."""
    import server

    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
    # Temporarily set a tiny limit (1 byte) so we can test with a real in-memory file
    monkeypatch.setattr(server, "MAX_FILE_SIZE_MB", 0)
    monkeypatch.setattr(server, "MAX_FILE_SIZE_BYTES", 1)  # 1 byte limit

    resp = client.post(
        "/v1/audio/transcriptions",
        files=[("file", ("big.wav", io.BytesIO(b"\x00" * 100), "audio/wav"))],
    )
    assert resp.status_code == 413
    body = resp.json()
    assert "too large" in body["detail"].lower() or "413" in str(resp.status_code)


# ---------------------------------------------------------------------------
# Test: security headers present on all responses
# ---------------------------------------------------------------------------

def test_security_headers_on_health(client):
    """Security headers must be present on /health responses."""
    resp = client.get("/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"


def test_security_headers_on_protected_endpoint(client):
    """Security headers must be present on protected endpoint responses too."""
    resp = client.get("/v1/jobs")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
