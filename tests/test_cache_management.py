"""
Unit tests for cache management API endpoints + bypass_cache flag.

Covers:
  GET    /cache                     — list entries
  DELETE /cache                     — clear all entries, returns count
  DELETE /cache?key=X               — clear specific entry by cache key
  bypass_cache=true on /v1/audio/transcriptions
  bypass_cache=true on /v1/diarize
  Auth enforced on all cache endpoints
"""

from __future__ import annotations

import io
import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 32

_CACHED_TRANSCRIPTION = {
    "task": "transcribe",
    "language": "en",
    "duration": 5.0,
    "text": "hello world",
    "segments": [],
}

_CACHED_DIARIZATION = {
    "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "is_overlap": False, "overlap_ratio": 0.0}],
    "num_speakers": 1,
    "duration": 5.0,
}


def _wav_file(name: str = "test.wav"):
    return ("file", (name, io.BytesIO(FAKE_WAV), "audio/wav"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# _reset_rate_limits fixture removed — per-IP rate limiter deleted.


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with no auth token, isolated temp DB."""
    from pathlib import Path
    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server
    # Explicitly override module-level path vars (monkeypatch.setenv isn't enough
    # since DB_PATH is resolved at import time)
    server.DATA_DIR = Path(tmp_path / "data")
    server.DB_PATH = Path(tmp_path / "jobs.db")
    server.UPLOAD_DIR = Path(tmp_path / "uploads")
    server.DATA_DIR.mkdir(parents=True, exist_ok=True)
    server.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server._init_db()

    with TestClient(server.app) as c:
        yield c


@pytest.fixture()
def authed_client(tmp_path, monkeypatch):
    """TestClient with a known bearer token."""
    from pathlib import Path
    monkeypatch.setenv("ACCELERATOR_TOKEN", "test-token")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server
    # Explicitly override module-level path vars (monkeypatch.setenv isn't enough
    # since DB_PATH is resolved at import time)
    server.DATA_DIR = Path(tmp_path / "data")
    server.DB_PATH = Path(tmp_path / "jobs.db")
    server.UPLOAD_DIR = Path(tmp_path / "uploads")
    server.DATA_DIR.mkdir(parents=True, exist_ok=True)
    server.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server._init_db()

    with TestClient(server.app) as c:
        yield c


def _insert_cache_entry(cache_key: str, job_type: str = "transcription", hit_count: int = 0) -> None:
    """Insert a fake cache entry directly via server helpers."""
    import server
    server.cache_store(cache_key, job_type, {"result": "fake"})
    if hit_count > 0:
        import server as s
        with s._db_lock:
            conn = s._get_db()
            try:
                conn.execute(
                    "UPDATE cache SET hit_count = ? WHERE cache_key = ?",
                    (hit_count, cache_key),
                )
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# GET /cache — list entries
# ---------------------------------------------------------------------------

class TestGetCache:
    def test_empty_cache_returns_empty_list(self, client):
        resp = client.get("/cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["entries"] == []
        assert body["total"] == 0

    def test_lists_all_entries(self, client):
        _insert_cache_entry("aaa111", job_type="transcription")
        _insert_cache_entry("bbb222", job_type="diarization")
        resp = client.get("/cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["entries"]) == 2
        keys = {e["key"] for e in body["entries"]}
        assert "aaa111" in keys
        assert "bbb222" in keys

    def test_entry_has_required_fields(self, client):
        _insert_cache_entry("ccc333", job_type="transcription", hit_count=3)
        resp = client.get("/cache")
        assert resp.status_code == 200
        entry = resp.json()["entries"][0]
        assert "key" in entry
        assert "job_type" in entry
        assert "created_at" in entry
        assert "expires_at" in entry
        assert "hit_count" in entry

    def test_limit_param_respected(self, client):
        for i in range(5):
            _insert_cache_entry(f"key{i:04d}", job_type="transcription")
        resp = client.get("/cache?limit=3")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["entries"]) == 3
        assert body["total"] == 5

    def test_auth_required_when_token_set(self, authed_client, monkeypatch):
        monkeypatch.setenv("ACCELERATOR_TOKEN", "test-token")
        # No auth header
        resp = authed_client.get("/cache")
        assert resp.status_code == 401

    def test_auth_accepted_with_valid_token(self, authed_client, monkeypatch):
        monkeypatch.setenv("ACCELERATOR_TOKEN", "test-token")
        resp = authed_client.get("/cache", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# DELETE /cache — clear all
# ---------------------------------------------------------------------------

class TestDeleteCacheAll:
    def test_clear_empty_cache(self, client):
        resp = client.delete("/cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == 0
        assert body["keys"] == []

    def test_clear_all_returns_count(self, client):
        _insert_cache_entry("d1d1d1", job_type="transcription")
        _insert_cache_entry("d2d2d2", job_type="diarization")
        _insert_cache_entry("d3d3d3", job_type="transcription")
        resp = client.delete("/cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == 3
        assert len(body["keys"]) == 3
        assert set(body["keys"]) == {"d1d1d1", "d2d2d2", "d3d3d3"}

    def test_cache_is_empty_after_clear(self, client):
        _insert_cache_entry("e1e1e1")
        client.delete("/cache")
        # Confirm cache is now empty
        resp = client.get("/cache")
        assert resp.json()["total"] == 0

    def test_auth_required_when_token_set(self, authed_client, monkeypatch):
        monkeypatch.setenv("ACCELERATOR_TOKEN", "test-token")
        resp = authed_client.delete("/cache")
        assert resp.status_code == 401

    def test_auth_accepted_with_valid_token(self, authed_client, monkeypatch):
        monkeypatch.setenv("ACCELERATOR_TOKEN", "test-token")
        resp = authed_client.delete("/cache", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# DELETE /cache?key=X — clear specific entry
# ---------------------------------------------------------------------------

class TestDeleteCacheByKey:
    def test_delete_specific_entry(self, client):
        _insert_cache_entry("f1f1f1")
        _insert_cache_entry("f2f2f2")
        resp = client.delete("/cache?key=f1f1f1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == 1
        assert body["keys"] == ["f1f1f1"]

    def test_other_entries_survive_specific_delete(self, client):
        _insert_cache_entry("g1g1g1")
        _insert_cache_entry("g2g2g2")
        client.delete("/cache?key=g1g1g1")
        resp = client.get("/cache")
        assert resp.json()["total"] == 1
        assert resp.json()["entries"][0]["key"] == "g2g2g2"

    def test_delete_nonexistent_key_returns_404(self, client):
        resp = client.delete("/cache?key=doesnotexist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_auth_required_for_delete_by_key(self, authed_client, monkeypatch):
        monkeypatch.setenv("ACCELERATOR_TOKEN", "test-token")
        resp = authed_client.delete("/cache?key=anything")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# bypass_cache=true on /v1/audio/transcriptions
# ---------------------------------------------------------------------------

class TestBypassCacheTranscription:
    def test_bypass_skips_cache_hit(self, client):
        """When bypass_cache=true, a cache hit must be ignored and a new job queued."""
        # Pre-seed the cache with a known key
        import server

        # Mock cache_lookup to return a hit (would normally return cached)
        with (
            patch.object(server, "cache_lookup", return_value=_CACHED_TRANSCRIPTION) as mock_lookup,
            patch.object(server, "job_create", return_value={
                "job_id": "test-job-001",
                "status": "queued",
                "type": "transcription",
                "created_at": server._utcnow(),
                "cache_hit": False,
                "poll_url": "/v1/jobs/test-job-001",
            }),
            patch.object(server, "_enqueue_job"),
            patch.object(server, "job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/audio/transcriptions?bypass_cache=true",
                files=[_wav_file()],
            )
        # Should get 202 (queued), NOT 200 cache hit
        assert resp.status_code == 202
        # cache_lookup should NOT have been called (bypassed)
        mock_lookup.assert_not_called()

    def test_bypass_false_uses_cache(self, client):
        """Without bypass_cache, a cache hit returns 200 immediately."""
        import server

        with (
            patch.object(server, "cache_lookup", return_value=_CACHED_TRANSCRIPTION),
            patch.object(server, "job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
            )
        assert resp.status_code == 200
        assert resp.json()["cache_hit"] is True

    def test_bypass_true_string_values(self, client):
        """bypass_cache accepts 'true', '1', 'yes'."""
        import server

        for val in ("true", "1", "yes"):
            with (
                patch.object(server, "cache_lookup", return_value=_CACHED_TRANSCRIPTION) as mock_lookup,
                patch.object(server, "job_create", return_value={
                    "job_id": f"test-{val}",
                    "status": "queued",
                    "type": "transcription",
                    "created_at": server._utcnow(),
                    "cache_hit": False,
                    "poll_url": f"/v1/jobs/test-{val}",
                }),
                patch.object(server, "_enqueue_job"),
                patch.object(server, "job_count_by_status", return_value=0),
            ):
                resp = client.post(
                    f"/v1/audio/transcriptions?bypass_cache={val}",
                    files=[_wav_file()],
                )
            assert resp.status_code == 202, f"bypass_cache={val} should give 202, got {resp.status_code}"
            mock_lookup.assert_not_called()


# ---------------------------------------------------------------------------
# bypass_cache=true on /v1/diarize
# ---------------------------------------------------------------------------

class TestBypassCacheDiarize:
    def test_bypass_skips_cache_hit(self, client):
        """When bypass_cache=true, a cache hit on /v1/diarize must be ignored."""
        import server

        with (
            patch.object(server, "cache_lookup", return_value=_CACHED_DIARIZATION) as mock_lookup,
            patch.object(server, "job_create", return_value={
                "job_id": "diarize-001",
                "status": "queued",
                "type": "diarization",
                "created_at": server._utcnow(),
                "cache_hit": False,
                "poll_url": "/v1/jobs/diarize-001",
            }),
            patch.object(server, "_enqueue_job"),
            patch.object(server, "job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/diarize?bypass_cache=true",
                files=[_wav_file()],
            )
        assert resp.status_code == 202
        mock_lookup.assert_not_called()

    def test_bypass_false_uses_cache(self, client):
        """Without bypass_cache, /v1/diarize returns a cached result."""
        import server

        with (
            patch.object(server, "cache_lookup", return_value=_CACHED_DIARIZATION),
            patch.object(server, "job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/diarize",
                files=[_wav_file()],
            )
        assert resp.status_code == 200
        assert resp.json()["cache_hit"] is True
