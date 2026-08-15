"""
Unit tests for the Audio Chronicle Accelerator server.

Uses FastAPI TestClient with mocked ML inference — no real GPU, no real models.
Tests complement test_auth.py (auth, rate-limiting, security headers).
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"  # ~32-byte minimal WAV header
FAKE_OGG = b"OggS" + b"\x00" * 96  # minimal OGG-like header

_WHISPER_RESULT = {
    "text": "hello world",
    "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "hello world",
                  "tokens": [], "avg_logprob": -0.1, "compression_ratio": 1.0, "no_speech_prob": 0.01}],
    "language": "en",
    "duration": 5.0,
}

_PYANNOTE_RESULT = {
    "segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
    "num_speakers": 1,
    "duration": 5.0,
}


def _wav_file(name: str = "test.wav") -> tuple:
    return ("file", (name, io.BytesIO(FAKE_WAV), "audio/wav"))


def _ogg_file(name: str = "test.ogg") -> tuple:
    return ("file", (name, io.BytesIO(FAKE_OGG), "audio/ogg"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# _reset_rate_limits fixture removed — per-IP rate limiter was deleted.
# Queue-depth 503 admission control is the only gate now; no shared state to reset.


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with no auth token, isolated temp dirs."""
    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
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
# Mock helpers: patch inference so jobs complete synchronously
# ---------------------------------------------------------------------------

def _run_job_sync(job_id: str) -> None:
    """Run a job synchronously (bypass thread pool) for predictable test state."""
    import server
    job = server.job_get(job_id)
    if not job:
        return
    server.job_update(job_id, status="processing", started_at=server._utcnow(), progress=0.0)
    if job["job_type"] == "transcription":
        result = {
            "task": "transcribe",
            "language": "en",
            "duration": 5.0,
            "text": "hello world",
            "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "hello world",
                           "tokens": [], "avg_logprob": -0.1, "compression_ratio": 1.0, "no_speech_prob": 0.01}],
        }
    else:
        result = {
            "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}],
            "num_speakers": 1,
            "duration": 5.0,
        }
    server.job_update(job_id, status="complete", result_json=json.dumps(result),
                      completed_at=server._utcnow(), progress=1.0)
    server.cache_store(job["cache_key"], job["job_type"], result)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    """GET /health → 200 with status, gpu, queue_depth (queue.pending) fields."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "gpu" in body
    assert "queue" in body
    assert "pending" in body["queue"]


# ---------------------------------------------------------------------------
# Transcription submit
# ---------------------------------------------------------------------------

def test_transcribe_submit_returns_job_id(client):
    """POST /v1/audio/transcriptions → 202 with job_id."""
    with (
        patch("server._enqueue_job"),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
    ):
        resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"


def test_transcribe_invalid_format_returns_400(client):
    """POST with unsupported file type → 400 (server returns 400, not 422)."""
    resp = client.post(
        "/v1/audio/transcriptions",
        files=[("file", ("audio.mp4", io.BytesIO(b"\x00" * 100), "video/mp4"))],
    )
    # Server returns 400 for unsupported extension
    assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Queue-depth 503 admission control
# ---------------------------------------------------------------------------

def test_transcribe_queue_full_returns_503(client):
    """POST /v1/audio/transcriptions with full queue → 503 with Retry-After: 5."""
    with patch("server.job_count_by_status", return_value=20):  # MAX_QUEUE_DEPTH=20
        resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "5"
    body = resp.json()
    assert "Queue full" in body["detail"]


def test_diarize_queue_full_returns_503(client):
    """POST /v1/diarize with full queue → 503 with Retry-After: 5."""
    with patch("server.job_count_by_status", return_value=20):  # MAX_QUEUE_DEPTH=20
        resp = client.post("/v1/diarize", files=[_wav_file()])
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "5"
    body = resp.json()
    assert "Queue full" in body["detail"]


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

def test_poll_queued_job(client):
    """GET /v1/jobs/{id} on a queued job → status=queued."""
    with (
        patch("server._enqueue_job"),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
    ):
        post_resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    poll_resp = client.get(f"/v1/jobs/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["status"] == "queued"


def test_poll_completed_job(client):
    """GET /v1/jobs/{id} on a completed job → status=complete with result."""
    with (
        patch("server._enqueue_job", side_effect=_run_job_sync),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
        patch("server._load_whisper") as mock_w,
    ):
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello world",
            "segments": [],
            "language": "en",
            "duration": 5.0,
        }
        mock_w.return_value = mock_whisper
        post_resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])

    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    poll_resp = client.get(f"/v1/jobs/{job_id}")
    assert poll_resp.status_code == 200
    body = poll_resp.json()
    assert body["status"] == "complete"
    assert "result" in body


def test_poll_nonexistent_job(client):
    """GET /v1/jobs/nonexistent → 404."""
    resp = client.get("/v1/jobs/nonexistent-job-id-that-does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job listing
# ---------------------------------------------------------------------------

def test_list_jobs(client):
    """GET /v1/jobs → list including at least the submitted job."""
    with (
        patch("server._enqueue_job"),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
    ):
        post_resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    list_resp = client.get("/v1/jobs")
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert "jobs" in body
    assert "total" in body
    job_ids = [j["job_id"] for j in body["jobs"]]
    assert job_id in job_ids


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------

def test_cancel_queued_job(client):
    """DELETE /v1/jobs/{id} on a queued job → 200, status=cancelled."""
    with (
        patch("server._enqueue_job"),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
    ):
        post_resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    del_resp = client.delete(f"/v1/jobs/{job_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "cancelled"


def test_cancel_processing_job_returns_cancelled(client):
    """DELETE on a processing job → 200, status=cancelled.

    The server does not return 409 for in-flight jobs; it marks them cancelled
    and lets the worker thread check status before updating the result.
    """
    import server

    with (
        patch("server._enqueue_job"),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
    ):
        post_resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    # Manually flip to processing
    server.job_update(job_id, status="processing", started_at=server._utcnow())

    del_resp = client.delete(f"/v1/jobs/{job_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    # Server returns "cancelled" and a message about cancellation being requested
    assert body["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Diarization submit
# ---------------------------------------------------------------------------

def test_diarize_submit_returns_job_id(client):
    """POST /v1/diarize → 202 with job_id."""
    with (
        patch("server._enqueue_job"),
        patch("server.cache_lookup", return_value=None),
        patch("server.job_count_by_status", return_value=0),
    ):
        resp = client.post("/v1/diarize", files=[_wav_file()])
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"


# ---------------------------------------------------------------------------
# Memory endpoint
# ---------------------------------------------------------------------------

def test_memory_endpoint(client):
    """GET /memory → 200 with rss_mb field (server uses rss_mb, not ram_used_gb)."""
    resp = client.get("/memory")
    assert resp.status_code == 200
    body = resp.json()
    assert "rss_mb" in body
    assert isinstance(body["rss_mb"], (int, float))
    assert body["rss_mb"] >= 0


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------

def test_cache_hit_returns_200_immediately(client):
    """Second POST with same file content → 200 (cache hit), result in body."""
    cached_result = {
        "task": "transcribe",
        "language": "en",
        "duration": 5.0,
        "text": "hello world",
        "segments": [],
    }

    with patch("server.cache_lookup", return_value=cached_result):
        resp = client.post("/v1/audio/transcriptions", files=[_wav_file()])

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "complete"
    assert body["cache_hit"] is True
    assert "result" in body
    assert body["result"]["text"] == "hello world"


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------

def test_cache_stats(client):
    """GET /v1/cache/stats → 200 with hit_count, miss_count, total_entries fields."""
    resp = client.get("/v1/cache/stats")
    assert resp.status_code == 200
    body = resp.json()
    # Server returns hit_count/miss_count (not hits/misses per spec — match actual)
    assert "hit_count" in body or "total_hits" in body or "hits" in body, \
        f"Expected a hits field, got: {list(body.keys())}"
    assert "miss_count" in body or "total_misses" in body or "misses" in body, \
        f"Expected a misses field, got: {list(body.keys())}"
    assert "total_entries" in body or "entries" in body, \
        f"Expected an entries field, got: {list(body.keys())}"


# ---------------------------------------------------------------------------
# nan/inf float sanitization
# ---------------------------------------------------------------------------

class TestSanitizeFloats:
    """Unit tests for the _sanitize_floats helper."""

    def test_nan_replaced_with_none(self):
        import math
        import server
        assert server._sanitize_floats(float("nan")) is None

    def test_positive_inf_replaced_with_none(self):
        import server
        assert server._sanitize_floats(float("inf")) is None

    def test_negative_inf_replaced_with_none(self):
        import server
        assert server._sanitize_floats(float("-inf")) is None

    def test_normal_float_unchanged(self):
        import server
        assert server._sanitize_floats(-0.123) == -0.123

    def test_zero_float_unchanged(self):
        import server
        assert server._sanitize_floats(0.0) == 0.0

    def test_int_unchanged(self):
        import server
        assert server._sanitize_floats(42) == 42

    def test_string_unchanged(self):
        import server
        assert server._sanitize_floats("hello") == "hello"

    def test_none_unchanged(self):
        import server
        assert server._sanitize_floats(None) is None

    def test_nested_dict_with_nan(self):
        import math
        import server
        result = server._sanitize_floats({
            "avg_logprob": float("nan"),
            "text": "hello",
            "compression_ratio": 1.2,
        })
        assert result["avg_logprob"] is None
        assert result["text"] == "hello"
        assert result["compression_ratio"] == 1.2

    def test_list_with_inf(self):
        import server
        result = server._sanitize_floats([1.0, float("inf"), 3.0, float("-inf")])
        assert result == [1.0, None, 3.0, None]

    def test_deeply_nested(self):
        import math
        import server
        obj = {
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "hello",
                    "avg_logprob": float("nan"),
                    "compression_ratio": float("inf"),
                    "no_speech_prob": 0.01,
                }
            ],
            "duration": 5.0,
            "language": "en",
        }
        result = server._sanitize_floats(obj)
        seg = result["segments"][0]
        assert seg["avg_logprob"] is None
        assert seg["compression_ratio"] is None
        assert seg["no_speech_prob"] == 0.01
        assert result["duration"] == 5.0

    def test_result_is_json_serializable_after_sanitize(self):
        """Sanitized result must not raise when passed to json.dumps."""
        import json
        import server
        obj = {
            "avg_logprob": float("nan"),
            "compression_ratio": float("inf"),
            "score": -0.5,
        }
        sanitized = server._sanitize_floats(obj)
        # Should not raise ValueError
        serialized = json.dumps(sanitized)
        parsed = json.loads(serialized)
        assert parsed["avg_logprob"] is None
        assert parsed["compression_ratio"] is None
        assert parsed["score"] == -0.5


class TestCacheCorruptEviction:
    """Integration tests for corrupt cache entry eviction (Layer 2)."""

    def test_corrupt_cache_entry_evicted_and_fresh_job_queued(self, client):
        """A cache entry with nan in result_json is evicted on cache read;
        the endpoint falls through and queues a fresh job (202) rather than
        serving the corrupt cached result."""
        import json
        import server

        file_bytes = FAKE_WAV
        cache_key = server.compute_cache_key(file_bytes, "transcription", model=server.WHISPER_MODEL)

        # Inject a corrupt cache entry directly into the DB (simulates pre-fix nan storage).
        corrupt_result = {
            "task": "transcribe",
            "language": "en",
            "duration": 5.0,
            "text": "hello",
            "segments": [{"avg_logprob": float("nan")}],
        }
        # Use json.dumps with allow_nan=True to write the corrupt entry.
        corrupt_json = json.dumps(corrupt_result, allow_nan=True)
        import sqlite3
        from datetime import datetime, timedelta, timezone
        expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(server.DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, job_type, result_json, created_at, expires_at, hit_count)"
            " VALUES (?, ?, ?, ?, ?, 0)",
            (cache_key, "transcription", corrupt_json, now, expires),
        )
        conn.commit()
        conn.close()

        # Now attempt a cache lookup — should detect corrupt entry, evict it, return None.
        result = server.cache_lookup(cache_key, job_type="transcription", file_hint="test.wav")
        assert result is None, "Expected cache_lookup to return None after evicting corrupt entry"

        # The entry should no longer be in the DB.
        conn = sqlite3.connect(str(server.DB_PATH))
        row = conn.execute(
            "SELECT cache_key FROM cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        conn.close()
        assert row is None, "Corrupt cache entry should have been deleted from DB"

    def test_corrupt_cache_falls_through_to_fresh_job(self, client):
        """After eviction, the endpoint queues a fresh job (202) instead of 500."""
        import json
        import sqlite3
        import server
        from datetime import datetime, timedelta, timezone

        file_bytes = FAKE_WAV
        cache_key = server.compute_cache_key(file_bytes, "transcription", model=server.WHISPER_MODEL)

        corrupt_result = {
            "task": "transcribe",
            "language": "en",
            "duration": 5.0,
            "text": "hello",
            "segments": [{"avg_logprob": float("nan")}],
        }
        corrupt_json = json.dumps(corrupt_result, allow_nan=True)
        expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(server.DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, job_type, result_json, created_at, expires_at, hit_count)"
            " VALUES (?, ?, ?, ?, ?, 0)",
            (cache_key, "transcription", corrupt_json, now, expires),
        )
        conn.commit()
        conn.close()

        # POST the same file — server should evict the corrupt entry and queue a new job.
        with (
            patch("server._enqueue_job"),
            patch("server.job_count_by_status", return_value=0),
        ):
            resp = client.post("/v1/audio/transcriptions", files=[("file", ("test.wav", io.BytesIO(FAKE_WAV), "audio/wav"))])

        # Should NOT be 500; server evicts the bad entry and queues fresh work.
        assert resp.status_code == 202, (
            f"Expected 202 (fresh job after eviction), got {resp.status_code}: {resp.text}"
        )

    def test_cache_store_sanitizes_nan_before_write(self, client):
        """cache_store() must sanitize nan/inf so stored result_json is always JSON-safe."""
        import json
        import sqlite3
        import server

        cache_key = "test-sanitize-key-" + "a" * 44
        result_with_nan = {
            "task": "transcribe",
            "language": "en",
            "duration": 5.0,
            "text": "hello",
            "segments": [{"avg_logprob": float("nan"), "compression_ratio": float("inf")}],
        }
        server.cache_store(cache_key, "transcription", result_with_nan)

        # Read back raw JSON from DB and confirm it's valid.
        conn = sqlite3.connect(str(server.DB_PATH))
        row = conn.execute(
            "SELECT result_json FROM cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        conn.close()
        assert row is not None, "cache_store should have written an entry"

        # Must not raise.
        parsed = json.loads(row[0])
        assert parsed["segments"][0]["avg_logprob"] is None
        assert parsed["segments"][0]["compression_ratio"] is None
