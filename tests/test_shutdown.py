"""Graceful shutdown + VRAM release + upload sweeper.

These tests exercise the new shutdown plumbing without spinning up
uvicorn or actually loading mlx-whisper. They focus on the pure-Python
helpers (``_sweep_orphaned_uploads``, ``_get_backend_if_loaded``,
``_shutdown_drain_jobs``, ``_shutdown_release_models``) so they run
green on Linux CI as well as Mac.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _sweep_orphaned_uploads
# ---------------------------------------------------------------------------


def test_sweep_orphaned_uploads_removes_old_files(tmp_path: Path) -> None:
    import server

    old = tmp_path / "old.wav"
    old.write_bytes(b"RIFF")
    # Backdate mtime to 2h ago (cutoff is 1h).
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))

    swept = server._sweep_orphaned_uploads(upload_dir=tmp_path)

    assert swept == 1
    assert not old.exists()


def test_sweep_orphaned_uploads_keeps_recent_files(tmp_path: Path) -> None:
    import server

    fresh = tmp_path / "fresh.wav"
    fresh.write_bytes(b"RIFF")
    # mtime is now → should be kept.
    swept = server._sweep_orphaned_uploads(upload_dir=tmp_path)

    assert swept == 0
    assert fresh.exists()


def test_sweep_orphaned_uploads_only_targets_wav(tmp_path: Path) -> None:
    """Non-.wav junk in UPLOAD_DIR must not be touched, even if old."""
    import server

    junk = tmp_path / "old.txt"
    junk.write_text("don't touch me")
    long_ago = time.time() - 7200
    os.utime(junk, (long_ago, long_ago))

    swept = server._sweep_orphaned_uploads(upload_dir=tmp_path)

    assert swept == 0
    assert junk.exists()


def test_sweep_orphaned_uploads_handles_missing_dir(tmp_path: Path) -> None:
    import server

    missing = tmp_path / "does-not-exist"
    # Should not raise.
    swept = server._sweep_orphaned_uploads(upload_dir=missing)
    assert swept == 0


# ---------------------------------------------------------------------------
# _get_backend_if_loaded
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_backend():
    """Save/restore module-level ``_inference_backend`` around a test."""
    import server

    original = server._inference_backend
    try:
        yield
    finally:
        server._inference_backend = original


def test_get_backend_if_loaded_returns_none_when_not_instantiated(
    restore_backend: None,
) -> None:
    import server

    server._inference_backend = None
    assert server._get_backend_if_loaded() is None


def test_get_backend_if_loaded_returns_none_when_backend_exists_but_not_loaded(
    restore_backend: None,
) -> None:
    import server

    fake = MagicMock()
    fake.is_loaded = False
    server._inference_backend = fake

    assert server._get_backend_if_loaded() is None
    # Crucially: we must NOT have triggered a load.
    fake.load.assert_not_called()


def test_get_backend_if_loaded_returns_backend_when_loaded(
    restore_backend: None,
) -> None:
    import server

    fake = MagicMock()
    fake.is_loaded = True
    server._inference_backend = fake

    assert server._get_backend_if_loaded() is fake
    fake.load.assert_not_called()


# ---------------------------------------------------------------------------
# _shutdown_release_models — never triggers a load
# ---------------------------------------------------------------------------


def test_shutdown_release_models_is_noop_when_nothing_loaded(
    restore_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import server

    server._inference_backend = None
    monkeypatch.setattr(server, "_pyannote_loaded", False, raising=False)

    # Must not raise, must not try to load anything.
    server._shutdown_release_models()


def test_shutdown_release_models_unloads_loaded_backend(
    restore_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import server

    fake = MagicMock()
    fake.is_loaded = True
    server._inference_backend = fake
    monkeypatch.setattr(server, "_pyannote_loaded", False, raising=False)

    server._shutdown_release_models()

    fake.unload.assert_called_once()
    fake.load.assert_not_called()


# ---------------------------------------------------------------------------
# _shutdown_drain_jobs — marks processing/queued as failed
# ---------------------------------------------------------------------------


def test_shutdown_drain_jobs_marks_inflight_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insert a fake processing+queued job, run drain, verify status flip."""
    import server

    # Point the DB at a tmp file and reinit schema.
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(server, "DB_PATH", db_path)
    server._init_db()

    # Insert two jobs via the existing helper so column shapes are right.
    server.job_create(
        job_id="j-proc",
        job_type="transcription",
        cache_key="ck-proc",
        file_path=str(tmp_path / "a.wav"),
        params={},
    )
    server.job_create(
        job_id="j-queued",
        job_type="transcription",
        cache_key="ck-queued",
        file_path=str(tmp_path / "b.wav"),
        params={},
    )
    # Flip the first to 'processing' so we cover that branch.
    server.job_update(job_id="j-proc", status="processing")

    n = server._shutdown_drain_jobs()

    assert n == 2
    assert server.job_get("j-proc")["status"] == "failed"
    assert server.job_get("j-queued")["status"] == "failed"
    assert "shutdown" in (server.job_get("j-proc")["error"] or "").lower()
