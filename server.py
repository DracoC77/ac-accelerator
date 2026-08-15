"""
Audio Chronicle Accelerator - Core FastAPI Server

Exposes an OpenAI-compatible REST API for speech-to-text transcription
(mlx-whisper on Metal) and speaker diarization (pyannote on MPS).
Jobs are processed asynchronously with polling-based result retrieval
and a content-addressed SHA256 cache.

## Log Level Semantics

DEBUG  — Verbose internals: per-chunk timings, memory snapshots before/after
         model load, cache key computation, DB row details, rate-limit bucket
         state. Enable with LOG_LEVEL=DEBUG env var; no code changes needed.

INFO   — Normal operational events: server start/stop, job lifecycle
         (queued→started→complete/failed), model load/unload with timing and
         memory delta, cache hit/miss with file and key, idle watchdog ticks,
         cache sweep results, startup crash-recovery counts.

WARNING — Recoverable anomalies: queue full (jobs rejected), job cancellation
          of in-flight work, rate limit trips, cache sweep exceptions, missing
          ACCELERATOR_TOKEN (open access mode).

ERROR  — Job failures and unhandled exceptions: transcription or diarization
         errors with job_id + elapsed + error message, DB errors, upload
         cleanup failures.

Environment:
    LOG_LEVEL       — One of DEBUG / INFO / WARNING / ERROR (default: INFO).
                      Controls the console / uvicorn log level.
    LOG_LEVEL_FILE  — One of DEBUG / INFO / WARNING / ERROR (default: INFO).
                      Controls the rotating file handler level independently
                      of LOG_LEVEL.  Example: LOG_LEVEL=INFO LOG_LEVEL_FILE=DEBUG
                      writes verbose file logs while keeping console clean.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import platform
import resource
import signal
import sqlite3
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION = "0.1.0"

ACCELERATOR_PORT = int(os.getenv("ACCELERATOR_PORT", "8765"))
ACCELERATOR_HOST = os.getenv("ACCELERATOR_HOST", "0.0.0.0")
ACCELERATOR_TOKEN = os.getenv("ACCELERATOR_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
DIARIZE_MODEL = os.getenv("DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")

# Per-turn is_overlap threshold. Previously this was hardcoded
# to ``overlap_duration > 0.0`` — any non-zero brush against an overlap
# region (even 1 ms) flagged the entire turn as is_overlap=True. Production
# validation against the main repo's local pyannote path (which uses 0.3)
# showed this was the dominant driver of the ~65% is_overlap rate on the
# accelerator path. Default 0.3 brings the accelerator to parity with the
# local CPU pyannote path; env-tunable for ops without code changes.
DIARIZE_TURN_OVERLAP_THRESHOLD = float(
    os.environ.get("DIARIZE_TURN_OVERLAP_THRESHOLD", "0.3")
)
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))
# Raised default 500 -> 2048 MB. Long recordings (2-3h
# morning sessions) produce 16kHz mono WAVs of 500-700 MB, and the prior
# 500 MB cap caused HTTP 413 -> slow CPU fallback. 2 GB covers ~6h of
# 16kHz mono audio. Still env-overridable via MAX_FILE_SIZE_MB.
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "2048"))
# Windows/RTX 5090 has plenty of VRAM for parallel jobs;
# Mac default stays at 2 to avoid Metal contention.
_default_max_jobs = 3 if platform.system() == "Windows" else 2
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", str(_default_max_jobs)))
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "20"))
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "30"))
CACHE_SWEEP_INTERVAL_MIN = int(os.getenv("CACHE_SWEEP_INTERVAL_MIN", "60"))
DATA_DIR = Path(os.getenv("DATA_DIR", os.path.expanduser("~/.audio-accelerator/data")))
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "jobs.db")))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "INFO").upper()
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in ("true", "1", "yes")

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".ogg", ".mp3", ".m4a", ".flac", ".webm", ".opus"}

# Per-IP req/min rate limiter removed.
# In a single-client LAN deployment the sliding-window counter was a false positive
# factory: cache hits and small-chunk pipelines easily exceed 10-30 req/60 s even
# for legitimate workloads → 429 → CPU fallback stall (confirmed: 4h 49 min).
# Queue-depth admission control (MAX_QUEUE_DEPTH=20) already provides the correct
# backpressure; anything that exceeds it now returns 503 with Retry-After: 5.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Configure console logging at module level.  Uvicorn will later call
# logging.config.dictConfig() which replaces handlers on the root logger;
# the rotating file handler is therefore added inside the FastAPI lifespan
# startup event (see _setup_file_logging), which runs after uvicorn
# initialises its own logging.
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("accelerator")


# ---------------------------------------------------------------------------
# Float sanitization — guards against nan/inf from mlx-whisper
# ---------------------------------------------------------------------------

def _sanitize_floats(obj: Any) -> Any:
    """Recursively replace nan/inf floats with None to ensure JSON compliance.

    mlx-whisper can produce nan/inf in avg_logprob, compression_ratio, etc.
    Leaving them in the result causes json.dumps() to raise ValueError, which
    crashes the cache write and — worse — any subsequent cache read that tries
    to re-serialize a stored corrupt entry.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    # Also handle numpy/mlx scalar types that have __float__ but aren't Python float
    if not isinstance(obj, (bool, int, str, bytes, type(None))) and hasattr(obj, '__float__'):
        try:
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (ValueError, TypeError):
            return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _setup_file_logging() -> None:
    """Attach a rotating file handler to the root logger.

    Must be called *after* uvicorn has called logging.config.dictConfig(),
    which is why this is invoked from the FastAPI lifespan startup event
    rather than at module level.

    The handler is attached to the root logger so that log records emitted
    by uvicorn.* and uvicorn.access.* are captured in addition to the
    accelerator.* hierarchy.

    LOG_LEVEL_FILE (default INFO) controls the file handler's level
    independently of LOG_LEVEL (which governs the console/uvicorn output).
    """
    import logging.handlers as _lh

    log_file = Path(os.getenv("DATA_DIR", str(DATA_DIR))) / "logs" / "accelerator.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate handlers if uvicorn calls startup more than once
    # (e.g. in test environments using TestClient lifespan).
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing, _lh.RotatingFileHandler) and Path(os.path.realpath(existing.baseFilename)) == Path(os.path.realpath(log_file)):
            return

    handler = _lh.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setLevel(getattr(logging, LOG_LEVEL_FILE, logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Ensure the root logger's effective level is at most the file handler's
    # level so records aren't filtered before they reach the handler.
    root_level = root.level
    file_level = getattr(logging, LOG_LEVEL_FILE, logging.INFO)
    if root_level == logging.NOTSET or root_level > file_level:
        root.setLevel(file_level)

    root.addHandler(handler)
    root.info("File logging initialised: %s (level=%s)", log_file, LOG_LEVEL_FILE)

# ---------------------------------------------------------------------------
# Database - SQLite (thread-safe, WAL mode)
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_db()
    try:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "    job_id       TEXT PRIMARY KEY,"
            "    job_type     TEXT NOT NULL,"
            "    status       TEXT NOT NULL,"
            "    cache_key    TEXT NOT NULL,"
            "    params_json  TEXT,"
            "    result_json  TEXT,"
            "    error        TEXT,"
            "    progress     REAL DEFAULT 0.0,"
            "    file_path    TEXT,"
            "    created_at   TEXT NOT NULL,"
            "    started_at   TEXT,"
            "    completed_at TEXT,"
            "    failed_at    TEXT"
            ");"
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_cache_key ON jobs(cache_key);"
            "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);"
            "CREATE TABLE IF NOT EXISTS cache ("
            "    cache_key    TEXT PRIMARY KEY,"
            "    job_type     TEXT NOT NULL,"
            "    result_json  TEXT NOT NULL,"
            "    created_at   TEXT NOT NULL,"
            "    expires_at   TEXT NOT NULL,"
            "    hit_count    INTEGER DEFAULT 0"
            ");"
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);"
            "CREATE TABLE IF NOT EXISTS cache_stats ("
            "    id           INTEGER PRIMARY KEY CHECK (id = 1),"
            "    total_hits   INTEGER DEFAULT 0,"
            "    total_misses INTEGER DEFAULT 0"
            ");"
            "INSERT OR IGNORE INTO cache_stats (id, total_hits, total_misses)"
            "    VALUES (1, 0, 0);"
        )
        conn.commit()
    finally:
        conn.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def compute_cache_key(file_bytes: bytes, job_type: str, **params: Any) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(job_type.encode())
    for k in sorted(params.keys()):
        if params[k] is not None:
            h.update(f"{k}={params[k]}".encode())
    return h.hexdigest()


def cache_lookup(cache_key: str, job_type: str = "", file_hint: str = "") -> Optional[dict]:
    now = _utcnow()
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT result_json, expires_at FROM cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row and row["expires_at"] > now:
                cached_result = json.loads(row["result_json"])
                # Validate the cached result is still JSON-serializable.
                # A prior run may have stored nan/inf floats (pre-fix), which
                # will cause json.dumps() to raise ValueError on serialization.
                # Evict the corrupt entry and fall through to a fresh computation.
                try:
                    # Use allow_nan=False to match FastAPI JSONResponse.render() behavior.
                    # Python's json module permits NaN/Infinity by default, but FastAPI
                    # rejects them — so a cached result that contains nan will cause a 500
                    # on every subsequent request until the entry is evicted.
                    json.dumps(cached_result, allow_nan=False)
                except (ValueError, TypeError):
                    logger.warning(
                        "Cache entry corrupt (nan/inf) — evicting key=%.12s, file=%s",
                        cache_key, file_hint or "unknown",
                    )
                    conn.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
                    conn.execute(
                        "UPDATE cache_stats SET total_misses = total_misses + 1 WHERE id = 1"
                    )
                    conn.commit()
                    return None
                conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.execute(
                    "UPDATE cache_stats SET total_hits = total_hits + 1 WHERE id = 1"
                )
                conn.commit()
                logger.info(
                    "Cache HIT (type=%s, key=%.12s, file=%s)",
                    job_type, cache_key, file_hint or "unknown",
                )
                return cached_result
            conn.execute(
                "UPDATE cache_stats SET total_misses = total_misses + 1 WHERE id = 1"
            )
            conn.commit()
            logger.debug(
                "Cache MISS (type=%s, key=%.12s, file=%s)",
                job_type, cache_key, file_hint or "unknown",
            )
            return None
        finally:
            conn.close()


def cache_store(cache_key: str, job_type: str, result: dict) -> None:
    # Sanitize nan/inf before writing so the DB never holds un-serializable data.
    result = _sanitize_floats(result)
    now = _utcnow()
    expires = (datetime.now(timezone.utc) + timedelta(days=CACHE_TTL_DAYS)).isoformat()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache"
                " (cache_key, job_type, result_json, created_at, expires_at, hit_count)"
                " VALUES (?, ?, ?, ?, ?, 0)",
                (cache_key, job_type, json.dumps(result), now, expires),
            )
            conn.commit()
        finally:
            conn.close()


def cache_evict_expired() -> int:
    now = _utcnow()
    with _db_lock:
        conn = _get_db()
        try:
            cur = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def cache_get_stats() -> dict:
    with _db_lock:
        conn = _get_db()
        try:
            stats_row = conn.execute(
                "SELECT total_hits, total_misses FROM cache_stats WHERE id = 1"
            ).fetchone()
            total_hits = stats_row["total_hits"] if stats_row else 0
            total_misses = stats_row["total_misses"] if stats_row else 0
            counts = conn.execute(
                "SELECT job_type, COUNT(*) as cnt FROM cache GROUP BY job_type"
            ).fetchall()
            type_counts = {r["job_type"]: r["cnt"] for r in counts}
            total_entries = sum(type_counts.values())
            total_requests = total_hits + total_misses
            oldest = conn.execute(
                "SELECT MIN(created_at) as oldest FROM cache"
            ).fetchone()
            size_row = conn.execute(
                "SELECT SUM(LENGTH(result_json)) as total_size FROM cache"
            ).fetchone()
            return {
                "total_entries": total_entries,
                "transcription_entries": type_counts.get("transcription", 0),
                "diarization_entries": type_counts.get("diarization", 0),
                "hit_count": total_hits,
                "miss_count": total_misses,
                "hit_rate": round(total_hits / total_requests, 3) if total_requests > 0 else 0.0,
                "cache_size_bytes": size_row["total_size"] or 0 if size_row else 0,
                "oldest_entry": oldest["oldest"] if oldest and oldest["oldest"] else None,
                "ttl_days": CACHE_TTL_DAYS,
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Job store helpers
# ---------------------------------------------------------------------------


def job_create(job_id: str, job_type: str, cache_key: str, file_path: str, params: dict) -> dict:
    now = _utcnow()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "INSERT INTO jobs"
                " (job_id, job_type, status, cache_key, params_json,"
                "  file_path, created_at, progress)"
                " VALUES (?, ?, 'queued', ?, ?, ?, ?, 0.0)",
                (job_id, job_type, cache_key, json.dumps(params), file_path, now),
            )
            conn.commit()
        finally:
            conn.close()
    return {
        "job_id": job_id,
        "status": "queued",
        "type": job_type,
        "created_at": now,
        "cache_hit": False,
        "poll_url": f"/v1/jobs/{job_id}",
    }


def job_get(job_id: str) -> Optional[dict]:
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def job_update(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
            conn.commit()
        finally:
            conn.close()


def job_list_query(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    where_parts: list[str] = []
    params: list[Any] = []
    if status:
        where_parts.append("status = ?")
        params.append(status)
    if job_type:
        where_parts.append("job_type = ?")
        params.append(job_type)
    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    with _db_lock:
        conn = _get_db()
        try:
            total = conn.execute(
                f"SELECT COUNT(*) FROM jobs{where_sql}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM jobs{where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            return [dict(r) for r in rows], total
        finally:
            conn.close()


def job_count_by_status(*statuses: str) -> int:
    placeholders = ",".join("?" for _ in statuses)
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status IN ({placeholders})",
                statuses,
            ).fetchone()
            return row[0]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Model management - lazy load with idle watchdog
# ---------------------------------------------------------------------------

_pyannote_lock = threading.Lock()
_pyannote_loaded: bool = False
_backend_init_lock = threading.Lock()
_last_pyannote_use: float = 0.0
_idle_watchdog_running: bool = False

_pyannote_pipeline: Any = None

# Inference backend is constructed at startup via
# accelerator.backends.create_backend(); see lifespan() below. Until then
# the binding is None so import-time doesn't pull mlx_whisper in.
from accelerator.backends import InferenceBackend, create_backend  # noqa: E402

_inference_backend: "InferenceBackend | None" = None
_last_whisper_use: float = 0.0


def _get_backend() -> InferenceBackend:
    """Return the live inference backend, lazy-creating it if startup
    hasn't run yet (e.g. in unit tests that import server module but
    don't run lifespan).

    If create_backend() can't auto-detect (e.g. running on Linux CI),
    fall back to a bare MlxWhisperBackend. That preserves the
    pre-refactor laziness: nothing actually imports mlx_whisper until
    backend.load() is called, and tests patch ``_load_whisper`` to
    short-circuit that anyway.

    Thread-safe: uses double-checked locking via ``_backend_init_lock``
    so two concurrent startup requests cannot both create a backend and
    silently discard one.
    """
    global _inference_backend
    if _inference_backend is not None:
        return _inference_backend
    with _backend_init_lock:
        if _inference_backend is None:  # double-checked locking
            try:
                _inference_backend = create_backend(model=WHISPER_MODEL)
            except RuntimeError:
                from accelerator.backends.mlx_backend import MlxWhisperBackend
                _inference_backend = MlxWhisperBackend(model=WHISPER_MODEL)
    return _inference_backend


def _whisper_is_loaded() -> bool:
    """Compatibility shim used by /health and /memory."""
    return _inference_backend is not None and _inference_backend.is_loaded


def _get_backend_if_loaded() -> Optional["InferenceBackend"]:
    """Return the backend only if already instantiated AND loaded.

    Unlike :func:`_get_backend`, this NEVER triggers a model load or
    backend creation — it is safe to call from shutdown paths where we
    just want to release VRAM if there's anything to release.
    """
    backend = _inference_backend
    if backend is not None and backend.is_loaded:
        return backend
    return None


def _detect_gpu() -> dict:
    info: dict[str, Any] = {
        "metal_available": False,
        "mps_available": False,
        "device_name": "cpu",
    }
    try:
        import mlx.core as mx  # noqa: F401
        info["metal_available"] = True
        info["device_name"] = "Apple Silicon (Metal)"
    except ImportError:
        pass
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            if not info["metal_available"]:
                info["device_name"] = "Apple Silicon (MPS)"
    except ImportError:
        pass
    return info


def _rss_mb() -> float:
    """Return current process RSS in MB (cross-platform)."""
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_bytes / (1024 * 1024)
    return rss_bytes / 1024  # Linux reports kB


def _load_whisper() -> Any:
    """Backwards-compatible wrapper: loads via the backend and returns
    the underlying whisper module (mlx-whisper exposes ``transcribe``
    directly on the module).

    Tests that monkeypatch ``server._load_whisper`` to return a mock module
    continue to work: when the patched return value is observed, it is
    pushed into the backend so subsequent ``backend.transcribe()`` calls
    invoke the mock's ``transcribe(...)`` and call_args can be inspected.
    """
    global _last_whisper_use
    backend = _get_backend()
    module = backend.load()
    if hasattr(backend, "last_use"):
        _last_whisper_use = backend.last_use
    else:
        _last_whisper_use = time.monotonic()
    return module


def _unload_whisper() -> None:
    backend = _inference_backend
    if backend is not None:
        backend.unload()


def _load_pyannote() -> Any:
    global _pyannote_pipeline, _pyannote_loaded, _last_pyannote_use
    with _pyannote_lock:
        if _pyannote_pipeline is None:
            pre_mb = _rss_mb()
            logger.debug("Memory before pyannote load: %.1f MB RSS", pre_mb)
            logger.info("Loading pyannote pipeline: %s", DIARIZE_MODEL)
            t0 = time.monotonic()
            import torch
            from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore[import-untyped]
            token = HF_TOKEN or None
            pipeline = PyannotePipeline.from_pretrained(
                DIARIZE_MODEL, token=token
            )
            device_str = "mps" if torch.backends.mps.is_available() else "cpu"
            pipeline.to(torch.device(device_str))
            _pyannote_pipeline = pipeline
            _pyannote_loaded = True
            elapsed = time.monotonic() - t0
            post_mb = _rss_mb()
            logger.info(
                "pyannote ready (model=%s, device=%s, load_time=%.2fs, memory_mb=%.1f, delta_mb=%.1f)",
                DIARIZE_MODEL, device_str, elapsed, post_mb, post_mb - pre_mb,
            )
        _last_pyannote_use = time.monotonic()
        return _pyannote_pipeline


def _unload_pyannote() -> None:
    global _pyannote_pipeline, _pyannote_loaded
    with _pyannote_lock:
        if _pyannote_pipeline is not None:
            pre_mb = _rss_mb()
            logger.info("Unloading pyannote pipeline (idle timeout, memory_mb=%.1f)", pre_mb)
            _pyannote_pipeline = None
            _pyannote_loaded = False


def _idle_watchdog() -> None:
    global _idle_watchdog_running
    _idle_watchdog_running = True
    logger.info("Idle watchdog started (timeout=%ds)", IDLE_TIMEOUT_SECONDS)
    while _idle_watchdog_running:
        time.sleep(30)
        now = time.monotonic()
        if _whisper_is_loaded():
            idle_s = now - _last_whisper_use
            logger.debug("Idle watchdog: whisper idle=%.0fs / %ds", idle_s, IDLE_TIMEOUT_SECONDS)
            if idle_s > IDLE_TIMEOUT_SECONDS:
                _unload_whisper()
        if _pyannote_loaded:
            idle_s = now - _last_pyannote_use
            logger.debug("Idle watchdog: pyannote idle=%.0fs / %ds", idle_s, IDLE_TIMEOUT_SECONDS)
            if idle_s > IDLE_TIMEOUT_SECONDS:
                _unload_pyannote()


def _stop_idle_watchdog() -> None:
    global _idle_watchdog_running
    _idle_watchdog_running = False


_cache_sweep_running: bool = False


def _cache_sweep_loop() -> None:
    global _cache_sweep_running
    _cache_sweep_running = True
    interval = CACHE_SWEEP_INTERVAL_MIN * 60
    logger.info("Cache sweep started (interval=%dm)", CACHE_SWEEP_INTERVAL_MIN)
    while _cache_sweep_running:
        time.sleep(interval)
        try:
            n = cache_evict_expired()
            if n:
                logger.info("Cache sweep evicted %d expired entries", n)
        except Exception:
            logger.exception("Cache sweep error")


def _stop_cache_sweep() -> None:
    global _cache_sweep_running
    _cache_sweep_running = False


# ---------------------------------------------------------------------------
# Orphaned upload sweeper + graceful shutdown helpers
# ---------------------------------------------------------------------------

ORPHANED_UPLOAD_TTL_SECONDS = 3600  # 1 hour


def _sweep_orphaned_uploads(upload_dir: Optional[Path] = None,
                            ttl_seconds: int = ORPHANED_UPLOAD_TTL_SECONDS) -> int:
    """Delete WAV upload files older than ``ttl_seconds`` from ``upload_dir``.

    Called at startup to clean up files left behind by a prior crash or
    forced shutdown (NSSM kill on Windows, SIGKILL on Mac).  Returns the
    number of files removed.  Silently ignores per-file errors so a
    transient permission glitch doesn't abort startup.
    """
    target = upload_dir if upload_dir is not None else UPLOAD_DIR
    now = time.time()
    swept = 0
    try:
        for f in Path(target).glob("*.wav"):
            try:
                if now - f.stat().st_mtime > ttl_seconds:
                    f.unlink()
                    swept += 1
            except OSError:
                # File vanished between glob and stat/unlink, or perms
                # — not worth aborting startup over.
                continue
    except OSError:
        # UPLOAD_DIR didn't exist or wasn't traversable.
        return 0
    if swept:
        logger.info("Startup sweep: removed %d orphaned upload(s) from %s", swept, target)
    return swept


def _shutdown_drain_jobs() -> int:
    """Mark any queued/processing jobs as failed during shutdown.

    Mirrors the crash-recovery write in lifespan startup so a clean
    shutdown leaves the DB in the same state a hard kill would (jobs
    not stuck in 'processing' forever).  Returns the number of rows
    updated.  Uses the existing ``_db_lock`` + sqlite pattern.
    """
    with _db_lock:
        conn = _get_db()
        try:
            cur = conn.execute(
                "UPDATE jobs SET status = 'failed', "
                "error = 'Server shutdown', "
                "failed_at = ? WHERE status IN ('processing', 'queued')",
                (_utcnow(),),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def _shutdown_release_models() -> None:
    """Unload whisper + pyannote backends if currently loaded.

    Uses :func:`_get_backend_if_loaded` so we never accidentally
    instantiate or load a model during shutdown.
    """
    backend = _get_backend_if_loaded()
    if backend is not None:
        logger.info("Shutdown: unloading whisper backend (VRAM release)")
        try:
            backend.unload()
            logger.info("Shutdown: whisper backend unloaded")
        except Exception:
            logger.exception("Shutdown: whisper backend unload failed")
    if _pyannote_loaded:
        logger.info("Shutdown: unloading pyannote pipeline (VRAM release)")
        try:
            _unload_pyannote()
            logger.info("Shutdown: pyannote pipeline unloaded")
        except Exception:
            logger.exception("Shutdown: pyannote unload failed")


# ---------------------------------------------------------------------------
# Windows CTRL handler so NSSM stop terminates the process
# cleanly.  NSSM sends CTRL_BREAK_EVENT via GenerateConsoleCtrlEvent;
# we translate it into a SIGINT so uvicorn's existing signal path runs
# the lifespan shutdown.
# ---------------------------------------------------------------------------

if platform.system() == "Windows":  # pragma: no cover - Windows-only
    import ctypes

    def _windows_ctrl_handler(ctrl_type: int) -> bool:
        # CTRL_C_EVENT = 0, CTRL_BREAK_EVENT = 1
        if ctrl_type in (0, 1):
            try:
                logger.info(
                    "Windows CTRL event received (type=%d) — initiating shutdown",
                    ctrl_type,
                )
            except Exception:
                pass
            try:
                signal.raise_signal(signal.SIGINT)
            except Exception:
                pass
            return True
        return False

    _HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    _ctrl_handler_ref = _HandlerRoutine(_windows_ctrl_handler)
    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler_ref, True)
    except Exception:
        # Best-effort — if the runtime doesn't expose kernel32 we still
        # want the server to start.
        pass


# ---------------------------------------------------------------------------
# Worker - job processing
# ---------------------------------------------------------------------------

_job_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)
_startup_time: float = 0.0


def _process_transcription(job: dict) -> dict:
    job_id = job["job_id"]
    file_path = job["file_path"]
    params = json.loads(job["params_json"]) if job["params_json"] else {}
    model = params.get("model", WHISPER_MODEL)

    logger.debug("Transcription stage start (job_id=%s, file=%s, model=%s)", job_id, file_path, model)
    t_load = time.monotonic()
    backend = _get_backend()
    # _load_whisper() keeps the legacy timestamp + log behavior intact.
    # The return value is the underlying whisper module; we inject it into
    # the backend so tests that monkeypatch ``server._load_whisper`` see
    # their mock's ``transcribe(...)`` invoked.
    whisper_module = _load_whisper()
    if hasattr(backend, "_module"):
        # TECH DEBT: _module injection for test mock compat;
        # replace with backend.set_module_loader() when test suite is migrated
        backend._module = whisper_module  # type: ignore[attr-defined]
        backend._loaded = True  # type: ignore[attr-defined]
    load_elapsed = time.monotonic() - t_load
    logger.debug("%s available (job_id=%s, model_load_time=%.2fs)", backend.backend_name, job_id, load_elapsed)

    # Normalize per-job request params into backend kwargs. Backends ignore
    # any kwargs they don't recognize.
    backend_kwargs: dict[str, Any] = {"model": model}
    language = params.get("language") or None
    # Temperature — accept scalar float or JSON-serialized list (tuple for fallback chain)
    if "temperature" in params and params["temperature"] is not None:
        raw_temp = params["temperature"]
        if isinstance(raw_temp, list):
            backend_kwargs["temperature"] = tuple(float(x) for x in raw_temp)
        else:
            backend_kwargs["temperature"] = float(raw_temp)
    # no_speech_threshold
    if "no_speech_threshold" in params and params["no_speech_threshold"] is not None:
        backend_kwargs["no_speech_threshold"] = float(params["no_speech_threshold"])
    # condition_on_previous_text — forwarded as string "true"/"false" from container
    if "condition_on_previous_text" in params and params["condition_on_previous_text"] is not None:
        raw_copt = params["condition_on_previous_text"]
        if isinstance(raw_copt, str):
            backend_kwargs["condition_on_previous_text"] = raw_copt.lower().strip() not in ("false", "0", "no")
        else:
            backend_kwargs["condition_on_previous_text"] = bool(raw_copt)
    logger.info("Transcription kwargs: temperature=%s, no_speech_threshold=%s, condition_on_previous_text=%s",
        backend_kwargs.get("temperature", "default"), backend_kwargs.get("no_speech_threshold", "default"),
        backend_kwargs.get("condition_on_previous_text", "default"))
    logger.info("Transcribing (job_id=%s, file=%s, model=%s)", job_id, file_path, model)
    t_infer = time.monotonic()
    result = backend.transcribe(file_path, language=language, **backend_kwargs)
    infer_elapsed = time.monotonic() - t_infer

    # Whisper quality fields (avg_logprob, compression_ratio,
    # no_speech_prob, temperature) can come back as NaN/Inf from mlx-whisper's
    # fp16 path — systematically for avg_logprob/compression_ratio in some
    # model builds. The global _sanitize_floats() pass below would turn those
    # into None, which makes downstream chronicle code see every segment as
    # "unknown quality". Coerce them to 0.0 here so the Optional[float]
    # contract stays a real float and downstream code can distinguish
    # "neutral signal" from "field missing entirely".
    def _quality(value: Any, field_name: str, segment_index: int) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(f) or math.isinf(f):
            logger.warning(
                "non-finite %s on segment %d (job_id=%s) — "
                "replaced with 0.0",
                field_name, segment_index, job_id,
            )
            return 0.0
        return f

    segments = []
    for i, seg in enumerate(result.segments):
        segments.append({
            "id": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "tokens": seg.tokens,
            "avg_logprob": _quality(seg.avg_logprob, "avg_logprob", i),
            "compression_ratio": _quality(seg.compression_ratio, "compression_ratio", i),
            "no_speech_prob": _quality(seg.no_speech_prob, "no_speech_prob", i),
            # expose per-segment temperature in the HTTP response.
            "temperature": _quality(getattr(seg, "temperature", 0.0), "temperature", i),
        })
    duration = 0.0
    if segments:
        duration = segments[-1]["end"]
    if result.duration:
        duration = result.duration

    logger.info(
        "Transcription inference complete (job_id=%s, file=%s, model=%s, "
        "duration_s=%.1f, infer_time=%.2fs, segments=%d, memory_mb=%.1f)",
        job_id, file_path, model, duration, infer_elapsed, len(segments), _rss_mb(),
    )
    return {
        "task": "transcribe",
        "language": result.language or "en",
        "duration": round(duration, 1),
        "text": result.text,
        "segments": segments,
    }


def _process_diarization(job: dict) -> dict:
    job_id = job["job_id"]
    file_path = job["file_path"]
    params = json.loads(job["params_json"]) if job["params_json"] else {}

    logger.debug("Diarization stage start (job_id=%s, file=%s, model=%s)", job_id, file_path, DIARIZE_MODEL)
    t_load = time.monotonic()
    pipeline = _load_pyannote()
    load_elapsed = time.monotonic() - t_load
    logger.debug("pyannote available (job_id=%s, model_load_time=%.2fs)", job_id, load_elapsed)

    kwargs: dict[str, Any] = {}
    if params.get("min_speakers") is not None:
        kwargs["min_speakers"] = int(params["min_speakers"])
    if params.get("max_speakers") is not None:
        kwargs["max_speakers"] = int(params["max_speakers"])
    logger.info("Diarizing (job_id=%s, file=%s, model=%s)", job_id, file_path, DIARIZE_MODEL)
    t_infer = time.monotonic()
    output = pipeline(file_path, **kwargs)
    infer_elapsed = time.monotonic() - t_infer

    # pyannote v4 API: pipeline() returns a SpeakerDiarizationOutput object.
    # Extract the Annotation from output.speaker_diarization.
    # v3 fallback: output IS the Annotation directly (has itertracks on it).
    if hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
    else:
        annotation = output

    # Compute per-turn overlap metrics so the main repo
    # can run crosstalk detection on the remote-diarization path.
    # get_overlap() returns a Timeline of segments where ≥2 speakers overlap.
    # .support() merges adjacent overlap regions to prevent double-counting when 3+ speakers overlap
    overlap_timeline = annotation.get_overlap().support()

    # Serialise the merged overlap regions so the main
    # repo can run the per-segment intersection (bug #2 fix)
    # on the accelerator path. Without this list the main repo passes
    # ``overlap_timeline=None`` into ``align_with_transcript()`` on every
    # remote run, leaving the per-segment fix dead code in production.
    overlap_regions = [
        {"start": round(seg.start, 4), "end": round(seg.end, 4)}
        for seg in overlap_timeline
    ]

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turn_duration = turn.end - turn.start
        # Sum the intersection of this turn with every overlap region.
        overlap_duration = 0.0
        for overlap_seg in overlap_timeline:
            inter_start = max(turn.start, overlap_seg.start)
            inter_end = min(turn.end, overlap_seg.end)
            if inter_end > inter_start:
                overlap_duration += inter_end - inter_start
        overlap_ratio = (
            round(overlap_duration / turn_duration, 4)
            if turn_duration > 0
            else 0.0
        )
        # Was ``overlap_duration > 0.0`` — any touch
        # flagged the whole turn (degenerate). Now matches the main repo
        # local pyannote path threshold (diarize.py line 360).
        is_overlap = overlap_ratio > DIARIZE_TURN_OVERLAP_THRESHOLD
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 4),
            "end": round(turn.end, 4),
            "is_overlap": is_overlap,
            "overlap_ratio": overlap_ratio,
        })
    duration = segments[-1]["end"] if segments else 0.0
    speakers = set(s["speaker"] for s in segments)

    logger.info(
        "Diarization inference complete (job_id=%s, file=%s, model=%s, "
        "duration_s=%.1f, infer_time=%.2fs, speakers=%d, segments=%d, memory_mb=%.1f)",
        job_id, file_path, DIARIZE_MODEL, duration, infer_elapsed,
        len(speakers), len(segments), _rss_mb(),
    )
    return {
        "segments": segments,
        # Enables per-segment intersection on the main
        # repo side (align_with_transcript bug #2 fix). Old clients that
        # don't know about this key simply ignore it.
        "overlap_regions": overlap_regions,
        "num_speakers": len(speakers),
        "duration": round(duration, 1),
    }


def _run_job(job_id: str) -> None:
    _job_semaphore.acquire()
    try:
        job = job_get(job_id)
        if not job or job["status"] != "queued":
            logger.debug("Job %s skipped (status=%s)", job_id, job["status"] if job else "not_found")
            return
        job_type = job["job_type"]
        file_path = job.get("file_path", "")
        job_update(job_id, status="processing", started_at=_utcnow(), progress=0.0)
        logger.info(
            "Job started (job_id=%s, type=%s, file=%s, memory_mb=%.1f)",
            job_id, job_type, file_path, _rss_mb(),
        )
        t0 = time.monotonic()
        try:
            if job_type == "transcription":
                result = _process_transcription(job)
            elif job_type == "diarization":
                result = _process_diarization(job)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            # Sanitize nan/inf floats (e.g. avg_logprob from mlx-whisper) before
            # storing to DB or cache — json.dumps() raises ValueError on nan/inf.
            result = _sanitize_floats(result)
            elapsed = time.monotonic() - t0
            job_update(
                job_id,
                status="complete",
                result_json=json.dumps(result),
                completed_at=_utcnow(),
                progress=1.0,
            )
            cache_store(job["cache_key"], job_type, result)
            logger.info(
                "Job complete (job_id=%s, type=%s, file=%s, duration_s=%.1f, memory_mb=%.1f)",
                job_id, job_type, file_path, elapsed, _rss_mb(),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            error_msg = f"{type(exc).__name__}: {exc}"
            job_update(job_id, status="failed", error=error_msg, failed_at=_utcnow())
            logger.error(
                "Job failed (job_id=%s, type=%s, file=%s, duration_s=%.1f, error=%s)",
                job_id, job_type, file_path, elapsed, error_msg,
            )
        finally:
            try:
                fp = job.get("file_path")
                if fp and os.path.exists(fp):
                    os.remove(fp)
                    logger.debug("Upload cleaned up (job_id=%s, file=%s)", job_id, fp)
            except OSError as oe:
                logger.error("Upload cleanup failed (job_id=%s, file=%s, error=%s)", job_id, fp, oe)
    finally:
        _job_semaphore.release()


def _enqueue_job(job_id: str) -> None:
    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Auth dependency — HTTPBearer with constant-time comparison
# ---------------------------------------------------------------------------

_http_bearer = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_http_bearer),
) -> None:
    """FastAPI dependency: enforce Bearer token auth when ACCELERATOR_TOKEN is set.

    GET /health is always exempt — do not add this dependency there.
    Uses hmac.compare_digest for constant-time comparison to resist timing attacks.
    """
    token = os.getenv("ACCELERATOR_TOKEN", "")
    if not token:
        return  # auth not configured — open access
    if not credentials or not hmac.compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate limiting — REMOVED
# ---------------------------------------------------------------------------
# The per-IP sliding-window rate limiter has been removed.  Queue-depth
# admission control (MAX_QUEUE_DEPTH=20 → 503 Service Unavailable) is the
# sole admission gate.  See the constant block above for the full rationale.


# ---------------------------------------------------------------------------
# FastAPI app with lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.monotonic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _init_db()
    # Set up rotating file logging *after* uvicorn has configured its own
    # handlers so our handler isn't wiped by dictConfig().
    _setup_file_logging()
    if not ACCELERATOR_TOKEN:
        logger.warning(
            "ACCELERATOR_TOKEN not set — running in open-access mode (no auth required). "
            "Set ACCELERATOR_TOKEN env var to enable bearer token authentication."
        )
    # Crash recovery: mark stale processing jobs as failed
    with _db_lock:
        conn = _get_db()
        try:
            cur = conn.execute(
                "UPDATE jobs SET status = 'failed', "
                "error = 'Server restarted during processing', "
                "failed_at = ? WHERE status = 'processing'",
                (_utcnow(),),
            )
            conn.commit()
            if cur.rowcount:
                logger.warning(
                    "Crash recovery: marked %d stale in-flight jobs as failed",
                    cur.rowcount,
                )
        finally:
            conn.close()
    # Sweep orphaned upload WAVs left behind by a prior
    # crash / forced shutdown.  Best-effort — swallow errors.
    try:
        _sweep_orphaned_uploads()
    except Exception:
        logger.exception("Startup sweep of orphaned uploads failed")
    # Select + instantiate the inference backend up front when
    # possible so misconfiguration surfaces at startup, not on first job.
    # Fall back to lazy creation on first use if selection fails here
    # (preserves existing Linux test behavior where mlx_whisper is
    # unavailable but no transcription jobs are run).
    global _inference_backend
    try:
        _inference_backend = create_backend(model=WHISPER_MODEL)
        logger.info("Inference backend selected: %s", _inference_backend.backend_name)
    except RuntimeError as exc:
        logger.warning(
            "Inference backend not yet selected at startup (%s); will retry on first job",
            exc,
        )
    threading.Thread(target=_idle_watchdog, daemon=True).start()
    threading.Thread(target=_cache_sweep_loop, daemon=True).start()
    if PRELOAD_MODELS:
        logger.info("Preloading models (PRELOAD_MODELS=true) ...")
        threading.Thread(target=_load_whisper, daemon=True).start()
        threading.Thread(target=_load_pyannote, daemon=True).start()
    gpu_info = _detect_gpu()
    logger.info(
        "Accelerator v%s started (host=%s, port=%d, log_level=%s, gpu=%s, "
        "whisper=%s, diarize=%s, cache_ttl=%dd, idle_timeout=%ds, preload=%s)",
        VERSION, ACCELERATOR_HOST, ACCELERATOR_PORT, LOG_LEVEL,
        gpu_info.get("device_name", "unknown"),
        WHISPER_MODEL, DIARIZE_MODEL, CACHE_TTL_DAYS, IDLE_TIMEOUT_SECONDS,
        PRELOAD_MODELS,
    )
    yield
    # Graceful shutdown — stop background threads, mark
    # in-flight jobs failed, and release VRAM so NSSM-triggered
    # restarts come back quickly without a hung GPU context.
    logger.info("Accelerator shutting down")
    _stop_idle_watchdog()
    _stop_cache_sweep()
    try:
        n = _shutdown_drain_jobs()
        if n:
            logger.warning(
                "Shutdown: marked %d in-flight/queued job(s) as failed", n
            )
    except Exception:
        logger.exception("Shutdown: draining in-flight jobs failed")
    try:
        _shutdown_release_models()
    except Exception:
        logger.exception("Shutdown: model release failed")
    logger.info("Accelerator shutdown complete")


app = FastAPI(title="Audio Chronicle Accelerator", version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _save_upload(file: UploadFile, request: Request) -> tuple[str, bytes]:
    """Save an uploaded audio file after validating extension and size.

    Size is checked against content-length header first for a fast-fail,
    then verified again on the actual bytes after reading.
    """
    filename = file.filename or "audio"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Fast-fail: check Content-Length header before reading the entire body
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
            if declared_size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large ({declared_size / (1024 * 1024):.1f} MB declared). "
                        f"Max: {MAX_FILE_SIZE_MB} MB"
                    ),
                )
        except ValueError:
            pass  # malformed header — verify after read

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes) / (1024 * 1024):.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / unique_name
    dest.write_bytes(raw_bytes)
    return str(dest), raw_bytes


def _format_job_response(job: dict) -> dict:
    resp: dict[str, Any] = {
        "job_id": job["job_id"],
        "status": job["status"],
        "type": job["job_type"],
        "created_at": job["created_at"],
    }
    if job.get("started_at"):
        resp["started_at"] = job["started_at"]
    if job["status"] == "processing":
        resp["progress"] = job.get("progress", 0.0)
    if job["status"] == "complete":
        resp["completed_at"] = job.get("completed_at")
        if job.get("started_at") and job.get("completed_at"):
            try:
                t_start = datetime.fromisoformat(job["started_at"])
                t_end = datetime.fromisoformat(job["completed_at"])
                resp["duration_seconds"] = round((t_end - t_start).total_seconds(), 1)
            except (ValueError, TypeError):
                pass
        if job.get("result_json"):
            resp["result"] = _sanitize_floats(json.loads(job["result_json"]))
    if job["status"] == "failed":
        resp["error"] = job.get("error")
        resp["failed_at"] = job.get("failed_at")
    return resp


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    gpu_info = _detect_gpu()
    pending = job_count_by_status("queued")
    processing = job_count_by_status("processing")
    uptime = time.monotonic() - _startup_time if _startup_time else 0
    backend = _inference_backend
    backend_name = backend.backend_name if backend is not None else "unconfigured"
    return {
        "status": "healthy",
        "version": VERSION,
        "gpu": gpu_info,
        "backend_name": backend_name,
        "models": {
            "whisper": WHISPER_MODEL,
            "whisper_loaded": _whisper_is_loaded(),
            "backend_name": backend_name,
            "diarization": DIARIZE_MODEL,
            "diarization_loaded": _pyannote_loaded,
        },
        "queue": {"pending": pending, "processing": processing},
        "uptime_seconds": round(uptime, 1),
    }


@app.get("/memory")
async def memory_endpoint(_auth: None = Depends(verify_token)):
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_mb = rss_bytes / (1024 * 1024)
    else:
        rss_mb = rss_bytes / 1024
    return {
        "rss_mb": round(rss_mb, 1),
        "models": {
            "whisper_loaded": _whisper_is_loaded(),
            "diarization_loaded": _pyannote_loaded,
        },
    }


@app.post("/v1/audio/transcriptions", status_code=202)
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),        # scalar float or JSON list for fallback
    no_speech_threshold: Optional[str] = Form(None),  # float threshold for no-speech filter
    bypass_cache: Optional[str] = Query(None, description="Set to 'true' to skip cache lookup and always run fresh"), 
    condition_on_previous_text: Optional[str] = Form(None),  # hallucination guard
    _auth: None = Depends(verify_token),
):
    pending = job_count_by_status("queued")
    if pending >= MAX_QUEUE_DEPTH:
        logger.warning(
            "Queue full — transcription rejected (pending=%d/%d); returning 503",
            pending, MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({pending}/{MAX_QUEUE_DEPTH} pending jobs). Try again later.",
            headers={"Retry-After": "5"},
        )
    file_path, raw_bytes = await _save_upload(file, request)
    cache_params = {"model": model or WHISPER_MODEL}
    if language:
        cache_params["language"] = language
    # Include temperature and NST in cache key so different param sets don't share cache
    if temperature is not None:
        cache_params["temperature"] = temperature
    if no_speech_threshold is not None:
        cache_params["no_speech_threshold"] = no_speech_threshold
    # Include condition_on_previous_text in cache key so true/false don't share entries
    if condition_on_previous_text is not None:
        cache_params["condition_on_previous_text"] = condition_on_previous_text
    cache_key = compute_cache_key(raw_bytes, "transcription", **cache_params)
    # Check cache (skip if bypass_cache=true)
    _bypass = bypass_cache is not None and bypass_cache.lower() in ("true", "1", "yes")
    if _bypass:
        logger.info("Cache bypass requested (type=transcription, file=%s)", file.filename or "")
    cached = None if _bypass else cache_lookup(cache_key, job_type="transcription", file_hint=file.filename or "")
    if cached is not None:
        # Belt-and-suspenders: sanitize in case a corrupt entry slipped past cache_lookup.
        cached = _sanitize_floats(cached)
        # Return cached result with a synthetic job_id
        return JSONResponse(
            status_code=200,
            content={
                "job_id": f"cache-{cache_key[:12]}",
                "status": "complete",
                "type": "transcription",
                "cache_hit": True,
                "result": cached,
            },
        )
    job_id = str(uuid.uuid4())
    params = {"model": model or WHISPER_MODEL}
    if language:
        params["language"] = language
    if response_format:
        params["response_format"] = response_format
    # Parse and forward temperature (scalar or JSON list) and no_speech_threshold
    if temperature is not None:
        try:
            parsed_temp = json.loads(temperature)  # handles "[0.0, 0.2, ...]" or "0.0"
            params["temperature"] = parsed_temp
        except (json.JSONDecodeError, ValueError):
            logger.warning("malformed temperature value %r — ignoring, accelerator will use default", temperature)
    if no_speech_threshold is not None:
        try:
            params["no_speech_threshold"] = float(no_speech_threshold)
        except ValueError:
            logger.warning("malformed no_speech_threshold value %r — ignoring, accelerator will use default", no_speech_threshold)
    # Forward condition_on_previous_text (string "true"/"false") to params
    if condition_on_previous_text is not None:
        params["condition_on_previous_text"] = condition_on_previous_text.lower().strip()
    logger.info(
        "Job queued (job_id=%s, type=transcription, file=%s, model=%s)",
        job_id, file.filename or "", model or WHISPER_MODEL,
    )
    result = job_create(job_id, "transcription", cache_key, file_path, params)
    _enqueue_job(job_id)
    return JSONResponse(status_code=202, content=result)


@app.post("/v1/diarize", status_code=202)
async def create_diarization(
    request: Request,
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    bypass_cache: Optional[str] = Query(None, description="Set to 'true' to skip cache lookup and always run fresh"), 
    _auth: None = Depends(verify_token),
):
    pending = job_count_by_status("queued")
    if pending >= MAX_QUEUE_DEPTH:
        logger.warning(
            "Queue full — diarization rejected (pending=%d/%d); returning 503",
            pending, MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({pending}/{MAX_QUEUE_DEPTH} pending jobs). Try again later.",
            headers={"Retry-After": "5"},
        )
    file_path, raw_bytes = await _save_upload(file, request)
    cache_params: dict[str, Any] = {}
    if min_speakers is not None:
        cache_params["min_speakers"] = min_speakers
    if max_speakers is not None:
        cache_params["max_speakers"] = max_speakers
    cache_key = compute_cache_key(raw_bytes, "diarization", **cache_params)
    # Check cache (skip if bypass_cache=true)
    _bypass = bypass_cache is not None and bypass_cache.lower() in ("true", "1", "yes")
    if _bypass:
        logger.info("Cache bypass requested (type=diarization, file=%s)", file.filename or "")
    cached = None if _bypass else cache_lookup(cache_key, job_type="diarization", file_hint=file.filename or "")
    if cached is not None:
        # Belt-and-suspenders: sanitize in case a corrupt entry slipped past cache_lookup.
        cached = _sanitize_floats(cached)
        return JSONResponse(
            status_code=200,
            content={
                "job_id": f"cache-{cache_key[:12]}",
                "status": "complete",
                "type": "diarization",
                "cache_hit": True,
                "result": cached,
            },
        )
    job_id = str(uuid.uuid4())
    params: dict[str, Any] = {}
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers
    logger.info(
        "Job queued (job_id=%s, type=diarization, file=%s, min_speakers=%s, max_speakers=%s)",
        job_id, file.filename or "", min_speakers, max_speakers,
    )
    result = job_create(job_id, "diarization", cache_key, file_path, params)
    _enqueue_job(job_id)
    return JSONResponse(status_code=202, content=result)


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, _auth: None = Depends(verify_token)):
    job = job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _format_job_response(job)


@app.get("/v1/jobs")
async def list_jobs(
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None, alias="type"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _auth: None = Depends(verify_token),
):
    jobs, total = job_list_query(status=status, job_type=type, limit=limit, offset=offset)
    return {
        "jobs": [_format_job_response(j) for j in jobs],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.delete("/v1/jobs/{job_id}")
async def delete_job(job_id: str, _auth: None = Depends(verify_token)):
    job = job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "queued":
        job_update(job_id, status="cancelled")
        logger.info("Job cancelled (job_id=%s, type=%s, was=queued)", job_id, job["job_type"])
        # Clean up file
        try:
            fp = job.get("file_path")
            if fp and os.path.exists(fp):
                os.remove(fp)
        except OSError:
            pass
        return {"job_id": job_id, "status": "cancelled", "message": "Job cancelled successfully"}
    elif job["status"] == "processing":
        # Can't interrupt a running thread safely; mark as cancelled
        # and the worker will check status
        job_update(job_id, status="cancelled")
        logger.warning(
            "Job cancellation requested for in-flight job (job_id=%s, type=%s) — "
            "thread will complete but result will be discarded",
            job_id, job["job_type"],
        )
        return {"job_id": job_id, "status": "cancelled", "message": "Job cancellation requested"}
    else:
        # Remove completed/failed/cancelled jobs from history
        with _db_lock:
            conn = _get_db()
            try:
                conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                conn.commit()
            finally:
                conn.close()
        return {"job_id": job_id, "status": "removed", "message": "Job removed from history"}


@app.get("/v1/cache/stats")
async def get_cache_stats(_auth: None = Depends(verify_token)):
    return cache_get_stats()


@app.get("/cache")
async def list_cache(
    limit: int = Query(100, ge=1, le=1000),
    _auth: None = Depends(verify_token),
):
    """List cache entries with key, job_type, created_at, expires_at, hit_count."""
    with _db_lock:
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT cache_key, job_type, created_at, expires_at, hit_count"
                " FROM cache ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        finally:
            conn.close()
    entries = [
        {
            "key": r["cache_key"],
            "job_type": r["job_type"],
            "created_at": r["created_at"],
            "expires_at": r["expires_at"],
            "hit_count": r["hit_count"],
        }
        for r in rows
    ]
    return {"entries": entries, "total": total, "limit": limit}


@app.delete("/cache")
async def clear_cache(
    key: Optional[str] = Query(None, description="Clear specific entry by cache key"),
    _auth: None = Depends(verify_token),
):
    """Clear all cache entries (or a specific one by key).

    DELETE /cache          — clears all entries, returns {"deleted": N, "keys": [...]}
    DELETE /cache?key=X    — clears a specific entry by cache_key
    """
    with _db_lock:
        conn = _get_db()
        try:
            if key is not None:
                # Delete specific entry
                cur = conn.execute(
                    "SELECT cache_key FROM cache WHERE cache_key = ?", (key,)
                )
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail=f"Cache entry not found: {key}")
                conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
                conn.commit()
                logger.info("Cache entry deleted (key=%.12s)", key)
                return {"deleted": 1, "keys": [key]}
            else:
                # Delete all entries
                cur = conn.execute("SELECT cache_key FROM cache")
                keys = [r["cache_key"] for r in cur.fetchall()]
                conn.execute("DELETE FROM cache")
                conn.commit()
                n = len(keys)
                logger.info("Cache cleared (%d entries deleted)", n)
                return {"deleted": n, "keys": keys}
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Log viewer endpoint
# ---------------------------------------------------------------------------

_MAX_LOG_LINES = 1000
_DEFAULT_LOG_LINES = 200


def _tail_file(path: Path, n: int) -> list[str]:
    """Return the last *n* lines of *path* using a memory-efficient deque tail.

    Returns an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    buf: deque[str] = deque(maxlen=n)
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            buf.append(line.rstrip("\n"))
    return list(buf)


def _resolve_server_log() -> Path:
    """Return the server log path to use, applying the accelerator.log→stderr.log fallback.

    Primary: DATA_DIR/logs/accelerator.log
    Fallback: DATA_DIR/logs/stderr.log  (when accelerator.log missing or 0 bytes)
    """
    primary = DATA_DIR / "logs" / "accelerator.log"
    if primary.exists() and primary.stat().st_size > 0:
        return primary
    return DATA_DIR / "logs" / "stderr.log"


def _log_source_info(path: Path, n: int) -> dict:
    """Build the per-source dict for the JSON response."""
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    content = _tail_file(path, n) if exists else []
    return {
        "path": str(path),
        "lines_returned": len(content),
        "content": content,
        "size_bytes": size,
        "exists": exists,
    }


@app.get("/logs")
async def get_logs(
    lines: int = Query(_DEFAULT_LOG_LINES, ge=1, description="Number of tail lines (max 1000)"),
    source: str = Query("all", description="Log source: server, menubar, or all"),
    format: str = Query("json", description="Response format: json or text"),
    _auth: None = Depends(verify_token),
):
    """Return the last N lines from accelerator log files.

    Auth: requires bearer token (same as all other non-/health endpoints).
    """
    if lines > _MAX_LOG_LINES:
        raise HTTPException(
            status_code=400,
            detail=f"lines parameter exceeds maximum ({_MAX_LOG_LINES}). Got {lines}.",
        )

    if source not in ("server", "menubar", "all"):
        raise HTTPException(
            status_code=400,
            detail="source must be one of: server, menubar, all",
        )

    if format not in ("json", "text"):
        raise HTTPException(
            status_code=400,
            detail="format must be one of: json, text",
        )

    server_log = _resolve_server_log()
    menubar_log = DATA_DIR / "logs" / "menubar.log"

    if format == "text":
        sections: list[str] = []
        if source in ("server", "all"):
            sections.append(f"=== {server_log.name} ===")
            if server_log.exists():
                tail = _tail_file(server_log, lines)
                sections.extend(tail if tail else [f"(empty: {server_log})"])
            else:
                sections.append(f"(file not found: {server_log})")
        if source in ("menubar", "all"):
            if source == "all":
                sections.append("")
            sections.append(f"=== {menubar_log.name} ===")
            if menubar_log.exists():
                tail = _tail_file(menubar_log, lines)
                sections.extend(tail if tail else [f"(empty: {menubar_log})"])
            else:
                sections.append(f"(file not found: {menubar_log})")
        return PlainTextResponse(content="\n".join(sections))

    # JSON format
    sources_payload: dict = {}
    if source in ("server", "all"):
        sources_payload["server"] = _log_source_info(server_log, lines)
    if source in ("menubar", "all"):
        sources_payload["menubar"] = _log_source_info(menubar_log, lines)

    return {
        "sources": sources_payload,
        "timestamp": _utcnow(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=ACCELERATOR_HOST,
        port=ACCELERATOR_PORT,
        log_level=LOG_LEVEL.lower(),
    )


