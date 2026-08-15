"""
E2E pytest tests for Audio Chronicle Accelerator.

Requires a live server instance:
  ACCELERATOR_URL=http://localhost:8765 pytest tests/e2e/ -v

Optional:
  ACCELERATOR_TOKEN=<bearer-token>  — auth token if server requires one
  FIXTURE_PATH=/path/to/audio.wav   — skip fixture download, use local file
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACCELERATOR_URL = os.getenv("ACCELERATOR_URL", "").rstrip("/")
ACCELERATOR_TOKEN = os.getenv("ACCELERATOR_TOKEN", "")

_POLL_INTERVAL = 5      # seconds between status polls
_POLL_TIMEOUT = 600     # 10 minutes max
_MIN_WORD_OVERLAP = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    if ACCELERATOR_TOKEN:
        return {"Authorization": f"Bearer {ACCELERATOR_TOKEN}"}
    return {}


def _poll_job(job_id: str) -> dict:
    """Poll GET /v1/jobs/{id} until terminal state or timeout. Returns final job dict."""
    deadline = time.monotonic() + _POLL_TIMEOUT
    while time.monotonic() < deadline:
        resp = requests.get(
            f"{ACCELERATOR_URL}/v1/jobs/{job_id}", headers=_headers(), timeout=30
        )
        resp.raise_for_status()
        job = resp.json()
        if job["status"] in ("complete", "failed", "cancelled"):
            return job
        time.sleep(_POLL_INTERVAL)
    pytest.fail(f"Job {job_id} did not complete within {_POLL_TIMEOUT}s")


def _word_overlap(predicted: str, reference: str) -> float:
    """Return the fraction of reference words found in the predicted text."""
    pred_words = set(predicted.lower().split())
    ref_words = reference.lower().split()
    if not ref_words:
        return 1.0
    matched = sum(1 for w in ref_words if w in pred_words)
    return matched / len(ref_words)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ACCELERATOR_URL, reason="ACCELERATOR_URL not set — skipping E2E")
def test_health_live(ground_truth_wav, ground_truth_answers):  # noqa: ARG001
    """Sanity check: live server health endpoint returns healthy."""
    resp = requests.get(f"{ACCELERATOR_URL}/health", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.skipif(not ACCELERATOR_URL, reason="ACCELERATOR_URL not set — skipping E2E")
def test_transcription_e2e(ground_truth_wav: Path, ground_truth_answers: dict):
    """Submit ground-truth audio → poll → assert ≥80% word overlap with reference text."""
    audio_path = ground_truth_wav
    assert audio_path.exists(), f"Fixture file not found: {audio_path}"

    # Submit
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{ACCELERATOR_URL}/v1/audio/transcriptions",
            headers=_headers(),
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=60,
        )
    assert resp.status_code in (200, 202), f"Unexpected status {resp.status_code}: {resp.text}"
    body = resp.json()

    # Cache hit: result is inline
    if resp.status_code == 200 and body.get("cache_hit"):
        result = body["result"]
    else:
        job = _poll_job(body["job_id"])
        assert job["status"] == "complete", f"Job failed: {job.get('error')}"
        result = job["result"]

    predicted = result.get("text", "")
    reference = ground_truth_answers["text"]
    overlap = _word_overlap(predicted, reference)
    assert overlap >= _MIN_WORD_OVERLAP, (
        f"Word overlap {overlap:.1%} below threshold {_MIN_WORD_OVERLAP:.0%}.\n"
        f"  Predicted: {predicted!r}\n"
        f"  Reference: {reference!r}"
    )


@pytest.mark.skipif(not ACCELERATOR_URL, reason="ACCELERATOR_URL not set — skipping E2E")
def test_diarization_e2e(ground_truth_wav: Path):
    """Submit ground-truth audio for diarization → poll → assert ≥1 speaker segment."""
    audio_path = ground_truth_wav
    assert audio_path.exists(), f"Fixture file not found: {audio_path}"

    # Submit
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{ACCELERATOR_URL}/v1/diarize",
            headers=_headers(),
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=60,
        )
    assert resp.status_code in (200, 202), f"Unexpected status {resp.status_code}: {resp.text}"
    body = resp.json()

    # Cache hit: result is inline
    if resp.status_code == 200 and body.get("cache_hit"):
        result = body["result"]
    else:
        job = _poll_job(body["job_id"])
        assert job["status"] == "complete", f"Job failed: {job.get('error')}"
        result = job["result"]

    segments = result.get("segments", [])
    assert len(segments) >= 1, f"Expected ≥1 diarization segment, got {len(segments)}"

    # Validate segment structure
    for seg in segments:
        assert "speaker" in seg, f"Missing 'speaker' key in segment: {seg}"
        assert "start" in seg and "end" in seg, f"Missing start/end in segment: {seg}"
        assert seg["end"] > seg["start"], f"Segment end ≤ start: {seg}"
