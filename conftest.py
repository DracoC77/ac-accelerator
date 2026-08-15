"""
conftest.py — Shared pytest fixtures for Audio Chronicle Accelerator.

Provides:
  - ground_truth_wav: uses the FIXTURE_PATH env var for a local copy of the
    synthetic OGG fixture (or downloads it when FIXTURE_URL/FIXTURE_TOKEN are
    provided), converts to WAV via ffmpeg if available, otherwise returns the
    .ogg directly (server accepts .ogg). When no fixture is available the E2E
    tests are skipped gracefully.
  - ground_truth_answers: synthesized answer key for E2E transcript comparison.

Note: the reference audio fixture is a synthetic dev clip (no real personal
audio). To run the E2E suite, point FIXTURE_PATH at a local copy, or set
FIXTURE_URL (and optionally FIXTURE_TOKEN for a protected host).
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ground-truth fixture constants
# ---------------------------------------------------------------------------

# Optional remote URL for the synthetic OGG fixture. Not set by default so the
# public test suite never depends on an external/protected host — provide a
# local file via FIXTURE_PATH instead, or set FIXTURE_URL to your own copy.
_FIXTURE_OGG_URL = os.getenv("FIXTURE_URL", "")

# Known transcription text for the synthetic fixture (from annotated_segments.json).
# Concatenated text across all segments — used for word overlap assertion in E2E tests.
_GROUND_TRUTH_TEXT = (
    "Hey yeah sure "
    "So the plan for today is to finish up the speaker ID module "
    "Yeah totally agree "
    "I was thinking we should double check the resemblyzer version compatibility "
    "Good call I will add a smoke test for the VoiceEncoder import "
    "Okay so the afternoon session We are going to review the code "
    "The sidecar approach makes sense We might want a sanity check"
)

_ANSWER_KEY: dict = {
    "text": _GROUND_TRUTH_TEXT,
    "min_word_overlap": 0.80,
    "expected_speakers": 2,
    "expected_min_segments": 1,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ground_truth_wav(tmp_path_factory):
    """Return a Path to a usable audio fixture for E2E tests.

    Priority:
    1. FIXTURE_PATH env var (pre-downloaded local file)
    2. FIXTURE_URL env var (download; optional FIXTURE_TOKEN for a protected host)
       — converted to WAV via ffmpeg if available, otherwise the .ogg is used
       directly (the server supports .ogg).

    If neither a local fixture nor a fixture URL is available, the E2E test is
    skipped rather than failing — this keeps the public suite green without a
    fixture, while remaining fully functional when one is provided.
    """
    fixture_path = os.getenv("FIXTURE_PATH", "")
    if fixture_path and Path(fixture_path).exists():
        return Path(fixture_path)

    if not _FIXTURE_OGG_URL:
        pytest.skip(
            "No audio fixture available: set FIXTURE_PATH to a local file or "
            "FIXTURE_URL to a downloadable synthetic fixture to run E2E tests."
        )

    token = os.getenv("FIXTURE_TOKEN", "")
    tmp = tmp_path_factory.mktemp("fixtures")
    ogg_path = tmp / "ground_truth.ogg"

    # Download OGG fixture; skip (don't fail) if it is unreachable.
    try:
        req = urllib.request.Request(_FIXTURE_OGG_URL)
        if token:
            req.add_header("Authorization", f"token {token}")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            ogg_path.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Audio fixture download failed ({exc!r}); skipping E2E test.")

    # Try converting to WAV via ffmpeg (optional — server accepts .ogg too)
    wav_path = tmp / "ground_truth.wav"
    try:
        result = subprocess.run(  # noqa: S603
            ["ffmpeg", "-y", "-i", str(ogg_path), str(wav_path)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and wav_path.exists():
            return wav_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # ffmpeg not available — use .ogg

    return ogg_path


@pytest.fixture(scope="session")
def ground_truth_answers() -> dict:
    """Return the E2E answer key dict with expected transcript text and thresholds."""
    return _ANSWER_KEY
