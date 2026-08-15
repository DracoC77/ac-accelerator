"""
Tests verifying temperature and no_speech_threshold are correctly included
in the cache key (Bug 1), forwarded to the inference worker params (Bug 2), and passed
through to mlx-whisper kwargs (Bug 3).

The bugs existed prior to a prior fix in server.py. These tests provide
regression coverage to prevent the fixes from regressing.
"""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal audio bytes
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"


def _wav_file(name: str = "test.wav") -> tuple:
    return ("file", (name, io.BytesIO(FAKE_WAV), "audio/wav"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# _reset_rate_limits fixture removed — per-IP rate limiter deleted.


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATOR_TOKEN", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    import server
    server.DATA_DIR.mkdir(parents=True, exist_ok=True)
    server.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server._init_db()

    with TestClient(server.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Bug 1: temperature/no_speech_threshold must affect the cache key
# ---------------------------------------------------------------------------

class TestCacheKeyInclusion:
    """Bug 1 — cache key must differ when temperature or no_speech_threshold differ."""

    def test_different_temperature_produces_different_cache_key(self, client):
        """Two requests with same audio but different temperature must get different cache keys."""
        captured_keys: list[str] = []

        def fake_cache_lookup(cache_key, **kwargs):
            captured_keys.append(cache_key)
            return None  # always miss

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", side_effect=fake_cache_lookup),
            patch("server.job_count_by_status", return_value=0),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"temperature": "0.0"})
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"temperature": "0.8"})

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1], (
            "Bug 1 regression: temperature is not included in the cache key"
        )

    def test_different_nst_produces_different_cache_key(self, client):
        """Two requests with same audio but different no_speech_threshold must get different cache keys."""
        captured_keys: list[str] = []

        def fake_cache_lookup(cache_key, **kwargs):
            captured_keys.append(cache_key)
            return None

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", side_effect=fake_cache_lookup),
            patch("server.job_count_by_status", return_value=0),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"no_speech_threshold": "0.3"})
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"no_speech_threshold": "0.9"})

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1], (
            "Bug 1 regression: no_speech_threshold is not included in the cache key"
        )

    def test_no_params_vs_with_params_differ(self, client):
        """Request without params and one with temperature should get different cache keys."""
        captured_keys: list[str] = []

        def fake_cache_lookup(cache_key, **kwargs):
            captured_keys.append(cache_key)
            return None

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", side_effect=fake_cache_lookup),
            patch("server.job_count_by_status", return_value=0),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()])
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"temperature": "0.5"})

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1], (
            "Bug 1 regression: adding temperature did not change the cache key"
        )


# ---------------------------------------------------------------------------
# Bug 2: temperature/no_speech_threshold must be in params forwarded to job_create
# ---------------------------------------------------------------------------

class TestParamsForwarding:
    """Bug 2 — params dict passed to job_create must include temperature and no_speech_threshold."""

    def test_temperature_forwarded_to_job_create(self, client):
        """Scalar temperature value must appear in params passed to job_create."""
        captured_params: list[dict] = []
        original_job_create = __import__("server").job_create

        def fake_job_create(job_id, job_type, cache_key, file_path, params):
            captured_params.append(dict(params))
            return original_job_create(job_id, job_type, cache_key, file_path, params)

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
            patch("server.job_create", side_effect=fake_job_create),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"temperature": "0.6"})

        assert len(captured_params) == 1
        params = captured_params[0]
        assert "temperature" in params, (
            "Bug 2 regression: temperature not forwarded to job_create params"
        )
        # JSON-parsed: could be float 0.6 or already parsed
        assert float(params["temperature"]) == pytest.approx(0.6)

    def test_nst_forwarded_to_job_create(self, client):
        """no_speech_threshold must appear in params passed to job_create."""
        captured_params: list[dict] = []
        original_job_create = __import__("server").job_create

        def fake_job_create(job_id, job_type, cache_key, file_path, params):
            captured_params.append(dict(params))
            return original_job_create(job_id, job_type, cache_key, file_path, params)

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
            patch("server.job_create", side_effect=fake_job_create),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"no_speech_threshold": "0.4"})

        assert len(captured_params) == 1
        params = captured_params[0]
        assert "no_speech_threshold" in params, (
            "Bug 2 regression: no_speech_threshold not forwarded to job_create params"
        )
        assert float(params["no_speech_threshold"]) == pytest.approx(0.4)

    def test_temperature_json_list_forwarded(self, client):
        """JSON-encoded temperature list must be parsed and forwarded as a list."""
        captured_params: list[dict] = []
        original_job_create = __import__("server").job_create

        def fake_job_create(job_id, job_type, cache_key, file_path, params):
            captured_params.append(dict(params))
            return original_job_create(job_id, job_type, cache_key, file_path, params)

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
            patch("server.job_create", side_effect=fake_job_create),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()],
                        data={"temperature": "[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]"})

        assert len(captured_params) == 1
        params = captured_params[0]
        assert "temperature" in params, "temperature list not forwarded"
        assert isinstance(params["temperature"], list), (
            "temperature should be parsed to a list from JSON"
        )
        assert params["temperature"] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_missing_params_not_forwarded(self, client):
        """When temperature/nst are omitted, they should NOT appear in params."""
        captured_params: list[dict] = []
        original_job_create = __import__("server").job_create

        def fake_job_create(job_id, job_type, cache_key, file_path, params):
            captured_params.append(dict(params))
            return original_job_create(job_id, job_type, cache_key, file_path, params)

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
            patch("server.job_create", side_effect=fake_job_create),
        ):
            client.post("/v1/audio/transcriptions", files=[_wav_file()])

        assert len(captured_params) == 1
        params = captured_params[0]
        assert "temperature" not in params
        assert "no_speech_threshold" not in params


# ---------------------------------------------------------------------------
# Bug 3: temperature/no_speech_threshold must reach whisper.transcribe() kwargs
# ---------------------------------------------------------------------------

class TestWhisperKwargsPassthrough:
    """Bug 3 — _process_transcription must pass temperature and no_speech_threshold to whisper.transcribe."""

    def _make_job(self, tmp_path, params: dict) -> dict:
        """Build a minimal job dict as _process_transcription expects."""
        # Write a fake audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(FAKE_WAV)
        return {
            "job_id": "test-job-123",
            "file_path": str(audio_file),
            "params_json": json.dumps(params),
        }

    def test_scalar_temperature_passed_to_whisper(self, tmp_path):
        """Scalar temperature must be passed as float kwarg to whisper.transcribe."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {"model": "base", "temperature": 0.6})

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "temperature" in call_kwargs, (
            "Bug 3 regression: temperature not passed to whisper.transcribe kwargs"
        )
        assert call_kwargs["temperature"] == pytest.approx(0.6)

    def test_nst_passed_to_whisper(self, tmp_path):
        """no_speech_threshold must be passed as float kwarg to whisper.transcribe."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {"model": "base", "no_speech_threshold": 0.4})

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "no_speech_threshold" in call_kwargs, (
            "Bug 3 regression: no_speech_threshold not passed to whisper.transcribe kwargs"
        )
        assert call_kwargs["no_speech_threshold"] == pytest.approx(0.4)

    def test_temperature_list_converted_to_tuple(self, tmp_path):
        """JSON-parsed temperature list must be converted to tuple for mlx-whisper fallback chain."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {
            "model": "base",
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "temperature" in call_kwargs, "temperature list not passed to whisper"
        assert isinstance(call_kwargs["temperature"], tuple), (
            "temperature list must be converted to tuple for mlx-whisper"
        )
        assert call_kwargs["temperature"] == pytest.approx((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    def test_missing_params_not_in_whisper_kwargs(self, tmp_path):
        """When temperature/nst are absent from params, they must not appear in whisper kwargs."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {"model": "base"})

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "temperature" not in call_kwargs
        assert "no_speech_threshold" not in call_kwargs

    def test_both_params_together(self, tmp_path):
        """Both temperature and no_speech_threshold forwarded together."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {
            "model": "base",
            "temperature": 0.2,
            "no_speech_threshold": 0.6,
        })

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs.get("temperature") == pytest.approx(0.2)
        assert call_kwargs.get("no_speech_threshold") == pytest.approx(0.6)
