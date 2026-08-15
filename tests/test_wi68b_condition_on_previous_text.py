"""
tests/test_wi68b_condition_on_previous_text.py -- WI68b: condition_on_previous_text support.

Verifies that:
  1. /v1/audio/transcriptions endpoint accepts condition_on_previous_text Form param
     (previously silently ignored — FastAPI would discard unknown form fields).
  2. condition_on_previous_text is included in the cache key (so true/false have
     separate cache entries and don't collide).
  3. condition_on_previous_text is forwarded to _process_transcription params dict.
  4. _process_transcription passes condition_on_previous_text to whisper.transcribe()
     kwargs as a bool.
  5. Both "true" and "false" string values are handled correctly.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal audio bytes (WAV header)
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"


def _wav_file(name: str = "test.wav") -> tuple:
    return ("file", (name, io.BytesIO(FAKE_WAV), "audio/wav"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# WI-ACC-28: _reset_rate_limits fixture removed — per-IP rate limiter deleted.


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
# 1. Endpoint accepts condition_on_previous_text without 422
# ---------------------------------------------------------------------------


class TestEndpointAcceptsParam:
    """Endpoint must accept condition_on_previous_text as a Form field."""

    def test_condition_false_accepted_without_422(self, client):
        """condition_on_previous_text=false must not cause a 422 parse error."""
        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "false"},
            )
        assert resp.status_code != 422, (
            "WI68b: condition_on_previous_text=false caused a 422 — "
            "it is not declared as a Form param in the endpoint"
        )
        assert resp.status_code in (200, 202)

    def test_condition_true_accepted_without_422(self, client):
        """condition_on_previous_text=true must not cause a 422 parse error."""
        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "true"},
            )
        assert resp.status_code != 422
        assert resp.status_code in (200, 202)

    def test_omitting_param_still_works(self, client):
        """Requests without condition_on_previous_text remain valid."""
        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", return_value=None),
            patch("server.job_count_by_status", return_value=0),
        ):
            resp = client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
            )
        assert resp.status_code in (200, 202)


# ---------------------------------------------------------------------------
# 2. condition_on_previous_text in cache key
# ---------------------------------------------------------------------------


class TestCacheKeyInclusion:
    """condition_on_previous_text must be part of the cache key."""

    def test_true_vs_false_produce_different_cache_keys(self, client):
        """Requests with true vs false condition_on_previous_text get different cache keys."""
        captured_keys: list[str] = []

        def fake_cache_lookup(cache_key, **kwargs):
            captured_keys.append(cache_key)
            return None  # always miss

        with (
            patch("server._enqueue_job"),
            patch("server.cache_lookup", side_effect=fake_cache_lookup),
            patch("server.job_count_by_status", return_value=0),
        ):
            client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "true"},
            )
            client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "false"},
            )

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1], (
            "WI68b: condition_on_previous_text=true and =false share the same cache key — "
            "different settings would serve each other's cached results"
        )

    def test_with_vs_without_produce_different_cache_keys(self, client):
        """Request without param vs with condition_on_previous_text=false differ in cache key."""
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
            client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "false"},
            )

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1], (
            "WI68b: omitting condition_on_previous_text and passing 'false' share the same key"
        )


# ---------------------------------------------------------------------------
# 3. condition_on_previous_text forwarded to job_create params
# ---------------------------------------------------------------------------


class TestParamsForwarding:
    """condition_on_previous_text must appear in params dict passed to job_create."""

    def test_false_forwarded_to_job_create(self, client):
        """condition_on_previous_text=false must appear in params."""
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
            client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "false"},
            )

        assert len(captured_params) == 1
        params = captured_params[0]
        assert "condition_on_previous_text" in params, (
            "WI68b: condition_on_previous_text not forwarded to job_create params"
        )
        assert params["condition_on_previous_text"] == "false"

    def test_true_forwarded_to_job_create(self, client):
        """condition_on_previous_text=true must appear in params."""
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
            client.post(
                "/v1/audio/transcriptions",
                files=[_wav_file()],
                data={"condition_on_previous_text": "true"},
            )

        assert len(captured_params) == 1
        params = captured_params[0]
        assert "condition_on_previous_text" in params
        assert params["condition_on_previous_text"] == "true"

    def test_omitted_param_not_in_params(self, client):
        """When condition_on_previous_text is omitted, it must not appear in params."""
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
        assert "condition_on_previous_text" not in params


# ---------------------------------------------------------------------------
# 4. _process_transcription passes condition_on_previous_text to whisper.transcribe()
# ---------------------------------------------------------------------------


class TestWhisperKwargsPassthrough:
    """_process_transcription must convert and pass condition_on_previous_text to whisper."""

    def _make_job(self, tmp_path, params: dict) -> dict:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(FAKE_WAV)
        return {
            "job_id": "test-job-wi68b",
            "file_path": str(audio_file),
            "params_json": json.dumps(params),
        }

    def test_false_string_passed_as_bool_false(self, tmp_path):
        """condition_on_previous_text='false' must reach whisper as bool False."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {
            "model": "base",
            "condition_on_previous_text": "false",
        })

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "condition_on_previous_text" in call_kwargs, (
            "WI68b: condition_on_previous_text not passed to whisper.transcribe kwargs"
        )
        assert call_kwargs["condition_on_previous_text"] is False, (
            f"Expected bool False, got {call_kwargs['condition_on_previous_text']!r}"
        )

    def test_true_string_passed_as_bool_true(self, tmp_path):
        """condition_on_previous_text='true' must reach whisper as bool True."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {
            "model": "base",
            "condition_on_previous_text": "true",
        })

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "condition_on_previous_text" in call_kwargs
        assert call_kwargs["condition_on_previous_text"] is True, (
            f"Expected bool True, got {call_kwargs['condition_on_previous_text']!r}"
        )

    def test_omitted_param_not_in_whisper_kwargs(self, tmp_path):
        """When condition_on_previous_text is absent, it must not appear in whisper kwargs."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {"model": "base"})

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert "condition_on_previous_text" not in call_kwargs

    def test_all_transcribe_params_together(self, tmp_path):
        """temperature + no_speech_threshold + condition_on_previous_text all forwarded."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {
            "model": "base",
            "temperature": 0.2,
            "no_speech_threshold": 0.3,
            "condition_on_previous_text": "false",
        })

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs.get("temperature") == pytest.approx(0.2)
        assert call_kwargs.get("no_speech_threshold") == pytest.approx(0.3)
        assert call_kwargs.get("condition_on_previous_text") is False

    def test_zero_string_treated_as_false(self, tmp_path):
        """condition_on_previous_text='0' is treated as False (falsy string)."""
        import server

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "hello", "segments": [], "language": "en", "duration": 1.0
        }

        job = self._make_job(tmp_path, {
            "model": "base",
            "condition_on_previous_text": "0",
        })

        with patch("server._load_whisper", return_value=mock_whisper):
            server._process_transcription(job)

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs.get("condition_on_previous_text") is False
