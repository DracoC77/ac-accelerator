"""WI-ACC-27b: FasterWhisperBackend tests.

These tests run on the Mac build machine where neither ``faster_whisper`` nor
CUDA-enabled torch is installed. We inject a stub ``faster_whisper`` module
into ``sys.modules`` *before* importing the backend, so the lazy import inside
``FasterWhisperBackend.load()`` picks up the mock instead of trying to import
the real wheel.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock the faster_whisper module before the backend imports it.
# The backend uses a lazy `from faster_whisper import WhisperModel` inside
# load(), so we just need this stub present in sys.modules by then.
# ---------------------------------------------------------------------------
_mock_fw = MagicMock()
sys.modules.setdefault("faster_whisper", _mock_fw)


from accelerator.backends.base import TranscriptResult, TranscriptSegment  # noqa: E402
from accelerator.backends.faster_whisper_backend import FasterWhisperBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Identity / properties
# ---------------------------------------------------------------------------

def test_backend_name():
    b = FasterWhisperBackend()
    assert b.backend_name == "faster-whisper"


def test_is_loaded_false_before_load():
    b = FasterWhisperBackend()
    assert not b.is_loaded


def test_default_construction_uses_env_defaults(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("FASTER_WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("FASTER_WHISPER_COMPUTE_TYPE", raising=False)
    b = FasterWhisperBackend()
    assert b._model_name == "large-v3-turbo"
    assert b._device == "cuda"
    assert b._compute_type == "int8_float16"


def test_construction_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("FASTER_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
    b = FasterWhisperBackend()
    assert b._model_name == "tiny"
    assert b._device == "cpu"
    assert b._compute_type == "int8"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_load_creates_model():
    b = FasterWhisperBackend()
    b.load()
    assert b.is_loaded


def test_load_is_idempotent():
    b = FasterWhisperBackend()
    b.load()
    first = b._model
    b.load()
    assert b._model is first  # second load() must not re-instantiate


def test_unload_releases_model():
    b = FasterWhisperBackend()
    b.load()
    b.unload()
    assert not b.is_loaded
    assert b._model is None


def test_unload_idempotent():
    b = FasterWhisperBackend()
    b.unload()  # not loaded yet — must not raise
    b.unload()  # second call also safe
    assert not b.is_loaded


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _make_mock_segment(start: float, end: float, text: str,
                       avg_logprob: float = -0.3, no_speech_prob: float = 0.01):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    seg.avg_logprob = avg_logprob
    seg.no_speech_prob = no_speech_prob
    return seg


def test_transcribe_returns_transcript_result():
    mock_seg = _make_mock_segment(0.0, 1.0, "hello")
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 1.0

    b = FasterWhisperBackend()
    b.load()
    b._model.transcribe.return_value = ([mock_seg], mock_info)

    result = b.transcribe("fake_path.wav")
    assert isinstance(result, TranscriptResult)
    assert len(result.segments) == 1
    assert result.segments[0].text == "hello"
    assert isinstance(result.segments[0], TranscriptSegment)
    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.99)


def test_transcribe_consumes_generator_fully():
    """The contract from the task spec: transcribe() must materialize the
    generator into a list, not return a lazy iterator."""
    segs = [
        _make_mock_segment(0.0, 1.0, "hello "),
        _make_mock_segment(1.0, 2.5, "world"),
        _make_mock_segment(2.5, 3.0, "!"),
    ]
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.97
    mock_info.duration = 3.0

    def _gen():
        for s in segs:
            yield s

    b = FasterWhisperBackend()
    b.load()
    b._model.transcribe.return_value = (_gen(), mock_info)

    result = b.transcribe("fake_path.wav")
    assert isinstance(result.segments, list)
    assert len(result.segments) == 3
    assert result.text == "hello world!"
    assert result.duration == pytest.approx(3.0)


def test_transcribe_without_load_raises():
    b = FasterWhisperBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        b.transcribe("fake_path.wav")


def test_transcribe_passes_language_and_prompt():
    mock_info = MagicMock()
    mock_info.language = "es"
    mock_info.language_probability = 0.88
    mock_info.duration = 0.0

    b = FasterWhisperBackend()
    b.load()
    b._model.transcribe.return_value = ([], mock_info)

    b.transcribe("x.wav", language="es", initial_prompt="hola")
    args, kwargs = b._model.transcribe.call_args
    assert args[0] == "x.wav"
    assert kwargs["language"] == "es"
    assert kwargs["initial_prompt"] == "hola"
    # VAD must remain disabled — pipeline handles VAD upstream.
    assert kwargs["vad_filter"] is False
    assert kwargs["word_timestamps"] is False


def test_transcribe_forwards_optional_kwargs():
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.9
    mock_info.duration = 0.0

    b = FasterWhisperBackend()
    b.load()
    b._model.transcribe.return_value = ([], mock_info)

    b.transcribe(
        "x.wav",
        temperature=0.2,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
    )
    _, kwargs = b._model.transcribe.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["no_speech_threshold"] == 0.5
    assert kwargs["condition_on_previous_text"] is False


def test_transcribe_ignores_unknown_kwargs():
    """The InferenceBackend ABC contract: unknown kwargs must be ignored,
    not raise. This is what lets the same call site work across backends."""
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.9
    mock_info.duration = 0.0

    b = FasterWhisperBackend()
    b.load()
    b._model.transcribe.return_value = ([], mock_info)

    # Should not raise.
    b.transcribe("x.wav", some_unknown_kwarg="ignored")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_create_backend_faster_whisper_env(monkeypatch):
    """INFERENCE_BACKEND=faster-whisper returns a FasterWhisperBackend."""
    from accelerator.backends import create_backend

    monkeypatch.setenv("INFERENCE_BACKEND", "faster-whisper")
    backend = create_backend()
    assert isinstance(backend, FasterWhisperBackend)


def test_create_backend_faster_whisper_underscore_alias(monkeypatch):
    """The underscore variant is also accepted (matches the planning doc)."""
    from accelerator.backends import create_backend

    monkeypatch.setenv("INFERENCE_BACKEND", "faster_whisper")
    backend = create_backend()
    assert isinstance(backend, FasterWhisperBackend)


def test_create_backend_mlx_still_works(monkeypatch):
    """No regression — explicit mlx selection still returns MlxWhisperBackend."""
    from accelerator.backends import create_backend
    from accelerator.backends.mlx_backend import MlxWhisperBackend

    monkeypatch.setenv("INFERENCE_BACKEND", "mlx")
    backend = create_backend(model="mlx-community/whisper-tiny")
    assert isinstance(backend, MlxWhisperBackend)


def test_create_backend_autodetect_cuda(monkeypatch):
    """Non-Darwin host with CUDA available auto-selects faster-whisper."""
    import platform as _platform

    import accelerator.backends as backends_pkg
    from accelerator.backends import create_backend

    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    monkeypatch.setattr(_platform, "system", lambda: "Windows")
    monkeypatch.setattr(_platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(backends_pkg, "_cuda_available", lambda: True)

    backend = create_backend()
    assert isinstance(backend, FasterWhisperBackend)


def test_create_backend_autodetect_no_cuda_raises(monkeypatch):
    """Non-Darwin host with no CUDA still raises (no silent CPU fallback)."""
    import platform as _platform

    import accelerator.backends as backends_pkg
    from accelerator.backends import create_backend

    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(backends_pkg, "_cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="No backend configured"):
        create_backend()


def test_create_backend_passes_device_and_compute_type(monkeypatch):
    """FASTER_WHISPER_DEVICE / FASTER_WHISPER_COMPUTE_TYPE env vars flow
    through to the backend constructor."""
    from accelerator.backends import create_backend

    monkeypatch.setenv("INFERENCE_BACKEND", "faster-whisper")
    monkeypatch.setenv("FASTER_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
    backend = create_backend()
    assert isinstance(backend, FasterWhisperBackend)
    assert backend._device == "cpu"
    assert backend._compute_type == "int8"
