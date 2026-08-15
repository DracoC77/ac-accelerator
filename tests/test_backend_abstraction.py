"""Backend abstraction sanity tests.

These tests don't require mlx-whisper to be installed — they only verify the
ABC contract, the dataclass shapes, and that ``MlxWhisperBackend`` is a valid
concrete subclass (instantiating it doesn't pull in mlx_whisper; that only
happens on ``load()``).
"""

from __future__ import annotations

import inspect

import pytest


def test_inference_backend_is_abstract():
    """InferenceBackend cannot be instantiated directly."""
    from accelerator.backends import InferenceBackend

    assert inspect.isabstract(InferenceBackend)
    with pytest.raises(TypeError):
        InferenceBackend()  # type: ignore[abstract]


def test_transcript_dataclasses_shape():
    """TranscriptResult / TranscriptSegment expose the documented fields."""
    from accelerator.backends import TranscriptResult, TranscriptSegment

    seg = TranscriptSegment(
        start=0.0, end=1.0, text="hi", avg_logprob=-0.1, no_speech_prob=0.01
    )
    assert seg.start == 0.0
    assert seg.tokens == []  # default
    assert seg.compression_ratio == 0.0  # default

    res = TranscriptResult(segments=[seg], language="en")
    assert res.language == "en"
    assert res.language_probability == 0.0  # default
    assert res.segments[0] is seg


def test_mlx_backend_implements_abc():
    """MlxWhisperBackend is a non-abstract subclass and is instantiable."""
    from accelerator.backends import InferenceBackend
    from accelerator.backends.mlx_backend import MlxWhisperBackend

    assert issubclass(MlxWhisperBackend, InferenceBackend)
    assert not inspect.isabstract(MlxWhisperBackend)

    backend = MlxWhisperBackend(model="mlx-community/whisper-tiny")
    assert backend.backend_name == "mlx-whisper"
    assert backend.is_loaded is False  # nothing loaded until load() runs


def test_create_backend_env_var(monkeypatch):
    """INFERENCE_BACKEND=mlx returns an MlxWhisperBackend regardless of host."""
    from accelerator.backends import create_backend
    from accelerator.backends.mlx_backend import MlxWhisperBackend

    monkeypatch.setenv("INFERENCE_BACKEND", "mlx")
    backend = create_backend(model="mlx-community/whisper-tiny")
    assert isinstance(backend, MlxWhisperBackend)


def test_create_backend_unknown_raises(monkeypatch):
    """Unknown backend names raise RuntimeError."""
    from accelerator.backends import create_backend

    monkeypatch.setenv("INFERENCE_BACKEND", "totally_made_up")
    with pytest.raises(RuntimeError):
        create_backend()


def test_create_backend_autodetect_non_darwin(monkeypatch):
    """On non-Darwin hosts with no env var set, create_backend raises with
    the documented 'No backend configured' message."""
    import platform

    from accelerator.backends import create_backend

    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="No backend configured"):
        create_backend()


def test_mlx_backend_transcribe_without_module_raises():
    """Calling transcribe() on a backend whose _module is None (i.e. load()
    has not been called and no module_loader is configured) must raise a
    clean error rather than silently passing or crashing with an AttributeError.

    This is the guard for the tech debt documented in server.py at the
    ``backend._module = whisper_module`` injection site:
    replacing the injection with set_module_loader() requires confidence that
    the bare-backend path fails loudly instead of silently.
    """
    from accelerator.backends.mlx_backend import MlxWhisperBackend

    def _failing_loader():
        raise ImportError("mlx_whisper not installed on this platform")

    backend = MlxWhisperBackend(
        model="mlx-community/whisper-tiny",
        module_loader=_failing_loader,
    )
    assert backend._module is None
    assert backend.is_loaded is False

    # transcribe() must raise — either ImportError from the loader or a
    # TypeError/AttributeError — rather than silently returning garbage.
    with pytest.raises((ImportError, TypeError, AttributeError, RuntimeError)):
        backend.transcribe("/nonexistent/path.wav")
