"""InferenceBackend ABC and result dataclasses.

This module defines the contract every transcription backend must implement.
Concrete backends (e.g. :mod:`accelerator.backends.mlx_backend`) own the model
object and are responsible for loading/unloading it and normalizing their
native output into :class:`TranscriptResult`.

Pure refactor, no behavior change vs. the previous inline
mlx-whisper code path in ``server.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TranscriptSegment:
    """A single transcript segment, normalized across backends.

    The first five fields are the canonical schema from the backend design §1.3.
    The trailing fields (``tokens``, ``compression_ratio``) are preserved
    here so that backends which expose them (mlx-whisper does) can keep
    the existing server.py response payload byte-identical. Backends that
    do not expose them leave the defaults in place.
    """

    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    # Extension fields — preserved for zero-behavior-change with current
    # server.py response payload. Optional for future backends.
    tokens: list[int] = field(default_factory=list)
    compression_ratio: float = 0.0
    # WI-BUG-20: temperature used by Whisper for this segment. mlx-whisper
    # surfaces this per-segment; faster-whisper also exposes it. Default 0.0
    # so backends that don't expose it produce a benign neutral value.
    temperature: float = 0.0


@dataclass
class TranscriptResult:
    """Full transcription result, normalized across backends.

    ``language_probability`` is kept for forward-compatibility with
    backends that expose it (e.g. faster-whisper). mlx-whisper does not
    surface a language probability today, so its backend sets ``0.0``.

    ``text`` and ``duration`` are extension fields that the current
    server.py response includes; backends populate them when available
    so the existing response payload is byte-identical.
    """

    segments: list[TranscriptSegment]
    language: str
    language_probability: float = 0.0
    # Extension fields — preserved for zero-behavior-change with current
    # server.py response payload.
    text: str = ""
    duration: float = 0.0


class InferenceBackend(ABC):
    """Stateful transcription backend. One instance per server process.

    Implementations own the model object. They MUST:
      * be safe to construct lazily (idle watchdog calls ``load()`` on demand),
      * release GPU/MPS memory on ``unload()`` (gaming-VM requirement),
      * be idempotent across repeated ``load()`` / ``unload()`` calls,
      * normalize native output into :class:`TranscriptResult`.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model into memory/VRAM. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model, release VRAM. Safe to call multiple times."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptResult:
        """Transcribe an audio file and return a :class:`TranscriptResult`.

        ``**kwargs`` is an escape hatch for backend-specific parameters
        (e.g. ``temperature``, ``no_speech_threshold``,
        ``condition_on_previous_text``). Unknown kwargs MUST be ignored
        rather than raising, so the same call site works across backends.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if the model is currently resident in memory/VRAM."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name (e.g. ``'mlx-whisper'``, ``'faster-whisper'``)."""
