"""mlx-whisper inference backend (Apple Silicon / Metal).

Extracted from ``server.py`` with zero behavior change.
Mirrors the exact load/unload semantics, logging strings, and per-segment
field set that ``_process_transcription`` produced previously.
"""

from __future__ import annotations

import logging
import os
import resource
import sys
import threading
import time
from typing import Any, Callable, Optional

from .base import InferenceBackend, TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    """Return current process RSS in MB (cross-platform).

    Mirrors the helper of the same name in server.py so log lines match
    byte-for-byte.
    """
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_bytes / (1024 * 1024)
    return rss_bytes / 1024  # Linux reports kB


class MlxWhisperBackend(InferenceBackend):
    """mlx-whisper backend. Lazy-loads the module on first ``load()``."""

    backend_name_str = "mlx-whisper"

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        module_loader: Optional[Callable[[], Any]] = None,
    ) -> None:
        # Default mirrors server.py's WHISPER_MODEL fallback.
        self._default_model = model or os.getenv(
            "WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
        )
        self._lock = threading.Lock()
        self._module: Any = None
        self._loaded: bool = False
        self._last_use: float = 0.0
        # Allow tests / server to swap how the mlx_whisper module is acquired.
        # The default importer mirrors the old _load_whisper() body exactly so
        # behavior is unchanged on the Mac.
        self._module_loader = module_loader or self._default_module_loader

    @staticmethod
    def _default_module_loader() -> Any:
        import mlx_whisper  # type: ignore[import-untyped]
        return mlx_whisper

    def set_module_loader(self, loader: Callable[[], Any]) -> None:
        """Override how the whisper module is acquired. Forces a re-load on
        next ``load()`` so the new loader takes effect."""
        with self._lock:
            self._module_loader = loader
            self._module = None
            self._loaded = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return self.backend_name_str

    @property
    def last_use(self) -> float:
        """monotonic timestamp of last transcribe() / load() call."""
        return self._last_use

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> Any:
        """Load mlx-whisper. Returns the underlying module for callers
        that want to invoke it directly (matches the previous
        ``_load_whisper()`` contract in server.py)."""
        with self._lock:
            if self._module is None:
                pre_mb = _rss_mb()
                logger.debug("Memory before mlx-whisper load: %.1f MB RSS", pre_mb)
                logger.info("Loading mlx-whisper model: %s", self._default_model)
                t0 = time.monotonic()
                self._module = self._module_loader()
                self._loaded = True
                elapsed = time.monotonic() - t0
                post_mb = _rss_mb()
                logger.info(
                    "mlx-whisper ready (model=%s, load_time=%.2fs, "
                    "memory_mb=%.1f, delta_mb=%.1f)",
                    self._default_model, elapsed, post_mb, post_mb - pre_mb,
                )
            self._last_use = time.monotonic()
            return self._module

    def unload(self) -> None:
        with self._lock:
            if self._module is not None:
                pre_mb = _rss_mb()
                logger.info(
                    "Unloading mlx-whisper (idle timeout, memory_mb=%.1f)", pre_mb
                )
                self._module = None
                self._loaded = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        *,
        model: Optional[str] = None,
        temperature: Any = None,
        no_speech_threshold: Optional[float] = None,
        condition_on_previous_text: Optional[bool] = None,
        **_unused: Any,
    ) -> TranscriptResult:
        """Run mlx-whisper on ``audio_path`` and normalize the output.

        Field-level parity with the previous inline implementation in
        ``server.py::_process_transcription`` is preserved: rounding,
        defaults for missing fields, and language fallback to ``"en"``.
        """
        whisper = self.load()

        effective_model = model or self._default_model
        kwargs: dict[str, Any] = {"path_or_hf_repo": effective_model}
        if language:
            kwargs["language"] = language
        if initial_prompt is not None:
            kwargs["initial_prompt"] = initial_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        if no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = no_speech_threshold
        if condition_on_previous_text is not None:
            kwargs["condition_on_previous_text"] = condition_on_previous_text

        raw = whisper.transcribe(audio_path, **kwargs)
        self._last_use = time.monotonic()

        segments: list[TranscriptSegment] = []
        for seg in raw.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start=round(seg.get("start", 0.0), 3),
                    end=round(seg.get("end", 0.0), 3),
                    text=seg.get("text", ""),
                    avg_logprob=seg.get("avg_logprob", 0.0),
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                    tokens=seg.get("tokens", []),
                    compression_ratio=seg.get("compression_ratio", 0.0),
                    # WI-BUG-20: surface per-segment temperature so the HTTP
                    # response can expose it alongside the other quality
                    # signals. `or 0.0` guards against a present-but-None value.
                    temperature=float(seg.get("temperature", 0.0) or 0.0),
                )
            )

        # Duration: prefer raw["duration"] if present, else last segment's end.
        duration = 0.0
        if segments:
            duration = segments[-1].end
        if raw.get("duration"):
            duration = raw["duration"]

        return TranscriptResult(
            segments=segments,
            language=raw.get("language", "en"),
            language_probability=0.0,  # mlx-whisper does not surface this
            text=raw.get("text", ""),
            duration=duration,
        )
