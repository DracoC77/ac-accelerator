"""faster-whisper (CTranslate2) inference backend for CUDA/Windows.

Companion to :mod:`accelerator.backends.mlx_backend`. Mirrors
the same load/unload/transcribe contract from :class:`InferenceBackend`,
targeting NVIDIA GPUs (RTX 5090) on Windows via the faster-whisper / CTranslate2
runtime. Heavy imports (``faster_whisper``, ``torch``) are deferred to
``load()`` / ``unload()`` so this module is importable on Macs / Linux boxes
without those packages installed (this is what lets the test suite mock it).
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import sys
import threading
import time
from typing import Any, Optional

from .base import InferenceBackend, TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    """Current process RSS in MB. Mirrors ``mlx_backend._rss_mb``."""
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_bytes / (1024 * 1024)
    return rss_bytes / 1024  # Linux/Windows-via-WSL reports kB


class FasterWhisperBackend(InferenceBackend):
    """faster-whisper inference backend.

    Targets CUDA on Windows but accepts arbitrary ``device`` / ``compute_type``
    values so callers can drop down to CPU for smoke-testing. The model
    handle is held under a lock and torn down explicitly on ``unload()`` so
    VRAM is returned to the gaming VM between idle windows.
    """

    backend_name_str = "faster-whisper"

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        # Defaults mirror the backend design doc — large-v3-turbo on CUDA
        # with int8_float16 quant for the 5090.
        self._model_name = model or os.getenv(
            "WHISPER_MODEL", "large-v3-turbo"
        )
        self._device = device or os.getenv("FASTER_WHISPER_DEVICE", "cuda")
        self._compute_type = compute_type or os.getenv(
            "FASTER_WHISPER_COMPUTE_TYPE", "int8_float16"
        )
        self._lock = threading.Lock()
        self._model: Any = None
        self._loaded: bool = False
        self._last_use: float = 0.0

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
        return self._last_use

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> Any:
        """Load the WhisperModel into VRAM. Idempotent."""
        with self._lock:
            if self._model is None:
                pre_mb = _rss_mb()
                logger.debug(
                    "Memory before faster-whisper load: %.1f MB RSS", pre_mb
                )
                logger.info(
                    "Loading faster-whisper model: %s (device=%s, compute_type=%s)",
                    self._model_name, self._device, self._compute_type,
                )
                t0 = time.monotonic()
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                self._loaded = True
                elapsed = time.monotonic() - t0
                post_mb = _rss_mb()
                logger.info(
                    "faster-whisper ready (model=%s, device=%s, "
                    "load_time=%.2fs, memory_mb=%.1f, delta_mb=%.1f)",
                    self._model_name, self._device, elapsed,
                    post_mb, post_mb - pre_mb,
                )
            self._last_use = time.monotonic()
            return self._model

    def unload(self) -> None:
        """Release the model and VRAM. Safe to call repeatedly."""
        with self._lock:
            if self._model is None:
                return
            pre_mb = _rss_mb()
            logger.info(
                "Unloading faster-whisper (idle timeout, memory_mb=%.1f)", pre_mb
            )
            del self._model
            self._model = None
            self._loaded = False
            gc.collect()
            # Best-effort VRAM release. Torch isn't a hard dep of this module
            # (the test suite mocks faster_whisper without torch present), so
            # any failure here is swallowed.
            try:
                import torch  # type: ignore[import-not-found]
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:  # pragma: no cover - depends on host
                pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        *,
        temperature: Any = None,
        no_speech_threshold: Optional[float] = None,
        condition_on_previous_text: Optional[bool] = None,
        **_unused: Any,
    ) -> TranscriptResult:
        """Transcribe ``audio_path`` and return a normalized
        :class:`TranscriptResult`.

        The faster-whisper API returns ``(segments, info)`` where
        ``segments`` is a lazy generator. We materialize it fully here so the
        caller receives a list (the upstream pipeline expects to iterate
        segments more than once).
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("FasterWhisperBackend: model not loaded")

            kwargs: dict[str, Any] = {
                "language": language,
                "initial_prompt": initial_prompt,
                "word_timestamps": False,
                # VAD is handled upstream by the pipeline (see design §3.2).
                "vad_filter": False,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if no_speech_threshold is not None:
                kwargs["no_speech_threshold"] = no_speech_threshold
            if condition_on_previous_text is not None:
                kwargs["condition_on_previous_text"] = condition_on_previous_text

            segments_gen, info = self._model.transcribe(audio_path, **kwargs)

            # Materialize the generator fully — must NOT return a lazy iterator.
            segments: list[TranscriptSegment] = []
            for seg in segments_gen:
                # WI-BUG-20: faster-whisper Segment exposes .temperature; use
                # getattr with a 0.0 fallback so older releases or test fakes
                # without the attribute don't break.
                seg_temperature = getattr(seg, "temperature", 0.0)
                try:
                    seg_temperature = float(seg_temperature) if seg_temperature is not None else 0.0
                except (TypeError, ValueError):
                    seg_temperature = 0.0
                segments.append(
                    TranscriptSegment(
                        start=round(float(seg.start), 3),
                        end=round(float(seg.end), 3),
                        text=seg.text,
                        avg_logprob=float(seg.avg_logprob),
                        no_speech_prob=float(seg.no_speech_prob),
                        temperature=seg_temperature,
                    )
                )

            self._last_use = time.monotonic()

            # Stitch the full text + duration for response-payload parity
            # with the mlx backend.
            full_text = "".join(s.text for s in segments)
            duration = segments[-1].end if segments else 0.0
            # faster-whisper's `info` exposes a duration field on real runs;
            # prefer it when present.
            info_duration = getattr(info, "duration", None)
            if isinstance(info_duration, (int, float)) and info_duration > 0:
                duration = float(info_duration)

            return TranscriptResult(
                segments=segments,
                language=getattr(info, "language", "en") or "en",
                language_probability=float(
                    getattr(info, "language_probability", 0.0) or 0.0
                ),
                text=full_text,
                duration=duration,
            )
