"""Inference backend implementations + selection logic.

Use :func:`create_backend` to instantiate the configured backend based on the
``INFERENCE_BACKEND`` environment variable, falling back to host autodetect
(Darwin/arm64 → mlx; CUDA-capable Linux/Windows → faster-whisper).
"""

from __future__ import annotations

import os
import platform

from .base import InferenceBackend, TranscriptResult, TranscriptSegment

__all__ = [
    "InferenceBackend",
    "TranscriptResult",
    "TranscriptSegment",
    "create_backend",
]


def _cuda_available() -> bool:
    """Best-effort CUDA availability check.

    ``torch`` is imported lazily so non-CUDA hosts (mac build machines, CI)
    can still import this module without the torch wheel being present.
    Any import failure is treated as "no CUDA" rather than bubbling up.
    """
    try:
        import torch  # type: ignore[import-not-found]
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _auto_detect_backend() -> str:
    """Return the auto-detected backend name based on the host.

    Order:
      1. Darwin/arm64 (Apple Silicon) → ``mlx``.
      2. CUDA available (Windows or Linux + NVIDIA) → ``faster-whisper``.
      3. Otherwise raise RuntimeError.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    if _cuda_available():
        return "faster-whisper"
    raise RuntimeError(
        "No backend configured — set INFERENCE_BACKEND "
        "(supported: 'mlx' on Darwin/arm64, 'faster-whisper' on CUDA hosts)"
    )


def create_backend(model: str | None = None) -> InferenceBackend:
    """Instantiate the configured inference backend.

    Order of selection:
      1. ``INFERENCE_BACKEND`` env var (``mlx`` or ``faster-whisper``).
      2. Auto-detect (see :func:`_auto_detect_backend`).
    """
    name = (os.getenv("INFERENCE_BACKEND") or "").strip().lower()
    if not name:
        name = _auto_detect_backend()

    if name == "mlx":
        from .mlx_backend import MlxWhisperBackend
        return MlxWhisperBackend(model=model)

    # Accept both spellings ("faster-whisper" canonical, "faster_whisper"
    # underscore variant) — they show up in env vars / docs interchangeably.
    if name in ("faster-whisper", "faster_whisper"):
        from .faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend(
            model=model,
            device=os.getenv("FASTER_WHISPER_DEVICE"),
            compute_type=os.getenv("FASTER_WHISPER_COMPUTE_TYPE"),
        )

    raise RuntimeError(
        f"Unknown INFERENCE_BACKEND={name!r}. "
        "Supported: 'mlx', 'faster-whisper'."
    )
