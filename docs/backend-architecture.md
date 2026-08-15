# Inference Backend Architecture

This document describes the pluggable inference backend system, including the
MLX backend and the faster-whisper / Windows backend.

---

## Overview

The accelerator server (`server.py`) is backend-agnostic. It calls a single
`InferenceBackend` instance obtained at startup via `create_backend()`.
Swapping the backend (mlx-whisper ↔ faster-whisper) requires only an
environment variable change — no server code changes.

```
server.py
  └── create_backend()        ← accelerator/backends/__init__.py
        ├── MlxWhisperBackend  ← accelerator/backends/mlx_backend.py
        └── FasterWhisperBackend ← accelerator/backends/faster_whisper_backend.py
```

---

## `InferenceBackend` ABC

Defined in `accelerator/backends/base.py`. Every backend must implement:

```python
class InferenceBackend(ABC):

    @abstractmethod
    def load(self) -> None:
        """Load model into memory/VRAM. Must be idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model and release VRAM. Must be safe to call multiple times."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptResult:
        """Transcribe audio. Unknown kwargs MUST be ignored, not raised."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if the model is currently resident in memory/VRAM."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier, e.g. 'mlx-whisper' or 'faster-whisper'."""
```

### Contract requirements

- **Lazy construction** — backends must be safe to instantiate before `load()`
  is called. The idle watchdog calls `load()` on demand.
- **Idempotency** — `load()` called twice is the same as calling it once.
  `unload()` called on an already-unloaded backend is a no-op.
- **VRAM release** — `unload()` must release GPU/MPS memory. This is a hard
  requirement for the gaming-VM use case where VRAM must be freed on demand.
- **Unknown kwargs ignored** — `transcribe(**kwargs)` must silently discard
  kwargs it doesn't understand so the same call site works across backends.

---

## Result dataclasses

Both are defined in `accelerator/backends/base.py`.

### `TranscriptSegment`

| Field | Type | Notes |
|---|---|---|
| `start` | `float` | Segment start time (seconds) |
| `end` | `float` | Segment end time (seconds) |
| `text` | `str` | Transcribed text for this segment |
| `avg_logprob` | `float` | Average log-probability |
| `no_speech_prob` | `float` | Probability that segment is silence/noise |
| `tokens` | `list[int]` | Token IDs (optional; populated by mlx-whisper) |
| `compression_ratio` | `float` | Optional; populated by mlx-whisper |

### `TranscriptResult`

| Field | Type | Notes |
|---|---|---|
| `segments` | `list[TranscriptSegment]` | All segments |
| `language` | `str` | Detected language code (e.g. `"en"`) |
| `language_probability` | `float` | Confidence (0–1); `0.0` if unavailable |
| `text` | `str` | Full concatenated transcript |
| `duration` | `float` | Audio duration in seconds |

---

## Available backends

### `MlxWhisperBackend` — macOS Apple Silicon

- **File:** `accelerator/backends/mlx_backend.py`
- **Engine:** [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
  (Apple MLX framework, runs on Metal/ANE)
- **Performance:** ~28–30× realtime on M2/M3
- **Default model:** `mlx-community/whisper-large-v3-turbo`
- **Platform requirement:** `platform.system() == "Darwin"` and
  `platform.machine() == "arm64"`

### `FasterWhisperBackend` — Windows / Linux NVIDIA

- **File:** `accelerator/backends/faster_whisper_backend.py`
- **Engine:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (CTranslate2, runs on CUDA)
- **Performance:** ~60–90× realtime on RTX 5090
- **Default model:** `large-v3-turbo`
- **Platform requirement:** CUDA-capable GPU with driver ≥ 580; CUDA 12.8
  wheels installed via `torch cu128` index

Additional env vars for this backend:

| Variable | Default | Description |
|---|---|---|
| `FASTER_WHISPER_DEVICE` | `cuda` | `cuda`, `cpu`, or `cuda:N` |
| `FASTER_WHISPER_COMPUTE_TYPE` | `float16` | `float16`, `int8_float16`, `int8` |

---

## Backend selection

Selection is implemented in `accelerator/backends/__init__.py` →
`create_backend()`.

```
INFERENCE_BACKEND env var set?
  ├─ "mlx"             → MlxWhisperBackend
  ├─ "faster-whisper"  → FasterWhisperBackend   (also accepts "faster_whisper")
  └─ (anything else)   → RuntimeError

INFERENCE_BACKEND not set → auto-detect:
  ├─ Darwin + arm64    → MlxWhisperBackend
  ├─ CUDA available    → FasterWhisperBackend
  └─ neither           → RuntimeError
```

CUDA availability is checked lazily (torch imported on-demand) so non-CUDA
hosts (e.g. Mac CI) can import the module without the CUDA torch wheel present.

---

## VRAM release

The server calls `backend.unload()` on:
- Graceful shutdown (SIGTERM / SIGINT)
- Idle timeout (configurable; the server unloads after N minutes of inactivity
  to reduce VRAM pressure)

On hard kill (`SIGKILL`, Task Manager "End Task"), Python's `unload()` does not
run, but the OS unconditionally reclaims all VRAM held by the terminated
process. VRAM is always freed within a few seconds of process exit regardless
of how the process exits.

---

## Adding a new backend

1. **Create** `accelerator/backends/your_backend.py` and implement
   `InferenceBackend`:

   ```python
   from accelerator.backends.base import InferenceBackend, TranscriptResult

   class MyBackend(InferenceBackend):
       def load(self): ...
       def unload(self): ...
       def transcribe(self, audio_path, language=None, initial_prompt=None, **kwargs) -> TranscriptResult: ...

       @property
       def is_loaded(self) -> bool: ...

       @property
       def backend_name(self) -> str:
           return "my-backend"
   ```

2. **Register** in `accelerator/backends/__init__.py` — add a branch in
   `create_backend()`:

   ```python
   if name == "my-backend":
       from .your_backend import MyBackend
       return MyBackend(model=model)
   ```

3. **Add dependencies** to `requirements.txt` with platform markers if
   platform-specific:

   ```
   my-inference-lib>=1.0; platform_system=="Linux"
   ```

4. **Document** the new backend here and in `README.md`.

---

## Related files

| File | Purpose |
|---|---|
| `accelerator/backends/base.py` | ABC + result dataclasses |
| `accelerator/backends/__init__.py` | `create_backend()` factory + auto-detect |
| `accelerator/backends/mlx_backend.py` | macOS / Apple Silicon backend |
| `accelerator/backends/faster_whisper_backend.py` | Windows / NVIDIA backend |
| `server.py` | FastAPI server — calls `create_backend()` at startup |
| `companion_client.py` | Shared HTTP polling + service control for tray/menubar |
| `tray_app.py` | Windows system tray (uses `companion_client`) |
| `menubar_app.py` | macOS menubar (standalone; future: migrate to `companion_client`) |
