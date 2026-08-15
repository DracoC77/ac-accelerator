"""
tests/test_overlap_parity.py — Unit tests for overlap parity.

Verifies that _process_diarization() emits is_overlap / overlap_ratio for
every segment so the main repo can run crosstalk detection on the
remote-diarization path.

All pyannote/mlx imports are mocked — no GPU, no real models needed.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake pyannote Annotation
# ---------------------------------------------------------------------------


class _FakeSegment:
    """Minimal pyannote Segment-alike with .start and .end."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeTimeline:
    """Minimal pyannote Timeline-alike — iterable list of Segments."""

    def __init__(self, segments: list) -> None:
        self._segments = segments

    def __iter__(self):
        return iter(self._segments)

    def support(self) -> "_FakeTimeline":
        """WI-80b: production code calls annotation.get_overlap().support().
        For the test fake, get_overlap() already returns non-adjacent merged
        regions, so support() is a no-op — just return self."""
        return self


class _FakeAnnotation:
    """Minimal pyannote Annotation-alike.

    itertracks() yields (Segment, track_id, label) triples.
    get_overlap() returns a Timeline of overlapping regions.
    """

    def __init__(self, turns: list[tuple]) -> None:
        # turns: [(start, end, speaker), ...]
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._turns:
            yield _FakeSegment(start, end), None, speaker

    def get_overlap(self) -> _FakeTimeline:
        """Compute overlap regions naively (O(n²)) for test purposes."""
        segments = [_FakeSegment(s, e) for s, e, _ in self._turns]
        overlap_segs = []
        n = len(segments)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = segments[i], segments[j]
                inter_start = max(a.start, b.start)
                inter_end = min(a.end, b.end)
                if inter_end > inter_start:
                    overlap_segs.append(_FakeSegment(inter_start, inter_end))
        return _FakeTimeline(overlap_segs)


# ---------------------------------------------------------------------------
# Import _process_diarization with ML deps stubbed
# ---------------------------------------------------------------------------


def _import_server():
    """Import server module with heavy ML libs stubbed out."""
    import sys
    import importlib

    # Stub the ML deps so we can import without GPU/models
    stubs = {
        "mlx_whisper": MagicMock(),
        "torch": MagicMock(),
        "pyannote": MagicMock(),
        "pyannote.audio": MagicMock(),
        "uvicorn": MagicMock(),
    }
    for name, stub in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = stub

    # Re-import so module-level code sees the stubs
    if "server" in sys.modules:
        del sys.modules["server"]
    import server as _server
    return _server


# ---------------------------------------------------------------------------
# Tests — _process_diarization overlap fields
# ---------------------------------------------------------------------------


class TestProcessDiarizationOverlapFields:
    """Verify is_overlap / overlap_ratio are emitted for each segment."""

    def _run(self, annotation, job_id="test-job"):
        """Call _process_diarization() with a mock pipeline returning annotation."""
        import sys
        # Ensure server is importable
        srv = _import_server()

        # Build a minimal job dict
        import json, tempfile, os
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tf.write(b"RIFF" + b"\x00" * 100)
        tf.close()

        job = {
            "job_id": job_id,
            "file_path": tf.name,
            "params_json": json.dumps({}),
        }

        mock_pipeline = MagicMock(return_value=annotation)

        try:
            with patch.object(srv, "_load_pyannote", return_value=mock_pipeline):
                result = srv._process_diarization(job)
        finally:
            os.unlink(tf.name)

        return result

    def test_no_overlap_segments_get_false_zero(self):
        """Two non-overlapping speakers → is_overlap=False, overlap_ratio=0.0."""
        annotation = _FakeAnnotation([
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ])
        result = self._run(annotation)
        segs = result["segments"]
        assert len(segs) == 2
        for seg in segs:
            assert "is_overlap" in seg, f"is_overlap missing from segment: {seg}"
            assert "overlap_ratio" in seg, f"overlap_ratio missing from segment: {seg}"
            assert seg["is_overlap"] is False
            assert seg["overlap_ratio"] == 0.0

    def test_overlapping_turns_get_true_nonzero(self):
        """Two turns that overlap for 2s each → is_overlap=True, overlap_ratio > 0."""
        # SPEAKER_00: 0–5s, SPEAKER_01: 3–8s → overlap 3–5 = 2s each
        annotation = _FakeAnnotation([
            (0.0, 5.0, "SPEAKER_00"),
            (3.0, 8.0, "SPEAKER_01"),
        ])
        result = self._run(annotation)
        segs = result["segments"]
        assert len(segs) == 2

        seg0 = segs[0]  # SPEAKER_00: turn_duration=5, overlap=2 → ratio=0.4
        assert seg0["is_overlap"] is True
        assert seg0["overlap_ratio"] == pytest.approx(0.4, abs=0.001)

        seg1 = segs[1]  # SPEAKER_01: turn_duration=5, overlap=2 → ratio=0.4
        assert seg1["is_overlap"] is True
        assert seg1["overlap_ratio"] == pytest.approx(0.4, abs=0.001)

    def test_partial_overlap_ratio_correct(self):
        """Only part of the turn overlaps — ratio reflects actual fraction.

        WI-80b: is_overlap now follows the ``> DIARIZE_TURN_OVERLAP_THRESHOLD``
        rule (default 0.3, mirroring the main repo's local pyannote path)
        instead of the legacy ``> 0.0`` (any touch flags the turn).
        """
        # SPEAKER_00: 0–10s, SPEAKER_01: 8–12s → overlap 8–10 = 2s
        # SPEAKER_00 ratio = 2/10 = 0.2, SPEAKER_01 ratio = 2/4 = 0.5
        annotation = _FakeAnnotation([
            (0.0, 10.0, "SPEAKER_00"),
            (8.0, 12.0, "SPEAKER_01"),
        ])
        result = self._run(annotation)
        segs = result["segments"]
        assert len(segs) == 2

        # SPEAKER_00 ratio=0.2 ≤ 0.3 → NOT overlap (post WI-80b).
        seg0 = next(s for s in segs if s["speaker"] == "SPEAKER_00")
        assert seg0["is_overlap"] is False
        assert seg0["overlap_ratio"] == pytest.approx(0.2, abs=0.001)

        # SPEAKER_01 ratio=0.5 > 0.3 → overlap.
        seg1 = next(s for s in segs if s["speaker"] == "SPEAKER_01")
        assert seg1["is_overlap"] is True
        assert seg1["overlap_ratio"] == pytest.approx(0.5, abs=0.001)

    def test_zero_duration_turn_no_crash(self):
        """A zero-duration turn (degenerate case) gets overlap_ratio=0.0, no div/0."""
        annotation = _FakeAnnotation([
            (5.0, 5.0, "SPEAKER_00"),  # zero-length
        ])
        result = self._run(annotation)
        segs = result["segments"]
        assert len(segs) == 1
        assert segs[0]["overlap_ratio"] == 0.0
        assert segs[0]["is_overlap"] is False

    def test_result_has_required_fields(self):
        """Result dict has segments, num_speakers, duration, overlap_regions."""
        annotation = _FakeAnnotation([
            (0.0, 3.0, "SPEAKER_00"),
        ])
        result = self._run(annotation)
        assert "segments" in result
        assert "num_speakers" in result
        assert "duration" in result
        # WI-80b: overlap_regions is the merged overlap timeline serialised
        # for the main repo's per-segment intersection.
        assert "overlap_regions" in result
        assert isinstance(result["overlap_regions"], list)
        assert result["num_speakers"] == 1

    def test_overlap_regions_emitted_for_overlapping_speech(self):
        """WI-80b: overlap_regions reflects the merged get_overlap().support() timeline."""
        # SPEAKER_00: 0–5s, SPEAKER_01: 3–8s → overlap region [3.0, 5.0]
        annotation = _FakeAnnotation([
            (0.0, 5.0, "SPEAKER_00"),
            (3.0, 8.0, "SPEAKER_01"),
        ])
        result = self._run(annotation)
        regions = result["overlap_regions"]
        assert isinstance(regions, list)
        assert len(regions) == 1
        region = regions[0]
        assert set(region.keys()) == {"start", "end"}
        assert region["start"] == pytest.approx(3.0, abs=0.001)
        assert region["end"] == pytest.approx(5.0, abs=0.001)

    def test_overlap_regions_empty_when_no_overlap(self):
        """WI-80b: overlap_regions is an empty list when there is no overlap."""
        annotation = _FakeAnnotation([
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ])
        result = self._run(annotation)
        assert result["overlap_regions"] == []

    def test_segment_has_all_fields(self):
        """Each segment has speaker, start, end, is_overlap, overlap_ratio."""
        annotation = _FakeAnnotation([
            (0.0, 5.0, "SPEAKER_00"),
        ])
        result = self._run(annotation)
        seg = result["segments"][0]
        assert set(seg.keys()) >= {"speaker", "start", "end", "is_overlap", "overlap_ratio"}
