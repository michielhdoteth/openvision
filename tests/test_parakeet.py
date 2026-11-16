"""Tests for openvision.providers.parakeet - Parakeet STT provider."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from openvision.providers.parakeet import (
    ParakeetProvider,
    Transcript,
    TranscriptSegment,
    _resolve_device,
    _split_into_segments,
    _read_wav_float32,
)


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    @patch("openvision.core.gpu.get_vram_info", return_value={"free_gb": 2.0})
    def test_auto_with_gpu(self, mock_vram):
        assert _resolve_device("auto") == "cuda"

    @patch("openvision.core.gpu.get_vram_info", return_value={"free_gb": 0.5})
    def test_auto_without_gpu(self, mock_vram):
        assert _resolve_device("auto") == "cpu"

    @patch("openvision.core.gpu.get_vram_info", side_effect=Exception("no gpu"))
    def test_auto_vram_query_fails(self, mock_vram):
        assert _resolve_device("auto") == "cpu"


class TestSplitIntoSegments:
    def test_empty_text(self):
        assert _split_into_segments("", 10.0) == []

    def test_single_sentence(self):
        segs = _split_into_segments("Hello world.", 5.0)
        assert len(segs) == 1
        assert segs[0].text == "Hello world."
        assert segs[0].start == 0.0
        assert segs[0].end == 5.0

    def test_multiple_sentences(self):
        segs = _split_into_segments("First sentence. Second sentence. Third.", 9.0)
        assert len(segs) == 3
        assert segs[0].text == "First sentence."
        assert segs[1].text == "Second sentence."
        assert segs[2].text == "Third."

    def test_even_distribution(self):
        segs = _split_into_segments("A. B. C.", 6.0)
        for seg in segs:
            assert seg.end - seg.start == pytest.approx(2.0)

    def test_confidence_always_one(self):
        segs = _split_into_segments("Hello.", 1.0)
        assert segs[0].confidence == 1.0

    def test_whitespace_only(self):
        segs = _split_into_segments("   ", 5.0)
        # Should fall back to single segment
        assert len(segs) == 1


class TestTranscript:
    def test_text_property(self):
        t = Transcript(segments=[
            TranscriptSegment(0, 1, "Hello"),
            TranscriptSegment(1, 2, "world"),
        ])
        assert t.text == "Hello world"

    def test_segment_count(self):
        t = Transcript(segments=[
            TranscriptSegment(0, 1, "a"),
            TranscriptSegment(1, 2, "b"),
            TranscriptSegment(2, 3, "c"),
        ])
        assert t.segment_count == 3

    def test_empty_segments(self):
        t = Transcript()
        assert t.text == ""
        assert t.segment_count == 0


class TestParakeetProvider:
    def test_init_default_config(self):
        p = ParakeetProvider()
        assert p.config == {}
        assert p.device in ("cpu", "cuda")
        assert not p.is_loaded

    def test_init_custom_config(self):
        p = ParakeetProvider({"device": "cpu", "model_dir": "/tmp/model"})
        assert p.device == "cpu"
        assert p.model_dir == Path("/tmp/model")

    def test_unload(self):
        p = ParakeetProvider()
        p._recognizer = MagicMock()
        p.unload()
        assert not p.is_loaded
        assert p._recognizer is None

    def test_transcribe_audio_file_not_found(self):
        p = ParakeetProvider()
        with pytest.raises(FileNotFoundError):
            p.transcribe_audio("/nonexistent/file.wav")

    def test_transcript_dataclass_fields(self):
        seg = TranscriptSegment(start=0.0, end=1.0, text="hi", confidence=0.9, speaker="SPEAKER_00")
        assert seg.start == 0.0
        assert seg.speaker == "SPEAKER_00"

    def test_transcript_default_speaker(self):
        seg = TranscriptSegment(start=0.0, end=1.0, text="hi")
        assert seg.speaker == ""
