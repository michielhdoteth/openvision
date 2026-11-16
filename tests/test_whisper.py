"""Tests for Parakeet provider."""
import pytest
from pathlib import Path
from openvision.providers.parakeet import ParakeetProvider, Transcript, TranscriptSegment


class TestTranscript:
    def test_empty_transcript(self):
        t = Transcript()
        assert t.text == ""
        assert t.segment_count == 0

    def test_with_segments(self):
        t = Transcript(
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="hello", confidence=0.9),
                TranscriptSegment(start=1.0, end=2.0, text="world", confidence=0.8),
            ],
            language="en",
            duration_seconds=2.0,
        )
        assert t.text == "hello world"
        assert t.segment_count == 2


class TestParakeetProvider:
    def test_init(self):
        provider = ParakeetProvider({"device": "cpu"})
        assert not provider.is_loaded

    def test_transcribe_no_audio(self, tmp_path):
        """transcribe() should return empty Transcript when video has no audio."""
        import subprocess

        video_path = tmp_path / "no_audio.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
             "-c:v", "libx264", "-an", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        if not video_path.exists():
            pytest.skip("Could not create test video")

        provider = ParakeetProvider({"device": "cpu"})
        result = provider.transcribe(str(video_path))
        assert isinstance(result, Transcript)
        assert result.segment_count == 0
        assert result.text == ""

    def test_transcribe_real_video(self, tmp_path):
        """Integration test: transcribe a short generated video (mocked model)."""
        import subprocess

        video_path = tmp_path / "real_video.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-shortest", "-c:v", "libx264", "-c:a", "aac",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        if not video_path.exists():
            pytest.skip("Could not create test video")

        from unittest.mock import MagicMock

        # Mock the recognizer to avoid large download / slow inference
        mock_recognizer = MagicMock()
        mock_stream = MagicMock()
        mock_stream.result.text = " hello "
        mock_recognizer.create_stream.return_value = mock_stream

        provider = ParakeetProvider({"device": "cpu"})
        provider._recognizer = mock_recognizer
        result = provider.transcribe(str(video_path))
        assert isinstance(result, Transcript)
        assert "hello" in result.text
        provider.unload()
