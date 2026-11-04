"""Tests for audio extraction."""
import os
import pytest
from pathlib import Path
from core.audio import (
    has_audio_stream,
    extract_audio,
    extract_audio_segment,
    get_audio_duration,
    cleanup_wav,
)


def _find_test_video() -> Path:
    """Find any available test video file."""
    videos = list(Path("test_data").glob("*.mp4")) if Path("test_data").exists() else []
    if not videos:
        downloads = Path(r"C:\Users\michi\Downloads\Videos")
        if downloads.exists():
            videos = list(downloads.glob("*.mp4"))
    if not videos:
        pytest.skip("No test video available")
    return videos[0]


class TestHasAudioStream:
    def test_nonexistent_file(self):
        assert not has_audio_stream("nonexistent.mp4")

    def test_real_video(self):
        """Test with a real video that has audio."""
        video = _find_test_video()
        result = has_audio_stream(str(video))
        assert isinstance(result, bool)


class TestGetAudioDuration:
    def test_nonexistent(self):
        assert get_audio_duration("nonexistent.mp4") == 0.0

    def test_real_video(self):
        """Get duration from real video."""
        video = _find_test_video()
        duration = get_audio_duration(str(video))
        assert duration > 0


class TestExtractAudio:
    def test_nonexistent_video(self):
        with pytest.raises(FileNotFoundError):
            extract_audio("nonexistent.mp4")

    def test_real_video(self, tmp_path):
        """Extract audio from real video."""
        video = _find_test_video()
        output = str(tmp_path / "audio.wav")
        result = extract_audio(str(video), output)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

        with open(result, "rb") as f:
            header = f.read(4)
            assert header == b"RIFF"

    def test_default_output_path(self, tmp_path):
        """Extract audio to default output path (same dir as video)."""
        video = _find_test_video()
        # Use a copy in tmp_path so default path logic works
        from shutil import copy2
        test_video = str(tmp_path / "test_video.mp4")
        copy2(str(video), test_video)

        result = extract_audio(test_video)
        expected = str(Path(test_video).with_suffix(".wav"))
        assert result == expected
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0


class TestExtractAudioSegment:
    def test_real_video_segment(self, tmp_path):
        """Extract a segment of audio from real video."""
        video = _find_test_video()
        duration = get_audio_duration(str(video))
        output = str(tmp_path / "segment.wav")

        start = 0.0
        end = min(3.0, duration)

        result = extract_audio_segment(str(video), start, end, output)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

        with open(result, "rb") as f:
            header = f.read(4)
            assert header == b"RIFF"

    def test_nonexistent_video_segment(self, tmp_path):
        """Extract segment from nonexistent file."""
        output = str(tmp_path / "no_source.wav")

        with pytest.raises(RuntimeError):
            extract_audio_segment("nonexistent_video.mp4", 0, 1, output)


class TestCleanupWav:
    def test_cleanup_existing_file(self, tmp_path):
        """Clean up a temporary WAV file."""
        wav_path = str(tmp_path / "temp.wav")
        Path(wav_path).write_bytes(b"RIFF data")
        cleanup_wav(wav_path)
        assert not Path(wav_path).exists()

    def test_cleanup_nonexistent_file(self):
        """Clean up a non-existent file (should not error)."""
        cleanup_wav(r"C:\nonexistent\temp.wav")
