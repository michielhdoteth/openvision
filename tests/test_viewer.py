"""Tests for openvision.core.viewer - self-contained HTML viewer."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from openvision.core.viewer import generate_viewer


class TestGenerateViewer:
    def test_basic_viewer(self, sample_frames, tmp_path):
        out = str(tmp_path / "viewer.html")
        result = generate_viewer(sample_frames, output_path=out)
        assert Path(result).exists()
        content = Path(result).read_text(encoding="utf-8")
        assert "OpenVision Viewer" in content

    def test_with_transcript(self, sample_frames, sample_transcript_segments, tmp_path):
        out = str(tmp_path / "viewer.html")
        result = generate_viewer(
            sample_frames, transcript_segments=sample_transcript_segments, output_path=out
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "Transcript" in content
        assert "Hello world" in content

    def test_with_summary(self, sample_frames, tmp_path):
        out = str(tmp_path / "viewer.html")
        result = generate_viewer(
            sample_frames, summary="A cat sits on a mat", output_path=out
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "A cat sits on a mat" in content

    def test_custom_title(self, sample_frames, tmp_path):
        out = str(tmp_path / "viewer.html")
        result = generate_viewer(sample_frames, title="My Video", output_path=out)
        content = Path(result).read_text(encoding="utf-8")
        assert "My Video" in content

    def test_frame_count_displayed(self, sample_frames, tmp_path):
        out = str(tmp_path / "viewer.html")
        generate_viewer(sample_frames, output_path=out)
        content = Path(out).read_text(encoding="utf-8")
        assert f"{len(sample_frames)} keyframes" in content

    def test_frames_without_image(self, tmp_path):
        frames = [{"timestamp": i * 1.0, "image": None} for i in range(3)]
        out = str(tmp_path / "viewer.html")
        result = generate_viewer(frames, output_path=out)
        assert Path(result).exists()

    def test_default_output_path(self, sample_frames):
        result = generate_viewer(sample_frames)
        assert Path(result).exists()
        Path(result).unlink()

    def test_creates_parent_dirs(self, sample_frames, tmp_path):
        out = str(tmp_path / "subdir" / "viewer.html")
        result = generate_viewer(sample_frames, output_path=out)
        assert Path(result).exists()

    def test_html_structure(self, sample_frames, tmp_path):
        out = str(tmp_path / "viewer.html")
        generate_viewer(sample_frames, output_path=out)
        content = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "frame-card" in content
        assert "data:image/jpeg;base64" in content

    def test_transcript_timestamps_formatted(self, sample_frames, sample_transcript_segments, tmp_path):
        out = str(tmp_path / "viewer.html")
        generate_viewer(sample_frames, transcript_segments=sample_transcript_segments, output_path=out)
        content = Path(out).read_text(encoding="utf-8")
        # Should have formatted timestamps
        assert "[00:00]" in content or "[0:00]" in content

    def test_empty_frames(self, tmp_path):
        out = str(tmp_path / "viewer.html")
        result = generate_viewer([], output_path=out)
        assert Path(result).exists()
