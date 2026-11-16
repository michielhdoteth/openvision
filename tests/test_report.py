"""Tests for openvision.core.report - keep/drop visualization HTML report."""
import pytest
import numpy as np
from pathlib import Path
from openvision.core.report import generate_keep_drop_report


class TestGenerateKeepDropReport:
    def test_basic_report(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        result = generate_keep_drop_report(
            sample_frames, kept_indices=[0, 2, 4], output_path=out
        )
        assert Path(result).exists()
        content = Path(result).read_text(encoding="utf-8")
        assert "Frame Selection Report" in content

    def test_all_kept(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        result = generate_keep_drop_report(
            sample_frames, kept_indices=list(range(len(sample_frames))), output_path=out
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "100%" in content or "100 %" in content

    def test_all_dropped(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        result = generate_keep_drop_report(
            sample_frames, kept_indices=[], output_path=out
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "0%" in content or "0 %" in content

    def test_dedup_stats(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        stats = {"total_extracted": 100, "used_for_analysis": 30}
        result = generate_keep_drop_report(
            sample_frames, kept_indices=[0, 1], dedup_stats=stats, output_path=out
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "total_extracted" in content

    def test_custom_title(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        result = generate_keep_drop_report(
            sample_frames, kept_indices=[0], title="My Report", output_path=out
        )
        content = Path(result).read_text(encoding="utf-8")
        assert "My Report" in content

    def test_frames_without_image(self, tmp_path):
        frames = [{"timestamp": i * 1.0, "image": None} for i in range(5)]
        out = str(tmp_path / "report.html")
        result = generate_keep_drop_report(frames, kept_indices=[0, 1], output_path=out)
        assert Path(result).exists()

    def test_default_output_path(self, sample_frames):
        result = generate_keep_drop_report(sample_frames, kept_indices=[0])
        assert Path(result).exists()
        Path(result).unlink()

    def test_creates_parent_dirs(self, sample_frames, tmp_path):
        out = str(tmp_path / "subdir" / "report.html")
        result = generate_keep_drop_report(sample_frames, kept_indices=[0], output_path=out)
        assert Path(result).exists()

    def test_html_structure(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        generate_keep_drop_report(sample_frames, kept_indices=[0, 2], output_path=out)
        content = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "timeline" in content
        assert "thumbs" in content

    def test_kept_indices_out_of_range(self, sample_frames, tmp_path):
        out = str(tmp_path / "report.html")
        result = generate_keep_drop_report(
            sample_frames, kept_indices=[0, 99], output_path=out
        )
        assert Path(result).exists()
