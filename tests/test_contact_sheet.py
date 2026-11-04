"""Tests for core.contact_sheet - grid contact sheet generation."""
import pytest
import numpy as np
from pathlib import Path
from core.contact_sheet import create_contact_sheet


class TestCreateContactSheet:
    def test_basic_3x3(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(sample_frames, grid_size=(3, 3), output_path=out)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_2x2_grid(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(sample_frames, grid_size=(2, 2), output_path=out)
        assert Path(result).exists()

    def test_fewer_frames_than_cells(self, sample_image, tmp_path):
        frames = [{"image": sample_image, "timestamp": 0.0}]
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(frames, grid_size=(3, 3), output_path=out)
        assert Path(result).exists()

    def test_more_frames_than_cells(self, sample_image, tmp_path):
        frames = [{"image": sample_image, "timestamp": i * 1.0} for i in range(20)]
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(frames, grid_size=(3, 3), output_path=out)
        assert Path(result).exists()

    def test_no_timestamps(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(sample_frames, show_timestamps=False, output_path=out)
        assert Path(result).exists()

    def test_custom_cell_size(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(sample_frames, cell_width=640, cell_height=360, output_path=out)
        assert Path(result).exists()

    def test_default_output_path(self, sample_frames):
        result = create_contact_sheet(sample_frames)
        assert Path(result).exists()
        Path(result).unlink()

    def test_output_is_jpeg(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(sample_frames, output_path=out)
        with open(result, "rb") as f:
            magic = f.read(2)
        assert magic == b'\xff\xd8'

    def test_timestamps_included(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(sample_frames, show_timestamps=True, output_path=out)
        assert Path(result).exists()

    def test_creates_parent_dirs(self, sample_frames, tmp_path):
        out = str(tmp_path / "subdir" / "deep" / "sheet.jpg")
        result = create_contact_sheet(sample_frames, output_path=out)
        assert Path(result).exists()

    def test_grid_dimensions(self, sample_frames, tmp_path):
        out = str(tmp_path / "sheet.jpg")
        create_contact_sheet(sample_frames, grid_size=(2, 3), cell_width=100, cell_height=80, output_path=out)
        import cv2
        img = cv2.imread(out)
        assert img.shape[:2] == (240, 200)  # 3 rows * 80, 2 cols * 100

    def test_empty_frame_image_skipped(self, tmp_path):
        frames = [{"image": None, "timestamp": 0.0}]
        out = str(tmp_path / "sheet.jpg")
        result = create_contact_sheet(frames, output_path=out)
        assert Path(result).exists()
