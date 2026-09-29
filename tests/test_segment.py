"""Tests for SAM provider and segment/track commands."""
import pytest
import numpy as np
from pathlib import Path
from typer.testing import CliRunner
from cli.app import app
from models import SegmentResult, SegmentObject, TrackResult, TrackPosition
from providers.sam import SamProvider, Sam3Provider, create_segmenter

runner = CliRunner()


class TestSegmentModels:
    def test_segment_result_creation(self):
        obj = SegmentObject(
            object_id="person_001",
            timestamps=["00:05", "00:10"],
            mask_paths=["/tmp/mask_1.png"],
            bbox=[100, 200, 300, 400],
        )
        result = SegmentResult(
            prompt="person",
            frames_processed=2,
            masks_dir="/tmp/masks",
            objects=[obj],
        )
        assert result.frames_processed == 2
        assert len(result.objects) == 1


class TestTrackModels:
    def test_track_result_creation(self):
        pos = TrackPosition(
            timestamp="00:05",
            time_seconds=5.0,
            bbox=[100, 200, 300, 400],
            confidence=0.85,
        )
        result = TrackResult(
            track_id="car_test",
            positions=[pos],
            summary="Car tracked across 1 frame.",
        )
        assert result.track_id == "car_test"
        assert len(result.positions) == 1


class TestSamProvider:
    def test_segment_masks_are_persisted_without_collisions(self, tmp_path, monkeypatch):
        provider = SamProvider({"model": "mobile-sam", "load_on_demand": True})
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        monkeypatch.setattr(provider, "segment", lambda *_: [
            {"mask": np.zeros((8, 8), dtype=np.uint8), "bbox": [0, 0, 2, 2], "score": 0.9},
            {"mask": np.ones((8, 8), dtype=np.uint8), "bbox": [2, 2, 4, 4], "score": 0.8},
        ])
        result = provider.segment_frames([(1.0, image)], "blue chart", tmp_path)
        paths = [Path(item["mask_path"]) for item in result["objects"]]
        assert len(paths) == 2
        assert paths[0] != paths[1]
        assert all(path.is_file() for path in paths)

    def test_init(self):
        config = {"model": "mobile-sam", "load_on_demand": True}
        provider = SamProvider(config)
        assert provider.model == "mobile-sam"
        assert not provider.is_loaded
        provider.unload()  # Cleanup

    def test_segment_no_model(self):
        """Without SAM installed, should fail gracefully."""
        config = {"model": "mobile-sam", "load_on_demand": True}
        provider = SamProvider(config)

        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        try:
            result = provider.segment(img, "object")
            # If it works, great. If not, should return empty list (not crash)
            assert isinstance(result, list)
        except (ImportError, RuntimeError):
            pass  # Expected if SAM deps not installed
        finally:
            provider.unload()


    def test_configured_text_prompt_backend_is_sam3(self):
        provider = create_segmenter({"provider": "sam3", "model": "sam3"})
        assert isinstance(provider, Sam3Provider)

    def test_sam3_segments_frames_and_persists_masks(self, tmp_path, monkeypatch):
        provider = Sam3Provider({})
        monkeypatch.setattr(provider, "segment", lambda *_: [
            {"mask": np.ones((5, 6), dtype=np.uint8), "bbox": [1, 2, 4, 5], "score": 0.9},
        ])
        image = np.zeros((5, 6, 3), dtype=np.uint8)
        result = provider.segment_frames([(3.0, image)], "person holding cup", tmp_path)
        assert result["frames_processed"] == 1
        assert result["objects"][0]["time_seconds"] == 3.0
        mask = Path(result["objects"][0]["mask_path"])
        assert mask.is_file()
        assert mask.parent.name == "masks"


class TestSegmentCli:
    def test_segment_nonexistent(self):
        result = runner.invoke(app, ["segment", "nonexistent.mp4", "person"])
        assert result.exit_code != 0

    def test_segment_empty_prompt(self):
        result = runner.invoke(app, ["segment", "test.mp4", ""])
        assert result.exit_code != 0


class TestTrackCli:
    def test_track_nonexistent(self):
        result = runner.invoke(app, ["track", "nonexistent.mp4", "car"])
        assert result.exit_code != 0

    def test_track_empty_prompt(self):
        result = runner.invoke(app, ["track", "test.mp4", ""])
        assert result.exit_code != 0
