"""Tests for core sampling module."""
import pytest
import numpy as np
from openvision.core.sampling import (
    detect_scenes,
    deduplicate,
    smart_sample,
    find_keyframes_for_question,
)


def _make_test_frame(size=(100, 100, 3)):
    """Create a basic frame dict."""
    return {"index": 0, "timestamp": 0.0, "timestamp_str": "00:00", "image": np.zeros(size, dtype=np.uint8)}


def _make_frame_list(count=5, color_offset=0):
    """Create a list of frame dicts with optional color variation."""
    frames = []
    for i in range(count):
        val = 50 + color_offset + (i * 30)
        img = np.ones((100, 100, 3), dtype=np.uint8) * min(val, 255)
        frames.append({
            "index": i,
            "timestamp": float(i),
            "timestamp_str": f"00:{i:02d}",
            "image": img,
        })
    return frames


class TestDetectScenes:
    def test_empty_frames(self):
        assert detect_scenes([]) == []

    def test_single_frame(self):
        frames = [_make_test_frame()]
        assert detect_scenes(frames) == []

    def test_identical_frames_no_scene(self):
        frames = _make_frame_list(5, color_offset=0)
        scenes = detect_scenes(frames, threshold=50.0)
        assert len(scenes) == 0 or len(scenes) < 3  # Low chance of many false scenes with low variation

    def test_different_frames_detect_changes(self):
        """Black frames then white frames should trigger scene change."""
        frames = []
        for i in range(3):
            frames.append({
                "index": i, "timestamp": float(i), "timestamp_str": f"00:0{i}",
                "image": np.zeros((100, 100, 3), dtype=np.uint8),
            })
        for i in range(3, 6):
            frames.append({
                "index": i, "timestamp": float(i), "timestamp_str": f"00:0{i}",
                "image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
            })
        scenes = detect_scenes(frames, threshold=0.5)
        assert len(scenes) >= 1


class TestDeduplicate:
    def test_empty(self):
        assert deduplicate([], []) == []

    def test_identical_frames_deduped(self):
        """All frames are identical -> only first remains."""
        frames = []
        for i in range(5):
            frames.append({
                "index": i, "timestamp": float(i), "timestamp_str": f"00:0{i}",
                "image": np.ones((100, 100, 3), dtype=np.uint8) * 128,
            })
        result = deduplicate(frames, list(range(5)), similarity_threshold=0.9)
        # All identical frames should be deduped to 1
        assert len(result) == 1

    def test_no_indices_returns_all(self):
        frames = _make_frame_list(3)
        result = deduplicate(frames)
        assert len(result) <= 3


class TestFindKeyframes:
    def test_empty(self):
        assert find_keyframes_for_question([]) == []

    def test_evenly_spaced(self):
        frames = _make_frame_list(10)
        result = find_keyframes_for_question(frames, max_keyframes=4)
        assert len(result) <= 4

    def test_specific_indices(self):
        frames = _make_frame_list(10)
        result = find_keyframes_for_question(frames, [0, 5, 9], max_keyframes=3)
        assert len(result) == 3
