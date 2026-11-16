"""Shared test fixtures for openvision test suite."""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample BGR image (numpy array)."""
    img = np.zeros((200, 320, 3), dtype=np.uint8)
    img[50:150, 80:240] = (255, 128, 64)  # colored rectangle
    return img


@pytest.fixture
def sample_frames(sample_image):
    """Create a list of frame dicts as returned by smart_sample."""
    frames = []
    for i in range(6):
        img = sample_image.copy()
        # Vary the color per frame
        img[:, :, i % 3] = 100 + i * 20
        frames.append({
            "image": img,
            "timestamp": i * 2.0,
            "timestamp_str": f"{i * 2:.1f}s",
            "reason": "sample",
        })
    return frames


@pytest.fixture
def tmp_run_cache(tmp_path):
    """Create a fresh RunCache in a temp directory."""
    from openvision.storage.cache import RunCache
    return RunCache(str(tmp_path / "runs"))


@pytest.fixture
def sample_transcript_segments():
    """Create sample transcript segments (dataclass-like objects)."""
    from openvision.providers.parakeet import TranscriptSegment
    return [
        TranscriptSegment(start=0.0, end=3.0, text="Hello world.", confidence=1.0),
        TranscriptSegment(start=3.0, end=6.0, text="This is a test.", confidence=1.0),
        TranscriptSegment(start=6.0, end=9.0, text="Goodbye.", confidence=1.0),
    ]


@pytest.fixture
def sample_video_meta():
    """Sample video metadata dict as returned by probe()."""
    return {
        "path": "/tmp/test.mp4",
        "duration": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "size_bytes": 1024000,
        "bitrate": 800000,
    }
