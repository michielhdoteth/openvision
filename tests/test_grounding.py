"""Tests for LocateAnything provider and grounding integration."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from openvision.providers.locate_anything import LocateAnythingProvider
from openvision.models import GroundingBox, GroundingFrame, GroundingResult, TokenUsage


class TestLocateAnythingProvider:
    def test_init_defaults(self):
        provider = LocateAnythingProvider({})
        assert provider.base_url == "http://localhost:8000"
        assert provider.model_name == "locate-anything-3b"
        assert provider.timeout == 30

    def test_init_custom(self):
        config = {
            "base_url": "http://gpu-server:9000",
            "model": "locate-anything-7b",
            "timeout": 60,
        }
        provider = LocateAnythingProvider(config)
        assert provider.base_url == "http://gpu-server:9000"
        assert provider.model_name == "locate-anything-7b"
        assert provider.timeout == 60

    def test_health_check_bad_url(self):
        provider = LocateAnythingProvider({"base_url": "http://localhost:19999", "timeout": 2})
        assert provider.check_health() is False

    def test_unload(self):
        provider = LocateAnythingProvider({})
        result = provider.unload()
        assert result is None  # unload returns None, just cleans up

    def test_locate_empty_frames(self):
        provider = LocateAnythingProvider({})
        result = provider.locate_frames([], "person")
        assert result["results"] == []
        assert result["frames_processed"] == 0

    @patch("openvision.providers.locate_anything.LocateAnythingProvider.client", new_callable=lambda: property(lambda self: self))
    def test_locate_single_frame(self, mock_client_prop):
        """Test locating objects in a single frame."""
        provider = LocateAnythingProvider({})

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "boxes": [
                {"label": "person", "score": 0.92, "bbox": [0.1, 0.2, 0.4, 0.6]},
                {"label": "cup", "score": 0.85, "bbox": [0.5, 0.3, 0.7, 0.5]},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch.object(type(provider), "client", new_callable=lambda: property(lambda self: mock_client)):
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            result = provider.locate_frames([(0.0, image)], "person holding cup")

            assert len(result["results"]) == 1
            r = result["results"][0]
            assert r["timestamp"] == 0.0
            assert len(r["boxes"]) == 2

            # Verify first box — provider normalizes bbox into individual keys
            box1 = r["boxes"][0]
            assert box1["label"] == "person"
            assert box1["score"] == 0.92
            assert box1["x1"] == 0.1
            assert box1["y1"] == 0.2
            assert box1["x2"] == 0.4
            assert box1["y2"] == 0.6

            # Verify second box
            box2 = r["boxes"][1]
            assert box2["label"] == "cup"
            assert box2["score"] == 0.85

            # Verify frames_processed
            assert result["frames_processed"] == 1

    def test_locate_multiple_frames(self):
        """Test locating objects across multiple frames."""
        provider = LocateAnythingProvider({})

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "boxes": [
                {"label": "person", "score": 0.95, "bbox": [0.1, 0.2, 0.4, 0.6]},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch.object(type(provider), "client", new_callable=lambda: property(lambda self: mock_client)):
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            frames = [(0.0, image), (1.0, image), (2.0, image)]
            result = provider.locate_frames(frames, "person")

            assert len(result["results"]) == 3
            assert result["frames_processed"] == 3
            for r in result["results"]:
                assert len(r["boxes"]) == 1
                assert r["boxes"][0]["label"] == "person"

    def test_locate_empty_response(self):
        """Test handling of empty grounding response."""
        provider = LocateAnythingProvider({})

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"boxes": []}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch.object(type(provider), "client", new_callable=lambda: property(lambda self: mock_client)):
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            result = provider.locate_frames([(0.0, image)], "car")

            assert len(result["results"]) == 1
            assert result["results"][0]["boxes"] == []

    def test_locate_server_error(self):
        """Test handling of server errors."""
        import httpx

        provider = LocateAnythingProvider({})

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="Internal Server Error"),
        )
        mock_client.post.return_value = mock_resp

        with patch.object(type(provider), "client", new_callable=lambda: property(lambda self: mock_client)):
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            result = provider.locate_frames([(0.0, image)], "person")

            # On error, locate() returns empty list, so boxes should be empty
            assert result["results"][0]["boxes"] == []


class TestGroundingModels:
    def test_grounding_box(self):
        box = GroundingBox(
            x1=0.1, y1=0.2, x2=0.4, y2=0.6,
            label="person", score=0.95,
        )
        assert box.label == "person"
        assert box.score == 0.95
        assert box.x1 == 0.1
        assert box.y1 == 0.2
        assert box.x2 == 0.4
        assert box.y2 == 0.6

    def test_grounding_box_from_dict(self):
        """GroundingBox can be constructed from provider output dict."""
        raw = {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6, "label": "cup", "score": 0.85}
        box = GroundingBox(**raw)
        assert box.label == "cup"
        assert box.x1 == 0.1

    def test_grounding_frame(self):
        frame = GroundingFrame(
            timestamp=10.5,
            timestamp_str="00:10",
            boxes=[
                GroundingBox(x1=0.1, y1=0.2, x2=0.3, y2=0.4, label="person", score=0.9)
            ],
            query="person",
        )
        assert frame.timestamp == 10.5
        assert frame.timestamp_str == "00:10"
        assert len(frame.boxes) == 1
        assert frame.boxes[0].label == "person"

    def test_grounding_result(self):
        result = GroundingResult(
            query="person holding cup",
            video_path="/path/to/video.mp4",
            frames=[],
            tokens=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            confidence="high",
            artifacts_dir="/tmp/run",
        )
        assert result.query == "person holding cup"
        assert result.confidence == "high"
        assert result.frames == []
        assert result.tokens.total_tokens == 120


class TestGroundCommandCLI:
    def test_ground_help(self):
        from typer.testing import CliRunner
        from openvision.cli.app import app
        runner = CliRunner()
        result = runner.invoke(app, ["ground", "--help"])
        assert result.exit_code == 0
        assert "Ground" in result.output or "ground" in result.output

    def test_ground_nonexistent_file(self):
        from typer.testing import CliRunner
        from openvision.cli.app import app
        runner = CliRunner()
        result = runner.invoke(app, ["ground", "nonexistent.mp4", "--query", "person"])
        assert result.exit_code != 0

    def test_ground_no_query(self):
        from typer.testing import CliRunner
        from openvision.cli.app import app
        runner = CliRunner()
        result = runner.invoke(app, ["ground", "nonexistent.mp4"])
        assert result.exit_code != 0


class TestObserveGroundFlag:
    def test_observe_ground_help(self):
        from typer.testing import CliRunner
        from openvision.cli.app import app
        runner = CliRunner()
        result = runner.invoke(app, ["observe", "--help"])
        assert result.exit_code == 0
        assert "--ground" in result.output or "ground" in result.output.lower()
