"""Tests for Open Vision's installable Python API."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile


class TestProcessResult:
    def test_public_package_exports_python_api(self):
        import openvision
        from openvision.api import process, transcribe
        from openvision.models import ObserveResult

        assert openvision.process is process
        assert openvision.transcribe is transcribe
        assert openvision.ProcessResult.__name__ == "ProcessResult"
        assert ObserveResult.__module__ == "openvision.models"

    def test_model_fields(self):
        from openvision import ProcessResult
        r = ProcessResult(
            summary="test",
            timeline=[],
            entities=[],
            transcript=None,
        )
        assert r.summary == "test"
        assert r.timeline == []

    def test_default_values(self):
        from openvision import ProcessResult
        r = ProcessResult()
        assert r.summary == ""
        assert r.timeline == []
        assert r.entities == []
        assert r.transcript is None
        assert r.frames_dir == ""
        assert r.duration_seconds == 0.0


class TestProcess:
    @patch("providers.registry.ProviderRegistry.create")
    @patch("core.sampling.smart_sample")
    @patch("core.video.detect_vfr", return_value={"mode": "cfr"})
    @patch("core.video.auto_fps", return_value=60)
    @patch("core.video.probe")
    def test_process_returns_result(self, mock_probe, mock_auto_fps, mock_vfr,
                                     mock_sample, mock_registry, tmp_path):
        from openvision import process, ProcessResult

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        mock_probe.return_value = {"duration_seconds": 5.0}
        mock_registry.return_value = MagicMock()

        import numpy as np
        mock_sample.return_value = [{"image": np.zeros((100, 100, 3), dtype=np.uint8), "timestamp": 0.0}]

        result = process(str(video))
        assert isinstance(result, ProcessResult)

    def test_process_file_not_found(self):
        from openvision import process
        with pytest.raises(FileNotFoundError):
            process("/nonexistent/video.mp4")


class TestTranscribe:
    @patch("providers.get_provider")
    def test_transcribe_returns_result(self, mock_get_provider):
        from openvision import transcribe
        from providers.parakeet import Transcript, TranscriptSegment

        mock_provider = MagicMock()
        mock_provider.transcribe.return_value = Transcript(
            segments=[TranscriptSegment(0, 1, "hello", 1.0)],
            language="en",
            duration_seconds=1.0,
        )
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            result = transcribe(f.name)

        assert result is not None
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello"

    def test_transcribe_file_not_found(self):
        from openvision import transcribe
        with pytest.raises(FileNotFoundError):
            transcribe("/nonexistent/video.mp4")
