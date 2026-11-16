"""Tests for openvision.core.dnn_filter - FFmpeg ONNX Runtime DNN pipeline."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from openvision.core.dnn_filter import (
    DNNFilterPipeline,
    detect_dnn_support,
    _has_dnn_onnxruntime,
)


class TestHasDnnOnnxruntime:
    @patch("openvision.core.dnn_filter.subprocess.run")
    def test_available(self, mock_run):
        mock_run.return_value = MagicMock(stdout="dnn_backend=onnxruntime\n", returncode=0)
        assert _has_dnn_onnxruntime() is True

    @patch("openvision.core.dnn_filter.subprocess.run")
    def test_not_available(self, mock_run):
        mock_run.return_value = MagicMock(stdout="scale=1.0\n", returncode=0)
        assert _has_dnn_onnxruntime() is False

    @patch("openvision.core.dnn_filter.subprocess.run", side_effect=FileNotFoundError)
    def test_ffmpeg_missing(self, mock_run):
        assert _has_dnn_onnxruntime() is False

    @patch("openvision.core.dnn_filter.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10)
        assert _has_dnn_onnxruntime() is False


class TestDetectDnnSupport:
    @patch("openvision.core.dnn_filter.subprocess.run")
    def test_full_support(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="dnn_backend=onnxruntime\ndnn_detect\n", returncode=0
        )
        result = detect_dnn_support()
        assert result["available"] is True
        assert result["supports_detection"] is True

    @patch("openvision.core.dnn_filter.subprocess.run")
    def test_no_support(self, mock_run):
        mock_run.return_value = MagicMock(stdout="scale=1.0\n", returncode=0)
        result = detect_dnn_support()
        assert result["available"] is False

    @patch("openvision.core.dnn_filter.subprocess.run", side_effect=Exception("fail"))
    def test_exception(self, mock_run):
        result = detect_dnn_support()
        assert result["available"] is False


class TestDNNFilterPipeline:
    def test_init_unavailable(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=False):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))
            assert pipeline.is_available is False

    def test_init_available(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=True):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))
            assert pipeline.is_available is True

    def test_extract_unavailable_raises(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=False):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))
            with pytest.raises(RuntimeError, match="not available"):
                pipeline.extract_with_detection("video.mp4")

    def test_extract_model_not_found(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=True):
            pipeline = DNNFilterPipeline(str(tmp_path / "nonexistent.onnx"))
            with pytest.raises(FileNotFoundError, match="ONNX model not found"):
                pipeline.extract_with_detection("video.mp4")

    def test_extract_video_not_found(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(b"fake")
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=True):
            pipeline = DNNFilterPipeline(str(model))
            with pytest.raises(FileNotFoundError, match="Video not found"):
                pipeline.extract_with_detection(str(tmp_path / "nonexistent.mp4"))

    def test_parse_dnn_output(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=True):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))

        stderr = """
        [Parsed_dnn_detect_0 @ 0x] frame:0 class:0 conf:0.87 bbox:[100,50,200,300]
        [Parsed_dnn_detect_0 @ 0x] frame:0 class:1 conf:0.65 bbox:[300,100,400,250]
        [Parsed_dnn_detect_0 @ 0x] frame:1 class:0 conf:0.92 bbox:[50,50,150,150]
        """
        result = pipeline._parse_dnn_output(stderr)
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert result[0][0]["class_id"] == 0
        assert result[0][0]["confidence"] == pytest.approx(0.87)
        assert result[0][0]["bbox"] == [100, 50, 200, 300]

    def test_parse_dnn_output_empty(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=True):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))
        result = pipeline._parse_dnn_output("")
        assert result == {}

    def test_parse_classify_output(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=True):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))

        stderr = """
        [Parsed_dnn_classify_0 @ 0x] frame:0 label:cat conf:0.95
        [Parsed_dnn_classify_0 @ 0x] frame:1 label:dog conf:0.88
        """
        result = pipeline._parse_classify_output(stderr)
        assert len(result) == 2
        assert result[0]["label"] == "cat"
        assert result[1]["confidence"] == pytest.approx(0.88)

    def test_extract_with_classification_unavailable(self, tmp_path):
        with patch("openvision.core.dnn_filter._has_dnn_onnxruntime", return_value=False):
            pipeline = DNNFilterPipeline(str(tmp_path / "model.onnx"))
            with pytest.raises(RuntimeError, match="not available"):
                pipeline.extract_with_classification("video.mp4", labels=["cat", "dog"])
