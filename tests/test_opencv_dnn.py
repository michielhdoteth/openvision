"""Tests for openvision.providers.opencv_dnn - OpenCV 5 DNN provider."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from openvision.providers.opencv_dnn import (
    OpenCVDNNProvider,
    detect_cuda_available,
    get_opencv_version,
    ENGINE_AUTO,
    ENGINE_NEW,
    ENGINE_CLASSIC,
)


class TestOpenCVDNNProvider:
    @pytest.fixture
    def provider(self):
        return OpenCVDNNProvider({"engine": "auto", "device": "cpu"})

    def test_init_default(self):
        p = OpenCVDNNProvider({})
        assert p.engine == "auto"
        assert p.device == "cpu"

    def test_init_custom(self):
        p = OpenCVDNNProvider({"engine": "new", "device": "cuda"})
        assert p.engine == "new"
        assert p.device == "cuda"

    def test_get_engine_auto(self, provider):
        assert provider._get_engine() == ENGINE_AUTO

    def test_get_engine_new(self):
        p = OpenCVDNNProvider({"engine": "new"})
        assert p._get_engine() == ENGINE_NEW

    def test_get_engine_classic(self):
        p = OpenCVDNNProvider({"engine": "classic"})
        assert p._get_engine() == ENGINE_CLASSIC

    def test_get_engine_unknown_falls_back(self):
        p = OpenCVDNNProvider({"engine": "unknown"})
        assert p._get_engine() == ENGINE_AUTO

    def test_list_models_empty(self, provider):
        assert provider.list_models() == []

    def test_unload_all(self, provider):
        provider._nets = {"model1": MagicMock(), "model2": MagicMock()}
        provider.unload()
        assert provider.list_models() == []

    def test_unload_single(self, provider):
        provider._nets = {"model1": MagicMock(), "model2": MagicMock()}
        provider.unload("model1")
        assert "model1" not in provider._nets
        assert "model2" in provider._nets

    def test_infer_model_not_loaded(self, provider):
        with pytest.raises(KeyError, match="not loaded"):
            provider.infer("nonexistent", np.zeros((1, 3, 640, 640), dtype=np.float32))

    def test_load_model_not_found(self, provider):
        with pytest.raises(FileNotFoundError):
            provider.load_model("/nonexistent/model.onnx")

    def test_load_model_success(self, provider, tmp_path):
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake onnx model")

        mock_net = MagicMock()
        with patch("cv2.dnn.readNetFromONNX", return_value=mock_net):
            key = provider.load_model(str(model_file))
            assert key == "test"
            assert "test" in provider.list_models()

    def test_load_model_custom_alias(self, provider, tmp_path):
        model_file = tmp_path / "yolov8.onnx"
        model_file.write_bytes(b"fake onnx model")

        mock_net = MagicMock()
        with patch("cv2.dnn.readNetFromONNX", return_value=mock_net):
            key = provider.load_model(str(model_file), alias="yolo")
            assert key == "yolo"
            assert "yolo" in provider.list_models()

    def test_load_model_idempotent(self, provider, tmp_path):
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake onnx model")

        mock_net = MagicMock()
        with patch("cv2.dnn.readNetFromONNX", return_value=mock_net):
            key1 = provider.load_model(str(model_file))
            key2 = provider.load_model(str(model_file))
            assert key1 == key2

    def test_detect_yolo_no_model(self, provider):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        with pytest.raises(KeyError):
            provider.detect_yolo("nonexistent", img)


class TestUtilityFunctions:
    def test_get_opencv_version(self):
        version = get_opencv_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_detect_cuda_available(self):
        result = detect_cuda_available()
        assert isinstance(result, bool)
