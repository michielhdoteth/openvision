"""OpenCV 5 DNN provider - runs ONNX models natively without PyTorch."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

ENGINE_AUTO = 3   # Try new engine first, fallback to classic
ENGINE_NEW = 2    # New graph engine (CPU, 80%+ ONNX)
ENGINE_CLASSIC = 1  # Old 4.x engine (supports CUDA/OpenVINO)


class OpenCVDNNProvider:
    """
    Run ONNX models via OpenCV 5's rewritten DNN engine.

    Supports YOLO (detection), SAM encoder (embeddings), and any ONNX model.
    Uses ENGINE_AUTO by default: new graph engine first, classic as fallback.
    """

    def __init__(self, config: dict):
        self.config = config
        self.engine = config.get("engine", "auto")
        self.device = config.get("device", "cpu")
        self._engine_map = {
            "auto": ENGINE_AUTO,
            "new": ENGINE_NEW,
            "classic": ENGINE_CLASSIC,
        }
        self._nets: dict[str, cv2.dnn.Net] = {}

    def _get_engine(self) -> int:
        return self._engine_map.get(self.engine, ENGINE_AUTO)

    def load_model(self, model_path: str, alias: Optional[str] = None) -> str:
        """
        Load an ONNX model into OpenCV's DNN engine.

        Args:
            model_path: Path to .onnx file
            alias: Optional name alias for later lookup

        Returns:
            Alias or model filename as key
        """
        key = alias or Path(model_path).stem
        if key in self._nets:
            return key

        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        engine_id = self._get_engine()

        try:
            net = cv2.dnn.readNetFromONNX(str(model), engine=engine_id)
        except Exception as e:
            if engine_id == ENGINE_AUTO:
                logger.warning(f"New engine failed for {key}, falling back to classic: {e}")
                net = cv2.dnn.readNetFromONNX(str(model), engine=ENGINE_CLASSIC)
            else:
                raise

        if self.device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._nets[key] = net
        logger.info(f"Loaded ONNX model: {key} (engine={self.engine}, device={self.device})")
        return key

    def infer(self, model_key: str, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on loaded model.

        Args:
            model_key: Key returned by load_model()
            input_data: Input tensor (N, C, H, W) or (N, H, W, C)

        Returns:
            Output tensor
        """
        if model_key not in self._nets:
            raise KeyError(f"Model '{model_key}' not loaded. Call load_model() first.")

        net = self._nets[model_key]

        if input_data.ndim == 4:
            blob = input_data
        else:
            blob = cv2.dnn.blobFromImage(input_data, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)

        net.setInput(blob)
        return net.forward()

    def detect_yolo(self, model_key: str, image: np.ndarray,
                    conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> list[dict]:
        """
        Run YOLO detection on a single image.

        Args:
            model_key: Loaded YOLO model key
            image: BGR numpy array (H, W, 3)
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold

        Returns:
            List of {class_id, confidence, bbox: [x, y, w, h]}
        """
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)

        net = self._nets[model_key]
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for det in output[0]:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < conf_threshold:
                    continue

                cx, cy, bw, bh = det[0:4]
                x = int((cx - bw / 2) * w / 640)
                y = int((cy - bh / 2) * h / 640)
                bw = int(bw * w / 640)
                bh = int(bh * h / 640)

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "class_id": class_ids[i],
                    "confidence": confidences[i],
                    "bbox": boxes[i],
                })

        return results

    def unload(self, model_key: Optional[str] = None):
        """Unload model(s) to free memory."""
        if model_key:
            self._nets.pop(model_key, None)
        else:
            self._nets.clear()

    def list_models(self) -> list[str]:
        return list(self._nets.keys())


def detect_cuda_available() -> bool:
    """Check if OpenCV was built with CUDA support."""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def get_opencv_version() -> str:
    """Get OpenCV version string."""
    return cv2.__version__
