"""FFmpeg DNN filter integration for inline ONNX model inference during frame extraction.

FFmpeg 9.0+ supports the dnn_backend=onnxruntime filter, which runs ONNX models
directly inside the ffmpeg pipeline. This avoids a separate Python inference step.
"""
import subprocess
import json
import os
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _has_dnn_onnxruntime() -> bool:
    """Check if ffmpeg was built with DNN + ONNX Runtime support."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10
        )
        # The dnn_* filters appear when DNN support is compiled in
        return "dnn" in proc.stdout.lower() or "onnxruntime" in proc.stdout.lower()
    except Exception:
        return False


def detect_dnn_support() -> dict:
    """
    Detect ffmpeg DNN/ONNX Runtime support level.

    Returns dict with: available, supports_detection, supports_classification
    """
    result = {
        "available": False,
        "supports_detection": False,
        "supports_classification": False,
    }

    try:
        proc = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10
        )
        output = proc.stdout.lower()

        # dnn_backend is available if any dnn filter exists
        result["available"] = "dnn" in output

        # Detection support: dnn_detect
        result["supports_detection"] = "dnn_detect" in output or "dnn_detection" in output

        # Classification support: dnn_classify
        result["supports_classification"] = "dnn_classify" in output

    except Exception:
        pass

    return result


class DNNFilterPipeline:
    """
    Run ffmpeg with inline DNN inference during frame extraction.

    Uses ffmpeg's dnn_backend=onnxruntime filter to run an ONNX model
    on each frame as it's extracted, avoiding a separate Python inference loop.

    Example usage:
        pipeline = DNNFilterPipeline(model_path="yolov8n.onnx")
        results = pipeline.extract_with_detection("video.mp4", output_dir="./dnn_out")
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: Path to .onnx model file
            device: "cpu" or "cuda" (requires ffmpeg built with CUDA DNN support)
        """
        self.model_path = Path(model_path)
        self.device = device
        self._available = _has_dnn_onnxruntime()

        if not self._available:
            logger.warning(
                "FFmpeg DNN/ONNX Runtime not available. "
                "Install ffmpeg with --enable-libonnxruntime and --enable-dnn."
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_with_detection(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        conf_threshold: float = 0.5,
        max_frames: int = 100,
    ) -> list[dict]:
        """
        Extract frames and run YOLO detection inline via ffmpeg DNN filter.

        Args:
            video_path: Path to input video
            output_dir: Directory for output frames + detection JSON
            conf_threshold: Confidence threshold for detections
            max_frames: Maximum frames to process

        Returns:
            List of {frame_index, timestamp, detections: [{class, confidence, bbox}]}
        """
        if not self._available:
            raise RuntimeError("FFmpeg DNN/ONNX Runtime not available")

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        out.mkdir(parents=True, exist_ok=True)

        # dnn_detect outputs bounding boxes to stderr as structured text
        model_str = str(self.model_path).replace("\\", "/")

        cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", f"dnn_backend=onnxruntime,model={model_str}:dnn_detect=confidence={conf_threshold}",
            "-frames:v", str(max_frames),
            "-vsync", "vfr",
            str(out / "frame_%06d.jpg"),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            logger.warning("DNN filter extraction timed out")
            return []

        detections = self._parse_dnn_output(result.stderr)

        frames = []
        frame_files = sorted(out.glob("frame_*.jpg"))
        for i, fpath in enumerate(frame_files):
            frame_dets = detections.get(i, [])
            frames.append({
                "frame_index": i,
                "frame_path": str(fpath),
                "detections": frame_dets,
            })

        return frames

    def extract_with_classification(
        self,
        video_path: str,
        labels: list[str],
        output_dir: Optional[str] = None,
        max_frames: int = 100,
    ) -> list[dict]:
        """
        Extract frames and classify each via ffmpeg DNN filter.

        Args:
            video_path: Path to input video
            labels: List of class labels
            output_dir: Directory for output
            max_frames: Max frames

        Returns:
            List of {frame_index, label, confidence}
        """
        if not self._available:
            raise RuntimeError("FFmpeg DNN/ONNX Runtime not available")

        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        out.mkdir(parents=True, exist_ok=True)

        model_str = str(self.model_path).replace("\\", "/")
        labels_str = ":".join(labels)

        cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", f"dnn_backend=onnxruntime,model={model_str}:dnn_classify=labels={labels_str}",
            "-frames:v", str(max_frames),
            "-vsync", "vfr",
            str(out / "frame_%06d.jpg"),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            return []

        return self._parse_classify_output(result.stderr)

    def _parse_dnn_output(self, stderr: str) -> dict[int, list[dict]]:
        """Parse dnn_detect output from ffmpeg stderr."""
        import re
        frames_dets: dict[int, list[dict]] = {}

        for line in stderr.split("\n"):
            if "dnn_detect" not in line:
                continue
            # Pattern: [Parsed_dnn_detect_...] frame:0 class:0 conf:0.87 bbox:[100,50,200,300]
            frame_match = re.search(r'frame:(\d+)', line)
            class_match = re.search(r'class:(\d+)', line)
            conf_match = re.search(r'conf:([\d.]+)', line)
            bbox_match = re.search(r'bbox:\[(\d+),(\d+),(\d+),(\d+)\]', line)

            if frame_match and class_match:
                frame_idx = int(frame_match.group(1))
                det = {
                    "class_id": int(class_match.group(1)),
                    "confidence": float(conf_match.group(1)) if conf_match else 0.0,
                }
                if bbox_match:
                    det["bbox"] = [
                        int(bbox_match.group(i))
                        for i in range(1, 5)
                    ]
                frames_dets.setdefault(frame_idx, []).append(det)

        return frames_dets

    def _parse_classify_output(self, stderr: str) -> list[dict]:
        """Parse dnn_classify output from ffmpeg stderr."""
        import re
        results = []

        for line in stderr.split("\n"):
            if "dnn_classify" not in line:
                continue
            frame_match = re.search(r'frame:(\d+)', line)
            label_match = re.search(r'label:(\w+)', line)
            conf_match = re.search(r'conf:([\d.]+)', line)

            if frame_match:
                results.append({
                    "frame_index": int(frame_match.group(1)),
                    "label": label_match.group(1) if label_match else "unknown",
                    "confidence": float(conf_match.group(1)) if conf_match else 0.0,
                })

        return results
