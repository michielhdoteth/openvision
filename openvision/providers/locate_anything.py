"""LocateAnything-3B grounding provider.

Calls a self-hosted vLLM + Worker API endpoint running LocateAnything-3B.
The model supports open-set detection, referring-expression grounding,
GUI element grounding, and point-based localization.

Setup: Deploy LocateAnything-3B via vLLM with Worker API.
Default endpoint: http://localhost:8000

Reference: https://github.com/NVlabs/Eagle/tree/main/Embodied
"""
import base64
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import ipaddress

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


def _validate_base_url(url: str) -> None:
    """Validate base URL to prevent SSRF attacks."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "::1", "[::1]"):
        return
    try:
        ip = ipaddress.ip_address(hostname)
        # Allow private/internal IPs (local vLLM usage)
        if ip.is_private or ip.is_loopback:
            return
        raise ValueError(f"Public IP addresses not allowed: {hostname}")
    except ValueError as e:
        if "Public IP" in str(e):
            raise
        # Domain name — allow
        return


class LocateAnythingProvider:
    """Grounding provider using NVIDIA LocateAnything-3B via Worker API.

    Unlike VLM providers, this does NOT go through ProviderRegistry.
    It is instantiated directly in commands (same pattern as SamProvider).

    Config keys:
        base_url: Worker API endpoint (default: http://localhost:8000)
        model: Model name (default: locate-anything-3b)
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8000")
        _validate_base_url(self.base_url)
        self.model_name = config.get("model", "locate-anything-3b")
        self.timeout = config.get("timeout", 30)
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def check_health(self) -> bool:
        """Check if the LocateAnything endpoint is reachable."""
        try:
            resp = self.client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def locate(self, image: np.ndarray, query: str) -> list[dict]:
        """Run grounding on a single image.

        Args:
            image: NumPy array in (H, W, C) RGB format.
            query: Text query (e.g. "person holding cup", "all people").

        Returns:
            List of dicts with keys: x1, y1, x2, y2, label, score.
        """
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "model": self.model_name,
            "image": img_b64,
            "query": query,
        }

        try:
            resp = self.client.post(
                f"{self.base_url}/v1/grounding",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            # Worker API may return boxes in different formats
            raw_boxes = data.get("boxes", data.get("predictions", []))
            boxes = []
            for item in raw_boxes:
                if "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4:
                    x1, y1, x2, y2 = item["bbox"]
                else:
                    x1 = float(item.get("x1", 0))
                    y1 = float(item.get("y1", 0))
                    x2 = float(item.get("x2", 0))
                    y2 = float(item.get("y2", 0))

                boxes.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "label": str(item.get("label", item.get("class", "object"))),
                    "score": float(item.get("score", item.get("confidence", 0.0))),
                })
            return boxes

        except httpx.HTTPStatusError as e:
            logger.error(f"LocateAnything API error: {e.response.status_code} {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"LocateAnything request failed: {e}")
            return []

    def locate_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        query: str,
        run_dir: Optional[str] = None,
    ) -> dict:
        """Run grounding on multiple frames.

        Args:
            frames: List of (timestamp_seconds, image_array) tuples.
            query: Text query for grounding.
            run_dir: Optional directory to save annotated frames.

        Returns:
            Dict with 'frames_processed' (int) and 'results' (list of dicts).
        """
        results = []
        for ts, image in frames:
            boxes = self.locate(image, query)
            results.append({
                "timestamp": ts,
                "boxes": boxes,
            })

            if run_dir:
                try:
                    annotated = self._draw_boxes(image.copy(), boxes)
                    frame_path = Path(run_dir) / f"ground_{ts:.2f}.jpg"
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(frame_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                except Exception as e:
                    logger.warning(f"Failed to save annotated frame: {e}")

        return {
            "frames_processed": len(frames),
            "results": results,
        }

    def _draw_boxes(self, image: np.ndarray, boxes: list[dict]) -> np.ndarray:
        """Draw bounding boxes on an image for visualization."""
        h, w = image.shape[:2]
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
        ]
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            x1 = int(box["x1"] * w)
            y1 = int(box["y1"] * h)
            x2 = int(box["x2"] * w)
            y2 = int(box["y2"] * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{box['label']} {box['score']:.2f}"
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image

    def unload(self) -> None:
        """Release HTTP client resources."""
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None
        logger.info("LocateAnything provider closed")
