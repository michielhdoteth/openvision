"""Image preprocessing: downscale, encode, thumbnail."""
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Optional, Union


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def is_image_file(path: Union[str, Path]) -> bool:
    """True when the path is a still image rather than a video stream.

    Callers must check this before cv2.VideoCapture(): probing a PNG as video
    emits decoder errors and can raise, since a still has no frame count or
    duration to speak of.
    """
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def downscale(image: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Resize image so longest side is max_side."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_base64(image: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """Encode numpy image array to base64 string for LM Studio API."""
    encode_params = []
    if fmt == ".jpg" or fmt == ".jpeg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    success, buffer = cv2.imencode(fmt, image, encode_params)
    if not success:
        raise RuntimeError("Failed to encode image")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if fmt in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64_str}"


def save_frame(image: np.ndarray, path: Union[str, Path], fmt: str = ".jpg", quality: int = 90) -> str:
    """Save frame to disk and return the path string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_params = []
    if fmt == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success = cv2.imwrite(str(path), image, encode_params)
    if not success:
        raise RuntimeError(f"Failed to save frame to {path}")
    return str(path)


def make_thumbnail(image: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Create a small thumbnail."""
    return downscale(image, max_side)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
