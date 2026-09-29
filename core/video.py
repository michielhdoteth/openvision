"""Video probing and frame extraction using OpenCV and ffmpeg."""
import cv2
import numpy as np
import subprocess
import tempfile
import os
import json
from pathlib import Path
from typing import Optional


def probe(path: str) -> dict:
    """
    Extract video metadata without full decode.

    Returns dict with: duration_seconds, width, height, fps, codec, frame_count,
    vfr_mode (cfr/vfr/unknown), color_range, audio_stream
    """
    cap = cv2.VideoCapture(path)
    opened = cap.isOpened()
    if not opened:
        enhanced = _probe_ffprobe(path)
        if enhanced.get("duration_seconds", 0) > 0 and (enhanced.get("has_audio") or enhanced.get("has_video")):
            return {
                "duration_seconds": enhanced["duration_seconds"],
                "width": enhanced.get("width", 0), "height": enhanced.get("height", 0),
                "fps": enhanced.get("fps", 0.0), "codec": enhanced.get("codec", "unknown"),
                "frame_count": enhanced.get("frame_count", 0),
                "vfr_mode": enhanced.get("vfr_mode", "unknown"),
                "color_range": enhanced.get("color_range", "unknown"),
                "has_audio": enhanced.get("has_audio", False),
                "has_video": enhanced.get("has_video", False),
                "bit_rate": enhanced.get("bit_rate", 0),
                "pixel_format": enhanced.get("pixel_format", "unknown"),
            }
        raise FileNotFoundError(f"Cannot open media file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4)) if codec_int else "unknown"

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    # Enhanced metadata from ffprobe (VFR detection, color range, audio)
    enhanced = _probe_ffprobe(path)
    duration = duration or enhanced.get("duration_seconds", 0)

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "frame_count": frame_count,
        "vfr_mode": enhanced.get("vfr_mode", "unknown"),
        "color_range": enhanced.get("color_range", "unknown"),
        "has_audio": enhanced.get("has_audio", False),
        "has_video": enhanced.get("has_video", True),
        "bit_rate": enhanced.get("bit_rate", 0),
        "pixel_format": enhanced.get("pixel_format", "unknown"),
    }


def _probe_ffprobe(path: str) -> dict:
    """
    Use ffprobe to extract enhanced metadata (VFR, color range, audio).
    Falls back gracefully if ffprobe unavailable.
    """
    result = {
        "vfr_mode": "unknown",
        "color_range": "unknown",
        "has_audio": False,
        "has_video": False,
        "duration_seconds": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "codec": "unknown",
        "bit_rate": 0,
        "pixel_format": "unknown",
    }

    try:
        # Get pixel format and color info
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,pix_fmt,color_range,color_space,color_transfer,color_primaries,width,height,avg_frame_rate,nb_frames:format=bit_rate,duration",
            "-of", "json",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return result

        data = json.loads(proc.stdout)

        # Check for audio stream
        streams = data.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "audio":
                result["has_audio"] = True

        # Get video stream info
        for s in streams:
            if s.get("codec_type") == "video":
                result["has_video"] = True
                result["pixel_format"] = s.get("pix_fmt", "unknown")
                result["color_range"] = s.get("color_range", "unknown")
                result["width"] = int(s.get("width", 0) or 0)
                result["height"] = int(s.get("height", 0) or 0)
                result["codec"] = s.get("codec_name", "unknown")
                try:
                    numerator, denominator = (s.get("avg_frame_rate", "0/1").split("/", 1) + ["1"])[:2]
                    result["fps"] = float(numerator) / float(denominator) if float(denominator) else 0.0
                except (ValueError, ZeroDivisionError):
                    result["fps"] = 0.0
                try:
                    result["frame_count"] = int(s.get("nb_frames", 0) or 0)
                except (ValueError, TypeError):
                    result["frame_count"] = 0

        fmt = data.get("format", {})
        result["bit_rate"] = int(fmt.get("bit_rate", 0))
        result["duration_seconds"] = float(fmt.get("duration", 0) or 0)

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        pass

    return result


def detect_vfr(path: str) -> dict:
    """
    Detect variable frame rate using ffmpeg's vfrdet filter.

    This is the most accurate way to detect VFR - it analyzes actual
    frame presentation timestamps rather than container metadata.

    Returns dict with:
        is_vfr: bool
        mode: "cfr" | "vfr"
        variable_frames: int (count of VFR frames)
        constant_frames: int (count of CFR frames)
        vfr_ratio: float (0.0 = pure CFR, 1.0 = pure VFR)
    """
    result = {
        "is_vfr": False,
        "mode": "unknown",
        "variable_frames": 0,
        "constant_frames": 0,
        "vfr_ratio": 0.0,
    }

    try:
        # Run vfrdet filter - outputs to stderr
        cmd = [
            "ffmpeg", "-i", path,
            "-vf", "vfrdet",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Parse vfrdet output from stderr
        # Format: [Parsed_vfrdet_0 @ ...] VFR detect: N frames ... V:count ... C:count ...
        for line in proc.stderr.split("\n"):
            if "vfrdet" in line and ("V:" in line or "C:" in line):
                import re
                # Extract V: and C: counts
                v_match = re.search(r'V:(\d+)', line)
                c_match = re.search(r'C:(\d+)', line)
                if v_match and c_match:
                    v_count = int(v_match.group(1))
                    c_count = int(c_match.group(1))
                    result["variable_frames"] = v_count
                    result["constant_frames"] = c_count
                    total = v_count + c_count
                    if total > 0:
                        result["vfr_ratio"] = v_count / total
                        result["is_vfr"] = v_count > c_count
                        result["mode"] = "vfr" if result["is_vfr"] else "cfr"
                break

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return result


def detect_color_range(path: str) -> dict:
    """
    Detect video color range (limited vs full) using ffmpeg's colordetect filter.

    Returns dict with:
        range_type: "limited" | "full" | "unknown"
        y_low: int (min Y value)
        y_high: int (max Y value)
    """
    result = {
        "range_type": "unknown",
        "y_low": 0,
        "y_high": 255,
    }

    try:
        cmd = [
            "ffmpeg", "-i", path,
            "-vf", "colordetect",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        for line in proc.stderr.split("\n"):
            if "colordetect" in line:
                import re
                # Parse Y range
                y_match = re.search(r'Y:(\d+)\s*->\s*(\d+)', line)
                if y_match:
                    y_low = int(y_match.group(1))
                    y_high = int(y_match.group(2))
                    result["y_low"] = y_low
                    result["y_high"] = y_high
                    # Limited range: 16-235, Full range: 0-255
                    if y_low >= 14 and y_high <= 240:
                        result["range_type"] = "limited"
                    elif y_low <= 2 and y_high >= 250:
                        result["range_type"] = "full"
                break

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return result


# ---------------------------------------------------------------------------
# FFmpeg 9.0 CUDA transpose support
# ---------------------------------------------------------------------------

_cuda_transpose_available: bool | None = None


def _has_cuda_transpose() -> bool:
    """Check if ffmpeg build includes the transpose_cuda filter (FFmpeg 9.0+)."""
    global _cuda_transpose_available
    if _cuda_transpose_available is not None:
        return _cuda_transpose_available

    try:
        proc = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10
        )
        _cuda_transpose_available = "transpose_cuda" in proc.stdout
    except Exception:
        _cuda_transpose_available = False

    return _cuda_transpose_available


def _detect_orientation(path: str) -> int:
    """
    Detect video rotation from metadata (side_data, tags).

    Returns:
        Rotation in degrees: 0, 90, 180, 270
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_side_data=rotation,tags:stream_tags=rotate",
            "-of", "json",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            return 0

        import json
        data = json.loads(proc.stdout)

        # Check side_data for rotation
        for stream in data.get("streams", []):
            for sd in stream.get("side_data_list", []):
                rot = sd.get("rotation")
                if rot is not None:
                    return abs(int(float(rot)))

            # Check tags
            tags = stream.get("tags", {})
            rotate = tags.get("rotate") or tags.get("tags.rotate")
            if rotate is not None:
                return abs(int(float(rotate)))

    except Exception:
        pass

    return 0


def _get_transpose_filter(rotation: int, use_cuda: bool = False) -> str:
    """
    Get the appropriate transpose filter for a given rotation.

    Args:
        rotation: 0, 90, 180, or 270 degrees
        use_cuda: Use CUDA-accelerated transpose if available

    Returns:
        ffmpeg filter string (e.g., "transpose_cuda=1") or "" if no transform needed
    """
    if rotation == 0:
        return ""

    suffix = "_cuda" if use_cuda and _has_cuda_transpose() else ""

    # transpose values: 0=90CCW+VFlip, 1=90CW, 2=90CCW, 3=90CW+VFlip
    filter_map = {
        90: f"transpose{suffix}=1",       # 90 CW
        180: f"transpose{suffix}=1,transpose{suffix}=1",  # 180 = two 90s
        270: f"transpose{suffix}=2",      # 90 CCW (= 270 CW)
    }

    return filter_map.get(rotation, "")


def probe_orientation(path: str) -> dict:
    """
    Probe video orientation and CUDA availability.

    Returns dict with: rotation, has_cuda_transpose, transpose_filter
    """
    rotation = _detect_orientation(path)
    has_cuda = _has_cuda_transpose()
    transpose_filter = _get_transpose_filter(rotation, use_cuda=has_cuda)

    return {
        "rotation": rotation,
        "has_cuda_transpose": has_cuda,
        "transpose_filter": transpose_filter,
    }


def extract_frames(
    path: str,
    fps: float = 0.5,
    max_frames: int = 60,
    max_resolution: int = 768,
    vfr_mode: str = "unknown",
) -> list[dict]:
    """
    Extract frames from video at given sampling rate.

    For VFR (variable frame rate) videos, uses timestamp-based extraction
    via ffmpeg for more accurate sampling. For CFR videos, uses OpenCV
    frame-index approach (faster).

    Args:
        path: Path to video file
        fps: Target frames per second to extract
        max_frames: Maximum number of frames to return
        max_resolution: Longest side in pixels for downscaling
        vfr_mode: "cfr", "vfr", or "unknown" (auto-detects if unknown)

    Returns:
        List of dicts: {index, timestamp, timestamp_str, image (np.ndarray)}
    """
    # Auto-detect VFR if not specified
    if vfr_mode == "unknown":
        vfr_info = detect_vfr(path)
        vfr_mode = vfr_info.get("mode", "cfr")

    # For VFR videos, use ffmpeg timestamp-based extraction (more accurate)
    if vfr_mode == "vfr":
        return _extract_frames_vfr(path, fps, max_frames, max_resolution)

    # Standard CFR extraction via OpenCV
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 30.0

    # Calculate stride that evenly samples across the ENTIRE video duration
    # This prevents only sampling the first N frames of a long video
    fps_interval = int(video_fps / fps) if fps > 0 else 1
    fps_interval = max(1, fps_interval)
    total_at_fps = total_frames // fps_interval

    if total_at_fps > max_frames:
        # Spread max_frames evenly across the full video
        stride = max(1, total_frames // max_frames)
    else:
        stride = fps_interval

    frames = []
    sample_idx = 0

    for frame_idx in range(0, total_frames, stride):
        if len(frames) >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        if not ret:
            continue

        timestamp = frame_idx / video_fps
        timestamp_str = _format_timestamp(timestamp)

        # Downscale
        image = _downscale(image, max_resolution)

        frames.append({
            "index": sample_idx,
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
            "image": image,
            "reason": "uniform",
        })
        sample_idx += 1

    cap.release()
    return frames


def _extract_frames_vfr(
    path: str,
    fps: float,
    max_frames: int,
    max_resolution: int,
) -> list[dict]:
    """
    Extract frames from VFR video using ffmpeg timestamp-based extraction.

    VFR videos have irregular frame timing, so frame-index-based sampling
    (OpenCV) produces inaccurate timestamps. Instead, we use ffmpeg's fps
    filter which correctly handles variable timestamps.
    """
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    meta = probe(path)
    duration = meta["duration_seconds"]

    # Calculate target fps to stay within frame budget
    target_fps = min(fps, max_frames / max(1, duration))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build filter chain with optional CUDA transpose for rotated videos
        rotation = _detect_orientation(path)
        transpose_filter = _get_transpose_filter(rotation, use_cuda=True)

        vf_parts = [f"fps={target_fps:.4f}"]
        if transpose_filter:
            vf_parts.append(transpose_filter)
        vf_chain = ",".join(vf_parts)

        cmd_with_info = [
            "ffmpeg", "-i", str(video),
            "-vf", vf_chain,
            "-vsync", "vfr",
            os.path.join(tmpdir, "frame_%06d.jpg"),
        ]

        try:
            result = subprocess.run(
                cmd_with_info, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            # Fallback: simpler command without transpose
            fallback_cmd = [
                "ffmpeg", "-i", str(video),
                "-vf", f"fps={target_fps:.4f}",
                "-vsync", "vfr",
                os.path.join(tmpdir, "frame_%06d.jpg"),
            ]
            subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=600)

        # Parse timestamps from showinfo
        timestamps = []
        for line in result.stderr.split("\n"):
            if "showinfo" in line and "pts_time:" in line:
                try:
                    pts_part = line.split("pts_time:")[1].split()[0]
                    timestamps.append(float(pts_part))
                except (IndexError, ValueError):
                    continue

        # Read extracted frames
        frames = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        for i, fpath in enumerate(frame_files):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, max_resolution)

            # Use parsed timestamp or fallback to estimation
            if i < len(timestamps):
                ts = timestamps[i]
            else:
                ts = i * (duration / max(1, len(frame_files)))

            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _format_timestamp(ts),
                "image": img,
                "reason": "vfr-uniform",
            })

    return frames


def extract_frames_by_indices(
    path: str,
    indices: list[int],
    max_resolution: int = 768,
) -> list[dict]:
    """Extract specific frame indices from video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for i, idx in enumerate(sorted(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, image = cap.read()
        if not ret:
            continue

        timestamp = idx / video_fps if video_fps > 0 else 0
        timestamp_str = _format_timestamp(timestamp)
        image = _downscale(image, max_resolution)

        frames.append({
            "index": i,
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
            "image": image,
        })

    cap.release()
    return frames


def extract_at_timestamps(
    path: str,
    timestamps: list[float],
    max_resolution: int = 768,
) -> list[dict]:
    """Extract a single frame at each specified timestamp. These are 'pinned' frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for i, ts in enumerate(sorted(timestamps)):
        frame_idx = int(ts * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        if not ret:
            continue

        image = _downscale(image, max_resolution)
        frames.append({
            "index": i,
            "timestamp": ts,
            "timestamp_str": _format_timestamp(ts),
            "image": image,
            "reason": "timestamp-cue",
            "pinned": True,
        })

    cap.release()
    return frames


def _downscale(image: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Resize image so longest side is max_side, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def parse_time(s: str) -> float:
    """Parse a timestamp string into float seconds.

    Accepts: 'SS', 'MM:SS', 'HH:MM:SS', 'HH:MM:SS.mmm'
    Examples: '45' -> 45.0, '1:30' -> 90.0, '1:30:00' -> 5400.0
    """
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp: {s}")


def estimate_processing_cost(metadata: dict, mode: str = "balanced") -> str:
    """Estimate processing cost based on video metadata and mode."""
    duration = metadata.get("duration_seconds", 0)
    resolution = metadata.get("width", 0) * metadata.get("height", 0)

    if mode == "fast":
        limit = 120  # 2 min
    elif mode == "detailed":
        limit = 30   # 30 sec
    else:
        limit = 60   # 1 min

    if duration <= limit and resolution <= 1280 * 720:
        return "low"
    elif duration <= limit * 3:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# ffmpeg-based extraction (for large/long videos where OpenCV is too slow)
# ---------------------------------------------------------------------------

def extract_scene_frames_ffmpeg(
    path: str,
    threshold: float = 0.20,
    max_frames: int = 100,
    max_resolution: int = 768,
    min_scene_gap: float = 0.5,
) -> list[dict]:
    """
    Extract frames at scene changes using ffmpeg's scene detection filter.

    Args:
        path: Video file path
        threshold: Scene change sensitivity 0-1 (lower = more scenes detected)
        max_frames: Maximum frames to return
        max_resolution: Longest side in pixels
        min_scene_gap: Minimum seconds between scene frames

    Returns:
        List of frame dicts: {index, timestamp, timestamp_str, image}
    """
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use ffmpeg scene detection with select filter
        cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr",
            "-frame_pts", "1",
            os.path.join(tmpdir, "frame_%06d.jpg"),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg scene detection timed out (10 min)")

        # Parse showinfo output to get timestamps
        timestamps = []
        for line in result.stderr.split("\n"):
            if "showinfo" in line and "pts_time:" in line:
                try:
                    pts_part = line.split("pts_time:")[1].split()[0]
                    timestamps.append(float(pts_part))
                except (IndexError, ValueError):
                    continue

        # Read extracted frames
        frames = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        for i, (fpath, ts) in enumerate(zip(frame_files, timestamps)):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, max_resolution)
            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _format_timestamp(ts),
                "image": img,
                "reason": "scene-change",
            })

    return frames


def extract_keyframes_ffmpeg(
    path: str,
    max_frames: int = 100,
    max_resolution: int = 768,
) -> list[dict]:
    """
    Extract I-frames (keyframes) using ffmpeg's skip_frame filter.

    Keyframes are the most informative frames in a video - they contain
    full image data without prediction from other frames.
    """
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    # Get video fps for timestamp calculation
    meta = probe(path)
    video_fps = meta.get("fps", 30.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", "select='eq(pict_type,I)'",
            "-vsync", "vfr",
            os.path.join(tmpdir, "keyframe_%06d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg keyframe extraction timed out")

        frames = []
        for i, fpath in enumerate(sorted(Path(tmpdir).glob("keyframe_*.jpg"))):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, max_resolution)
            # Estimate timestamp from frame index (approximate)
            ts = i * (meta["duration_seconds"] / max(1, len(list(Path(tmpdir).glob("keyframe_*.jpg")))))
            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _format_timestamp(ts),
                "image": img,
                "reason": "keyframe",
            })

    return frames


def dedupe_frames_ffmpeg(
    frames: list[dict],
    threshold: float = 2.0,
) -> list[dict]:
    """
    Lightweight perceptual deduplication using ffmpeg 16x16 thumbnails.

    This is much faster than full histogram comparison. It:
    1. Scales each frame to 16x16 grayscale via ffmpeg
    2. Computes mean absolute pixel difference between consecutive frames
    3. Keeps frames where difference > threshold

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        threshold: Minimum pixel difference to keep (default 2.0)

    Returns:
        Deduplicated list of frames
    """
    if len(frames) <= 1:
        return frames

    result = [frames[0]]
    prev_thumb = _make_thumbnail(frames[0]["image"])

    for frame in frames[1:]:
        thumb = _make_thumbnail(frame["image"])
        diff = np.mean(np.abs(prev_thumb.astype(float) - thumb.astype(float)))
        if diff > threshold:
            result.append(frame)
            prev_thumb = thumb

    return result


def dedupe_frames_sliding(
    frames: list[dict],
    threshold: float = 2.0,
    window: int = 3,
) -> list[dict]:
    """
    Sliding-window perceptual deduplication.

    Like dedupe_frames_ffmpeg() but compares against the last N kept frames
    instead of just the last one. This catches A-B-A cutaways where frame A
    appears, then B, then A again.

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        threshold: Minimum pixel difference to keep (default 2.0)
        window: Number of recent kept frames to compare against (default 3)

    Returns:
        Deduplicated list of frames
    """
    if len(frames) <= 1:
        return frames

    result = [frames[0]]
    thumbnails = [_make_thumbnail(frames[0]["image"])]

    for frame in frames[1:]:
        thumb = _make_thumbnail(frame["image"])

        # Compare against last N kept frames
        is_duplicate = False
        for prev_thumb in thumbnails[-window:]:
            diff = np.mean(np.abs(prev_thumb.astype(float) - thumb.astype(float)))
            if diff <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(frame)
            thumbnails.append(thumb)

    return result


def _make_thumbnail(image: np.ndarray, size: int = 16) -> np.ndarray:
    """Create a small grayscale thumbnail for fast comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def auto_fps(duration_seconds: float) -> int:
    """
    Calculate frame budget based on video duration.

    Returns maximum number of frames to extract:
    - <=30s: 12 frames
    - <=60s: 40 frames
    - <=3min: 60 frames
    - <=10min: 80 frames
    - >10min: 100 frames
    """
    if duration_seconds <= 30:
        return 12
    elif duration_seconds <= 60:
        return 40
    elif duration_seconds <= 180:
        return 60
    elif duration_seconds <= 600:
        return 80
    else:
        return 100


def auto_fps_focus(duration_seconds: float) -> int:
    """
    Calculate dense frame budget for a focus range window.

    When user specifies --start/--end, we want much denser extraction
    since they're zooming into a specific section.

    Returns maximum number of frames:
    - <=5s: 30 frames (6 fps)
    - <=15s: 60 frames (4 fps)
    - <=30s: 60 frames (2 fps)
    - <=60s: 60 frames (1 fps)
    - >60s: 80 frames
    """
    if duration_seconds <= 5:
        return 30
    elif duration_seconds <= 15:
        return 60
    elif duration_seconds <= 30:
        return 60
    elif duration_seconds <= 60:
        return 60
    else:
        return 80
