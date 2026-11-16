"""Audio extraction from video files using ffmpeg."""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional


def has_audio_stream(video_path: str) -> bool:
    """
    Check if video file has an audio stream using ffprobe.

    Returns True if at least one audio stream exists.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0",
             video_path],
            capture_output=True, text=True, timeout=30,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """
    Extract audio from video file as WAV using ffmpeg.

    Args:
        video_path: Path to video file
        output_path: Optional output path. If None, creates temp file.
        sample_rate: Output sample rate (Whisper uses 16kHz)
        mono: Convert to mono (Whisper expects mono)

    Returns:
        Path to extracted WAV file

    Raises:
        FileNotFoundError: If video doesn't exist or has no audio
        RuntimeError: If ffmpeg extraction fails
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not output_path:
        output_path = str(video.with_suffix(".wav"))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    channels = "1" if mono else "2"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),  # Sample rate
        "-ac", channels,  # Channels
        str(output),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed: {result.stderr[:500]}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg audio extraction timed out (15 min)")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or ensure it's in PATH."
        )

    return str(output)


def extract_audio_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    sample_rate: int = 16000,
) -> str:
    """
    Extract a segment of audio from video.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output WAV path
        sample_rate: Sample rate for output

    Returns:
        Path to extracted WAV segment
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Audio segment extraction failed: {result.stderr[:500]}")

    return output_path


def get_audio_duration(video_path: str) -> float:
    """
    Get audio duration in seconds using ffprobe.

    Returns 0 if no audio stream or error.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0


def cleanup_wav(path: str):
    """Delete a temporary WAV file."""
    try:
        os.remove(path)
    except (OSError, FileNotFoundError):
        pass


def save_full_audio(video_path: str, output_path: str) -> str:
    """
    Save the full soundtrack from a video file (lossless copy when possible).

    Args:
        video_path: Path to video file
        output_path: Output path (e.g., video_dir/audio.m4a)

    Returns:
        Path to saved audio file
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-acodec", "copy",  # lossless copy
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", str(video),
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                str(output),
            ]
            result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Audio save failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio save timed out (5 min)")

    return str(output)
