"""
OpenVision Python API - programmatic access to video analysis and transcription.

Usage:
    from openvision import process, transcribe

    # Observe a video (extract frames + VLM analysis)
    result = process("video.mp4", mode="balanced")
    print(result.summary)

    # Transcribe a video
    transcript = transcribe("video.mp4", provider="parakeet")
    for seg in transcript.segments:
        print(f"[{seg.start:.1f}s] {seg.text}")
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ProcessResult:
    """Result from process()."""
    summary: str = ""
    timeline: list[dict] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    transcript: Optional[object] = None
    frames_dir: str = ""
    duration_seconds: float = 0.0


def process(
    path: str,
    mode: str = "balanced",
    detail: str = "balanced",
    question: Optional[str] = None,
    prompt: Optional[str] = None,
    max_frames: int = 60,
    fps: float = 0.5,
    resolution: int = 768,
    device: str = "auto",
    transcribe_audio: bool = False,
    stt_provider: str = "parakeet",
    keep_audio: bool = False,
    provider: Optional[str] = None,
    force: bool = False,
) -> ProcessResult:
    """
    Analyze a video and return structured observations.

    Args:
        path: Path to video file or URL
        mode: Processing mode (fast, balanced, detailed)
        detail: Extraction detail (transcript, efficient, balanced, token-burner)
        question: Specific question to answer
        prompt: Custom prompt (overrides default)
        max_frames: Maximum frames to process
        fps: Frames per second to sample
        resolution: Max frame resolution in pixels
        device: Device for STT (auto, cpu, cuda)
        transcribe_audio: Whether to transcribe audio
        stt_provider: STT provider (parakeet, whisper)
        keep_audio: Save full soundtrack
        provider: VLM provider name
        force: Force reprocess (ignore cache)

    Returns:
        ProcessResult with summary, timeline, entities
    """
    from core.video import probe, extract_frames, auto_fps, detect_vfr
    from core.sampling import smart_sample, adaptive_sample
    from providers.registry import ProviderRegistry
    from storage.config import load_config, get_mode_config
    from openvision.models import ObserveResult, TimelineEntry

    input_path = Path(path)
    if not input_path.exists():
        from core.download import is_url, download_video
        if is_url(path):
            config = load_config()
            from storage.paths import downloads_dir
            dl_dir = downloads_dir(config)
            dl_result = download_video(path, output_dir=str(dl_dir))
            input_path = Path(dl_result["path"])
        else:
            raise FileNotFoundError(f"File not found: {path}")

    config = load_config()
    mode_config = get_mode_config(config, mode)

    video_meta = probe(str(input_path))
    duration = video_meta["duration_seconds"]

    vfr_info = detect_vfr(str(input_path))
    vfr_mode = vfr_info.get("mode", "cfr")
    auto_max = auto_fps(duration)
    actual_max = min(max_frames, auto_max)

    if duration > 120:
        frames = adaptive_sample(str(input_path), max_frames=actual_max, max_resolution=resolution, vfr_mode=vfr_mode)
    else:
        frames = smart_sample(str(input_path), fps=fps, max_frames=actual_max, max_resolution=resolution, vfr_mode=vfr_mode)

    vlm_config = config.get("models", {}).get("vlm", {})
    if provider:
        vlm_config["provider"] = provider

    vlm = ProviderRegistry.create(vlm_config)

    timeline = []
    for frame in frames:
        entry = TimelineEntry(
            time_seconds=frame["timestamp"],
            timestamp=frame.get("timestamp_str", ""),
            observation="",
            frame_path=None,
        )
        timeline.append(entry)

    transcript = None
    if transcribe_audio:
        from providers import get_provider
        stt = get_provider(stt_provider, {"device": device})
        transcript = stt.transcribe(str(input_path))

    return ProcessResult(
        summary="",
        timeline=[{"time": e.time_seconds, "ts": e.timestamp} for e in timeline],
        entities=[],
        transcript=transcript,
        frames_dir="",
        duration_seconds=duration,
    )


def transcribe(
    path: str,
    provider: str = "parakeet",
    device: str = "auto",
    language: Optional[str] = None,
    model: str = "base",
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    diarize: bool = False,
):
    """
    Transcribe audio from a video file.

    Args:
        path: Path to video file or URL
        provider: STT provider (parakeet, whisper)
        device: Device (auto, cpu, cuda)
        language: Language code (e.g., 'en'). Auto-detect if None.
        model: Model size (whisper only)
        from_time: Start time (e.g., '1:30')
        to_time: End time (e.g., '2:00')
        diarize: Add speaker labels

    Returns:
        Transcript with segments, language, duration
    """
    from providers import get_provider
    from core.video import parse_time

    input_path = Path(path)
    if not input_path.exists():
        from core.download import is_url, download_video
        if is_url(path):
            from storage.config import load_config
            from storage.paths import downloads_dir
            config = load_config()
            dl_dir = downloads_dir(config)
            dl_result = download_video(path, output_dir=str(dl_dir))
            input_path = Path(dl_result["path"])
        else:
            raise FileNotFoundError(f"File not found: {path}")

    config = {"device": device}
    if provider == "whisper":
        config["model_size"] = model

    stt = get_provider(provider, config)

    window_start = parse_time(from_time) if from_time else None
    window_end = parse_time(to_time) if to_time else None

    if window_start is not None or window_end is not None:
        from core.audio import extract_audio_segment
        import tempfile as _tf
        seg_tmp = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
        seg_path = seg_tmp.name
        seg_tmp.close()
        try:
            start_s = window_start or 0
            end_s = window_end or 999999
            extract_audio_segment(str(input_path), start_s, end_s, seg_path)
            transcript = stt.transcribe_audio(seg_path, language)
            if window_start and transcript.segments:
                for seg in transcript.segments:
                    seg.start += window_start
                    seg.end += window_start
            if window_end and transcript.segments:
                transcript.segments = [s for s in transcript.segments if s.start < window_end]
        finally:
            import os
            if os.path.exists(seg_path):
                os.unlink(seg_path)
    else:
        transcript = stt.transcribe(str(input_path), language)

    return transcript
