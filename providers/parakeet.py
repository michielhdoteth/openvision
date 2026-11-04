"""Parakeet transcription provider using NVIDIA Parakeet TDT via sherpa-onnx."""
import json
import logging
import re
import struct
import subprocess
import tempfile
import wave
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from core.audio import extract_audio, has_audio_stream, cleanup_wav

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "parakeet-tdt-0.6b-v3-int8"
# Fallback only — the shared resolution chain (config override -> home cache ->
# repo checkout) in core.models decides where the model really lives.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / _DEFAULT_MODEL_NAME


def _resolve_device(requested: str) -> str:
    """Resolve device string: 'auto' picks CUDA if available, else CPU."""
    if requested != "auto":
        return requested
    try:
        from core.gpu import get_vram_info
        vram = get_vram_info()
        if vram.get("free_gb", 0) > 1.0:
            logger.info(f"GPU auto-detect: using CUDA ({vram['free_gb']:.1f}GB free)")
            return "cuda"
        logger.info(f"GPU auto-detect: insufficient VRAM ({vram.get('free_gb', 0):.1f}GB), using CPU")
    except Exception:
        logger.info("GPU auto-detect: failed to query VRAM, using CPU")
    return "cpu"


def _read_wav_float32(wav_path: str) -> tuple[list[float], int]:
    """Read a WAV file and return (samples as float32 list, sample_rate).

    Works with 16-bit PCM WAV (what ffmpeg produces).
    """
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        # 16-bit PCM -> float32 [-1, 1]
        samples = struct.unpack(f"<{len(raw) // 2}h", raw)
        scale = 1.0 / 32768.0
        audio = [s * scale for s in samples]
    elif sampwidth == 4:
        # 32-bit int or float
        samples = struct.unpack(f"<{len(raw) // 4}i", raw)
        scale = 1.0 / 2147483648.0
        audio = [s * scale for s in samples]
    else:
        raise ValueError(f"Unsupported WAV bit depth: {sampwidth * 8}")

    if n_channels > 1:
        audio = [audio[i] for i in range(0, len(audio), n_channels)]

    return audio, sample_rate


@dataclass
class TranscriptSegment:
    """A single transcribed segment with timing."""
    start: float
    end: float
    text: str
    confidence: float = 0.0
    speaker: str = ""  # Speaker label from diarization (e.g., "SPEAKER_00")


@dataclass
class Transcript:
    """Full transcription result."""
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    duration_seconds: float = 0.0

    @property
    def text(self) -> str:
        """Full concatenated text."""
        return " ".join(s.text for s in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


class ParakeetProvider:
    """
    Speech-to-text provider using NVIDIA Parakeet TDT 0.6B v3 via sherpa-onnx.

    Model: nvidia/parakeet-tdt-0.6b-v3 (INT8 quantized, ~640MB)
    Supports 25 European languages with automatic language detection.
    Device can be 'auto' (picks CUDA if available), 'cpu', or 'cuda'.
    """

    def __init__(self, config: Optional[dict] = None):
        from storage.config import load_config
        try:
            app_config = load_config()
        except Exception:
            app_config = {}
        self.config = config or {}
        from core.models import resolve_model_dir
        resolved, origin = resolve_model_dir("parakeet", _DEFAULT_MODEL_NAME, app_config)
        override = self.config.get("model_dir")
        self.model_dir = Path(override) if override else resolved
        self._model_origin = origin
        self._requested_device = self.config.get("device", "auto")
        self.device = _resolve_device(self._requested_device)
        self._recognizer = None

    def _ensure_model(self):
        """Lazy-load the Parakeet model via sherpa-onnx.

        Auto-downloads the model if not found locally.
        """
        if self._recognizer is not None:
            return

        required_files = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        missing = [f for f in required_files if not (self.model_dir / f).exists()]
        if missing:
            logger.info(f"Parakeet model not found at {self.model_dir}, downloading...")
            try:
                from core.models import download_and_extract_parakeet
                self.model_dir = download_and_extract_parakeet()
            except Exception as e:
                from core.errors import OpenVisionError
                raise OpenVisionError(
                    code="STT_MODEL_UNAVAILABLE",
                    message=(
                        f"Parakeet model could not be loaded from {self.model_dir} "
                        f"({e}). The model may not be installed or the download failed."
                    ),
                    suggested_retry={"command": "openvision install get parakeet"},
                )

        import sherpa_onnx

        encoder = str(self.model_dir / "encoder.int8.onnx")
        decoder = str(self.model_dir / "decoder.int8.onnx")
        joiner = str(self.model_dir / "joiner.int8.onnx")
        tokens = str(self.model_dir / "tokens.txt")

        logger.info(
            f"Loading Parakeet TDT 0.6B v3 on {self.device} "
            f"(model_dir={self.model_dir})"
        )

        provider = "cuda" if self.device == "cuda" else "cpu"
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider=provider,
        )

        # Pre-create a stream to warm up the model
        stream = self._recognizer.create_stream()
        logger.info("Parakeet model loaded and warmed up")

    def transcribe(self, video_path: str, language: Optional[str] = None) -> Transcript:
        """
        Transcribe audio from a video file.

        Steps:
        1. Extract audio to WAV using ffmpeg
        2. Transcribe with sherpa-onnx Parakeet
        3. Clean up temp WAV

        Args:
            video_path: Path to video file
            language: Optional language code (e.g., "en"). Auto-detect if None.

        Returns:
            Transcript with segments, language, duration
        """
        if not has_audio_stream(video_path):
            logger.warning(f"No audio stream in {video_path}")
            return Transcript(duration_seconds=0.0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            wav_path = extract_audio(video_path, output_path=tmp_path)
            return self._transcribe_wav(wav_path, language)
        finally:
            cleanup_wav(tmp_path)

    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Transcript:
        """
        Transcribe from an already-extracted WAV file.

        Args:
            audio_path: Path to WAV file (16kHz mono PCM)
            language: Optional language code
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self._transcribe_wav(audio_path, language)

    def _transcribe_wav(self, wav_path: str, language: Optional[str] = None) -> Transcript:
        """Internal: transcribe a WAV file."""
        self._ensure_model()

        audio, sample_rate = _read_wav_float32(wav_path)

        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)

        result = stream.result
        text = result.text.strip() if result.text else ""

        transcript = Transcript(
            language=language or "",
            duration_seconds=duration,
        )

        if text:
            # sherpa-onnx returns timestamps as tokens; for offline we get
            # a single result. We split into sentence-level segments.
            segments = _split_into_segments(text, duration)
            transcript.segments = segments

        logger.info(
            f"Transcribed {len(transcript.segments)} segments "
            f"({transcript.duration_seconds:.1f}s, lang={transcript.language})"
        )

        return transcript

    def try_captions(self, video_path: str) -> Optional[Transcript]:
        """Try to extract existing captions from a video file.

        Checks for embedded subtitles first. Returns None if no captions found
        (caller should fall back to transcribe()).
        """
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
                 "-select_streams", "s", video_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            if not streams:
                return None

            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                subprocess.run(
                    ["ffmpeg", "-i", video_path, "-map", "0:s:0", tmp_path],
                    capture_output=True, text=True, timeout=60,
                )

                content = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
                from core.captions import parse_vtt

                caps = parse_vtt(content)
                if not caps:
                    caps = _parse_srt(content)

                if not caps:
                    return None

                segments = [
                    TranscriptSegment(start=c.start, end=c.end, text=c.text, confidence=1.0)
                    for c in caps
                ]
                return Transcript(
                    segments=segments,
                    language="",
                    duration_seconds=segments[-1].end if segments else 0,
                )
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.debug(f"Caption extraction failed: {e}")
            return None

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def unload(self):
        """Unload the model to free memory."""
        self._recognizer = None
        import gc
        gc.collect()


def _split_into_segments(text: str, total_duration: float) -> list[TranscriptSegment]:
    """Split transcribed text into sentence-level segments with estimated timing."""
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [TranscriptSegment(start=0.0, end=total_duration, text=text, confidence=1.0)]

    # Distribute time evenly across segments (rough but functional)
    n = len(sentences)
    seg_duration = total_duration / n

    segments = []
    for i, sentence in enumerate(sentences):
        start = i * seg_duration
        end = (i + 1) * seg_duration
        segments.append(TranscriptSegment(
            start=round(start, 3),
            end=round(end, 3),
            text=sentence,
            confidence=1.0,
        ))

    return segments


def _parse_srt(content: str) -> list:
    """Simple SRT parser as fallback."""
    from core.captions import CaptionSegment
    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())
    ts_pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})'
    )

    for block in blocks:
        lines = block.strip().split('\n')
        for i, line in enumerate(lines):
            m = ts_pattern.search(line)
            if m:
                start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
                end = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7)) + int(m.group(8)) / 1000
                text = ' '.join(l.strip() for l in lines[i + 1:] if l.strip())
                if text:
                    segments.append(CaptionSegment(start=start, end=end, text=text))
    return segments
