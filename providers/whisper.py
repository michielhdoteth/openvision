"""Whisper transcription provider using faster-whisper."""
import json
import logging
import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from core.audio import extract_audio, has_audio_stream, cleanup_wav

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """Resolve device string: 'auto' picks CUDA if VRAM > 2GB, else CPU."""
    if requested != "auto":
        return requested
    try:
        from core.gpu import get_vram_info
        vram = get_vram_info()
        if vram.get("free_gb", 0) > 2.0:
            logger.info(f"GPU auto-detect: using CUDA ({vram['free_gb']:.1f}GB free)")
            return "cuda"
        logger.info(f"GPU auto-detect: insufficient VRAM ({vram.get('free_gb', 0):.1f}GB), using CPU")
    except Exception:
        logger.info("GPU auto-detect: failed to query VRAM, using CPU")
    return "cpu"


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


class WhisperProvider:
    """
    Speech-to-text provider using faster-whisper.

    Device can be 'auto' (picks CUDA if >2GB free VRAM, else CPU),
    'cpu', or 'cuda'. Default is 'auto'.
    """

    def __init__(self, config: Optional[dict] = None):
        from storage.config import load_config
        try:
            app_config = load_config()
        except Exception:
            app_config = {}
        self.config = config or {}
        app_stt = (app_config.get("models", {}).get("whisper", {}) or {})
        # Per-call config wins; fall back to config file so the CLI need not
        # thread extras: models.whisper.model_size / .model_dir.
        self.model_size = self.config.get("model_size") or app_stt.get("model_size", "base")
        self._requested_device = self.config.get("device", "auto")
        self.device = _resolve_device(self._requested_device)
        self.compute_type = self.config.get(
            "compute_type", "float16" if self.device == "cuda" else "int8"
        )
        self._model = None
        # Bring-your-own model: config models.whisper.model_dir wins over the
        # shared resolution chain, so an existing download is never re-fetched.
        self._app_config = app_config

    def _resolve_whisper_model_ref(self) -> tuple[str, str]:
        """Return (model_reference_for_faster_whisper, origin).

        Prefers an existing local model: explicit config override -> home cache
        -> repo checkout. Only downloads when nothing is found locally.
        """
        from core.models import adopt_repo_model, download_whisper_model, resolve_model_dir
        name = f"whisper-{self.model_size}"
        override = self.config.get("model_dir") or (
            self._app_config.get("models", {}).get("whisper", {}) or {}
        ).get("model_dir")
        if override:
            return str(override), "override"
        resolved, origin = resolve_model_dir("whisper", name, self._app_config)
        if origin in ("home", "override"):
            return str(resolved), origin
        if origin == "repo":
            adopted = adopt_repo_model("whisper", name, self._app_config)
            return str(adopted or resolved), "repo"
        path = download_whisper_model(self.model_size)
        return str(path), "downloaded"

    def _ensure_model(self):
        """Lazy-load the Whisper model from a local directory when possible.

        faster-whisper auto-downloads models from Hugging Face Hub only when no
        local copy resolves; a user-pointed directory (config models.whisper.
        model_dir) is loaded directly.
        """
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError(
                    "faster-whisper is not installed. Run: pip install faster-whisper"
                )

            try:
                model_ref, origin = self._resolve_whisper_model_ref()
            except Exception as exc:
                from core.errors import OpenVisionError
                raise OpenVisionError(
                    code="STT_MODEL_UNAVAILABLE",
                    message=(
                        f"Whisper model '{self.model_size}' could not be resolved or "
                        f"downloaded: {exc}. Point models.whisper.model_dir at an "
                        "existing model directory to skip downloads."
                    ),
                    suggested_retry={"command": f"openvision install get whisper --model {self.model_size}"},
                )

            logger.info(
                f"Loading Whisper model '{self.model_size}' on {self.device} "
                f"(compute={self.compute_type}, model_dir={model_ref})"
            )
            self._model = WhisperModel(
                model_ref,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(self, video_path: str, language: Optional[str] = None) -> Transcript:
        """
        Transcribe audio from a video file.

        Steps:
        1. Extract audio to WAV using ffmpeg
        2. Transcribe with faster-whisper
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

        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5,
            ),
        )

        transcript = Transcript(
            language=info.language if info else "",
            duration_seconds=info.duration if info else 0.0,
        )

        for seg in segments:
            transcript.segments.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0,
            ))

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

            # Try to extract first subtitle stream
            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                subprocess.run(
                    ["ffmpeg", "-i", video_path, "-map", "0:s:0", tmp_path],
                    capture_output=True, text=True, timeout=60,
                )

                content = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
                from core.captions import parse_vtt

                # Try VTT parse first, fall back to SRT-like parsing
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
        return self._model is not None

    def unload(self):
        """Unload the Whisper model to free memory."""
        self._model = None
        import gc
        gc.collect()


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
