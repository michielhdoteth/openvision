"""Speaker diarization provider - identifies who speaks when."""
import logging
import os
import wave
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A diarized segment identifying a speaker."""
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    """Full diarization result."""
    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0
    speakers: list[str] = field(default_factory=list)


class DiarizerProvider:
    """
    Local speaker diarization using pyannote.audio Community-1.

    The model is downloaded from Hugging Face on first use and requires the
    account to have accepted the model conditions. ``token=True`` uses the
    token configured by ``hf auth login``; HF_TOKEN can override it.
    """

    MODEL_ID = "pyannote/speaker-diarization-community-1"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._pipeline = None

    def _ensure_pipeline(self):
        """Load the local diarization pipeline and place it on the best device."""
        if self._pipeline is None:
            try:
                import torch
                from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError(
                    "Speaker diarization requires pyannote.audio. Install with: uv sync --extra diarize"
                ) from None

            token = self.config.get("hf_token") or os.getenv("HF_TOKEN") or True
            self._pipeline = Pipeline.from_pretrained(self.MODEL_ID, token=token)
            device = self.config.get("device")
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline.to(torch.device(device))

    def diarize(self, audio_path: str,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)

        Returns:
            DiarizationResult with speaker segments
        """
        self._ensure_pipeline()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = self._read_waveform(audio_path)

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        logger.info(f"Running pyannote Community-1 diarization on {audio_path}")
        result = self._pipeline(audio, **kwargs)
        annotation = getattr(result, "speaker_diarization", result)
        segments = [
            SpeakerSegment(start=turn.start, end=turn.end, speaker=str(speaker))
            for turn, _track, speaker in annotation.itertracks(yield_label=True)
        ]

        speakers = list(set(s.speaker for s in segments))
        speakers.sort()

        diar_result = DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
            speakers=speakers,
        )

        logger.info(f"Diarization complete: {len(segments)} segments, {len(speakers)} speakers")
        return diar_result

    @staticmethod
    def _read_waveform(audio_path: str) -> dict:
        """Decode OpenVision's PCM WAV into pyannote's in-memory audio contract.

        Passing waveform data directly avoids TorchCodec's native FFmpeg DLL
        loader, which is not reliable for the standard FFmpeg Windows builds.
        """
        import numpy as np
        import torch

        with wave.open(audio_path, "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frames = stream.readframes(stream.getnframes())

        if sample_width == 1:
            samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif sample_width == 2:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
        elif sample_width == 4:
            samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width * 8} bits")

        samples = samples.reshape(-1, channels)
        if channels > 1:
            samples = samples.mean(axis=1, keepdims=True)
        waveform = torch.from_numpy(samples.T.copy())
        return {"waveform": waveform, "sample_rate": sample_rate}

    def close(self):
        """Release model references and return cached CUDA allocations."""
        self._pipeline = None
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def merge_transcript_with_diarization(
    transcript_segments: list,
    diarization_segments: list[SpeakerSegment],
    fill_nearest: bool = True,
) -> list:
    """
    Merge whisper transcript segments with speaker diarization output.

    For each transcript segment, finds the diarization segment with the
    greatest temporal overlap and assigns that speaker label.

    Args:
        transcript_segments: List of TranscriptSegment from faster-whisper
        diarization_segments: List of SpeakerSegment from diarizer
        fill_nearest: If True, assign nearest speaker when no overlap exists

    Returns:
        List of TranscriptSegment with speaker labels filled in
    """
    diarization_segments = sorted(diarization_segments, key=lambda x: x.start)
    merged = []

    for seg in transcript_segments:
        seg_start = seg.start
        seg_end = seg.end
        speaker_overlap: dict[str, float] = {}

        for dia in diarization_segments:
            # Calculate temporal intersection
            intersection = min(dia.end, seg_end) - max(dia.start, seg_start)
            if intersection <= 0:
                continue

            speaker = dia.speaker
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + intersection

        if speaker_overlap:
            # Assign speaker with greatest overlap
            speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]
        elif fill_nearest and diarization_segments:
            # Fallback: find nearest diarization segment by midpoint
            midpoint = (seg_start + seg_end) / 2
            nearest = min(
                diarization_segments,
                key=lambda x: abs(((x.start + x.end) / 2) - midpoint),
            )
            speaker = nearest.speaker
        else:
            speaker = ""

        # Create new segment with speaker label
        merged.append(type(seg)(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            confidence=seg.confidence,
            speaker=speaker,
        ))

    return merged
