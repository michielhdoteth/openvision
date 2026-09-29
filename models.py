"""Pydantic schemas for Open Vision tools."""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

# Re-export canonical TokenUsage from providers.base (Pydantic-compatible)
from providers.base import TokenUsage


class Frame(BaseModel):
    """A single video frame reference."""
    index: int
    timestamp: float
    timestamp_str: str
    path: Optional[str] = None


class TimelineEntry(BaseModel):
    """An observation at a specific point in time."""
    time_seconds: float
    timestamp: str
    observation: str
    frame_path: Optional[str] = None
    speaker: Optional[str] = None
    modality: Literal["visual", "audio", "segmentation", "grounding"] = "visual"
    boxes: list = Field(default_factory=list)  # list of GroundingBox dicts when modality="grounding"


class RawFrame(BaseModel):
    """A raw extracted frame with its path and timestamp."""
    index: int
    timestamp: float
    path: str
    base64: Optional[str] = None


class ObserveResult(BaseModel):
    """Result of a video_observe operation."""
    summary: str
    timeline: list[TimelineEntry]
    entities: list[str]
    artifacts_dir: str
    source_path: Optional[str] = None
    confidence: Literal["low", "medium", "high"]
    tokens: Optional[TokenUsage] = None
    raw_frames: Optional[list[RawFrame]] = None
    audio_segments: list[dict] = Field(default_factory=list)
    segmentation: Optional[dict] = None
    stages: dict[str, dict] = Field(default_factory=dict)
    duration_seconds: float = 0.0


class TranscribeResult(BaseModel):
    """Result of a video_transcribe operation."""
    segments: list[dict]
    language: str = "unknown"
    duration_seconds: float = 0.0
    device: str = "auto"
    tokens: Optional[TokenUsage] = None


class GroundingBox(BaseModel):
    """A single bounding box from LocateAnything-3B grounding."""
    x1: float  # normalized [0,1] left coordinate
    y1: float  # normalized [0,1] top coordinate
    x2: float  # normalized [0,1] right coordinate
    y2: float  # normalized [0,1] bottom coordinate
    label: str  # object label (e.g. "person", "cup")
    score: float  # confidence [0,1]


class GroundingFrame(BaseModel):
    """Grounding results for a single frame."""
    timestamp: float
    timestamp_str: str  # "MM:SS" formatted
    frame_path: Optional[str] = None
    boxes: list[GroundingBox] = Field(default_factory=list)
    query: str  # the query that produced these boxes


class GroundingResult(BaseModel):
    """Complete grounding result from `openvision ground`."""
    query: str
    video_path: str
    frames: list[GroundingFrame] = Field(default_factory=list)
    summary: str = ""  # optional VLM summary of grounding results
    tokens: Optional[TokenUsage] = None
    confidence: Literal["low", "medium", "high"] = "high"
    artifacts_dir: Optional[str] = None


class AskResult(BaseModel):
    """Result of a video_ask operation."""
    answer: str
    evidence: list[dict] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    tokens: Optional[TokenUsage] = None


class ProbeResult(BaseModel):
    """Metadata from video probing."""
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    frame_count: int
    estimated_processing_cost: Literal["low", "medium", "high"]
    vfr_mode: str = "unknown"
    color_range: str = "unknown"
    has_audio: bool = False
    has_video: bool = True
    pixel_format: str = "unknown"
    bit_rate: int = 0
    # Deep analysis fields (populated with --deep flag)
    vfr_variable_frames: int = 0
    vfr_constant_frames: int = 0
    vfr_ratio: float = 0.0


class SegmentObject(BaseModel):
    """A segmented object within a frame."""
    object_id: str
    timestamps: list[str]
    mask_paths: list[str]
    bbox: list[float]


class SegmentResult(BaseModel):
    """Result of a video_segment operation."""
    prompt: str
    frames_processed: int
    masks_dir: str
    objects: list[SegmentObject]


class TrackPosition(BaseModel):
    """Object position at a point in time."""
    timestamp: str
    time_seconds: float
    bbox: list[float]
    confidence: float


class TrackResult(BaseModel):
    """Result of a video_track operation."""
    track_id: str
    positions: list[TrackPosition]
    summary: str


class ImageResult(BaseModel):
    """Result of a single image observation."""
    description: str
    objects: list[str] = Field(default_factory=list)
    text_detected: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    tokens: Optional[TokenUsage] = None
    stages: dict[str, dict] = Field(default_factory=dict)


class StatusResult(BaseModel):
    """System status report."""
    gpu: str
    vram_total_gb: float
    vram_free_gb: float
    loaded_models: list[str]
    safe_mode: bool
    queue: int = 0


class OpenVisionConfig(BaseModel):
    """Full configuration model."""
    machine: dict = Field(default_factory=dict)
    defaults: dict = Field(default_factory=lambda: {
        "mode": "balanced",
        "max_resolution": 768,
        "fps": 0.5,
        "max_frames": 60,
    })
    models: dict = Field(default_factory=dict)
    gpu_policy: dict = Field(default_factory=dict)
    cache: dict = Field(default_factory=lambda: {"enabled": True, "directory": "./runs"})
    modes: dict = Field(default_factory=dict)


