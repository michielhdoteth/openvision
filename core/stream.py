"""Streaming JSONL output for agent/pipeline integration."""
import json
import sys
from typing import Optional, Any
from datetime import datetime


class JsonlWriter:
    """
    Write JSONL (JSON Lines) output for streaming results.

    Each event is a single JSON object written as one line, enabling
    incremental consumption by agents and pipelines.

    Events:
        {"event": "start", "data": {...}, "ts": "..."}
        {"event": "probe", "data": {...}, "ts": "..."}
        {"event": "frame", "data": {...}, "ts": "..."}
        {"event": "transcript", "data": {...}, "ts": "..."}
        {"event": "stage", "data": {"name": ..., "status": ...}, "ts": "..."}
        {"event": "result", "data": {...complete result...}, "ts": "..."}
        {"event": "summary", "data": {...}, "ts": "..."}
        {"event": "error", "data": {...}, "ts": "..."}
        {"event": "done", "data": {...}, "ts": "..."}
    """

    def __init__(self, file=None):
        """
        Args:
            file: Output file object (default: sys.stdout)
        """
        self._file = file or sys.stdout
        self._buffer = []

    def _ts(self) -> str:
        return datetime.now().isoformat()

    def emit(self, event: str, data: Any = None):
        """
        Write a single event line.

        Args:
            event: Event type (start, probe, frame, transcript, summary, error, done)
            data: Event payload (must be JSON-serializable)
        """
        obj = {
            "event": event,
            "data": data,
            "ts": self._ts(),
        }
        line = json.dumps(obj, default=str, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()

    def emit_start(self, video_path: str, mode: str = "balanced"):
        self.emit("start", {"video": video_path, "mode": mode})

    def emit_probe(self, metadata: dict):
        self.emit("probe", metadata)

    def emit_frame(self, index: int, timestamp: float, path: str = "", detections: list = None):
        data = {"index": index, "timestamp": timestamp}
        if path:
            data["path"] = path
        if detections:
            data["detections"] = detections
        self.emit("frame", data)

    def emit_transcript(self, start: float, end: float, text: str, speaker: str = None):
        data = {"start": start, "end": end, "text": text}
        if speaker:
            data["speaker"] = speaker
        self.emit("transcript", data)

    def emit_summary(self, summary: str, entities: list = None):
        data = {"summary": summary}
        if entities:
            data["entities"] = entities
        self.emit("summary", data)

    def emit_error(self, message: str, code: str = None):
        data = {"message": message}
        if code:
            data["code"] = code
        self.emit("error", data)

    def emit_done(self, stats: dict = None):
        self.emit("done", stats or {})

    def flush(self):
        if hasattr(self._file, "flush"):
            self._file.flush()


def create_jsonl_writer(enabled: bool, file=None) -> Optional[JsonlWriter]:
    """Create a JsonlWriter if enabled, otherwise return None."""
    if enabled:
        return JsonlWriter(file)
    return None
