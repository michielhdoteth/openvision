"""Tests for openvision.core.stream - JSONL streaming output."""
import json
import io
import pytest
from openvision.core.stream import JsonlWriter, create_jsonl_writer


class TestJsonlWriter:
    @pytest.fixture
    def output(self):
        return io.StringIO()

    @pytest.fixture
    def writer(self, output):
        return JsonlWriter(file=output)

    def test_emit_single_event(self, writer, output):
        writer.emit("test", {"key": "value"})
        output.seek(0)
        line = output.getvalue().strip()
        obj = json.loads(line)
        assert obj["event"] == "test"
        assert obj["data"] == {"key": "value"}
        assert "ts" in obj

    def test_emit_multiple_events(self, writer, output):
        writer.emit("start", {"video": "a.mp4"})
        writer.emit("frame", {"index": 0})
        writer.emit("done", {})
        output.seek(0)
        lines = [l for l in output.getvalue().strip().split("\n") if l]
        assert len(lines) == 3
        events = [json.loads(l)["event"] for l in lines]
        assert events == ["start", "frame", "done"]

    def test_emit_start(self, writer, output):
        writer.emit_start("video.mp4", mode="detailed")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["event"] == "start"
        assert obj["data"]["video"] == "video.mp4"
        assert obj["data"]["mode"] == "detailed"

    def test_emit_probe(self, writer, output):
        writer.emit_probe({"duration": 10.0, "fps": 30})
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["event"] == "probe"
        assert obj["data"]["duration"] == 10.0

    def test_emit_frame_with_path(self, writer, output):
        writer.emit_frame(5, 2.5, path="/tmp/frame.jpg")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"]["index"] == 5
        assert obj["data"]["timestamp"] == 2.5
        assert obj["data"]["path"] == "/tmp/frame.jpg"

    def test_emit_frame_without_path(self, writer, output):
        writer.emit_frame(0, 0.0)
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert "path" not in obj["data"]

    def test_emit_frame_with_detections(self, writer, output):
        dets = [{"class_id": 0, "confidence": 0.9, "bbox": [10, 20, 30, 40]}]
        writer.emit_frame(0, 1.0, detections=dets)
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"]["detections"] == dets

    def test_emit_transcript_with_speaker(self, writer, output):
        writer.emit_transcript(0.0, 3.0, "Hello", speaker="SPEAKER_00")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"]["speaker"] == "SPEAKER_00"
        assert obj["data"]["text"] == "Hello"

    def test_emit_transcript_without_speaker(self, writer, output):
        writer.emit_transcript(0.0, 3.0, "Hello")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert "speaker" not in obj["data"]

    def test_emit_summary_with_entities(self, writer, output):
        writer.emit_summary("A cat sits on a mat", entities=["cat", "mat"])
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"]["entities"] == ["cat", "mat"]

    def test_emit_summary_without_entities(self, writer, output):
        writer.emit_summary("Some summary")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert "entities" not in obj["data"]

    def test_emit_error_with_code(self, writer, output):
        writer.emit_error("File not found", code="ENOENT")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["event"] == "error"
        assert obj["data"]["code"] == "ENOENT"

    def test_emit_error_without_code(self, writer, output):
        writer.emit_error("Something broke")
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert "code" not in obj["data"]

    def test_emit_done_with_stats(self, writer, output):
        writer.emit_done({"frames": 10, "duration": 5.0})
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"]["frames"] == 10

    def test_emit_done_empty(self, writer, output):
        writer.emit_done()
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"] == {}

    def test_flush(self, writer, output):
        writer.emit("test", {})
        writer.flush()
        # Should not raise

    def test_unicode_data(self, writer, output):
        writer.emit("test", {"text": "Héllo Wörld 你好"})
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        assert obj["data"]["text"] == "Héllo Wörld 你好"

    def test_timestamp_is_iso(self, writer, output):
        writer.emit("test", {})
        output.seek(0)
        obj = json.loads(output.getvalue().strip())
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(obj["ts"])

    def test_each_line_is_valid_json(self, writer, output):
        for i in range(10):
            writer.emit("event", {"i": i})
        output.seek(0)
        for line in output.getvalue().strip().split("\n"):
            obj = json.loads(line)
            assert "event" in obj
            assert "data" in obj
            assert "ts" in obj


class TestCreateJsonlWriter:
    def test_enabled_returns_writer(self):
        writer = create_jsonl_writer(True, file=io.StringIO())
        assert isinstance(writer, JsonlWriter)

    def test_disabled_returns_none(self):
        writer = create_jsonl_writer(False)
        assert writer is None
