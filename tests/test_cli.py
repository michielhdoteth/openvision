"""Tests for CLI."""
import pytest
from typer.testing import CliRunner
from cli.app import app

runner = CliRunner()


class TestCliSmoke:
    def test_help_shows(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "openvision" in result.output or "visual perception" in result.output

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "openvision" in result.output.lower()
        assert "home:" in result.output.lower() or "Home" in result.output
    
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer may exit 0 (help shown) or 2 (error) depending on version
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output
    
    def test_probe_nonexistent(self):
        result = runner.invoke(app, ["probe", "nonexistent.mp4"])
        assert result.exit_code != 0  # Should error
    
    def test_probe_with_test_video(self):
        """If test video exists, probe it."""
        import os
        test_video = "test_data/sample.mp4"
        if not os.path.exists(test_video):
            # Check for any mp4 in test_data
            from pathlib import Path
            videos = list(Path("test_data").glob("*.mp4")) if Path("test_data").exists() else []
            if not videos:
                pytest.skip("No test video available")
            test_video = str(videos[0])
        
        result = runner.invoke(app, ["probe", test_video])
        assert result.exit_code == 0
        assert "Duration" in result.output
        assert "Resolution" in result.output
    
    def test_probe_json_output(self):
        """Test JSON output format."""
        from pathlib import Path
        videos = list(Path("test_data").glob("*.mp4")) if Path("test_data").exists() else []
        if not videos:
            pytest.skip("No test video available")
        
        result = runner.invoke(app, ["probe", str(videos[0]), "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "duration_seconds" in data
        assert "width" in data

    def test_probe_jsonl_is_only_machine_events(self, tmp_path, monkeypatch):
        import json
        from cli.commands import probe as probe_command
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        monkeypatch.setattr(probe_command, "probe", lambda _: {
            "duration_seconds": 3.0, "width": 10, "height": 10, "fps": 1.0,
            "codec": "test", "frame_count": 3, "vfr_mode": "cfr",
            "color_range": "unknown", "has_audio": True, "pixel_format": "yuv420p",
            "bit_rate": 100,
        })
        result = runner.invoke(app, ["probe", str(video), "--jsonl"])
        assert result.exit_code == 0
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert [event["event"] for event in events] == ["probe", "done"]
        assert result.stderr == ""
    
    def test_status_stub(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
    
    def test_observe_nonexistent(self):
        result = runner.invoke(app, ["observe", "test.mp4"])
        assert result.exit_code != 0  # File not found
    
    def test_ask_nonexistent(self):
        result = runner.invoke(app, ["ask", "test.mp4", "what is this?"])
        assert result.exit_code != 0  # File not found

    def test_observations_help(self):
        result = runner.invoke(app, ["observations", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "view" in result.output
        assert "delete" in result.output

    def test_observations_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENVISION_HOME", str(tmp_path / "openvision-home"))
        result = runner.invoke(app, ["observations", "list"])
        assert result.exit_code == 0
        assert "No saved observations" in result.output

    def test_observations_view_no_name(self):
        result = runner.invoke(app, ["observations", "view"])
        assert result.exit_code != 0

    def test_observations_delete_no_name(self):
        result = runner.invoke(app, ["observations", "delete"])
        assert result.exit_code != 0

    def test_observations_reject_traversal(self, tmp_path, monkeypatch):
        from cli.commands import observations
        outside = tmp_path / "outside.md"
        outside.write_text("keep", encoding="utf-8")
        monkeypatch.setattr(observations, "_obs_dir", lambda: tmp_path / "notes")
        result = runner.invoke(app, ["observations", "delete", "--name", "..\\outside"])
        assert result.exit_code != 0
        assert outside.read_text(encoding="utf-8") == "keep"

    def test_observations_reject_ambiguous_fuzzy_match(self, tmp_path, monkeypatch):
        from cli.commands import observations
        obs = tmp_path / "notes"
        obs.mkdir()
        (obs / "one-report.md").write_text("one", encoding="utf-8")
        (obs / "two-report.md").write_text("two", encoding="utf-8")
        monkeypatch.setattr(observations, "_obs_dir", lambda: obs)
        result = runner.invoke(app, ["observations", "delete", "--name", "report"])
        assert result.exit_code != 0
        assert len(list(obs.glob("*.md"))) == 2
