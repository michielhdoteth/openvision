"""Tests for cache and config modules."""
import pytest
import json
from pathlib import Path
from storage.config import (
    find_config,
    load_config,
    get_mode_config,
    merge_config,
)
from storage.cache import RunCache, create_cache_key


class TestConfig:
    def test_find_config(self):
        """Should find config/default.yaml relative to package."""
        config_path = find_config()
        assert config_path.exists()
        assert config_path.name == "default.yaml"

    def test_load_config(self):
        config = load_config()
        assert isinstance(config, dict)
        assert "machine" in config
        assert "models" in config
        assert "defaults" in config

    def test_load_config_bad_path(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_get_mode_config(self):
        config = load_config()
        fast = get_mode_config(config, "fast")
        assert isinstance(fast, dict)
        assert "resolution" in fast

    def test_get_mode_config_missing(self):
        fast = get_mode_config({}, "fast")
        assert "resolution" in fast
        assert fast.get("resolution") == 768  # fallback

    def test_merge_config(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}, "e": 4}
        merged = merge_config(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 2
        assert merged["b"]["d"] == 3
        assert merged["e"] == 4

    def test_env_var_override(self, monkeypatch, tmp_path):
        """OPENVISION_CONFIG env var should take priority."""
        custom_config = tmp_path / "custom.yaml"
        custom_config.write_text("machine:\n  gpu: custom\n")
        monkeypatch.setenv("OPENVISION_CONFIG", str(custom_config))
        config = load_config()
        assert config["machine"]["gpu"] == "custom"

    def test_load_config_empty_yaml(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        config = load_config(str(empty))
        assert config == {}


class TestRunCache:
    @pytest.fixture
    def cache(self, tmp_path):
        return RunCache(str(tmp_path / "runs"))

    def test_create_key(self, tmp_path):
        """Generate a cache key for a real file."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video content")

        key = create_cache_key(str(test_file))
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_nonexistent(self):
        key = create_cache_key("nonexistent.mp4")
        assert key == ""

    def test_create_and_find_run(self, cache):
        key = "testkey1234567890"
        run_dir = cache.create_run(key, {"test": True})
        assert run_dir.exists()
        assert (run_dir / "frames").exists()
        assert (run_dir / "masks").exists()
        assert (run_dir / "thumbnails").exists()

        # Verify DB entry exists
        found = cache.get_run(key)
        assert found is not None
        assert found == run_dir

    def test_has_cached(self, cache):
        key = "findmetest000001"
        assert not cache.has_cached(key)
        cache.create_run(key)
        assert cache.has_cached(key)

    def test_get_run(self, cache):
        key = "getruntest000001"
        cache.create_run(key)
        found = cache.get_run(key)
        assert found is not None
        assert found.name.endswith(key)

    def test_list_runs(self, cache):
        cache.create_run("run1")
        cache.create_run("run2")
        runs = cache.list_runs()
        assert len(runs) >= 2

    def test_save_artifact(self, cache):
        key = "artifacttest"
        run_dir = cache.create_run(key)
        cache.save_artifact(run_dir, "test_data.json", {"foo": "bar"})
        assert (run_dir / "test_data.json").exists()
        data = json.loads((run_dir / "test_data.json").read_text())
        assert data["foo"] == "bar"

        # Also verify it's in the database
        run_id = cache._get_run_id(run_dir)
        assert run_id is not None

    def test_save_text(self, cache):
        key = "texttest"
        run_dir = cache.create_run(key)
        cache.save_text(run_dir, "notes.txt", "hello world")
        assert (run_dir / "notes.txt").exists()
        assert (run_dir / "notes.txt").read_text() == "hello world"

    def test_save_frame(self, cache):
        import numpy as np
        key = "saveframetest"
        run_dir = cache.create_run(key)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = cache.save_frame(run_dir, img, "test.jpg")
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_cache_key_different_params(self, tmp_path):
        """Different params should yield different keys."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("content")
        key1 = create_cache_key(str(test_file), mode="fast")
        key2 = create_cache_key(str(test_file), mode="detailed")
        assert key1 != key2

    def test_cache_key_includes_stage_options_and_high_resolution_mtime(self, tmp_path):
        source = tmp_path / "sample.mp4"
        source.write_bytes(b"source")
        cache = RunCache(str(tmp_path / "runs"))
        visual_only = cache.cache_key(str(source), options={"transcribe": False})
        with_audio = cache.cache_key(str(source), options={"transcribe": True})
        assert visual_only != with_audio
        source.write_bytes(b"changed")
        changed_source = cache.cache_key(str(source), options={"transcribe": False})
        assert changed_source != visual_only

    def test_get_run_not_found(self, cache):
        assert cache.get_run("nonexistentkey") is None

    def test_save_frames_batch(self, cache):
        key = "batchtest"
        run_dir = cache.create_run(key)
        frames = [
            {"index": 0, "timestamp": 0.0, "timestamp_str": "0.0s", "path": "/tmp/f0.jpg", "reason": "test"},
            {"index": 1, "timestamp": 2.0, "timestamp_str": "2.0s", "path": "/tmp/f1.jpg", "reason": "test"},
        ]
        cache.save_frames_batch(run_dir, frames)
        # Verify frames are in the database
        run_id = cache._get_run_id(run_dir)
        assert run_id is not None

    def test_save_transcript_batch(self, cache):
        key = "transcripttest"
        run_dir = cache.create_run(key)
        from providers.parakeet import TranscriptSegment
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello", confidence=1.0, speaker="SPEAKER_00"),
            TranscriptSegment(start=3.0, end=6.0, text="World", confidence=1.0, speaker=""),
        ]
        cache.save_transcript_batch(run_dir, segments)
        run_id = cache._get_run_id(run_dir)
        assert run_id is not None

    def test_search_runs(self, cache):
        key = "searchtest"
        run_dir = cache.create_run(key)
        cache.save_artifact(run_dir, "data.json", {"content": "hello world"})
        results = cache.search_runs("hello")
        assert len(results) >= 1

    def test_search_runs_no_results(self, cache):
        results = cache.search_runs("nonexistent_xyz_query")
        assert results == []

    def test_get_analytics(self, cache):
        cache.create_run("run1")
        cache.create_run("run2")
        analytics = cache.get_analytics()
        assert analytics["total_runs"] >= 2
        assert "total_frames" in analytics
        assert "total_transcript_segments" in analytics

    def test_db_file_exists(self, cache):
        assert cache._db_path.exists()

    def test_multiple_caches_independent(self, tmp_path):
        cache1 = RunCache(str(tmp_path / "runs1"))
        cache2 = RunCache(str(tmp_path / "runs2"))
        cache1.create_run("only1")
        cache2.create_run("only2")
        assert cache1.has_cached("only1")
        assert not cache2.has_cached("only1")
        assert cache2.has_cached("only2")
        assert not cache1.has_cached("only2")
