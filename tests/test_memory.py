"""Tests for openvision.core.memory - SQLite cross-video search index."""
import pytest
from openvision.core.memory import VideoMemory, MemoryEntry


class TestVideoMemory:
    @pytest.fixture
    def mem(self, tmp_path):
        return VideoMemory(str(tmp_path / "test_memory.db"))

    def test_init_creates_db(self, tmp_path):
        db = tmp_path / "new.db"
        VideoMemory(str(db))
        assert db.exists()

    def test_index_video_basic(self, mem):
        frames = [
            {"timestamp": 0.0, "timestamp_str": "0.0s", "path": "/tmp/f0.jpg"},
            {"timestamp": 2.0, "timestamp_str": "2.0s", "path": "/tmp/f1.jpg"},
        ]
        mem.index_video("/tmp/test.mp4", frames, summary="A test video")
        stats = mem.stats()
        assert stats["total_entries"] == 2
        assert stats["total_videos"] == 1

    def test_index_video_with_transcript(self, mem, sample_transcript_segments):
        frames = [
            {"timestamp": 0.0, "path": "/tmp/f0.jpg"},
            {"timestamp": 2.0, "path": "/tmp/f1.jpg"},
        ]
        mem.index_video("/tmp/test.mp4", frames, transcript_segments=sample_transcript_segments)
        # Search for transcript text
        results = mem.search("Hello")
        assert len(results) >= 1

    def test_search_by_transcript(self, mem):
        frames = [{"timestamp": 0.0, "path": ""}]
        mem.index_video("/tmp/v.mp4", frames, transcript_segments=[])
        # Manually insert transcript text
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        conn.execute("UPDATE entries SET transcript_text = 'The quick brown fox' WHERE video_path = '/tmp/v.mp4'")
        conn.commit()
        conn.close()

        results = mem.search("quick")
        assert len(results) == 1
        assert "fox" in results[0]["transcript_text"]

    def test_search_by_summary(self, mem):
        frames = [{"timestamp": 0.0, "path": ""}]
        mem.index_video("/tmp/v.mp4", frames, summary="A beautiful sunset over the ocean")
        results = mem.search("sunset")
        assert len(results) == 1

    def test_search_by_tags(self, mem):
        frames = [{"timestamp": 0.0, "path": ""}]
        mem.index_video("/tmp/v.mp4", frames, tags=["nature", "ocean"])
        results = mem.search("nature")
        assert len(results) == 1

    def test_search_with_video_filter(self, mem):
        frames1 = [{"timestamp": 0.0, "path": ""}]
        frames2 = [{"timestamp": 0.0, "path": ""}]
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        conn.execute("INSERT INTO entries (video_path, timestamp, tags) VALUES (?, ?, ?)",
                     ("/tmp/a.mp4", 0.0, '["cat"]'))
        conn.execute("INSERT INTO entries (video_path, timestamp, tags) VALUES (?, ?, ?)",
                     ("/tmp/b.mp4", 0.0, '["dog"]'))
        conn.commit()
        conn.close()

        results = mem.search("cat", video_path="/tmp/a.mp4")
        assert len(results) == 1
        assert results[0]["video_path"] == "/tmp/a.mp4"

    def test_search_limit(self, mem):
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        for i in range(30):
            conn.execute("INSERT INTO entries (video_path, timestamp, transcript_text) VALUES (?, ?, ?)",
                         ("/tmp/v.mp4", i * 1.0, "hello world"))
        conn.commit()
        conn.close()

        results = mem.search("hello", limit=5)
        assert len(results) == 5

    def test_list_videos(self, mem):
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/a.mp4", 0.0))
        conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/a.mp4", 5.0))
        conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/b.mp4", 1.0))
        conn.commit()
        conn.close()

        videos = mem.list_videos()
        assert len(videos) == 2
        paths = [v["video_path"] for v in videos]
        assert "/tmp/a.mp4" in paths
        assert "/tmp/b.mp4" in paths

    def test_list_videos_entry_counts(self, mem):
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        for i in range(5):
            conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/v.mp4", i * 1.0))
        conn.commit()
        conn.close()

        videos = mem.list_videos()
        assert videos[0]["entries"] == 5

    def test_get_context(self, mem):
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        for i in range(10):
            conn.execute(
                "INSERT INTO entries (video_path, timestamp, transcript_text) VALUES (?, ?, ?)",
                ("/tmp/v.mp4", i * 1.0, f"segment {i}"),
            )
        conn.commit()
        conn.close()

        ctx = mem.get_context("/tmp/v.mp4", 5.0, window=2.0)
        assert len(ctx["at"]) >= 1
        assert len(ctx["before"]) >= 1
        assert len(ctx["after"]) >= 1

    def test_get_context_empty(self, mem):
        ctx = mem.get_context("/tmp/nonexistent.mp4", 0.0)
        assert ctx["before"] == []
        assert ctx["at"] == []
        assert ctx["after"] == []

    def test_stats_empty(self, mem):
        stats = mem.stats()
        assert stats["total_entries"] == 0
        assert stats["total_videos"] == 0

    def test_stats_multiple_videos(self, mem):
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/a.mp4", 0.0))
        conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/b.mp4", 0.0))
        conn.execute("INSERT INTO entries (video_path, timestamp) VALUES (?, ?)", ("/tmp/b.mp4", 1.0))
        conn.commit()
        conn.close()

        stats = mem.stats()
        assert stats["total_entries"] == 3
        assert stats["total_videos"] == 2

    def test_search_no_results(self, mem):
        results = mem.search("nonexistent_query_xyz")
        assert results == []

    def test_multiple_queries(self, mem):
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        conn.execute("INSERT INTO entries (video_path, timestamp, transcript_text, summary_text) VALUES (?, ?, ?, ?)",
                     ("/tmp/v.mp4", 0.0, "The cat sat on the mat", ""))
        conn.execute("INSERT INTO entries (video_path, timestamp, transcript_text, summary_text) VALUES (?, ?, ?, ?)",
                     ("/tmp/v.mp4", 5.0, "", "A dog runs in the park"))
        conn.commit()
        conn.close()

        assert len(mem.search("cat")) == 1
        assert len(mem.search("dog")) == 1
        assert len(mem.search("the")) == 2
