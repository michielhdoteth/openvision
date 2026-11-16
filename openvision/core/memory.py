"""Cross-video memory: SQLite index for searching frames and transcripts across videos."""
import sqlite3
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class MemoryEntry:
    """A single indexed entry from a video."""
    video_path: str
    timestamp: float
    timestamp_str: str
    frame_path: str
    transcript_text: str
    summary_text: str
    tags: list[str]


class VideoMemory:
    """SQLite-backed cross-video search index."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                timestamp REAL NOT NULL,
                timestamp_str TEXT,
                frame_path TEXT,
                transcript_text TEXT,
                summary_text TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_video ON entries(video_path)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON entries(timestamp)
        """)
        conn.commit()
        conn.close()

    def index_video(
        self,
        video_path: str,
        frames: list[dict],
        transcript_segments: Optional[list] = None,
        summary: str = "",
        tags: Optional[list[str]] = None,
    ):
        """
        Index frames and transcript from a video into the memory database.

        Args:
            video_path: Path to the video file
            frames: List of frame dicts with 'image', 'timestamp', 'path'
            transcript_segments: Optional transcript segments
            summary: Optional summary text
            tags: Optional tags for categorization
        """
        conn = sqlite3.connect(self.db_path)
        tags_json = json.dumps(tags or [])

        transcript_map = {}
        if transcript_segments:
            for seg in transcript_segments:
                # Round to nearest 0.5s for matching
                key = round(getattr(seg, "start", 0), 0)
                text = getattr(seg, "text", "")
                if key in transcript_map:
                    transcript_map[key] += " " + text
                else:
                    transcript_map[key] = text

        for frame in frames:
            ts = frame.get("timestamp", 0)
            ts_str = frame.get("timestamp_str", f"{ts:.1f}s")
            frame_path = frame.get("path", "")

            ts_key = round(ts, 0)
            transcript_text = transcript_map.get(ts_key, "")
            if not transcript_text and transcript_map:
                nearest = min(transcript_map.keys(), key=lambda k: abs(k - ts), default=None)
                if nearest is not None and abs(nearest - ts) < 2.0:
                    transcript_text = transcript_map[nearest]

            conn.execute(
                "INSERT INTO entries (video_path, timestamp, timestamp_str, frame_path, transcript_text, summary_text, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (video_path, ts, ts_str, frame_path, transcript_text, summary, tags_json),
            )

        conn.commit()
        conn.close()

    def search(
        self,
        query: str,
        video_path: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search across all indexed videos.

        Args:
            query: Search text (matches against transcript and summary)
            video_path: Optional filter by video path
            limit: Max results

        Returns:
            List of matching entries
        """
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT video_path, timestamp, timestamp_str, frame_path, transcript_text, summary_text, tags FROM entries WHERE (transcript_text LIKE ? OR summary_text LIKE ? OR tags LIKE ?)"
        params = [f"%{query}%", f"%{query}%", f"%{query}%"]

        if video_path:
            sql += " AND video_path = ?"
            params.append(video_path)

        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "video_path": row[0],
                "timestamp": row[1],
                "timestamp_str": row[2],
                "frame_path": row[3],
                "transcript_text": row[4],
                "summary_text": row[5],
                "tags": json.loads(row[6]) if row[6] else [],
            })
        return results

    def list_videos(self) -> list[dict]:
        """List all indexed videos with entry counts."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT video_path, COUNT(*) as entries, MIN(timestamp) as start, MAX(timestamp) as end FROM entries GROUP BY video_path ORDER BY video_path"
        ).fetchall()
        conn.close()
        return [{"video_path": r[0], "entries": r[1], "start": r[2], "end": r[3]} for r in rows]

    def get_context(self, video_path: str, timestamp: float, window: float = 5.0) -> dict:
        """
        Get context around a specific timestamp in a video.

        Args:
            video_path: Video to search
            timestamp: Target timestamp
            window: Seconds of context around the timestamp

        Returns:
            Dict with before, at, after entries
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT timestamp, timestamp_str, frame_path, transcript_text FROM entries WHERE video_path = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (video_path, timestamp - window, timestamp + window),
        ).fetchall()
        conn.close()

        before = [{"timestamp": r[0], "ts": r[1], "frame": r[2], "text": r[3]} for r in rows if r[0] < timestamp]
        at = [{"timestamp": r[0], "ts": r[1], "frame": r[2], "text": r[3]} for r in rows if abs(r[0] - timestamp) < 0.5]
        after = [{"timestamp": r[0], "ts": r[1], "frame": r[2], "text": r[3]} for r in rows if r[0] > timestamp]

        return {"before": before, "at": at, "after": after}

    def stats(self) -> dict:
        """Get index statistics."""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        videos = conn.execute("SELECT COUNT(DISTINCT video_path) FROM entries").fetchone()[0]
        conn.close()
        return {"total_entries": total, "total_videos": videos}
