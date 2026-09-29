"""Run cache and artifact management for Open Vision (SQLite-backed)."""
import os
import json
import hashlib
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


class RunCache:
    """
    Manages run directories and SQLite index for video processing artifacts.

    Storage:
    - Frames/images: saved as files in runs/YYYY-MM-DD_HHMMSS_<hash>/frames/
    - Metadata, transcripts, summaries: stored in runs/runs.db (SQLite)
    - Backward compatible: auto-imports existing JSON runs on first access

    Directory structure (on-disk):
    runs/
        runs.db              <- SQLite index
        YYYY-MM-DD_HHMMSS_<hash>/
            frames/
            thumbnails/
            masks/
    """

    def __init__(self, base_dir: str = "./runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.base_dir / "runs.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                dir_name TEXT NOT NULL,
                dir_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                summary_json TEXT DEFAULT '{}',
                config_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at REAL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
                UNIQUE(run_id, filename)
            );

            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp REAL,
                timestamp_str TEXT,
                file_path TEXT,
                reason TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                segment_index INTEGER,
                start_time REAL,
                end_time REAL,
                text TEXT,
                speaker TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_runs_key ON runs(cache_key);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_frames_run ON frames(run_id);
            CREATE INDEX IF NOT EXISTS idx_transcripts_run ON transcripts(run_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
        """)

        conn.commit()
        conn.close()

        # Auto-import legacy JSON runs
        self._import_legacy_json_runs()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def cache_key(self, path: str, mode: str = "balanced", fps: float = 0.5,
                  resolution: int = 768, options: Optional[dict] = None) -> str:
        """Generate a cache key from source identity and all result-changing options."""
        video_path = Path(path)
        if not video_path.exists():
            return ""
        stat = video_path.stat()
        option_json = json.dumps(options or {}, sort_keys=True, separators=(",", ":"), default=str)
        raw = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{mode}:{fps}:{resolution}:{option_json}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def has_cached(self, key: str) -> bool:
        """Check if a cached run exists for this key."""
        conn = self._conn()
        row = conn.execute("SELECT 1 FROM runs WHERE cache_key = ?", (key,)).fetchone()
        conn.close()
        return row is not None

    def get_run(self, key: str) -> Optional[Path]:
        """Find existing run directory by cache key."""
        conn = self._conn()
        row = conn.execute("SELECT dir_path FROM runs WHERE cache_key = ?", (key,)).fetchone()
        conn.close()
        if row:
            return Path(row["dir_path"])
        return None

    def create_run(self, key: str, metadata: Optional[dict] = None) -> Path:
        """Create a new run directory and register it in the database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{key}"
        run_dir = self.base_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "frames").mkdir(exist_ok=True)
        (run_dir / "thumbnails").mkdir(exist_ok=True)
        (run_dir / "masks").mkdir(exist_ok=True)

        # Register in database
        conn = self._conn()
        conn.execute(
            "INSERT INTO runs (cache_key, dir_name, dir_path, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (key, dir_name, str(run_dir), run_dir.stat().st_mtime, json.dumps(metadata or {}, default=str)),
        )
        conn.commit()
        conn.close()

        return run_dir

    def _get_run_id(self, run_dir: Path) -> Optional[int]:
        """Get database run_id from directory path."""
        conn = self._conn()
        row = conn.execute("SELECT id FROM runs WHERE dir_path = ?", (str(run_dir),)).fetchone()
        conn.close()
        return row["id"] if row else None

    def save_artifact(self, run_dir: Path, filename: str, data: dict):
        """Save a JSON artifact to the run directory (database + optional file)."""
        run_id = self._get_run_id(run_dir)
        if run_id:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, filename, data_json) VALUES (?, ?, ?)",
                (run_id, filename, json.dumps(data, indent=2, default=str)),
            )
            conn.commit()
            conn.close()

        # Also save as file for backward compatibility and CLI output
        file_path = run_dir / filename
        file_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def save_text(self, run_dir: Path, filename: str, text: str):
        """Save a text file to the run directory."""
        (run_dir / filename).write_text(text, encoding="utf-8")

    def save_frame(self, run_dir: Path, image, filename: str) -> Path:
        """Save an image frame to the run directory. Returns the path."""
        import cv2
        frame_path = run_dir / "frames" / filename
        cv2.imwrite(str(frame_path), image)
        return frame_path

    def save_frames_batch(self, run_dir: Path, frames: list[dict]):
        """Save frame metadata to database in bulk."""
        run_id = self._get_run_id(run_dir)
        if not run_id:
            return

        conn = self._conn()
        for frame in frames:
            conn.execute(
                "INSERT INTO frames (run_id, frame_index, timestamp, timestamp_str, file_path, reason) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, frame.get("index", 0), frame.get("timestamp", 0),
                 frame.get("timestamp_str", ""), frame.get("path", ""), frame.get("reason", "")),
            )
        conn.commit()
        conn.close()

    def save_transcript_batch(self, run_dir: Path, segments: list):
        """Save transcript segments to database in bulk."""
        run_id = self._get_run_id(run_dir)
        if not run_id:
            return

        conn = self._conn()
        for i, seg in enumerate(segments):
            conn.execute(
                "INSERT INTO transcripts (run_id, segment_index, start_time, end_time, text, speaker) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, i, getattr(seg, "start", 0), getattr(seg, "end", 0),
                 getattr(seg, "text", ""), getattr(seg, "speaker", None)),
            )
        conn.commit()
        conn.close()

    def list_runs(self) -> list[dict]:
        """List all cached runs with basic metadata."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT dir_name, dir_path, created_at, metadata_json FROM runs ORDER BY created_at DESC"
        ).fetchall()
        conn.close()

        runs = []
        for row in rows:
            runs.append({
                "name": row["dir_name"],
                "path": row["dir_path"],
                "created": row["created_at"],
                "metadata": json.loads(row["metadata_json"]),
            })
        return runs

    def search_runs(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across run metadata, artifacts, and transcripts."""
        conn = self._conn()
        like_q = f"%{query}%"

        # Search artifacts
        rows = conn.execute("""
            SELECT DISTINCT r.dir_name, r.dir_path, r.created_at, r.metadata_json, 'artifact' as source
            FROM runs r
            JOIN artifacts a ON a.run_id = r.id
            WHERE a.data_json LIKE ? OR a.filename LIKE ?
            ORDER BY r.created_at DESC LIMIT ?
        """, (like_q, like_q, limit)).fetchall()

        # Search transcripts
        rows2 = conn.execute("""
            SELECT DISTINCT r.dir_name, r.dir_path, r.created_at, r.metadata_json, 'transcript' as source
            FROM runs r
            JOIN transcripts t ON t.run_id = r.id
            WHERE t.text LIKE ?
            ORDER BY r.created_at DESC LIMIT ?
        """, (like_q, limit)).fetchall()

        conn.close()

        results = []
        seen = set()
        for row in list(rows) + list(rows2):
            key = row["dir_path"]
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "name": row["dir_name"],
                "path": row["dir_path"],
                "created": row["created_at"],
                "metadata": json.loads(row["metadata_json"]),
                "matched_via": row["source"],
            })

        return results

    def get_analytics(self) -> dict:
        """Get aggregate statistics across all runs."""
        conn = self._conn()
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        frame_count = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        transcript_count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        artifact_count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

        # Total duration (sum of last timestamp per run)
        total_duration = conn.execute("""
            SELECT COALESCE(SUM(max_ts), 0) FROM (
                SELECT run_id, MAX(timestamp) as max_ts FROM frames GROUP BY run_id
            )
        """).fetchone()[0]

        conn.close()

        return {
            "total_runs": run_count,
            "total_frames": frame_count,
            "total_transcript_segments": transcript_count,
            "total_artifacts": artifact_count,
            "total_duration_seconds": total_duration or 0,
        }

    def _import_legacy_json_runs(self):
        """Import existing JSON-based runs into SQLite (one-time migration)."""
        if not self.base_dir.exists():
            return

        conn = self._conn()
        existing_keys = {row[0] for row in conn.execute("SELECT cache_key FROM runs").fetchall()}

        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue

            meta_path = run_dir / "input_metadata.json"
            if not meta_path.exists():
                continue

            # Extract cache key from directory name (last 16 chars after last _)
            dir_name = run_dir.name
            cache_key = dir_name.split("_")[-1] if "_" in dir_name else dir_name

            if cache_key in existing_keys:
                continue

            # Import
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

            conn.execute(
                "INSERT OR IGNORE INTO runs (cache_key, dir_name, dir_path, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (cache_key, dir_name, str(run_dir), run_dir.stat().st_mtime, json.dumps(meta, default=str)),
            )

        conn.commit()
        conn.close()


def create_cache_key(path: str, mode: str = "balanced", fps: float = 0.5,
                     resolution: int = 768) -> str:
    """Standalone helper to generate a cache key."""
    cache = RunCache()
    return cache.cache_key(path, mode, fps, resolution)
