"""Local hierarchical, time-grounded memory for observed videos.

Memory stores evidence, not inferred facts: speech, visual observations, and
segmentation geometry stay separate and are co-located only by source time.
Speaker-to-visible-person identity is deliberately marked unverified.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable


CHUNK_SECONDS = 30
SECTION_SECONDS = 300


def _value(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _fingerprint(source_path: str) -> str:
    path = Path(source_path).expanduser().resolve()
    stat = path.stat()
    raw = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class VideoMemoryStore:
    """Index observation evidence in a local SQLite database with FTS5."""

    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS video_memory_nodes (
                    node_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('root','section','chunk')),
                    parent_id TEXT,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    visual_text TEXT NOT NULL DEFAULT '',
                    audio_text TEXT NOT NULL DEFAULT '',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    speakers_json TEXT NOT NULL DEFAULT '[]',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    identity_links_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_video_memory_source
                    ON video_memory_nodes(fingerprint, kind, start_sec);
                CREATE VIRTUAL TABLE IF NOT EXISTS video_memory_fts USING fts5(
                    node_id UNINDEXED, fingerprint UNINDEXED, source_path UNINDEXED,
                    kind UNINDEXED, start_sec UNINDEXED, end_sec UNINDEXED,
                    summary, visual_text, audio_text, entities, speakers,
                    tokenize='unicode61'
                );
            """)

    def _connect(self):
        conn = sqlite3.connect(str(self.database), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def build(
        self,
        source_path: str,
        run_dir: str | Path,
        summary: str,
        timeline: Iterable,
        entities: Iterable[str] = (),
        transcript_segments: Iterable = (),
        segmentation: dict | None = None,
        duration_seconds: float = 0.0,
    ) -> dict:
        """Build/replace one source's hierarchy from timestamped evidence."""
        source = str(Path(source_path).expanduser().resolve())
        run = str(Path(run_dir).resolve())
        fingerprint = _fingerprint(source)
        entities = sorted({str(e).strip() for e in entities if str(e).strip()})
        visuals: list[dict] = []
        for entry in timeline:
            modality = _value(entry, "modality", "visual")
            text = str(_value(entry, "observation", "") or "").strip()
            if modality == "visual" and text.startswith("[SPEECH]"):
                modality = "audio"  # compatibility with cached pre-modality runs
            if modality in ("visual", "grounding"):
                visuals.append({
                    "time": float(_value(entry, "time_seconds", 0) or 0),
                    "text": ("[GROUNDING] " if modality == "grounding" else "") + text,
                    "modality": modality,
                    "frame_path": _value(entry, "frame_path"),
                })

        speech: list[dict] = []
        for segment in transcript_segments:
            text = str(_value(segment, "text", "") or "").strip()
            if not text:
                continue
            speech.append({
                "start": float(_value(segment, "start", 0) or 0),
                "end": float(_value(segment, "end", _value(segment, "start", 0)) or 0),
                "text": text,
                "speaker": str(_value(segment, "speaker", "") or "").strip() or None,
            })

        segmented = []
        for obj in (segmentation or {}).get("objects", []):
            segmented.append({
                "time": float(obj.get("time_seconds", 0) or 0),
                "bbox": obj.get("bbox", []),
                "score": obj.get("score"),
                "mask_path": obj.get("mask_path"),
                "label": obj.get("label"),
            })

        max_time = max(
            [duration_seconds or 0]
            + [v["time"] for v in visuals]
            + [s["end"] for s in speech]
            + [o["time"] for o in segmented]
        )
        bucket_count = max(1, int(max_time // CHUNK_SECONDS) + 1)
        buckets = []
        for index in range(bucket_count):
            start = index * CHUNK_SECONDS
            end = min(start + CHUNK_SECONDS, max_time) if max_time else start + CHUNK_SECONDS
            bucket_visual = [v for v in visuals if start <= v["time"] < end or (index == bucket_count - 1 and v["time"] == end)]
            bucket_audio = [s for s in speech if s["start"] < end and s["end"] >= start]
            bucket_segments = [o for o in segmented if start <= o["time"] < end or (index == bucket_count - 1 and o["time"] == end)]
            if not (bucket_visual or bucket_audio or bucket_segments):
                continue
            visual_text = " ".join(v["text"] for v in bucket_visual)
            audio_text = " ".join(s["text"] for s in bucket_audio)
            speaker_names = sorted({s["speaker"] for s in bucket_audio if s["speaker"]})
            evidence = [
                {"modality": v["modality"], "time_seconds": v["time"], "text": v["text"], "frame_path": v["frame_path"]}
                for v in bucket_visual
            ] + [
                {"modality": "audio", **s, "speaker_to_visible_person": "unverified"}
                for s in bucket_audio
            ] + [
                {"modality": "segmentation", **o}
                for o in bucket_segments
            ]
            buckets.append({
                "index": index, "start": start, "end": end,
                "visual_text": visual_text, "audio_text": audio_text,
                "speakers": speaker_names,
                "entities": [e for e in entities if e.casefold() in (visual_text + " " + audio_text).casefold()],
                "evidence": evidence,
                "summary": _chunk_summary(visual_text, audio_text),
            })

        sections = []
        for start in range(0, max(1, int(max_time // SECTION_SECONDS) + 1) * SECTION_SECONDS, SECTION_SECONDS):
            children = [b for b in buckets if start <= b["start"] < start + SECTION_SECONDS]
            if children:
                sections.append({
                    "index": start // SECTION_SECONDS, "start": start,
                    "end": min(start + SECTION_SECONDS, max_time) if max_time else start + SECTION_SECONDS,
                    "children": children,
                })

        node_rows = []
        fts_rows = []

        def add_node(node_id, kind, parent, start, end, node_summary, visual, audio, node_entities, speakers, evidence):
            identity_links = [{"speaker": s, "visible_person": None, "status": "unverified"} for s in speakers]
            node_rows.append((node_id, fingerprint, source, run, kind, parent, start, end,
                              node_summary, visual, audio, json.dumps(node_entities),
                              json.dumps(speakers), json.dumps(evidence), json.dumps(identity_links)))
            fts_rows.append((node_id, fingerprint, source, kind, str(start), str(end), node_summary,
                             visual, audio, " ".join(node_entities), " ".join(speakers)))

        root_id = f"{fingerprint}:root"
        all_visual = " ".join(v["text"] for v in visuals)
        all_audio = " ".join(s["text"] for s in speech)
        all_speakers = sorted({s["speaker"] for s in speech if s["speaker"]})
        add_node(root_id, "root", None, 0, max_time, summary, all_visual, all_audio,
                 entities, all_speakers, [])
        for section in sections:
            section_id = f"{fingerprint}:section:{section['index']}"
            sec_visual = " ".join(c["visual_text"] for c in section["children"])
            sec_audio = " ".join(c["audio_text"] for c in section["children"])
            sec_speakers = sorted({p for c in section["children"] for p in c["speakers"]})
            sec_entities = sorted({e for c in section["children"] for e in c["entities"]})
            sec_summary = _chunk_summary(sec_visual, sec_audio)
            add_node(section_id, "section", root_id, section["start"], section["end"], sec_summary,
                     sec_visual, sec_audio, sec_entities, sec_speakers, [])
            for chunk in section["children"]:
                chunk_id = f"{fingerprint}:chunk:{chunk['index']}"
                add_node(chunk_id, "chunk", section_id, chunk["start"], chunk["end"], chunk["summary"],
                         chunk["visual_text"], chunk["audio_text"], chunk["entities"], chunk["speakers"], chunk["evidence"])

        with self._connect() as conn:
            old = [r[0] for r in conn.execute("SELECT node_id FROM video_memory_nodes WHERE source_path=?", (source,))]
            if old:
                conn.executemany("DELETE FROM video_memory_fts WHERE node_id=?", [(node_id,) for node_id in old])
                conn.execute("DELETE FROM video_memory_nodes WHERE source_path=?", (source,))
            conn.executemany("""
                INSERT INTO video_memory_nodes
                (node_id,fingerprint,source_path,run_dir,kind,parent_id,start_sec,end_sec,summary,visual_text,audio_text,
                 entities_json,speakers_json,evidence_json,identity_links_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, node_rows)
            conn.executemany("""
                INSERT INTO video_memory_fts
                (node_id,fingerprint,source_path,kind,start_sec,end_sec,summary,visual_text,audio_text,entities,speakers)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, fts_rows)

        return {"source": source, "fingerprint": fingerprint, "chunks": len(buckets),
                "sections": len(sections), "speakers": all_speakers, "database": str(self.database)}

    def search(self, query: str, limit: int = 8, source_path: str | None = None) -> list[dict]:
        """Search 30-second evidence chunks with SQLite BM25 ranking."""
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        sql = """
            SELECT f.node_id,f.source_path,f.start_sec,f.end_sec,f.summary,f.visual_text,
                   f.audio_text,b.entities_json,b.speakers_json,b.evidence_json,b.identity_links_json,
                   b.run_dir,b.kind,bm25(video_memory_fts,0,0,0,0,0,0,4.0,3.0,1.0,1.5,1.5) AS rank
            FROM video_memory_fts f JOIN video_memory_nodes b ON b.node_id=f.node_id
            WHERE video_memory_fts MATCH ? AND f.kind='chunk'
        """
        params: list = [match]
        if source_path:
            sql += " AND f.source_path=?"
            params.append(str(Path(source_path).expanduser().resolve()))
        sql += " ORDER BY rank LIMIT ?"
        params.append(max(1, min(int(limit), 100)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{
            "source_path": r["source_path"], "start_seconds": float(r["start_sec"]),
            "end_seconds": float(r["end_sec"]), "summary": r["summary"],
            "visual_evidence": r["visual_text"], "speech_evidence": r["audio_text"],
            "entities": json.loads(r["entities_json"]), "speakers": json.loads(r["speakers_json"]),
            "speaker_to_visible_person": "unverified",
            "evidence": json.loads(r["evidence_json"]),
            "identity_links": json.loads(r["identity_links_json"]),
            "run_dir": r["run_dir"], "kind": r["kind"],
        } for r in rows]

    def status(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT kind,COUNT(*) AS n FROM video_memory_nodes GROUP BY kind").fetchall()
            sources = conn.execute("SELECT COUNT(DISTINCT fingerprint) FROM video_memory_nodes").fetchone()[0]
        counts = {row["kind"]: row["n"] for row in rows}
        return {"sources": sources, "root_nodes": counts.get("root", 0),
                "sections": counts.get("section", 0), "chunks": counts.get("chunk", 0),
                "database": str(self.database)}

    def list_sources(self) -> list[dict]:
        """List every indexed source with its summary and stored hierarchy size."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT root.source_path, root.run_dir, root.start_sec, root.end_sec,
                       root.summary, root.created_at,
                       SUM(CASE WHEN child.kind='section' THEN 1 ELSE 0 END) AS sections,
                       SUM(CASE WHEN child.kind='chunk' THEN 1 ELSE 0 END) AS chunks
                FROM video_memory_nodes AS root
                LEFT JOIN video_memory_nodes AS child
                  ON child.fingerprint=root.fingerprint
                WHERE root.kind='root'
                GROUP BY root.node_id
                ORDER BY root.created_at DESC, root.source_path COLLATE NOCASE
            """).fetchall()
        return [{
            "source_path": row["source_path"], "run_dir": row["run_dir"],
            "start_seconds": float(row["start_sec"]), "end_seconds": float(row["end_sec"]),
            "summary": row["summary"], "indexed_at": row["created_at"],
            "sections": int(row["sections"] or 0), "chunks": int(row["chunks"] or 0),
        } for row in rows]


def _chunk_summary(visual: str, audio: str) -> str:
    parts = []
    if visual:
        parts.append("Visual: " + visual[:500])
    if audio:
        parts.append("Speech: " + audio[:500])
    return " | ".join(parts) or "No textual evidence in this interval."
