"""Video downloading from URLs using yt-dlp (Python API, CLI fallback)."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Check if a string is a URL rather than a local file path."""
    return bool(re.match(r"https?://", path))


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    try:
        import yt_dlp

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}
    except Exception as api_err:
        logger.debug("yt_dlp API info failed, falling back to CLI: %s", api_err)

    cmd = ["yt-dlp", "--dump-json", "--no-download", "--no-warnings", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")
        import json

        return json.loads(result.stdout)
    except FileNotFoundError:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")


def download_video(
    url: str,
    output_dir: Optional[str] = None,
    max_height: int = 720,
    audio_only: bool = False,
    write_subs: bool = True,
    cookies_file: Optional[str] = None,
) -> dict:
    """Download video from URL.

    Uses the yt-dlp Python package when available (native), otherwise the
    ``yt-dlp`` CLI. Returns dict with path, title, duration, info, subs_file.
    """
    if output_dir is None:
        # Prefer stable Open Vision downloads dir; fall back to temp
        try:
            from storage.paths import downloads_dir

            output_dir = str(downloads_dir())
        except Exception:
            output_dir = tempfile.mkdtemp(prefix="openvision_dl_")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prefer video id for stable filenames (vault-friendly)
    outtmpl = str(output_path / "%(id)s.%(ext)s")

    try:
        return _download_via_api(
            url=url,
            outtmpl=outtmpl,
            output_path=output_path,
            max_height=max_height,
            audio_only=audio_only,
            write_subs=write_subs,
            cookies_file=cookies_file,
        )
    except Exception as api_err:
        logger.warning("yt_dlp Python API failed (%s); falling back to CLI", api_err)
        return _download_via_cli(
            url=url,
            outtmpl=outtmpl,
            output_path=output_path,
            max_height=max_height,
            audio_only=audio_only,
            write_subs=write_subs,
            cookies_file=cookies_file,
        )


def _download_via_api(
    url: str,
    outtmpl: str,
    output_path: Path,
    max_height: int,
    audio_only: bool,
    write_subs: bool,
    cookies_file: Optional[str],
) -> dict:
    import yt_dlp

    if audio_only:
        ydl_opts: dict = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }
            ],
        }
    else:
        ydl_opts = {
            "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/best",
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
        }

    if write_subs:
        ydl_opts.update(
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "srt/best",
                "convertsubs": "srt",
            }
        )
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    logger.info("Downloading via yt_dlp API: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True) or {}

    return _resolve_download_result(output_path, info, url)


def _download_via_cli(
    url: str,
    outtmpl: str,
    output_path: Path,
    max_height: int,
    audio_only: bool,
    write_subs: bool,
    cookies_file: Optional[str],
) -> dict:
    if audio_only:
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "64K",
            "-o",
            outtmpl,
        ]
    else:
        cmd = [
            "yt-dlp",
            "-f",
            f"bv*[height<={max_height}]+ba/b[height<={max_height}]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            outtmpl,
        ]

    if write_subs:
        cmd.extend(
            [
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang",
                "en",
                "--convert-subs",
                "srt",
            ]
        )
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    cmd.append(url)

    logger.info("Downloading via yt-dlp CLI: %s", url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp download failed: {result.stderr[:500]}")
    except FileNotFoundError:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timed out (10 min)")

    try:
        info = get_video_info(url)
    except Exception:
        info = {}
    return _resolve_download_result(output_path, info, url)


def _resolve_download_result(output_path: Path, info: dict, url: str) -> dict:
    downloaded_files = [f for f in output_path.glob("*") if f.is_file()]
    if not downloaded_files:
        raise RuntimeError("Download completed but no files found")

    # Prefer matching video id when known
    vid = info.get("id")
    video_files = [
        f
        for f in downloaded_files
        if f.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4a"}
    ]
    audio_files = [
        f
        for f in downloaded_files
        if f.suffix.lower() in {".mp3", ".wav", ".opus"}
    ]

    main_file = None
    if vid:
        for f in video_files + audio_files:
            if f.stem.startswith(vid) or f.stem == vid:
                main_file = f
                break
    if main_file is None:
        main_file = (
            video_files[0]
            if video_files
            else (audio_files[0] if audio_files else downloaded_files[0])
        )

    sub_files = list(output_path.glob("*.srt")) + list(output_path.glob("*.vtt"))
    if vid:
        preferred = [s for s in sub_files if vid in s.name]
        if preferred:
            sub_files = preferred
    subs_file = str(sub_files[0]) if sub_files else None

    return {
        "path": str(main_file),
        "title": info.get("title", main_file.stem),
        "duration": info.get("duration", 0) or 0,
        "info": info,
        "subs_file": subs_file,
        "id": vid or main_file.stem,
    }


def fetch_captions(url: str, lang: str = "en") -> Optional[str]:
    """Fetch captions from URL without downloading the video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pattern = os.path.join(tmpdir, "subs")
        try:
            import yt_dlp

            opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [lang],
                "subtitlesformat": "vtt/best",
                "outtmpl": out_pattern,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if vtt_files:
                return vtt_files[0].read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("API caption fetch failed: %s", e)

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--no-warnings",
            "--write-auto-sub",
            "--sub-lang",
            lang,
            "--convert-subs",
            "vtt",
            "-o",
            out_pattern,
            url,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if vtt_files:
            return vtt_files[0].read_text(encoding="utf-8", errors="replace")
        return None
