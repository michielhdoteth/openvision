"""Stable filesystem paths for Open Vision data (observations, downloads, runs).

Observations were previously written relative to the process CWD, so
``openvision observations list`` looked empty whenever the shell was in a
different directory from the observe run. Paths now resolve under
``OPENVISION_HOME`` (default ``~/.openvision``) unless overridden in config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_HOME_NAME = ".openvision"


def get_home(config: Optional[dict] = None) -> Path:
    """Return the Open Vision data home directory (created if missing).

    Priority:
    1. ``OPENVISION_HOME`` env var (or legacy ``OPHANIM_HOME`` for backward compat)
    2. ``paths.home`` in config
    3. ``~/.openvision``
    """
    env = os.environ.get("OPENVISION_HOME") or os.environ.get("OPHANIM_HOME")
    if env:
        home = Path(env).expanduser().resolve()
    else:
        cfg_home = None
        if config:
            cfg_home = (config.get("paths") or {}).get("home")
        if cfg_home:
            home = Path(cfg_home).expanduser().resolve()
        else:
            home = Path.home() / DEFAULT_HOME_NAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def _subpath(config: Optional[dict], key: str, default_name: str) -> Path:
    """Resolve a named data subdirectory under home (or absolute override)."""
    override = None
    if config:
        override = (config.get("paths") or {}).get(key)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = get_home(config) / path
    else:
        path = get_home(config) / default_name
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def observations_dir(config: Optional[dict] = None) -> Path:
    """Directory for saved observation markdown notes (``observations/videos``).

    Resolves to ``~/.openvision/observations/videos/`` by default.
    Migrates from the legacy ``~/.openvision/memory/`` path if present.
    """
    _migrate_memory_to_observations(config)

    base = _subpath(config, "observations", "observations")
    videos = base / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    return videos


def _migrate_memory_to_observations(config: Optional[dict] = None) -> None:
    """Migrate data from ~/.openvision/memory/ to ~/.openvision/observations/.

    Silent no-op if migration is not needed.
    """
    home = get_home(config)
    old_dir = home / "memory"
    new_base = home / "observations"
    if old_dir.exists() and not new_base.exists():
        import shutil
        new_base.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(new_base))





def downloads_dir(config: Optional[dict] = None) -> Path:
    """Directory for URL downloads via native yt-dlp."""
    return _subpath(config, "downloads", "downloads")


def runs_dir(config: Optional[dict] = None) -> Path:
    """Directory for observe run caches / frame artifacts."""
    return _subpath(config, "runs", "runs")


def run_cache_dir(config: Optional[dict] = None) -> Path:
    """Resolve the configured RunCache directory used by all CLI commands."""
    cache = (config or {}).get("cache") or {}
    configured = Path(cache.get("directory") or "runs").expanduser()
    if not configured.is_absolute():
        configured = runs_dir(config)
    configured.mkdir(parents=True, exist_ok=True)
    return configured.resolve()


def runs_db_path(config: Optional[dict] = None) -> Path:
    """Path to the SQLite runs database used by RunCache and video memory."""
    return run_cache_dir(config) / "runs.db"


def migrate_from_legacy() -> Optional[Path]:
    """Migrate data from ~/.ophanim to ~/.openvision if it exists.

    Returns the old path if migration was performed, None otherwise.
    """
    old_home = Path.home() / ".ophanim"
    new_home = Path.home() / ".openvision"

    if old_home.exists() and not new_home.exists():
        import shutil
        shutil.move(str(old_home), str(new_home))
        return old_home
    return None
