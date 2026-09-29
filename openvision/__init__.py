"""Open Vision - Self-hosted, privacy-first AI vision tool for CLI and agents."""

__version__ = "1.0.0"

from .api import process, transcribe, ProcessResult

__all__ = ["process", "transcribe", "ProcessResult"]
