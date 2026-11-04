"""openvision observe <path> - Observe video or image."""
import tempfile
import typer
import json
import os
import re
import cv2
from pathlib import Path
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

from core.video import probe, extract_frames, estimate_processing_cost, auto_fps, _downscale, detect_vfr, detect_color_range
from core.image import downscale, encode_base64, save_frame, load_image, is_image_file
from core.sampling import smart_sample
from core.gpu import auto_downgrade_mode, log_vram
from providers.registry import ProviderRegistry
from providers import get_provider, list_providers
from providers.parakeet import Transcript
from providers.base import VlmResponse, TokenUsage
from storage.cache import RunCache
from storage.config import load_config, get_mode_config
from openvision.models import (
    ObserveResult, ImageResult, TimelineEntry, ProbeResult,
)

console = Console()


def _fmt_time(seconds: float) -> str:
    """Format seconds to MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_tokens(tokens: TokenUsage) -> str:
    """Format token usage for display."""
    parts = [f"{tokens.prompt_tokens} prompt"]
    if tokens.reasoning_tokens > 0:
        parts.append(f"{tokens.completion_tokens} completion ({tokens.reasoning_tokens} reasoning)")
    else:
        parts.append(f"{tokens.completion_tokens} completion")
    return f"{tokens.total_tokens} total | " + " + ".join(parts)


def _transcript_from_captions(captions_file: str) -> Optional[Transcript]:
    """Load downloaded SRT/VTT captions into the shared transcript model."""
    caption_path = Path(captions_file)
    if not caption_path.is_file():
        return None
    from core.captions import parse_srt, parse_vtt
    from providers.parakeet import TranscriptSegment

    content = caption_path.read_text(encoding="utf-8", errors="replace")
    parser = parse_vtt if caption_path.suffix.lower() == ".vtt" else parse_srt
    captions = parser(content)
    if not captions:
        return None
    return Transcript(
        segments=[TranscriptSegment(start=c.start, end=c.end, text=c.text, confidence=1.0) for c in captions],
        language="en",
        duration_seconds=captions[-1].end,
    )


def _transcribe_or_use_captions(path: Path, provider_name: str, device: str,
                               captions_file: Optional[str] = None) -> Optional[Transcript]:
    """Prefer source captions; otherwise run the configured local STT model."""
    if captions_file:
        try:
            caption_transcript = _transcript_from_captions(captions_file)
            if caption_transcript and caption_transcript.segments:
                return caption_transcript
        except Exception as exc:
            console.print(f"[yellow]Could not use downloaded captions: {exc}[/yellow]")
    stt = get_provider(provider_name, {"device": device})
    try:
        return stt.transcribe(str(path))
    finally:
        close = getattr(stt, "close", None)
        if callable(close):
            close()


def _warn_transcription_skipped() -> None:
    """Note that a speech track was left untranscribed, so nothing looks incomplete.

    Transcription is on by default, so this only fires on an explicit
    ``--no-transcribe``. Kept low-key because the user asked for it.
    """
    console.print(
        "[dim]Audio track present but transcription was skipped "
        "(drop --no-transcribe, or add --transcribe, to include speech evidence).[/dim]"
    )


def _sampling_coverage(frames: list, duration: float) -> dict:
    """Which part of the source the sampled frames actually cover.

    Reported on every run because a frame budget that is silently spent on the
    first few minutes of a long video is otherwise indistinguishable from a
    complete analysis.
    """
    timestamps = [float(f["timestamp"]) for f in frames
                  if isinstance(f, dict) and f.get("timestamp") is not None]
    duration = float(duration or 0.0)
    if not timestamps or duration <= 0:
        return {"frames": len(timestamps), "duration_seconds": duration,
                "sampled_start_seconds": 0.0, "sampled_end_seconds": 0.0,
                "sampled_span_seconds": 0.0, "coverage_percent": 0.0}
    start, end = min(timestamps), max(timestamps)
    span = max(0.0, end - start)
    return {
        "frames": len(timestamps),
        "duration_seconds": duration,
        "sampled_start_seconds": round(start, 3),
        "sampled_end_seconds": round(end, 3),
        "sampled_span_seconds": round(span, 3),
        "coverage_percent": round(min(100.0, span / duration * 100.0), 1),
    }


def _fmt_coverage(coverage: dict) -> str:
    """One-line human summary of how much of the source was sampled."""
    return (f"Sampled span: {_fmt_time(coverage.get('sampled_start_seconds', 0))}-"
            f"{_fmt_time(coverage.get('sampled_end_seconds', 0))} of "
            f"{_fmt_time(coverage.get('duration_seconds', 0))} "
            f"({coverage.get('coverage_percent', 0.0):.0f}% coverage, "
            f"{coverage.get('frames', 0)} frames)")


def _diarize_transcript(path: Path, transcript: Optional[Transcript], enabled: bool) -> tuple[Optional[Transcript], dict]:
    """Best-effort speaker labeling; never treats speaker labels as visual identity."""
    if not enabled or not transcript or not transcript.segments:
        return transcript, {"status": "skipped", "reason": "disabled or no transcript segments"}
    tmp_path = None
    diarizer = None
    try:
        from providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
        from core.audio import extract_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        extract_audio(str(path), output_path=tmp_path)
        diarizer = DiarizerProvider()
        dia_result = diarizer.diarize(tmp_path)
        transcript.segments = merge_transcript_with_diarization(transcript.segments, dia_result.segments)
        return transcript, {"status": "completed", "speakers": dia_result.speakers}
    except ImportError as exc:
        console.print(f"[yellow]Diarization unavailable: {exc}[/yellow]")
        return transcript, {"status": "unavailable", "reason": str(exc)}
    except Exception as exc:
        console.print(f"[yellow]Diarization failed: {exc}[/yellow]")
        return transcript, {"status": "failed", "reason": str(exc)}
    finally:
        if diarizer is not None:
            close = getattr(diarizer, "close", None)
            if callable(close):
                close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Contact sheet grid ceiling: n x n cells at the default cell size must stay a
# readable image rather than an unusable mosaic.
MAX_GRID_SIZE = 12

# Subtitle extensions, in the precedence order they are tried.
SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass")
# The language/mark tag a yt-dlp style subtitle name puts between the video stem
# and the extension: a 2-4 letter language code, then optional script/region
# parts, optionally chained with dots (`en`, `en-US`, `en-orig`, `pt-BR`,
# `es-419`, `zh-Hans`, `orig.en`). Anchoring on this keeps an unrelated
# `clip.otherepisode.en-orig.vtt` from being borrowed by `clip.mp4`.
_SUBTITLE_TAG_PART = r"[A-Za-z]{2,4}(?:[-_][A-Za-z]{2,4}|[0-9]{3})*"
_SUBTITLE_TAG = re.compile(rf"^{_SUBTITLE_TAG_PART}(?:\.{_SUBTITLE_TAG_PART})*$")


def _subtitle_candidates(path: Path) -> list[Path]:
    """Subtitle files that belong to a video, in deterministic precedence order.

    Precedence, highest first:

    1. extension order: ``.srt``, then ``.vtt``, then ``.ass``
    2. within an extension, the bare ``<stem><ext>`` name
    3. then any ``<stem>.<language-tag>...<ext>`` variant, sorted by filename

    Only the bare name used to be considered, so a yt-dlp download such as
    ``workaholic.en-orig.vtt`` beside ``workaholic.mp4`` was invisible to
    ``--text-anchors`` and had to be renamed by hand. Bare names still win, so
    setups that already work are unchanged.
    """
    candidates: list[Path] = []
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        return candidates
    prefix = f"{path.stem}."
    for ext in SUBTITLE_EXTENSIONS:
        bare = path.with_suffix(ext)
        candidates.append(bare)
        variants = sorted(
            (p for p in siblings
             if p.is_file() and p.name.startswith(prefix)
             and p.suffix.lower() == ext and p.name != bare.name
             and _SUBTITLE_TAG.match(p.name[len(prefix):-len(ext)])),
            key=lambda p: p.name,
        )
        candidates.extend(variants)
    return candidates


def _subtitle_anchor_times(path: Path) -> list[float]:
    """Cue start times from the highest-precedence subtitle file for this video."""
    from core.captions import parse_srt, parse_vtt

    for sub_file in _subtitle_candidates(path):
        if not sub_file.is_file():
            continue
        content = sub_file.read_text(encoding="utf-8", errors="replace")
        if sub_file.suffix.lower() == ".srt":
            captions = parse_srt(content)
        else:
            captions = parse_vtt(content)
        # Use the start time of each cue as an anchor.
        return [c.start for c in captions if c.text.strip()]
    return []


def _transcript_data(transcript: Optional[Transcript]) -> list[dict]:
    if not transcript:
        return []
    return [segment.model_dump() if hasattr(segment, "model_dump") else {
        "start": segment.start, "end": segment.end, "text": segment.text,
        "confidence": getattr(segment, "confidence", None),
        "speaker": getattr(segment, "speaker", None),
    } for segment in transcript.segments]


def _visual_prompt(prompt: Optional[str], read_text: bool, default: str) -> str:
    """Add explicit, cautious text-reading instructions to the existing VLM pass."""
    base = prompt or default
    if not read_text:
        return base
    return (base + "\n\nAlso transcribe only clearly legible text visible in this frame. "
            "Quote it exactly under a 'Visible text:' label, preserve line breaks where useful, "
            "and mark uncertain characters as [unclear]. Do not guess obscured or unreadable text.")


def _parse_transcript_txt(transcript_file: Path) -> list:
    """Parse a saved transcript.txt back into timed segments.

    Accepts MM:SS and HH:MM:SS prefixes (transcripts over an hour used to be
    misread as minutes:seconds and land at wrong times in memory).
    """
    import re
    segments = []
    if not transcript_file.is_file():
        return segments
    for line in transcript_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\]\s+(?:\[([^]]+)\]\s+)?(.*)$", line)
        if match and match.group(5).strip():
            start = int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
            segments.append({"start": start, "end": start + 0.01,
                             "speaker": match.group(4), "text": match.group(5)})
    return segments


def _index_video_memory(cache, source_path: Path, run_dir: Path, result, transcript=None,
                       segmentation: Optional[dict] = None, duration_seconds: float = 0.0,
                       writer=None, deep: bool = False) -> dict:
    """Persist synchronized evidence for later local retrieval.

    deep=True (``--memory`` runs) also builds the whole-video profile the root
    memory node summarises from, upgraded with the decision tier when reachable.
    """
    try:
        from core.video_memory import VideoMemoryStore
        timeline = _value_for_memory(result, "timeline", [])
        summary = _value_for_memory(result, "summary", "")
        entities = _value_for_memory(result, "entities", [])
        transcript_segments = (list(transcript) if isinstance(transcript, list)
                               else list(transcript.segments) if transcript else [])
        if not transcript_segments:
            transcript_segments = _parse_transcript_txt(run_dir / "transcript.txt")
        if deep:
            profile = _video_profile(result, transcript_segments, duration_seconds, entities)
            synthesized = _synthesize_profile_summary(profile, summary)
            if synthesized:
                summary = synthesized
                profile["synthesized_summary"] = synthesized
            try:
                cache.save_artifact(run_dir, "video_profile.json", profile)
            except Exception:
                pass
        store = VideoMemoryStore(cache._db_path)
        indexed = store.build(str(source_path), run_dir, summary, timeline, entities,
                              transcript_segments, segmentation, duration_seconds)
        stage = {"status": "completed", **indexed}
        _set_memory_stage(result, stage, str(Path(source_path).expanduser().resolve()))
        cache.save_artifact(run_dir, "observations.json", _memory_serializable_result(result))
        cache.save_artifact(run_dir, "video_memory.json", indexed)
        if writer:
            writer.emit("stage", {"name": "memory", **stage})
            writer.emit("memory", indexed)
        return stage
    except Exception as exc:
        stage = {"status": "failed", "reason": str(exc)}
        _set_memory_stage(result, stage, str(Path(source_path).expanduser().resolve()))
        try:
            cache.save_artifact(run_dir, "observations.json", _memory_serializable_result(result))
        except Exception:
            pass
        if writer:
            writer.emit("stage", {"name": "memory", **stage})
        console.print(f"[yellow]Video memory indexing failed: {exc}[/yellow]")
        return stage


def _set_memory_stage(result, stage: dict, source_path: str) -> None:
    if isinstance(result, dict):
        stages = result.get("stages")
        if not isinstance(stages, dict):
            stages = {}
            result["stages"] = stages
        stages["memory"] = stage
        result["source_path"] = source_path
        return
    stages = getattr(result, "stages", None)
    if isinstance(stages, dict):
        stages["memory"] = stage
    if hasattr(result, "source_path"):
        result.source_path = source_path


def _video_profile(result, transcript_segments: list, duration_seconds: float,
                   entities: list) -> dict:
    """Whole-video snapshot: the global picture memory answers are built from.

    Deterministic only (no model call): counts, span, speakers, entities and
    top terms from the indexed evidence, so even raw-frames runs leave a real
    profile behind.
    """
    import re as _re
    from collections import Counter as _Counter

    timeline = _value_for_memory(result, "timeline", [])
    coverage = _value_for_memory(result, "sampling", {}) or {}
    texts = [str(_value_for_memory(entry, "observation", "") or "") for entry in timeline]
    words = _re.findall(r"[A-Za-z][\w'-]{2,}", " ".join(texts))
    top_terms = [w for w, _ in _Counter(w.casefold() for w in words).most_common(15)]
    speakers = sorted({str(s.get("speaker")) for s in transcript_segments if s.get("speaker")})
    visual = sum(1 for e in timeline if _value_for_memory(e, "modality", "") == "visual")
    audio = sum(1 for e in timeline if _value_for_memory(e, "modality", "") == "audio")
    return {
        "duration_seconds": duration_seconds,
        "timeline_entries": len(timeline),
        "visual_entries": visual,
        "audio_entries": audio,
        "speech_segments": len(transcript_segments),
        "speakers": speakers,
        "entities": list(entities or []),
        "top_terms": top_terms,
        "coverage_percent": coverage.get("coverage_percent"),
        "frames_sampled": coverage.get("frames", visual),
        "summary": _value_for_memory(result, "summary", ""),
    }


def _synthesize_profile_summary(profile: dict, base_summary: str) -> Optional[str]:
    """Ask the decision tier for a whole-video snapshot.

    Returns None (keeping the inherited summary) whenever the tier is missing,
    unreachable, or the answer is an error string — indexing never fails.
    """
    try:
        from providers.registry import ProviderRegistry
        from storage.config import load_config

        config = load_config()
        tier = dict(config.get("models", {}).get("decision")
                    or config.get("models", {}).get("vlm", {}))
        provider = ProviderRegistry.create(tier)
        try:
            if not provider.check_health():
                return None
            digest = (
                f"Video duration: {profile['duration_seconds']:.0f}s; "
                f"{profile['frames_sampled']} frames sampled "
                f"({profile['coverage_percent'] or 0:.0f}% coverage).\n"
                f"Observed summary: {str(base_summary or '')[:1500]}\n"
                f"Top terms: {', '.join(profile['top_terms'])}\n"
                f"Speakers: {', '.join(profile['speakers']) or 'unknown'}\n"
                f"Entities: {', '.join(profile['entities']) or 'none'}\n"
                f"Speech segments: {profile['speech_segments']}; "
                f"visual entries: {profile['visual_entries']}.\n\n"
                "Write a compact global picture of this video (3-6 sentences): what it "
                "is about, main subjects and actions, notable moments with timestamps, "
                "and what remains unverified. Use only the evidence above."
            )
            response = provider.query_text(digest)
        finally:
            provider.close()
        answer = (response.content or "").strip()
        if len(answer) < 20 or answer.lower().startswith(
                ("could not connect", "request to ", "unexpected error")):
            return None
        return answer
    except Exception:
        return None


def _memory_serializable_result(result):
    return result.model_dump() if hasattr(result, "model_dump") else result


def _value_for_memory(value, key: str, default):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


class _EntityResult:
    """Helper to carry entities + tokens from _extract_entities."""
    def __init__(self, entities: list[str], tokens: TokenUsage):
        self.entities = entities
        self.tokens = tokens


def observe_cmd(
    path: str = typer.Argument(..., help="Path to image or video file"),
    question: str = typer.Option(None, "--question", "-q", help="Specific question to answer (hardcoded prompt)"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Entirely custom prompt (overrides default)"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Processing mode: fast, balanced, detailed"),
    detail: str = typer.Option("balanced", "--detail", help="Extraction detail level: transcript, efficient, balanced, token-burner"),
    fps: float = typer.Option(None, "--fps", help="Frames per second to sample"),
    max_frames: int = typer.Option(None, "--max-frames", help="Maximum frames to process"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Override max tokens for VLM response"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, cuda"),
    start_time: str = typer.Option(None, "--start", help="Start time for focus range (e.g., 1:30, 45, 0:15:00)"),
    end_time: str = typer.Option(None, "--end", help="End time for focus range (e.g., 2:00, 90, 0:20:00)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Streaming JSONL output for agents/pipelines"),
    save_observations: bool = typer.Option(False, "--save-observations", help="Save observation as markdown ledger"),
    transcribe_audio: bool = typer.Option(True, "--transcribe", "-t", help="Transcribe audio when present (on by default; the first run downloads a local STT model)"),
    no_transcribe: bool = typer.Option(False, "--no-transcribe", help="Skip audio transcription"),
    stt_provider: str = typer.Option("parakeet", "--stt-provider", help=f"STT provider: {', '.join(list_providers())}"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Save full soundtrack as audio.m4a"),
    grid: bool = typer.Option(False, "--grid", help="Generate a contact sheet from keyframes"),
    grid_size: int = typer.Option(3, "--grid-size", help=f"Contact sheet grid dimensions, n x n (1-{MAX_GRID_SIZE})"),
    text_anchors: bool = typer.Option(False, "--text-anchors", help="Force frames at subtitle-cue timestamps"),
    viewer: bool = typer.Option(False, "--viewer", help="Generate local HTML viewer with keyframes + transcript"),
    report: bool = typer.Option(False, "--report", help="Generate keep/drop frame selection report"),
    dnn_model: str = typer.Option(None, "--dnn-model", help="ONNX model path for inline DNN inference during extraction"),
    cookies: str = typer.Option(None, "--cookies", help="Netscape cookie file for authenticated videos"),
    cookies_from_browser: str = typer.Option(None, "--cookies-from-browser", help="Read cookies from browser (chrome, firefox, edge, safari)"),
    diarize_audio: bool = typer.Option(True, "--diarize", help="Add speaker labels when the optional diarizer is available"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip speaker diarization"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocess (ignore cache)"),
    timestamps: str = typer.Option(None, "--timestamps", help="Extract frames at specific timestamps (comma-separated, e.g., '0:30,2:15,5:00')"),
    provider_name: str = typer.Option(None, "--provider", help="VLM provider: auto, lmstudio, llamacpp, gemini, openai, groq, together, vllm, localai"),
    segment: bool = typer.Option(None, "--segment", "-s", help="Run SAM segmentation on extracted frames (respects mode settings)"),
    read_text: bool = typer.Option(False, "--read-text", help="Ask the VLM to transcribe clearly legible text in images/frames"),
    memory_build: bool = typer.Option(False, "--memory", help="Build searchable audiovisual memory; transcribe/diarize, target one visual sample per 30s (max 300), and segment sampled frames"),
    raw_frames: bool = typer.Option(False, "--raw-frames", help="Skip VLM — return raw frame paths and audio timeline for vision-capable agents"),
    ground: Optional[str] = typer.Option(None, "--ground", "-g", help="Ground a query using LocateAnything-3B and merge results into the timeline"),
):
    """Analyze a video or image and return observations."""
    global console
    console = Console(stderr=json_output or jsonl)
    if json_output and jsonl:
        console.print("[red]Choose either --json or --jsonl, not both.[/red]")
        raise typer.Exit(code=2)
    # Flag validation first: a bad usage error should not look like a missing file.
    if not 1 <= grid_size <= MAX_GRID_SIZE:
        console.print(f"[red]Error:[/red] --grid-size must be between 1 and {MAX_GRID_SIZE} "
                      f"(got {grid_size}).")
        raise typer.Exit(code=2)
    from core.download import is_url
    input_path = Path(path)
    is_remote = is_url(path)
    if not is_remote and not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    config = load_config()
    mode_config = get_mode_config(config, mode)
    if memory_build and (raw_frames or detail == "transcript"):
        console.print("[red]--memory requires visual analysis; it cannot be combined with --raw-frames or --detail transcript.[/red]")
        raise typer.Exit(code=2)
    if memory_build:
        transcribe_audio = not no_transcribe
        diarize_audio = not no_diarize
        if segment is None:
            segment = True
    elif no_transcribe:
        transcribe_audio = False
    if no_diarize:
        diarize_audio = False
    if raw_frames and read_text:
        console.print("[red]--read-text requires VLM analysis and cannot be combined with --raw-frames.[/red]")
        raise typer.Exit(code=2)

    cookies_file = cookies
    owned_cookie_file = None
    if cookies_from_browser and not cookies_file:
        try:
            import browser_cookie3
            import tempfile as _tf
            cj = getattr(browser_cookie3, cookies_from_browser.replace("-", "_").lower())()
            cookie_tmp = _tf.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            cookie_tmp.write("# Netscape HTTP Cookie File\n")
            for c in cj:
                secure = "TRUE" if c.secure else "FALSE"
                domain = c.domain if c.domain.startswith(".") else "." + c.domain
                cookie_tmp.write(f"{domain}\tTRUE\t{c.path}\t{secure}\t{int(c.expires or 0)}\t{c.name}\t{c.value}\n")
            cookie_tmp.close()
            cookies_file = cookie_tmp.name
            owned_cookie_file = cookie_tmp.name
            console.print(f"[dim]Loaded cookies from {cookies_from_browser}[/dim]")
        except ImportError:
            console.print("[yellow]browser-cookie3 not installed. Install with: pip install browser-cookie3[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Failed to read browser cookies: {e}[/yellow]")

    # Download from URL if remote (native yt-dlp into openvision_HOME/downloads)
    dl_result = {}
    if is_remote:
        from core.download import download_video
        from storage.paths import downloads_dir
        dl_dir = downloads_dir(config)
        console.print(f"[dim]Downloading from URL into {dl_dir}...[/dim]")
        try:
            dl_result = download_video(path, output_dir=str(dl_dir), max_height=720, cookies_file=cookies_file)
            input_path = Path(dl_result["path"])
            dur = dl_result.get("duration") or 0
            console.print(
                f"[dim]Downloaded: {dl_result.get('title', input_path.name)} "
                f"({dur:.0f}s) -> {input_path}[/dim]"
            )
            if dl_result.get("subs_file"):
                console.print(f"[dim]Subtitles: {dl_result['subs_file']}[/dim]")
        except Exception as e:
            console.print(f"[red]Download failed:[/red] {e}")
            raise typer.Exit(code=1)
        finally:
            if owned_cookie_file:
                try:
                    os.unlink(owned_cookie_file)
                except FileNotFoundError:
                    pass
    elif owned_cookie_file:
        try:
            os.unlink(owned_cookie_file)
        except FileNotFoundError:
            pass

    actual_mode, warning = auto_downgrade_mode(mode, config)
    if warning:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    if actual_mode != mode:
        mode = actual_mode
        mode_config = get_mode_config(config, mode)

    # A still image is not a video: routing it through cv2.VideoCapture for a
    # duration probe prints decoder errors before the image path even runs.
    is_image = (not is_remote and is_image_file(input_path))

    actual_fps = fps or mode_config.get("fps", 0.5)
    actual_max_frames = max_frames or mode_config.get("max_frames", 60)
    # Balanced overviews stay concise for long-form material; explicit user
    # budgets and detailed mode remain authoritative.
    if not is_image and not max_frames and mode != "detailed" and detail == "balanced":
        try:
            if probe(str(input_path))["duration_seconds"] > 120:
                actual_max_frames = min(actual_max_frames, 12)
        except Exception:
            pass
    if memory_build:
        # Coverage floor: memory targets one visual sample per 30 s. An
        # explicit small --max-frames silently produced near-empty memory
        # hierarchies on long videos, so the floor now wins and is reported.
        try:
            duration = probe(str(input_path))["duration_seconds"]
            floor = min(300, int((duration + 29) // 30))
            if actual_max_frames < floor:
                console.print(
                    f"[yellow]--memory raised the frame budget {actual_max_frames} -> {floor} "
                    f"for full {duration:.0f}s coverage (one sample per 30s).[/yellow]"
                )
                actual_max_frames = floor
        except Exception:
            pass
    actual_resolution = mode_config.get("resolution", 768)

    focus_start = None
    focus_end = None
    if start_time:
        from core.video import parse_time
        focus_start = parse_time(start_time)
    if end_time:
        from core.video import parse_time
        focus_end = parse_time(end_time)
    has_focus = focus_start is not None or focus_end is not None

    cue_timestamps = None
    if timestamps:
        from core.video import parse_time
        try:
            cue_timestamps = [parse_time(t.strip()) for t in timestamps.split(",")]
        except ValueError as e:
            console.print(f"[red]Invalid timestamp: {e}[/red]")
            raise typer.Exit(code=1)

    if is_image:
        _handle_image(input_path, question, prompt, json_output, config, actual_resolution, max_tokens,
                      provider_name, jsonl, read_text)
    else:
        _handle_video(input_path, question, prompt, json_output, save_observations, transcribe_audio, diarize_audio, force,
                      config, mode, detail, actual_fps, actual_max_frames, actual_resolution, max_tokens, device,
                      focus_start, focus_end, cue_timestamps, provider_name, segment, raw_frames, ground, stt_provider,
                      keep_audio, grid, text_anchors, viewer, report, dnn_model, jsonl,
                      dl_result.get("subs_file"), memory_build, read_text,
                      no_transcribe, no_diarize, grid_size)


def _handle_image(path: Path, question: Optional[str], custom_prompt: Optional[str],
                  json_output: bool, config: dict, resolution: int,
                  override_max_tokens: Optional[int] = None,
                  provider_name: Optional[str] = None,
                  jsonl: bool = False,
                  read_text: bool = False):
    """Process a single image."""
    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_start(str(path))
        writer.emit_probe({"type": "image", "path": str(path)})
    image = load_image(str(path))
    image = downscale(image, resolution)

    vlm_config = config.get("models", {}).get("vlm", {})
    if override_max_tokens:
        vlm_config = {**vlm_config, "max_tokens": override_max_tokens}
    if provider_name:
        vlm_config["provider"] = provider_name

    try:
        provider = ProviderRegistry.create(vlm_config)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

    if not provider.check_health():
        provider_name_display = vlm_config.get("provider", "auto")
        console.print(f"[red]Error: {provider_name_display} is not running or not reachable.[/red]")
        console.print("Start your VLM provider and load a vision model, then try again.")
        raise typer.Exit(code=1)

    if not provider.supports_images():
        console.print(
            f"[red]Error:[/red] Model {provider.model_name} at "
            f"{getattr(provider, 'base_url', '')} does not accept image input. "
            f"Load a vision-capable model with its projector (mmproj), "
            f"or run with --raw-frames so a vision-capable agent can read the frames."
        )
        provider.close()
        raise typer.Exit(code=1)

    effective_prompt = None
    if custom_prompt:
        effective_prompt = _visual_prompt(custom_prompt, read_text, "Describe this image accurately and concisely.")
    elif question:
        effective_prompt = _visual_prompt(
            f"Answer this question about the image: {question}\nBe concise and specific.",
            read_text, "Describe this image accurately and concisely.")
    elif read_text:
        effective_prompt = _visual_prompt(None, True, "Describe important objects, people, and actions in this image.")

    try:
        response = provider.describe_image(image, effective_prompt)
        description = response.content
        tokens = response.tokens
    except Exception as e:
        console.print(f"[red]Error querying VLM:[/red] {e}")
        raise typer.Exit(code=1)
    finally:
        provider.close()

    result = ImageResult(
        description=description,
        objects=[],
        text_detected=_visible_text_lines(description) if read_text else [],
        confidence="medium",
        tokens={"prompt_tokens": tokens.prompt_tokens, "completion_tokens": tokens.completion_tokens, "reasoning_tokens": tokens.reasoning_tokens, "total_tokens": tokens.total_tokens},
        stages={"visual": {"status": "completed"},
                "text_reading": {"status": "completed" if read_text else "skipped",
                                 "method": "vlm_prompt" if read_text else None}},
    )

    if json_output:
        typer.echo(json.dumps(result.model_dump(), indent=2))
    elif writer:
        writer.emit("stage", {"name": "visual", **result.stages["visual"]})
        writer.emit("stage", {"name": "text_reading", **result.stages["text_reading"]})
        writer.emit("result", result.model_dump())
        writer.emit_done({"type": "image", "has_text": bool(result.text_detected)})
        writer.flush()
    elif not jsonl:
        console.print(Panel(description, title="[bold cyan]Image Observation[/bold cyan]"))
        console.print(f"[dim]Tokens: {_fmt_tokens(tokens)}[/dim]")


def _visible_text_lines(description: str) -> list[str]:
    """Read the explicit Visible text section emitted under --read-text."""
    import re
    match = re.search(r"(?im)^\s*Visible text:\s*(.*?)(?=^\s*[^\n:]{1,40}:|\Z)", description or "", re.S)
    if not match:
        return []
    value = match.group(1).strip()
    if not value or value.casefold() in {"none", "none visible", "no legible text", "[none]"}:
        return []
    return [line.strip(" -*\t") for line in value.splitlines() if line.strip(" -*\t")]


def _handle_audio_only(path: Path, metadata: dict, cache: RunCache, key: str,
                       transcribe: bool, diarize: bool, stt_provider: str,
                       device: str, subtitles_file: Optional[str], json_output: bool,
                       jsonl: bool, writer, memory_build: bool) -> None:
    """Run the compatible audio stages when a source has no video stream."""
    from openvision.models import ObserveResult, TimelineEntry
    run_dir = cache.create_run(key, metadata)
    transcript = None
    stages = {"visual": {"status": "skipped", "reason": "input contains no video stream"},
              "segmentation": {"status": "skipped", "reason": "input contains no video frames"},
              "grounding": {"status": "skipped", "reason": "input contains no video frames"}}
    if transcribe:
        try:
            transcript = _transcribe_or_use_captions(path, stt_provider, device, subtitles_file)
            stages["transcription"] = {"status": "completed" if transcript and transcript.segments else "empty"}
        except Exception as exc:
            stages["transcription"] = {"status": "failed", "reason": str(exc)}
            console.print(f"[yellow]Audio transcription failed: {exc}[/yellow]")
    else:
        stages["transcription"] = {"status": "skipped", "reason": "disabled"}
    transcript, stages["diarization"] = _diarize_transcript(path, transcript, diarize)
    timeline = [TimelineEntry(
        time_seconds=segment.start, timestamp=_fmt_time(segment.start),
        observation=f"[SPEECH] {'[' + segment.speaker + '] ' if segment.speaker else ''}{segment.text}",
        modality="audio") for segment in (transcript.segments if transcript else [])]
    summary = (f"Audio-only analysis: {len(timeline)} transcript segments."
               if timeline else "Audio-only input; no transcript was produced.")
    result = ObserveResult(
        summary=summary, timeline=timeline, entities=[], artifacts_dir=str(run_dir),
        confidence="medium" if timeline else "low", audio_segments=_transcript_data(transcript),
        stages=stages, duration_seconds=metadata.get("duration_seconds", 0),
        sampling=_sampling_coverage([], metadata.get("duration_seconds", 0)),
    )
    cache.save_artifact(run_dir, "observations.json", result.model_dump())
    _index_video_memory(cache, path, run_dir, result, transcript,
                        duration_seconds=metadata.get("duration_seconds", 0), writer=writer, deep=True)
    cache.save_text(run_dir, "summary.md", f"# Observation Summary\n\n{summary}\n")
    if transcript and transcript.segments:
        transcript_text = "\n".join(
            f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
            for s in transcript.segments)
        cache.save_text(run_dir, "transcript.txt", transcript_text)
    if writer:
        writer.emit("stage", {"name": "visual", **stages["visual"]})
        writer.emit("stage", {"name": "transcription", **stages["transcription"]})
        writer.emit("stage", {"name": "diarization", **stages["diarization"]})
        for segment in result.audio_segments:
            writer.emit_transcript(segment["start"], segment["end"], segment["text"], segment.get("speaker"))
        writer.emit("result", result.model_dump())
        writer.emit_summary(summary, [])
        writer.emit_done({"timeline_entries": len(timeline), "has_transcript": bool(timeline)})
        writer.flush()
    if json_output:
        typer.echo(json.dumps(result.model_dump(), indent=2))
    elif not jsonl:
        _display_observation(result.model_dump())


def _extract_focus_range(path: str, start: Optional[float], end: Optional[float],
                         max_frames: int, resolution: int) -> list[dict]:
    """Extract frames from a specific time range using ffmpeg seek."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["ffmpeg", "-i", path]
        if start is not None:
            cmd.extend(["-ss", str(start)])
        if end is not None:
            cmd.extend(["-to", str(end)])

        duration = (end or 0) - (start or 0)
        if duration > 0 and max_frames > 0:
            target_fps = min(6.0, max_frames / duration)
        else:
            target_fps = 1.0

        cmd.extend([
            "-vf", f"fps={target_fps:.2f}",
            "-vsync", "vfr",
            os.path.join(tmpdir, "frame_%06d.jpg"),
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: focus range extraction timed out[/yellow]")
            return []

        frames = []
        for i, fpath in enumerate(sorted(Path(tmpdir).glob("frame_*.jpg"))):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, resolution)
            ts = (start or 0) + i * (duration / max(1, len(list(Path(tmpdir).glob("frame_*.jpg")))))
            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _fmt_time(ts),
                "image": img,
                "reason": "focus-range",
            })

        return frames


def _handle_video(path: Path, question: Optional[str], custom_prompt: Optional[str],
                  json_output: bool, save_observations: bool, transcribe_audio: bool, diarize_audio: bool, force: bool,
                  config: dict, mode: str, detail: str = "balanced",
                  fps: float = 0.5, max_frames: int = 60, resolution: int = 768,
                  override_max_tokens: Optional[int] = None,
                   stt_device: str = "auto",
                  focus_start: Optional[float] = None,
                  focus_end: Optional[float] = None,
                  cue_timestamps: Optional[list[float]] = None,
                  provider_name: Optional[str] = None,
                  segment_flag: Optional[bool] = None,
                  raw_frames: bool = False,
                  ground_query: Optional[str] = None,
                  stt_provider: str = "parakeet",
                  keep_audio: bool = False,
                  generate_grid: bool = False,
                  text_anchors: bool = False,
                  generate_viewer: bool = False,
                  generate_report: bool = False,
                  dnn_model: str = None,
                  jsonl: bool = False,
                  subtitles_file: Optional[str] = None,
                  memory_build: bool = False,
                  read_text: bool = False,
                   no_transcribe: bool = False,
                   no_diarize: bool = False,
                   grid_size: int = 3,
                   base_url: Optional[str] = None):
    """Process audiovisual media through one cached, evidence-preserving pipeline."""
    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_start(str(path), mode=mode)

    from storage.paths import run_cache_dir
    cache_dir = str(run_cache_dir(config))
    cache = RunCache(cache_dir)
    mode_config = get_mode_config(config, mode)
    seg_mode = mode_config.get("segmentation", False)
    should_segment = segment_flag is True or (segment_flag is None and seg_mode is True)
    vlm_config = config.get("models", {}).get("vlm", {})
    if provider_name:
        vlm_config["provider"] = provider_name
    segmentation_config = config.get("models", {}).get("segmentation", {})
    cache_options = {
        "question": question, "prompt": custom_prompt, "detail": detail,
        "max_frames": max_frames, "focus_start": focus_start, "focus_end": focus_end,
        "cue_timestamps": cue_timestamps, "transcribe": transcribe_audio and not no_transcribe,
        "diarize": diarize_audio and not no_diarize, "read_text": read_text, "segment": should_segment,
        "ground": ground_query, "raw_frames": raw_frames, "memory": memory_build,
        "subtitles": (str(Path(subtitles_file).resolve()), Path(subtitles_file).stat().st_size,
                      Path(subtitles_file).stat().st_mtime_ns)
                     if subtitles_file and Path(subtitles_file).is_file() else None,
        "stt_provider": stt_provider, "stt_device": stt_device,
        "max_tokens": override_max_tokens, "save_observations": save_observations,
        "keep_audio": keep_audio, "grid": generate_grid, "grid_size": grid_size,
        "text_anchors": text_anchors,
        "viewer": generate_viewer, "report": generate_report, "dnn_model": dnn_model,
    "vlm": {key: (provider_name if key == "provider" and provider_name else vlm_config.get(key))
            for key in ("provider", "model")},
        "segmentation": {key: segmentation_config.get(key) for key in ("provider", "model", "fallback_model")},
    }
    key = cache.cache_key(str(path), mode, fps, resolution, options=cache_options)
    if force and key and cache.has_cached(key):
        import hashlib
        import time
        key = hashlib.sha256(f"{key}:{time.time_ns()}".encode()).hexdigest()[:16]

    if not force and key and cache.has_cached(key):
        cached_dir = cache.get_run(key)
        cached_file = cached_dir / "observations.json" if cached_dir else None
        if cached_file and cached_file.exists():
            import json as j
            data = j.loads(cached_file.read_text())
            _index_video_memory(cache, path, cached_dir, data,
                                transcript=data.get("audio_segments", []),
                                segmentation=data.get("segmentation"),
                                duration_seconds=data.get("duration_seconds", 0), writer=writer)
            if writer:
                for name, stage in data.get("stages", {}).items():
                    writer.emit("stage", {"name": name, **stage})
                writer.emit("result", data)
                writer.emit_summary(data.get("summary", ""), data.get("entities", []))
                writer.emit_done({"cached": True, "timeline_entries": len(data.get("timeline", []))})
                writer.flush()
            if json_output:
                typer.echo(j.dumps(data, indent=2))
            elif not jsonl:
                console.print("[green]Using cached result.[/green]")
                _display_observation(data)
            return

    log_vram("video_probe")
    video_meta = probe(str(path))
    if not video_meta.get("has_video", True):
        if video_meta.get("has_audio") and not transcribe_audio:
            _warn_transcription_skipped()
        _handle_audio_only(path, video_meta, cache, key, transcribe_audio and not no_transcribe,
                           diarize_audio and not no_diarize, stt_provider, stt_device,
                           subtitles_file, json_output, jsonl, writer, memory_build)
        return
    if not video_meta.get("has_audio") and not subtitles_file:
        transcribe_audio = False
    elif video_meta.get("has_audio") and not transcribe_audio:
        # Transcription is on by default, so a skipped speech track would
        # otherwise look accidentally incomplete.
        _warn_transcription_skipped()
    if writer:
        writer.emit_probe(video_meta)

    vfr_info = detect_vfr(str(path))
    vfr_mode = vfr_info.get("mode", "cfr")
    if vfr_mode == "vfr":
        console.print(f"[yellow]VFR detected: {vfr_info.get('variable_frames', 0)} variable / "
                      f"{vfr_info.get('constant_frames', 0)} constant frames "
                      f"({vfr_info.get('vfr_ratio', 0):.0%} VFR)[/yellow]")
        console.print("[dim]Using ffmpeg timestamp-based extraction for accurate sampling.[/dim]")

    color_info = detect_color_range(str(path))
    if color_info.get("range_type") != "unknown":
        console.print(f"[dim]Color range: {color_info['range_type']}[/dim]")

    # Smart frame budget based on duration
    duration = video_meta["duration_seconds"]
    auto_max = auto_fps(duration)
    actual_max = min(max_frames, max(auto_max, max_frames if memory_build else 0))
    console.print(f"[dim]Video: {duration:.0f}s, budget: {actual_max} frames[/dim]")

    has_focus = focus_start is not None or focus_end is not None
    if not has_focus:
        cost = estimate_processing_cost(video_meta, mode)
        if cost == "high":
            console.print(f"[yellow]Video is long ({duration:.0f}s). "
                          f"Use --start/--end to focus on a specific section.[/yellow]")

    log_vram("frame_sampling")
    has_focus = focus_start is not None or focus_end is not None

    if has_focus:
        from core.video import auto_fps_focus
        focus_duration = (focus_end or duration) - (focus_start or 0)
        focus_max = min(max_frames, auto_fps_focus(focus_duration))
        console.print(f"[dim]Focus range: {_fmt_time(focus_start or 0)} - {_fmt_time(focus_end or duration)} ({focus_duration:.0f}s, budget: {focus_max} frames)[/dim]")
        frames = _extract_focus_range(str(path), focus_start, focus_end, focus_max, resolution)
    elif detail == "transcript":
        console.print("[dim]Transcript-only mode: skipping frame extraction[/dim]")
        frames = []
    elif detail == "efficient":
        from core.video import extract_keyframes_ffmpeg, dedupe_frames_ffmpeg
        console.print("[dim]Efficient mode: extracting keyframes (cap 50)[/dim]")
        frames = extract_keyframes_ffmpeg(str(path), max_frames=50, max_resolution=resolution)
        frames = dedupe_frames_ffmpeg(frames)
    elif detail == "token-burner":
        from core.video import extract_scene_frames_ffmpeg, dedupe_frames_ffmpeg
        console.print("[dim]Token-burner mode: full scene detection (uncapped)[/dim]")
        frames = extract_scene_frames_ffmpeg(str(path), max_frames=9999, max_resolution=resolution)
        frames = dedupe_frames_ffmpeg(frames)
    else:
        # balanced (default)
        if duration > 120:
            from core.sampling import adaptive_sample
            frames = adaptive_sample(str(path), max_frames=actual_max, max_resolution=resolution, vfr_mode=vfr_mode)
        else:
            frames = smart_sample(str(path), fps=fps, max_frames=actual_max, max_resolution=resolution, vfr_mode=vfr_mode)

    if dnn_model and frames:
        try:
            from core.dnn_filter import DNNFilterPipeline
            dnn = DNNFilterPipeline(dnn_model)
            if dnn.is_available:
                console.print(f"[dim]Running inline DNN detection: {dnn_model}[/dim]")
                dnn_results = dnn.extract_with_detection(str(path), max_frames=len(frames))
                for dnn_frame in dnn_results:
                    idx = dnn_frame["frame_index"]
                    if idx < len(frames):
                        frames[idx]["detections"] = dnn_frame.get("detections", [])
                        frames[idx]["frame_path_dnn"] = dnn_frame.get("frame_path")
                console.print(f"[dim]DNN: processed {len(dnn_results)} frames[/dim]")
            else:
                console.print("[yellow]FFmpeg DNN/ONNX Runtime not available, skipping inline inference[/yellow]")
        except Exception as e:
            console.print(f"[yellow]DNN filter failed: {e}[/yellow]")

    # Transcript-only early return: skip VLM, just run audio processing
    if detail == "transcript" and not frames:
        run_dir = cache.create_run(key, video_meta)
        transcript = None
        timeline = []
        stages = {"visual": {"status": "skipped", "reason": "transcript-only mode"}}

        if transcribe_audio:
            try:
                console.print(f"[dim]Transcribing audio with {stt_provider.title()} ({stt_device})...[/dim]")
                transcript = _transcribe_or_use_captions(path, stt_provider, stt_device, subtitles_file)
                stages["transcription"] = {"status": "completed" if transcript and transcript.segments else "empty"}
            except Exception as exc:
                stages["transcription"] = {"status": "failed", "reason": str(exc)}
                console.print(f"[yellow]Audio transcription failed: {exc}[/yellow]")
        else:
            stages["transcription"] = {"status": "skipped", "reason": "disabled"}

        transcript, stages["diarization"] = _diarize_transcript(path, transcript, diarize_audio)
        stages["segmentation"] = {"status": "skipped", "reason": "transcript-only mode"}
        stages["grounding"] = {"status": "skipped", "reason": "transcript-only mode"}

        if transcript and transcript.segments:
            for seg in transcript.segments:
                ts = _fmt_time(seg.start)
                speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
                timeline.append(TimelineEntry(
                    time_seconds=seg.start,
                    timestamp=ts,
                    observation=f"[SPEECH] {speaker_prefix}{seg.text}",
                    frame_path=None,
                    modality="audio",
                ))

        if timeline:
            summary = f"Transcript-only analysis: {len(transcript.segments)} speech segments transcribed."
        else:
            summary = "Transcript-only mode: no audio transcribed."

        result = ObserveResult(
            summary=summary,
            timeline=timeline,
            entities=[],
            artifacts_dir=str(run_dir),
            confidence="medium" if transcript and transcript.segments else "low",
            tokens={"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
            audio_segments=_transcript_data(transcript), stages=stages,
            duration_seconds=video_meta.get("duration_seconds", 0),
            sampling=_sampling_coverage([], video_meta.get("duration_seconds", 0)),
        )

        cache.save_artifact(run_dir, "observations.json", result.model_dump())
        _index_video_memory(cache, path, run_dir, result, transcript,
                            duration_seconds=video_meta.get("duration_seconds", 0), writer=writer, deep=True)
        cache.save_text(run_dir, "summary.md", f"# Observation Summary\n\n{summary}\n")

        timeline_md = "# Timeline\n\n"
        for entry in timeline:
            timeline_md += f"- **{entry.timestamp}** - {entry.observation}\n"
        cache.save_text(run_dir, "timeline.md", timeline_md)

        if transcript and transcript.segments:
            transcript_text = "\n".join(
                f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
                for s in transcript.segments
            )
            cache.save_text(run_dir, "transcript.txt", transcript_text)
            if writer:
                for seg in transcript.segments:
                    writer.emit_transcript(seg.start, seg.end, seg.text, getattr(seg, "speaker", None))
        if writer:
            for name, stage in result.stages.items():
                writer.emit("stage", {"name": name, **stage})
            writer.emit("sampling", result.sampling)
            writer.emit_summary(summary, [])
            writer.emit("result", result.model_dump())
            writer.emit_done({"timeline_entries": len(timeline), "has_transcript": bool(transcript and transcript.segments)})
            writer.flush()

        if save_observations:
            _save_observation_md(path, result, config, transcript)

        log_vram("observation_complete")

        if json_output:
            typer.echo(json.dumps(result.model_dump(), indent=2))
        elif not jsonl:
            _display_observation(result.model_dump())
        return

    if not frames:
        console.print("[red]No frames extracted from video.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Extracted {len(frames)} frames for analysis[/dim]")

    if generate_grid and frames:
        try:
            from core.contact_sheet import create_contact_sheet
            grid_path = str(path.parent / f"{path.stem}_contact_sheet.jpg")
            create_contact_sheet(frames, grid_size=(grid_size, grid_size), output_path=grid_path)
            console.print(f"[dim]Contact sheet ({grid_size}x{grid_size}): {grid_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Contact sheet failed: {e}[/yellow]")

    pinned_frames = []
    if cue_timestamps:
        from core.video import extract_at_timestamps
        pinned_frames = extract_at_timestamps(str(path), cue_timestamps, resolution)

    if pinned_frames:
        existing_times = {round(f["timestamp"], 1) for f in frames}
        for pf in pinned_frames:
            if round(pf["timestamp"], 1) not in existing_times:
                frames.append(pf)
                existing_times.add(round(pf["timestamp"], 1))
        frames.sort(key=lambda f: f["timestamp"])

    # Text-anchors: force frames at subtitle-cue timestamps
    if text_anchors:
        try:
            from core.video import extract_at_timestamps
            # Subtitle timestamps from downloaded subs or captions beside the video,
            # including yt-dlp language/mark names such as <stem>.en-orig.vtt.
            anchor_times = _subtitle_anchor_times(path)
            if anchor_times:
                if len(anchor_times) > 30:
                    step = len(anchor_times) / 30
                    anchor_times = [anchor_times[int(i * step)] for i in range(30)]
                anchor_frames = extract_at_timestamps(str(path), anchor_times, resolution)
                existing_times = {round(f["timestamp"], 1) for f in frames}
                added = 0
                for af in anchor_frames:
                    if round(af["timestamp"], 1) not in existing_times:
                        frames.append(af)
                        existing_times.add(round(af["timestamp"], 1))
                        added += 1
                if added:
                    frames.sort(key=lambda f: f["timestamp"])
                    console.print(f"[dim]Text-anchors: added {added} frames at subtitle cues[/dim]")
            else:
                console.print(f"[yellow]Text-anchors: no subtitle file found for "
                              f"{path.name} (looked for {', '.join(c.name for c in _subtitle_candidates(path))})[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Text-anchors failed: {e}[/yellow]")

    # Report what the sample actually covers. Frames are final here (pinned cues
    # and text-anchors merged), so this is the span the VLM will see.
    coverage = _sampling_coverage(frames, duration)
    if not json_output and not jsonl:
        console.print(f"[dim]{_fmt_coverage(coverage)}[/dim]")

    if duration > 600 and len(frames) < 30:
        console.print(f"[yellow]Warning: {duration/60:.0f}-minute video with only {len(frames)} frames. "
                      f"Consider --start/--end to focus, or --detail token-burner for full scene detection.[/yellow]")

    run_dir = cache.create_run(key, video_meta)

    frame_paths = []
    for i, frame in enumerate(frames):
        fname = f"frame_{i:04d}.jpg"
        fpath = cache.save_frame(run_dir, frame["image"], fname)
        frame_paths.append(str(fpath))
        frame["path"] = str(fpath)
        if writer:
            writer.emit_frame(i, frame.get("timestamp", 0), str(fpath), frame.get("detections"))

    if generate_report and frames:
        try:
            from core.report import generate_keep_drop_report
            report_path = str(run_dir / "report.html")
            generate_keep_drop_report(
                frames, list(range(len(frames))),
                dedup_stats={"total_extracted": len(frames), "used_for_analysis": len(frames)},
                output_path=report_path,
                title=f"Frame Report - {path.stem}",
            )
            console.print(f"[dim]Report: {report_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Report failed: {e}[/yellow]")

    transcript = None
    transcription_stage = None
    if raw_frames:
        from openvision.models import RawFrame, TimelineEntry
        if transcribe_audio and subtitles_file and Path(subtitles_file).is_file():
            try:
                transcript = _transcript_from_captions(subtitles_file)
            except Exception as e:
                console.print(f"[yellow]Could not use downloaded captions: {e}[/yellow]")
        if transcribe_audio and transcript is None:
            try:
                transcript = _transcribe_or_use_captions(path, stt_provider, stt_device)
            except Exception as e:
                # Report the failure in the stage, not as a silent empty transcript.
                transcription_stage = {"status": "failed", "reason": str(e)}
                console.print(f"[yellow]Audio transcription failed: {e}[/yellow]")
        transcript, diarization_stage = _diarize_transcript(path, transcript, diarize_audio)
        raw = [
            RawFrame(index=i, timestamp=f["timestamp"], path=f["path"])
            for i, f in enumerate(frames)
        ]

        # entries carry the frame path (marked unanalyzed) so indexed speech
        # can be matched to frames later, instead of a speech-only memory.
        timeline: list[TimelineEntry] = []
        if transcript and transcript.segments:
            for seg in transcript.segments:
                timeline.append(TimelineEntry(
                    time_seconds=seg.start,
                    timestamp=_fmt_time(seg.start),
                    observation=seg.text,
                    speaker=getattr(seg, "speaker", None),
                    modality="audio",
                ))
        for i, f in enumerate(frames):
            timeline.append(TimelineEntry(
                time_seconds=float(f["timestamp"]),
                timestamp=_fmt_time(f["timestamp"]),
                observation="[FRAME] not yet described",
                frame_path=f["path"],
                modality="visual",
            ))
        timeline.sort(key=lambda entry: entry.time_seconds)

        result = ObserveResult(
            summary=f"Raw frames extracted: {len(frames)} frames from {path.name}",
            timeline=timeline,
            entities=[],
            artifacts_dir=str(run_dir),
            confidence="medium" if frames else "low",
            raw_frames=raw,
            audio_segments=_transcript_data(transcript),
            stages={"visual": {"status": "skipped", "reason": "raw-frames mode"},
                    "transcription": transcription_stage or {
                        "status": "completed" if transcript and transcript.segments
                        else "empty" if transcribe_audio else "skipped"},
                    "diarization": diarization_stage,
                    "segmentation": {"status": "skipped", "reason": "raw-frames mode"},
                    "grounding": {"status": "skipped", "reason": "raw-frames mode"}},
            duration_seconds=video_meta.get("duration_seconds", 0),
            sampling=coverage,
        )

        cache.save_artifact(run_dir, "observations.json", result.model_dump())
        _index_video_memory(cache, path, run_dir, result, transcript,
                            duration_seconds=video_meta.get("duration_seconds", 0), writer=writer, deep=True)
        if writer:
            for name, stage in result.stages.items():
                writer.emit("stage", {"name": name, **stage})
            writer.emit("sampling", coverage)
            writer.emit("result", result.model_dump())
            writer.emit_summary(result.summary, [])
            writer.emit_done({"raw_frames": len(raw), "transcript_segments": len(timeline),
                              "coverage_percent": coverage["coverage_percent"]})
            writer.flush()

        if save_observations:
            _save_observation_md(path, result, config, transcript)

        if json_output:
            typer.echo(json.dumps(result.model_dump(), indent=2))
        elif not jsonl:
            console.print(f"[green]Extracted {len(frames)} raw frames to {run_dir}[/green]")
            if transcript and transcript.segments:
                console.print(f"[dim]Audio timeline: {len(transcript.segments)} segments with speaker labels[/dim]")
            console.print(f"[dim]{_fmt_coverage(coverage)}[/dim]")
            console.print("[dim]Ready for vision-capable agent processing.[/dim]")
        return

    from openvision.models import TimelineEntry

    vlm_config = config.get("models", {}).get("vlm", {})
    if override_max_tokens:
        vlm_config = {**vlm_config, "max_tokens": override_max_tokens}
    if provider_name:
        vlm_config["provider"] = provider_name

    provider = None
    visual_stage = {"status": "completed"}
    timeline = []
    total_tokens = TokenUsage()
    try:
        provider = ProviderRegistry.create(vlm_config)
        if not provider.check_health():
            raise RuntimeError(f"{vlm_config.get('provider', 'VLM')} is not reachable")
    except Exception as exc:
        visual_stage = {"status": "failed", "reason": str(exc)}
        console.print(f"[yellow]Visual analysis unavailable: {exc}[/yellow]")

    if provider:
        log_vram("vlm_observation")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(description=f"Analyzing {len(frames)} frames with VLM...", total=len(frames))
            failed_frames = 0
            for i, frame in enumerate(frames):
                progress.update(task, advance=1, description=f"Frame {i+1}/{len(frames)} at {frame['timestamp_str']}")
                default_prompt = "Describe what is happening in this frame. Focus on objects, people, and actions. Be concise (1-2 sentences)."
                if question:
                    default_prompt = f"Answer this question about the image: {question}\nBe concise and specific."
                frame_prompt = _visual_prompt(custom_prompt, read_text, default_prompt)
                try:
                    resp = provider.describe_image(frame["image"], frame_prompt)
                    total_tokens.prompt_tokens += resp.tokens.prompt_tokens
                    total_tokens.completion_tokens += resp.tokens.completion_tokens
                    total_tokens.reasoning_tokens += resp.tokens.reasoning_tokens
                    total_tokens.total_tokens += resp.tokens.total_tokens
                    content = (resp.content or "").strip()
                    if content and not content.lower().startswith(("could not connect", "request to ", "unexpected error contacting")) and " returned http " not in content.lower():
                        timeline.append(TimelineEntry(
                            time_seconds=frame["timestamp"], timestamp=frame["timestamp_str"],
                            observation=content, frame_path=frame_paths[i] if i < len(frame_paths) else None))
                    else:
                        failed_frames += 1
                except Exception as exc:
                    failed_frames += 1
                    console.print(f"[yellow]VLM failed on frame {frame['timestamp_str']}: {exc}[/yellow]")
            if failed_frames:
                visual_stage = {"status": "partial" if timeline else "failed", "failed_frames": failed_frames}
            if timeline and visual_stage["status"] == "completed":
                visual_stage["frames"] = len(timeline)

    timeline_text = "\n".join(f"{t.timestamp}: {t.observation}" for t in timeline)
    entities_resp = _extract_entities(provider, timeline_text) if provider and timeline_text.strip() else _EntityResult([], TokenUsage())
    total_tokens.prompt_tokens += entities_resp.tokens.prompt_tokens
    total_tokens.completion_tokens += entities_resp.tokens.completion_tokens
    total_tokens.reasoning_tokens += entities_resp.tokens.reasoning_tokens
    total_tokens.total_tokens += entities_resp.tokens.total_tokens

    if not timeline:
        summary = "No reliable visual observations were returned by the selected model. Check model output and frame decoding, then retry."
    elif question:
        summary = _build_qa_summary(timeline, question)
    elif provider:
        try:
            summary_resp = _generate_summary(provider, timeline)
            summary = summary_resp.content
            total_tokens.prompt_tokens += summary_resp.tokens.prompt_tokens
            total_tokens.completion_tokens += summary_resp.tokens.completion_tokens
            total_tokens.reasoning_tokens += summary_resp.tokens.reasoning_tokens
            total_tokens.total_tokens += summary_resp.tokens.total_tokens
        except Exception as exc:
            summary = f"Visual observations collected from {len(timeline)} frames; summary generation failed: {exc}"
            visual_stage = {"status": "partial", "reason": "summary generation failed", "frames": len(timeline)}
    else:
        summary = "Visual model unavailable; returning any successfully processed audio evidence."

    if provider:
        provider.close()

    transcript = None
    transcription_stage = {"status": "skipped", "reason": "no audio stream or disabled"}
    if transcribe_audio:
        try:
            console.print(f"[dim]Transcribing audio with {stt_provider.title()} ({stt_device})...[/dim]")
            transcript = _transcribe_or_use_captions(path, stt_provider, stt_device, subtitles_file)
            transcription_stage = {"status": "completed" if transcript and transcript.segments else "empty"}
            if transcript and transcript.segments:
                console.print(f"[dim]Transcribed {len(transcript.segments)} speech segments[/dim]")
        except Exception as exc:
            transcription_stage = {"status": "failed", "reason": str(exc)}
            console.print(f"[yellow]Audio transcription failed: {exc}[/yellow]")
    transcript, diarization_stage = _diarize_transcript(path, transcript, diarize_audio)

    if transcript and transcript.segments:
        audio_entries = []
        for seg in transcript.segments:
            ts = _fmt_time(seg.start)
            speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
            audio_entries.append(TimelineEntry(
                time_seconds=seg.start,
                timestamp=ts,
                observation=f"[SPEECH] {speaker_prefix}{seg.text}",
                frame_path=None,
                modality="audio",
            ))

        all_entries = timeline + audio_entries
        all_entries.sort(key=lambda e: e.time_seconds)
        timeline = all_entries

    segmentation_data = {"status": "skipped", "reason": "disabled by mode/options", "frames_processed": 0, "objects": []}
    if should_segment and frames:
        segmentation_data = _run_segmentation(path, frames, config, mode, run_dir,
                                              entities_resp.entities, memory_build)

    if isinstance(segmentation_data, dict):
        for obj in segmentation_data.get("objects", []):
            time_seconds = float(obj.get("time_seconds", 0) or 0)
            nearest = min(frames, key=lambda frame: abs(float(frame["timestamp"]) - time_seconds)) if frames else None
            label = obj.get("label") or "object"
            score = float(obj.get("score", 0) or 0)
            timeline.append(TimelineEntry(
                time_seconds=time_seconds, timestamp=_fmt_time(time_seconds),
                observation=f"[SEGMENT] {label} ({score:.0%})",
                frame_path=nearest.get("path") if nearest else None,
                modality="segmentation",
                boxes=[{"bbox": obj.get("bbox", []), "label": label, "score": score,
                        "mask_path": obj.get("mask_path")}],
            ))

    if ground_query:
        grounding_stage = _run_grounding(ground_query, frames, timeline, run_dir, config)
    else:
        grounding_stage = {"status": "skipped", "reason": "not requested"}
    timeline.sort(key=lambda entry: entry.time_seconds)

    result = ObserveResult(
        summary=summary,
        timeline=timeline,
        entities=entities_resp.entities,
        artifacts_dir=str(run_dir),
        confidence="medium" if timeline else "low",
        tokens={"prompt_tokens": total_tokens.prompt_tokens, "completion_tokens": total_tokens.completion_tokens, "reasoning_tokens": total_tokens.reasoning_tokens, "total_tokens": total_tokens.total_tokens},
        audio_segments=_transcript_data(transcript),
        segmentation=segmentation_data,
        stages={"visual": visual_stage, "transcription": transcription_stage,
                "diarization": diarization_stage,
                "segmentation": {key: value for key, value in segmentation_data.items() if key != "objects"},
                "grounding": grounding_stage},
        duration_seconds=video_meta.get("duration_seconds", 0),
        sampling=coverage,
    )

    cache.save_artifact(run_dir, "observations.json", result.model_dump())
    _index_video_memory(cache, path, run_dir, result, transcript, segmentation_data,
                        video_meta.get("duration_seconds", 0), writer, deep=memory_build)
    cache.save_text(run_dir, "summary.md", f"# Observation Summary\n\n{summary}\n")

    timeline_md = "# Timeline\n\n"
    for entry in timeline:
        timeline_md += f"- **{entry.timestamp}** - {entry.observation}\n"
    cache.save_text(run_dir, "timeline.md", timeline_md)

    if transcript and transcript.segments:
        transcript_text = "\n".join(
            f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
            for s in transcript.segments
        )
        cache.save_text(run_dir, "transcript.txt", transcript_text)
        if writer:
            for seg in transcript.segments:
                writer.emit_transcript(seg.start, seg.end, seg.text, getattr(seg, "speaker", None))

    if keep_audio:
        try:
            from core.audio import save_full_audio
            audio_out = str(run_dir / "audio.m4a")
            save_full_audio(str(path), audio_out)
            console.print(f"[dim]Audio saved: {audio_out}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to save audio: {e}[/yellow]")

    if generate_viewer and frames:
        try:
            from core.viewer import generate_viewer
            viewer_path = str(run_dir / "viewer.html")
            t_segments = transcript.segments if transcript and transcript.segments else None
            generate_viewer(frames, t_segments, summary, viewer_path, title=path.stem)
            console.print(f"[dim]Viewer: {viewer_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Viewer failed: {e}[/yellow]")

    if save_observations:
        _save_observation_md(path, result, config, transcript)

    if writer:
        for name, stage in result.stages.items():
            writer.emit("stage", {"name": name, **stage})
        writer.emit("sampling", coverage)
        writer.emit("result", result.model_dump())
        writer.emit_summary(result.summary, [str(entity) for entity in result.entities])
        writer.emit_done({"timeline_entries": len(result.timeline), "entities": len(result.entities),
                          "has_transcript": bool(transcript and transcript.segments),
                          "coverage_percent": coverage["coverage_percent"]})
        writer.flush()

    log_vram("observation_complete")

    if json_output:
        typer.echo(json.dumps(result.model_dump(), indent=2))
    elif not jsonl:
        _display_observation(result.model_dump())


def _generate_summary(provider, timeline: list) -> VlmResponse:
    """Use VLM to compress the timeline into a concise summary."""
    if not timeline:
        return VlmResponse(content="No observations recorded.", tokens=TokenUsage())

    timeline_text = "\n".join(
        f"{t.timestamp}: {t.observation}" for t in timeline
    )
    # Truncate to ~700 chars to avoid empty responses from LM Studio
    if len(timeline_text) > 700:
        timeline_text = timeline_text[:700] + "..."

    prompt = (
        "Compress this video timeline into 2-3 concise sentences describing "
        "what happened. Focus on key events, objects, and changes.\n\n"
        f"{timeline_text}"
    )

    try:
        resp = provider.query_text(prompt)
        if resp.content and not resp.content.startswith("[VLM"):
            return resp
        return _build_timeline_summary_fallback(timeline)
    except Exception:
        return _build_timeline_summary_fallback(timeline)


def _build_timeline_summary_fallback(timeline: list) -> VlmResponse:
    """Fallback summary if VLM summarization fails."""
    observations = [t.observation for t in timeline if t.observation]
    if not observations:
        return VlmResponse(content="No observations recorded.", tokens=TokenUsage())
    if len(observations) == 1:
        return VlmResponse(content=observations[0], tokens=TokenUsage())
    return VlmResponse(
        content=f"The video shows: {observations[0]} Towards the end, {observations[-1]}",
        tokens=TokenUsage(),
    )


def _extract_entities(provider, timeline_text: str) -> _EntityResult:
    """Use VLM to extract real entities from timeline descriptions."""
    # Truncate to ~700 chars to fit model context limit
    if len(timeline_text) > 700:
        timeline_text = timeline_text[:700] + "..."

    prompt = (
        "From these video observations, extract a comma-separated list of the "
        "main objects, people, and entities visible. Return ONLY the comma-separated list, no explanation.\n\n"
        f"{timeline_text}"
    )
    try:
        resp = provider.query_text(prompt)
        if resp.content and not resp.content.startswith("[VLM"):
            entities = [e.strip().lower() for e in resp.content.split(",") if e.strip()]
            return _EntityResult(entities=entities[:20], tokens=resp.tokens)
        return _EntityResult(entities=[], tokens=TokenUsage())
    except Exception:
        return _EntityResult(entities=[], tokens=TokenUsage())


def _build_qa_summary(timeline: list, question: str) -> str:
    """Compile answers to a specific question across timeline."""
    if not timeline:
        return "No frames available to answer the question."

    answers = [t.observation for t in timeline if t.observation and len(t.observation) > 5]

    if not answers:
        return "Could not determine answer from video frames."

    best = max(answers, key=len)
    return f"Based on video analysis: {best}"


def _run_segmentation(path: Path, frames: list, config: dict, mode: str, run_dir: Path,
                      entities: Optional[list[str]] = None, memory_build: bool = False) -> Optional[dict]:
    """Run SAM segmentation on extracted frames. Returns segmentation data or None."""
    try:
        from providers.sam import create_segmenter
    except ImportError as exc:
        console.print(f"[yellow]Segmentation unavailable: {exc}[/yellow]")
        return {"status": "unavailable", "reason": str(exc), "frames_processed": 0, "objects": []}

    mode_config = get_mode_config(config, mode)
    resolution = mode_config.get("resolution", 768)
    seg_fps = config.get("defaults", {}).get("segmentation_fps", 0.25)

    log_vram("before_segmentation")
    sam_config = config.get("models", {}).get("segmentation", {})
    try:
        provider = create_segmenter(sam_config)
    except Exception as exc:
        console.print(f"[yellow]Segmentation unavailable: {exc}[/yellow]")
        return {"status": "unavailable", "reason": str(exc), "frames_processed": 0, "objects": []}

    try:
        # concrete entities; an unlabelled "all objects" result is hard to use.
        frame_cap = 12 if memory_build else 20
        seg_frames = [(f["timestamp"], f["image"]) for f in frames[:frame_cap]]
        prompts = (entities or [])[:3]
        if not prompts:
            prompts = ["person"]
        console.print(f"[dim]Running SAM segmentation on {len(seg_frames)} frames...[/dim]")
        objects = []
        for prompt in prompts:
            result = provider.segment_frames(
                seg_frames,
                prompt,
                run_dir,
            )
            for obj in result.get("objects", []):
                obj["label"] = prompt
            objects.extend(result.get("objects", []))
        log_vram("after_segmentation")
        return {"status": "completed", "frames_processed": len(seg_frames), "objects": objects}
    except Exception as e:
        console.print(f"[yellow]Segmentation failed: {e}[/yellow]")
        log_vram("after_segmentation_error")
        status = "unavailable" if isinstance(e, ImportError) else "failed"
        return {"status": status, "reason": str(e), "frames_processed": 0, "objects": []}
    finally:
        if config.get("gpu_policy", {}).get("unload_after_job", True):
            try:
                provider.unload()
            except Exception:
                pass


def _run_grounding(query: str, frames: list, timeline: list, run_dir, config: dict) -> dict:
    """Run LocateAnything grounding and merge results into the timeline."""
    try:
        from providers.locate_anything import LocateAnythingProvider
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc), "matches": 0}
    from openvision.models import TimelineEntry

    locate_config = config.get("models", {}).get("locate", {})

    console.print(f"[dim]Running LocateAnything grounding: '{query}'...[/dim]")

    provider = LocateAnythingProvider(locate_config)

    try:
        if not provider.check_health():
            console.print("[yellow]LocateAnything endpoint not available — skipping grounding.[/yellow]")
            return {"status": "unavailable", "reason": "endpoint is not reachable", "matches": 0}

        log_vram("before_grounding")
        ground_frames = [(f["timestamp"], f["image"]) for f in frames]
        raw_result = provider.locate_frames(ground_frames, query, str(run_dir))
        log_vram("after_grounding")

        grounding_timeline = []
        for r in raw_result["results"]:
            if not r["boxes"]:
                continue

            labels = [b["label"] for b in r["boxes"]]
            scores = [f"{b['score']:.0%}" for b in r["boxes"]]

            entry = TimelineEntry(
                time_seconds=r["timestamp"],
                timestamp=_fmt_time(r["timestamp"]),
                observation=f"[GROUND] {query}: {', '.join(labels)} (confidence: {', '.join(scores)})",
                frame_path=next((f.get("path") for f in frames
                                 if abs(float(f["timestamp"]) - float(r["timestamp"])) < 0.1), None),
                modality="grounding",
                boxes=r["boxes"],
            )
            timeline.append(entry)
            grounding_timeline.append(entry)

        timeline.sort(key=lambda e: e.time_seconds)

        if grounding_timeline:
            console.print(f"[dim]Grounding found {len(grounding_timeline)} frames with matches.[/dim]")
        else:
            console.print(f"[dim]No matches found for '{query}'.[/dim]")
        return {"status": "completed", "matches": sum(len(r.get("boxes", [])) for r in raw_result.get("results", []))}

    except Exception as e:
        console.print(f"[yellow]Grounding failed: {e}[/yellow]")
        return {"status": "failed", "reason": str(e), "matches": 0}
    finally:
        if config.get("gpu_policy", {}).get("unload_after_job", True):
            try:
                provider.unload()
            except Exception:
                pass


def _display_observation(data: dict):
    """Display observation result in rich format."""
    console.print(Panel(data.get("summary", ""), title="[bold cyan]Observation Summary[/bold cyan]"))

    sampling = data.get("sampling") or {}
    if sampling.get("duration_seconds"):
        # Also the cached path, so an undersampled run is visible on re-read.
        console.print(f"[dim]{_fmt_coverage(sampling)}[/dim]")

    if data.get("timeline"):
        console.print("\n[bold]Timeline:[/bold]")
        table = Table()
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Observation", style="white")

        for entry in data["timeline"]:
            obs = entry.get("observation", "")
            ts = entry.get("timestamp", "??:??")
            if len(obs) > 80:
                obs = obs[:80] + "..."
            table.add_row(f"[{ts}]", obs)

        # Only show first 15 rows to avoid flooding
        console.print(table)
        if len(data["timeline"]) > 15:
            console.print(f"  [dim]... and {len(data['timeline']) - 15} more entries[/dim]")

    if data.get("entities"):
        ents = data.get("entities", [])
        console.print(f"\n[bold]Entities:[/bold] {', '.join(ents[:15])}")
        if len(ents) > 15:
            console.print(f"  [dim]... and {len(ents) - 15} more[/dim]")

    if data.get("stages"):
        stages = Table(title="Pipeline stages")
        stages.add_column("Stage", style="cyan")
        stages.add_column("Status", style="yellow")
        stages.add_column("Details")
        for name, info in data["stages"].items():
            detail = info.get("reason") or (f"{info.get('frames')} frames" if info.get("frames") else "")
            stages.add_row(name, str(info.get("status", "unknown")), detail)
        console.print(stages)

    segmentation = data.get("segmentation") or {}
    if segmentation.get("objects"):
        console.print(f"[dim]Segmentation: {len(segmentation['objects'])} mask(s) saved in {data.get('artifacts_dir', '')}[/dim]")

    if data.get("tokens"):
        tokens = data["tokens"]
        console.print(f"\n[dim]Tokens: {_fmt_tokens(TokenUsage(**tokens))}[/dim]")

    if data.get("artifacts_dir"):
        console.print(f"\n[dim]Artifacts: {data['artifacts_dir']}[/dim]")


def _save_observation_md(path: Path, result: ObserveResult, config: dict,
                    transcript: Optional[Transcript] = None):
    """Save observation as markdown ledger file under openvision_HOME (stable path)."""
    from datetime import date
    from storage.paths import observations_dir

    today = date.today().isoformat()
    video_name = path.stem

    obs_dir = observations_dir(config)
    obs_path = obs_dir / f"{today}-{video_name}.md"
    lines = [
        f"# Video Observation: {path.name}",
        "",
        f"Date: {today}",
        f"Source: `{path}`",
        f"Artifacts: `{result.artifacts_dir}`",
        "",
        "## Summary",
        "",
        result.summary or "",
        "",
        "## Timeline",
        "",
    ]
    for entry in result.timeline:
        lines.append(f"- **{entry.timestamp}** - {entry.observation}")

    if result.entities:
        lines.extend(["", "## Entities", ""])
        for entity in result.entities:
            lines.append(f"- {entity}")

    # Full transcript when available (not truncated to 30 for vault reuse)
    if transcript and transcript.segments:
        lines.extend(["", "## Transcript", ""])
        for seg in transcript.segments:
            ts = _fmt_time(seg.start)
            speaker = f"[{seg.speaker}] " if getattr(seg, "speaker", None) else ""
            lines.append(f"- **{ts}** {speaker}{seg.text}")

    lines.extend(["", "## Artifacts", "", f"- Frames: `{result.artifacts_dir}/frames/`"])
    lines.extend(["", f"Observation path: `{obs_path}`"])

    obs_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Memory saved:[/green] {obs_path}")
