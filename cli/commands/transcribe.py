"""openvision transcribe <path> - Transcribe audio from video."""
import os
import tempfile
import time
import sys
import typer
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from providers import get_provider, list_providers

console = Console()


def _safe_print(msg: str):
    """Print a message safely, falling back to plain print on Unicode errors."""
    try:
        console.print(msg)
    except UnicodeEncodeError:
        import re
        plain = re.sub(r'\[/?\w+(?:=.*?)?\]', '', msg)
        print(plain)


def transcribe_cmd(
    path: str = typer.Argument(..., help="Path to video file"),
    provider_name: str = typer.Option("parakeet", "--provider", "-p", help=f"STT provider: {', '.join(list_providers())}"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g., 'en'). Auto-detect if not set."),
    model: str = typer.Option("base", "--model", "-m", help="Model size (whisper only, ignored for parakeet)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, cuda"),
    from_time: str = typer.Option(None, "--from", help="Start time (e.g., 1:30, 45, 0:15:00)"),
    to_time: str = typer.Option(None, "--to", help="End time (e.g., 2:00, 90, 0:20:00)"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Save full soundtrack as audio.m4a"),
    cookies: str = typer.Option(None, "--cookies", help="Netscape cookie file for authenticated videos"),
    cookies_from_browser: str = typer.Option(None, "--cookies-from-browser", help="Read cookies from browser (chrome, firefox, edge, safari)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Streaming JSONL output"),
    save_text: bool = typer.Option(False, "--save", help="Save transcript to text file beside the input"),
    save_to: str = typer.Option(None, "--save-to", help="Save transcript text to this exact path"),
    no_run: bool = typer.Option(False, "--no-run", help="Do not write this transcript to the run cache or video memory"),
    diarize_audio: bool = typer.Option(False, "--diarize", help="Add speaker labels via diarization"),
    min_speakers: int = typer.Option(None, "--min-speakers", help="Minimum number of speakers"),
    max_speakers: int = typer.Option(None, "--max-speakers", help="Maximum number of speakers"),
):
    """Transcribe audio from a video file."""
    global console
    console = Console(stderr=json_output or jsonl)
    from core.download import is_url
    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    input_path = Path(path)
    is_remote = is_url(path)
    if not is_remote and not input_path.exists():
        _safe_print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    if writer:
        writer.emit_start(path, mode=provider_name)

    _safe_print(f"[dim]Loading {provider_name.title()} model on {device}...[/dim]")

    from core.video import parse_time
    window_start = parse_time(from_time) if from_time else None
    window_end = parse_time(to_time) if to_time else None
    if window_start is not None or window_end is not None:
        _safe_print(f"[dim]Time window: {_fmt_time(window_start or 0)} - {_fmt_time(window_end or 0)}[/dim]")

    config = {
        "device": device,
    }
    if provider_name == "whisper":
        config["model_size"] = model
    provider = get_provider(provider_name, config)

    cookies_file = cookies
    owned_cookie_file = None
    if cookies_from_browser and not cookies_file:
        try:
            import browser_cookie3
            cj = getattr(browser_cookie3, cookies_from_browser.replace("-", "_").lower())()
            cookie_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            cookie_tmp.write("# Netscape HTTP Cookie File\n")
            for c in cj:
                secure = "TRUE" if c.secure else "FALSE"
                domain = c.domain if c.domain.startswith(".") else "." + c.domain
                cookie_tmp.write(f"{domain}\tTRUE\t{c.path}\t{secure}\t{int(c.expires or 0)}\t{c.name}\t{c.value}\n")
            cookie_tmp.close()
            cookies_file = cookie_tmp.name
            owned_cookie_file = cookie_tmp.name
            _safe_print(f"[dim]Loaded cookies from {cookies_from_browser}[/dim]")
        except ImportError:
            _safe_print("[yellow]browser-cookie3 not installed. Install with: pip install browser-cookie3[/yellow]")
        except Exception as e:
            _safe_print(f"[yellow]Failed to read browser cookies: {e}[/yellow]")

    # Download from URL if remote (native yt-dlp into stable downloads dir)
    dl_result = {}
    if is_remote:
        from core.download import download_video
        from storage.paths import downloads_dir
        from storage.config import load_config
        try:
            cfg = load_config()
        except Exception:
            cfg = None
        dl_dir = downloads_dir(cfg)
        _safe_print(f"[dim]Downloading from URL into {dl_dir}...[/dim]")
        try:
            dl_result = download_video(
                path, output_dir=str(dl_dir), audio_only=False, write_subs=True, cookies_file=cookies_file
            )
            input_path = Path(dl_result["path"])
            _safe_print(f"[dim]Downloaded: {dl_result.get('title', input_path.name)}[/dim]")
        except Exception as e:
            _safe_print(f"[red]Download failed:[/red] {e}")
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

    if is_remote and dl_result.get("subs_file") and os.path.exists(dl_result["subs_file"]):
        try:
            from core.captions import parse_vtt, filter_range
            sub_content = Path(dl_result["subs_file"]).read_text(encoding="utf-8", errors="replace")
            caps = parse_vtt(sub_content)
            if caps:
                from providers.parakeet import Transcript, TranscriptSegment
                transcript = Transcript(
                    segments=[TranscriptSegment(start=c.start, end=c.end, text=c.text, confidence=1.0) for c in caps],
                    language="en",
                    duration_seconds=caps[-1].end if caps else 0,
                )
                _safe_print(f"[green]Using downloaded captions ({len(transcript.segments)} segments)[/green]")
            else:
                transcript = None
        except Exception as e:
            _safe_print(f"[dim]Caption parsing failed: {e}[/dim]")
            transcript = None
    else:
        transcript = None

    # When time window specified, extract audio segment for faster transcription
    if (window_start is not None or window_end is not None) and transcript is None:
        seg_path = None
        try:
            from core.audio import extract_audio_segment
            seg_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            seg_path = seg_tmp.name
            seg_tmp.close()
            start_s = window_start or 0
            end_s = window_end or 999999
            _safe_print(f"[dim]Extracting audio segment {_fmt_time(start_s)} - {_fmt_time(end_s)}...[/dim]")
            extract_audio_segment(str(input_path), start_s, end_s, seg_path)
            t0 = time.time()
            transcript = provider.transcribe_audio(seg_path, language)
            elapsed = time.time() - t0
            _safe_print(f"[dim]Transcription finished in {elapsed:.1f}s[/dim]")
            # Adjust timestamps to be relative to source video
            if window_start and transcript.segments:
                for seg in transcript.segments:
                    seg.start += window_start
                    seg.end += window_start
            if window_end and transcript.segments:
                transcript.segments = [s for s in transcript.segments if s.start < window_end]
        except Exception as e:
            _safe_print(f"[dim]Segment extraction failed ({e}), transcribing full file...[/dim]")
            transcript = None
        finally:
            if seg_path and os.path.exists(seg_path):
                os.unlink(seg_path)

    # Try existing captions first (faster, no GPU needed)
    if transcript is None:
        try:
            captions = provider.try_captions(str(input_path))
            if captions and captions.segments:
                _safe_print(f"[green]Found embedded captions ({len(captions.segments)} segments)[/green]")
                transcript = captions
            else:
                _safe_print("[dim]No embedded captions found, transcribing...[/dim]")
                t0 = time.time()
                transcript = provider.transcribe(str(input_path))
                elapsed = time.time() - t0
                _safe_print(f"[dim]Transcription finished in {elapsed:.1f}s[/dim]")
        except Exception as e:
            _safe_print(f"[dim]Caption extraction failed ({e}), transcribing...[/dim]")
            t0 = time.time()
            transcript = provider.transcribe(str(input_path))
            elapsed = time.time() - t0
            _safe_print(f"[dim]Transcription finished in {elapsed:.1f}s[/dim]")

    if not transcript.segments:
        _safe_print("[yellow]No speech detected in the video audio track.[/yellow]")
        return

    # Filter transcript to time window if specified
    if (window_start is not None or window_end is not None) and transcript.segments:
        before = len(transcript.segments)
        transcript.segments = [
            s for s in transcript.segments
            if (window_start is None or s.end >= window_start) and (window_end is None or s.start <= window_end)
        ]
        if len(transcript.segments) < before:
            _safe_print(f"[dim]Filtered to {len(transcript.segments)} segments in time window[/dim]")

    if diarize_audio and transcript.segments:
        try:
            from providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
            from core.audio import extract_audio

            _safe_print("[dim]Running speaker diarization...[/dim]")

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                extract_audio(str(input_path), output_path=tmp_path)
                diarizer = DiarizerProvider()
                dia_result = diarizer.diarize(
                    tmp_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                transcript.segments = merge_transcript_with_diarization(
                    transcript.segments, dia_result.segments
                )
                _safe_print(f"[dim]Found {dia_result.num_speakers} speakers: {', '.join(dia_result.speakers)}[/dim]")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except ImportError as e:
            _safe_print(f"[yellow]Diarization unavailable: {e}[/yellow]")
        except Exception as e:
            _safe_print(f"[yellow]Diarization failed: {e}[/yellow]")

    # Persist the transcript before printing it: stdout is lossy when a harness
    # truncates a long run, so the run cache and video memory are the recovery
    # path. Both write by source fingerprint, so frames, captions, observations
    # and ask evidence already stored for this video stay and this transcript
    # merges into them instead of replacing them.
    if not no_run:
        try:
            from storage.cache import RunCache
            from storage.config import load_config as _load_cfg
            from storage.paths import run_cache_dir
            from core.video_memory import VideoMemoryStore

            try:
                cfg = _load_cfg()
            except FileNotFoundError:
                cfg = {}
            cache = RunCache(str(run_cache_dir(cfg)))
            options = {"command": "transcribe", "provider": provider_name,
                       "language": language, "from": str(window_start),
                       "to": str(window_end), "diarize": bool(diarize_audio)}
            key = cache.cache_key(str(input_path), mode="transcribe", options=options)
            metadata = {
                "source": str(input_path), "command": "transcribe",
                "provider": provider_name, "language": transcript.language,
                "duration_seconds": transcript.duration_seconds,
                "segments": len(transcript.segments),
            }
            run_dir = cache.create_run(key, metadata=metadata)
            cache.save_transcript_batch(run_dir, transcript.segments)
            cache.save_text(run_dir, "transcript.txt", transcript.text or "")
            indexed = VideoMemoryStore(cache._db_path).add_transcript(
                str(input_path), run_dir, transcript.segments,
                duration_seconds=transcript.duration_seconds,
            )
            _safe_print(
                f"[green]Transcript cached:[/green] {run_dir} "
                f"[dim]({indexed.get('chunks', 0)} memory chunks, run {run_dir.name})[/dim]"
            )
        except Exception as e:
            _safe_print(f"[yellow]Could not persist transcript ({e}); it exists only in stdout now.[/yellow]")

    if writer:
        for seg in transcript.segments:
            writer.emit_transcript(seg.start, seg.end, seg.text, getattr(seg, "speaker", None))
        writer.emit_done({"segments": len(transcript.segments), "language": transcript.language,
                          "duration": transcript.duration_seconds})
        writer.flush()

    if json_output:
        output = {
            "language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "confidence": s.confidence, "speaker": s.speaker}
                for s in transcript.segments
            ],
            "full_text": transcript.text,
        }
        typer.echo(json.dumps(output, indent=2))
    elif not jsonl:
        _display_transcript(transcript)

    text_path = Path(save_to) if save_to else (input_path.with_suffix(".txt") if save_text else None)
    if text_path:
        lines = [
            f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
            for s in transcript.segments
        ]
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("\n".join(lines), encoding="utf-8")
        _safe_print(f"[green]Transcript saved:[/green] {text_path}")

    if keep_audio:
        try:
            from core.audio import save_full_audio
            audio_path = str(input_path.with_name(input_path.stem + "_audio.m4a"))
            save_full_audio(str(input_path), audio_path)
            _safe_print(f"[green]Audio saved:[/green] {audio_path}")
        except Exception as e:
            _safe_print(f"[yellow]Failed to save audio: {e}[/yellow]")


def _display_transcript(transcript):
    """Display transcript in rich format."""
    summary = (
        f"Language: {transcript.language} | "
        f"Duration: {transcript.duration_seconds:.0f}s | "
        f"Segments: {len(transcript.segments)}"
    )
    console.print(Panel(summary, title="[bold cyan]Transcription[/bold cyan]"))

    table = Table()
    table.add_column("Time", style="cyan", width=14)
    table.add_column("Speaker", style="magenta", width=12)
    table.add_column("Text", style="white")
    table.add_column("Conf", style="green", width=6)

    for seg in transcript.segments:
        ts = f"{_fmt_time(seg.start)} -> {_fmt_time(seg.end)}"
        speaker = seg.speaker if seg.speaker else "-"
        conf = f"{seg.confidence:.2f}" if seg.confidence else "-"
        text = seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
        table.add_row(ts, speaker, text, conf)

    console.print(table)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
