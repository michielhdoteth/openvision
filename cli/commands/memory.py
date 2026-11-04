"""Search and inspect OpenVision's local, timestamped video memories."""
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from core.video_memory import VideoMemoryStore
from storage.config import load_config
from storage.paths import runs_db_path

app = typer.Typer(name="memory", help="Search local, time-grounded video memories", no_args_is_help=True)
console = Console()


def _store() -> VideoMemoryStore:
    return VideoMemoryStore(runs_db_path(load_config()))


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Terms to find in speech, visual observations, entities, or speakers"),
    video: str = typer.Option(None, "--video", help="Restrict search to this source video"),
    limit: int = typer.Option(8, "--limit", min=1, max=100, help="Maximum matching 30-second intervals"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Find time ranges with separate speech and visual evidence."""
    global console
    console = Console(stderr=json_output)
    results = _store().search(query, limit=limit, source_path=video)
    if json_output:
        typer.echo(json.dumps(results, indent=2))
        return
    if not results:
        console.print("No matching memory intervals. Run [cyan]openvision observe VIDEO --memory[/cyan] for higher visual coverage and transcript indexing.")
        return
    for result in results:
        start = _format_time(result["start_seconds"])
        end = _format_time(result["end_seconds"])
        console.print(f"\n[bold cyan]{_console_safe(result['source_path'])}[/bold cyan] [{start}–{end}]")
        if result["visual_evidence"]:
            console.print("[bold]Visual[/bold]  " + _console_safe(result["visual_evidence"]))
        if result["speech_evidence"]:
            speakers = ", ".join(result["speakers"]) or "speaker unknown"
            console.print(f"[bold]Speech ({_console_safe(speakers)})[/bold]  " + _console_safe(result["speech_evidence"]))
        if result["entities"]:
            console.print("[bold]Entities[/bold]  " + _console_safe(", ".join(result["entities"])))
        if result["speakers"]:
            console.print("[dim]Speaker-to-visible-person link: unverified; replay source to confirm.[/dim]")


@app.command("outline")
def outline_cmd(
    video: str = typer.Option(..., "--video", help="Video source path"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Show the root, 5-minute sections, and 30-second evidence hierarchy."""
    global console
    console = Console(stderr=json_output)
    source = str(Path(video).expanduser().resolve())
    store = _store()
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT node_id,kind,parent_id,start_sec,end_sec,summary,visual_text,audio_text "
            "FROM video_memory_nodes WHERE source_path=? ORDER BY start_sec,kind",
            (source,),
        ).fetchall()
    data = [dict(row) for row in rows]
    if json_output:
        typer.echo(json.dumps(data, indent=2))
        return
    if not data:
        console.print("No memory found for this source. Run [cyan]openvision observe VIDEO[/cyan] first.")
        return
    root = next((row for row in data if row["kind"] == "root"), None)
    if root:
        console.print(f"[bold cyan]{_console_safe(source)}[/bold cyan]")
        console.print(_console_safe(root["summary"] or "Video memory"))
    for section in (row for row in data if row["kind"] == "section"):
        console.print(f"\n[bold]{_format_time(section['start_sec'])}–{_format_time(section['end_sec'])}[/bold]  {_console_safe(section['summary'])}")
        for chunk in (row for row in data if row["kind"] == "chunk" and row["parent_id"] == section["node_id"]):
            console.print(f"  [dim]{_format_time(chunk['start_sec'])}[/dim] {_console_safe(chunk['summary'])}")


@app.command("ask")
def ask_cmd(
    video: str = typer.Argument(..., help="Video whose indexed memory should answer the question"),
    question: str = typer.Argument(..., help="Question about the video"),
    limit: int = typer.Option(6, "--limit", min=1, max=12, help="Evidence intervals to retrieve and verify"),
    provider_name: str = typer.Option(None, "--provider", help="VLM provider override"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Answer from retrieved memory and verify against saved source frames."""
    global console
    console = Console(stderr=json_output)
    hits = _store().search(question, limit=limit, source_path=video)
    if not hits:
        result = {"answer": "No matching indexed evidence. Build richer coverage with `openvision observe VIDEO --memory --max-frames N`.",
                  "confidence": "low", "evidence": []}
        _emit_result(result, json_output)
        return

    config = load_config()
    # Decision tier (falls back to the observe-time VLM when unconfigured) so
    # verification questions run on the cheap/on-device backend by default.
    vlm_config = dict(config.get("models", {}).get("decision")
                      or config.get("models", {}).get("vlm", {}))
    if provider_name:
        vlm_config["provider"] = provider_name
    from providers.registry import ProviderRegistry
    try:
        provider = ProviderRegistry.create(vlm_config)
        if getattr(provider, "kind", "chat") != "chat":
            provider.close()
            raise RuntimeError(
                "models.decision points at a typed decision model, which returns "
                "probabilities instead of text. Set models.decision.provider to a "
                "chat provider (llamacpp, gemini, ...) and use `ask --decision` "
                "for typed answers."
            )
        if not provider.check_health():
            provider.close()
            raise RuntimeError("selected vision-language provider is not reachable")
    except Exception as exc:
        result = {"answer": f"Could not verify retrieved evidence with a VLM: {exc}", "confidence": "low", "evidence": []}
        _emit_result(result, json_output)
        return

    checked = []
    seen_frames = set()
    try:
        for hit in hits:
            frame = next((e for e in hit["evidence"]
                          if e.get("modality") == "visual" and e.get("frame_path")
                          and e["frame_path"] not in seen_frames), None)
            context = (
                f"Question: {question}\n"
                f"Transcript in this exact interval: {hit['speech_evidence'] or '[no transcript evidence]'}\n"
                f"Speaker labels: {', '.join(hit['speakers']) or 'unavailable'}\n"
                f"Prior visual notes: {hit['visual_evidence'] or '[no visual notes]'}\n\n"
                "Answer only from visible or quoted evidence. Distinguish speaker labels from visible identity; "
                "do not claim that a visible person said a line unless the evidence directly verifies it. "
                "Cite the interval time in your answer."
            )
            if frame:
                from core.image import load_image
                response = provider.describe_image(load_image(frame["frame_path"]), context)
                seen_frames.add(frame["frame_path"])
            else:
                response = provider.query_text(context)
            answer = (response.content or "").strip()
            if answer and not answer.lower().startswith(("could not connect", "request to ", "unexpected error contacting")) and " returned http " not in answer.lower():
                checked.append({"interval": [_format_time(hit["start_seconds"]), _format_time(hit["end_seconds"])],
                                "answer": answer, "speech_evidence": hit["speech_evidence"],
                                "visual_evidence": hit["visual_evidence"], "speakers": hit["speakers"],
                                "identity_link": hit["speaker_to_visible_person"], "visual_frame_verified": bool(frame),
                                "source": hit["source_path"]})
            if len(checked) >= limit:
                break
    except Exception as exc:
        console.print(f"[yellow]Evidence verification stopped: {exc}[/yellow]")
    finally:
        provider.close()

    if checked:
        answer = "\n".join(f"[{item['interval'][0]}] {item['answer']}" for item in checked)
        confidence = "medium" if any(item["visual_frame_verified"] for item in checked) else "low"
    else:
        answer = "Retrieved intervals did not yield a reliable model response; inspect the evidence or replay the source video."
        confidence = "low"
    _emit_result({"answer": answer, "confidence": confidence, "evidence": checked}, json_output)


@app.command("import")
def import_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Index saved observation runs into video memory (no re-analysis).

    Walks the local run cache, and for every recorded run whose source video
    still exists on disk, rebuilds its memory hierarchy from the saved
    artifacts — transcripts are parsed with HH:MM:SS support.
    """
    global console
    console = Console(stderr=json_output)
    from storage.paths import run_cache_dir
    from cli.commands.observe import _parse_transcript_txt

    runs_path = run_cache_dir(load_config())
    summary = {"imported": 0, "skipped": 0, "sources": []}
    for run_dir in sorted(p for p in runs_path.glob("*") if p.is_dir()):
        artifact = run_dir / "observations.json"
        if not artifact.is_file():
            summary["skipped"] += 1
            continue
        try:
            data = json.loads(artifact.read_text(encoding="utf-8", errors="replace"))
            source = data.get("source_path")
            if not source or not Path(source).is_file():
                summary["skipped"] += 1
                continue
            segments = _parse_transcript_txt(run_dir / "transcript.txt")
            result = _store().build(
                source, run_dir, data.get("summary", ""), data.get("timeline", []),
                data.get("entities", []), segments, None,
                data.get("duration_seconds", 0.0),
            )
            summary["imported"] += 1
            summary["sources"].append(str(source))
        except Exception as exc:
            summary["skipped"] += 1
            console.print(f"[yellow]Skipped {run_dir.name}: {exc}[/yellow]")
    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    else:
        console.print(f"[green]Imported {summary['imported']} run(s) into video memory"
                      f"[/green] [dim]({summary['skipped']} skipped)[/dim]")
        for source in summary["sources"]:
            console.print(f"  [cyan]{Path(source).name}[/cyan]")


@app.command("note")
def note_cmd(
    video: str = typer.Argument(..., help="Video whose memory the note belongs to (source path)"),
    text: str = typer.Argument(..., help="What the agent observed, e.g. 'red dial on the left panel'"),
    at: str = typer.Option("0:00", "--at", "-a", help="Timestamp the note describes (MM:SS, HH:MM:SS, or seconds)"),
    frame: str = typer.Option(None, "--frame", help="Frame path this note describes, if any"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Write an agent observation back into video memory.

    Vision agents that read frames themselves use this so what they see
    joins the searchable global picture instead of living only in chat.
    """
    global console
    console = Console(stderr=json_output)
    from core.video import parse_time
    try:
        seconds = parse_time(at)
    except Exception as exc:
        console.print(f"[red]Invalid --at time: {exc}[/red]")
        raise typer.Exit(code=1)
    try:
        recorded = _store().note(video, seconds, text, frame_path=frame)
    except FileNotFoundError:
        console.print(f"[red]Video not found: {video}[/red] (the source file must exist so its fingerprint matches)")
        raise typer.Exit(code=1)
    if json_output:
        typer.echo(json.dumps(recorded, indent=2))
    else:
        console.print(f"[green]Noted {_format_time(recorded['at_seconds'])} in "
                      f"{Path(recorded['source']).name}[/green] -> {recorded['note']}")


@app.command("notes")
def notes_cmd(
    video: str = typer.Option(None, "--video", help="Restrict notes to this source video"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """List agent-written observations stored in video memory."""
    global console
    console = Console(stderr=json_output)
    notes = _store().list_notes(video)
    if json_output:
        typer.echo(json.dumps(notes, indent=2))
        return
    if not notes:
        console.print("No agent notes yet. Write one with: "
                      "[cyan]openvision memory note VIDEO \"...\" --at 1:23[/cyan]")
        return
    for note in notes:
        console.print(f"[bold cyan]{Path(note['source_path']).name}[/bold cyan] "
                      f"[{_format_time(note['start_seconds'])}] {_console_safe(note['text'])}")


@app.command("status")
def status_cmd(json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON")):
    """Show indexed source and hierarchy counts."""
    global console
    console = Console(stderr=json_output)
    status = _store().status()
    if json_output:
        typer.echo(json.dumps(status, indent=2))
    else:
        table = Table(title="Local Video Memory")
        table.add_column("Sources", style="cyan")
        table.add_column("Sections", style="yellow")
        table.add_column("Evidence intervals", style="green")
        table.add_row(str(status["sources"]), str(status["sections"]), str(status["chunks"]))
        console.print(table)
        console.print(f"[dim]Database: {status['database']}[/dim]")


@app.command("sources")
def sources_cmd(json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON")):
    """List videos indexed in the local memory database."""
    global console
    console = Console(stderr=json_output)
    sources = _store().list_sources()
    if json_output:
        typer.echo(json.dumps(sources, indent=2))
        return
    if not sources:
        console.print("No indexed videos yet. Run [cyan]openvision observe VIDEO[/cyan] to analyze and index one.")
        return
    table = Table(title="Indexed Videos")
    table.add_column("Video", style="cyan")
    table.add_column("Duration")
    table.add_column("Sections", justify="right")
    table.add_column("Intervals", justify="right")
    table.add_column("Summary")
    for source in sources:
        summary = source["summary"].replace("\n", " ")
        table.add_row(_console_safe(Path(source["source_path"]).name), _format_time(source["end_seconds"]),
                      str(source["sections"]), str(source["chunks"]), _console_safe(summary[:100]))
    console.print(table)


def _format_time(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _emit_result(result: dict, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        console.print(f"[bold]{result['confidence'].upper()} confidence[/bold]")
        console.print(_console_safe(result["answer"]))
        for evidence in result["evidence"]:
            console.print(f"[dim]Evidence: {_console_safe(evidence['source'])} [{evidence['interval'][0]}–{evidence['interval'][1]}] · speaker identity link {evidence['identity_link']}[/dim]")


def _console_safe(value: str, encoding: str | None = None) -> str:
    """Keep machine-readable JSON exact while making terminal output encoding-safe."""
    encoding = encoding or getattr(console.file, "encoding", None)
    if not encoding:
        return value
    return value.encode(encoding, errors="backslashreplace").decode(encoding)
