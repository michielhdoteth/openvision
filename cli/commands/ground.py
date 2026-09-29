"""openvision ground <path> --query <query> - Detect and locate objects using LocateAnything-3B."""
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from openvision.models import GroundingBox, GroundingFrame, GroundingResult, TokenUsage
from storage.config import load_config, get_mode_config
from core.gpu import auto_downgrade_mode, log_vram

console = Console()


def ground_cmd(
    path: str = typer.Argument(..., help="Video or image file to ground"),
    query: str = typer.Option(..., "--query", "-q", help="Grounding query (e.g. 'person holding cup')"),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Frame sampling interval in seconds"),
    max_frames: int = typer.Option(60, "--max-frames", help="Maximum frames to analyze"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
    provider_name: Optional[str] = typer.Option(None, "--provider", help="Override grounding provider"),
):
    """Detect and locate objects in video frames.

    Uses LocateAnything-3B to find objects matching your query.
    Returns bounding boxes with labels and confidence scores.

    Examples:
        openvision ground video.mp4 --query "person holding cup"
        openvision ground video.mp4 --query "all people" --interval 1.0
        openvision ground frame.jpg --query "laptop on desk"
    """
    global console
    console = Console(stderr=json_output)
    from core.video import probe
    from core.sampling import smart_sample
    from providers.locate_anything import LocateAnythingProvider
    from storage.cache import RunCache
    from core.errors import OpenVisionError

    try:
        file_path = Path(path).resolve()
        if not file_path.exists():
            console.print(f"[red]Error:[/red] File not found: {path}")
            raise typer.Exit(code=1)

        if not query.strip():
            console.print("[red]Error:[/red] Query cannot be empty.")
            raise typer.Exit(code=1)

        # Load config
        config = load_config()
        actual_mode, warning = auto_downgrade_mode("fast", config)
        mode_config = get_mode_config(config, actual_mode)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        # Get locate provider config
        locate_config = config.get("models", {}).get("locate", {})
        if provider_name:
            locate_config["provider"] = provider_name

        # Probe video
        video_meta = probe(str(file_path))
        is_image = video_meta.get("is_image", False)

        # Extract frames
        if is_image:
            from core.image import load_image
            image = load_image(str(file_path))
            frames = [(0.0, image)]
        else:
            log_vram("before_frame_extraction")
            frames_raw = smart_sample(
                str(file_path),
                fps=1.0 / interval,
                max_frames=max_frames,
                max_resolution=mode_config.get("resolution", 768),
            )
            frames = [(f["timestamp"], f["image"]) for f in frames_raw]
            log_vram("after_frame_extraction")

        if not frames:
            console.print("[yellow]No frames extracted.[/yellow]")
            raise typer.Exit(code=1)

        console.print(f"[dim]Grounding '{query}' across {len(frames)} frames...[/dim]")

        # Set up cache
        from storage.paths import run_cache_dir
        cache_dir = str(run_cache_dir(config))
        cache = RunCache(cache_dir)
        import re
        safe_query = re.sub(r'[^a-zA-Z0-9_]', '_', query)[:20].strip('_')
        run_dir = cache.create_run(f"ground_{safe_query}", video_meta)

        # Run grounding
        provider = LocateAnythingProvider(locate_config)
        try:
            if not provider.check_health():
                console.print(
                    "[red]Error:[/red] LocateAnything endpoint not available. "
                    "Start your vLLM+Worker server first."
                )
                console.print(f"[dim]Expected endpoint: {locate_config.get('base_url', 'http://localhost:8000')}[/dim]")
                raise typer.Exit(code=1)

            log_vram("before_grounding")
            raw_result = provider.locate_frames(frames, query, str(run_dir))
            log_vram("after_grounding")

        finally:
            if config.get("gpu_policy", {}).get("unload_after_job", True):
                provider.unload()

        # Build result model
        grounding_frames = []
        for r in raw_result["results"]:
            boxes = [GroundingBox(**b) for b in r["boxes"]]
            # Find corresponding frame path
            frame_path = None
            candidate = run_dir / f"ground_{r['timestamp']:.2f}.jpg"
            if candidate.exists():
                frame_path = str(candidate)

            grounding_frames.append(GroundingFrame(
                timestamp=r["timestamp"],
                timestamp_str=_fmt_time(r["timestamp"]),
                frame_path=frame_path,
                boxes=boxes,
                query=query,
            ))

        result = GroundingResult(
            query=query,
            video_path=str(file_path),
            frames=grounding_frames,
            tokens=TokenUsage(),
            confidence="high",
            artifacts_dir=str(run_dir),
        )

        # Save artifacts
        cache.save_artifact(run_dir, "grounding.json", result.model_dump())

        # Display
        if json_output:
            typer.echo(json.dumps(result.model_dump(), indent=2))
        else:
            _display_grounding(result)

    except OpenVisionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _display_grounding(result: GroundingResult) -> None:
    """Display grounding results in a rich table."""
    console.print(Panel(
        f"Query: [bold]{result.query}[/bold]\n"
        f"Video: {result.video_path}\n"
        f"Frames analyzed: {len(result.frames)}",
        title="[bold cyan]Grounding Complete[/bold cyan]",
    ))

    if not result.frames:
        console.print("[yellow]No objects found matching the query.[/yellow]")
        return

    # Summary
    total_boxes = sum(len(f.boxes) for f in result.frames)
    unique_labels = set()
    for f in result.frames:
        for b in f.boxes:
            unique_labels.add(b.label)

    console.print(f"[bold]Total detections:[/bold] {total_boxes}")
    if unique_labels:
        console.print(f"[bold]Unique labels:[/bold] {', '.join(sorted(unique_labels))}")

    # Per-frame details
    table = Table()
    table.add_column("Time", style="cyan")
    table.add_column("Objects", style="green")
    table.add_column("Confidence", style="yellow")

    for frame in result.frames:
        if frame.boxes:
            objects = ", ".join(b.label for b in frame.boxes)
            scores = ", ".join(f"{b.score:.2f}" for b in frame.boxes)
            table.add_row(frame.timestamp_str, objects, scores)

    console.print(table)
    console.print(f"\n[dim]Annotated frames saved to: {result.artifacts_dir}[/dim]")
