"""openvision track <path> <prompt> - Track objects in video."""
import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.video import probe
from core.sampling import smart_sample
from core.gpu import auto_downgrade_mode, log_vram
from providers.sam import SamProvider
from storage.cache import RunCache
from storage.config import load_config, get_mode_config
from openvision.models import TrackResult, TrackPosition

console = Console()


def track_cmd(
    path: str = typer.Argument(..., help="Path to video file"),
    prompt: str = typer.Argument(..., help="Object to track"),
    start_time: float = typer.Option(0.0, "--start", "-s", help="Start time in seconds"),
    end_time: float = typer.Option(0.0, "--end", "-e", help="End time in seconds (0 = end of video)"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Processing mode"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Track an object in a video by prompt."""
    global console
    console = Console(stderr=json_output)
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    if not prompt.strip():
        console.print("[red]Error:[/red] Prompt cannot be empty.")
        raise typer.Exit(code=1)

    # Load config
    config = load_config()
    mode_config = get_mode_config(config, mode)

    # Check GPU
    actual_mode, warning = auto_downgrade_mode(mode, config)
    if warning:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    track_fps = config.get("defaults", {}).get("tracking_fps", 0.5)
    resolution = mode_config.get("resolution", 768)

    # Probe video
    video_meta = probe(str(input_path))
    actual_end = end_time if end_time > 0 else video_meta["duration_seconds"]

    console.print(f"[dim]Tracking '{prompt}' from {start_time}s to {actual_end}s[/dim]")

    # Sample frames
    frames = smart_sample(
        str(input_path),
        fps=track_fps,
        max_frames=60,
        max_resolution=resolution,
    )

    # Filter by time range
    frames = [f for f in frames if start_time <= f["timestamp"] <= actual_end]

    if not frames:
        console.print(f"[red]No frames found in time range {start_time}-{actual_end}s.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Analyzing {len(frames)} frames for tracking '{prompt}'...[/dim]")

    # Load SAM for detection per frame
    log_vram("before_tracking")
    sam_config = config.get("models", {}).get("segmentation", {})
    provider = SamProvider(sam_config)

    positions = []

    try:
        for frame in frames:
            objects = provider.segment(frame["image"], prompt)
            if objects:
                # Use highest confidence detection
                best = max(objects, key=lambda o: o.get("score", 0))
                positions.append(TrackPosition(
                    timestamp=frame["timestamp_str"],
                    time_seconds=frame["timestamp"],
                    bbox=best.get("bbox", [0, 0, 0, 0]),
                    confidence=best.get("score", 0.0),
                ))
    except ImportError as e:
        console.print(f"[red]Tracking unavailable:[/red] {e}")
        console.print("Install dependencies: pip install transformers ultralytics")
        raise typer.Exit(code=1)
    finally:
        if config.get("gpu_policy", {}).get("unload_after_job", True):
            provider.unload()

    log_vram("after_tracking")

    # Generate summary
    if positions:
        first_pos = positions[0]
        last_pos = positions[-1]
        summary = (
            f"The '{prompt}' was tracked across {len(positions)} frames "
            f"from {positions[0].timestamp} to {positions[-1].timestamp}."
        )
    else:
        summary = f"The '{prompt}' was not detected in the sampled frames."

    track_id = f"{prompt.replace(' ', '_')}_{input_path.stem}"

    result = TrackResult(
        track_id=track_id,
        positions=positions,
        summary=summary,
    )

    # Cache result
    from storage.paths import run_cache_dir
    cache_dir = str(run_cache_dir(config))
    cache = RunCache(cache_dir)
    run_dir = cache.create_run(f"track_{prompt.replace(' ', '_')[:20]}", video_meta)
    cache.save_artifact(run_dir, "tracking.json", result.model_dump())

    if json_output:
        typer.echo(json.dumps(result.model_dump(), indent=2))
    else:
        _display_track_result(result)


def _display_track_result(result: TrackResult):
    """Display tracking result."""
    console.print(Panel(result.summary, title=f"[bold cyan]Track: {result.track_id}[/bold cyan]"))

    if result.positions:
        table = Table()
        table.add_column("Time", style="cyan")
        table.add_column("BBox", style="yellow")
        table.add_column("Confidence", style="green")

        for pos in result.positions:
            bbox_str = f"[{', '.join(f'{b:.0f}' for b in pos.bbox[:4])}]" if len(pos.bbox) >= 4 else "N/A"
            conf_str = f"{pos.confidence:.2f}"
            table.add_row(pos.timestamp, bbox_str, conf_str)

        console.print(table)
