"""openvision segment <path> <prompt> - Segment objects in video."""
import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.video import probe, extract_frames, extract_frames_by_indices
from core.sampling import smart_sample
from core.gpu import auto_downgrade_mode, log_vram
from providers.sam import create_segmenter
from storage.cache import RunCache
from storage.config import load_config, get_mode_config
from openvision.models import SegmentResult, SegmentObject

console = Console()


def segment_cmd(
    path: str = typer.Argument(..., help="Path to video file"),
    prompt: str = typer.Argument(..., help="Object to segment"),
    start_time: float = typer.Option(0.0, "--start", "-s", help="Start time in seconds"),
    end_time: float = typer.Option(0.0, "--end", "-e", help="End time in seconds (0 = end of video)"),
    fps: float = typer.Option(None, "--fps", help="Frames per second for segmentation"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Processing mode"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Streaming JSONL output"),
):
    """Segment objects in a video by prompt."""
    global console
    console = Console(stderr=json_output or jsonl)
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    if not prompt.strip():
        console.print("[red]Error:[/red] Prompt cannot be empty.")
        raise typer.Exit(code=1)

    config = load_config()
    mode_config = get_mode_config(config, mode)

    actual_mode, warning = auto_downgrade_mode(mode, config)
    if warning:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    seg_fps = fps or config.get("defaults", {}).get("segmentation_fps", 0.25)
    resolution = mode_config.get("resolution", 768)

    video_meta = probe(str(input_path))
    actual_end = end_time if end_time > 0 else video_meta["duration_seconds"]

    console.print(f"[dim]Segmenting '{prompt}' from {start_time}s to {actual_end}s[/dim]")

    frames = smart_sample(
        str(input_path),
        fps=seg_fps,
        max_frames=30,
        max_resolution=resolution,
    )

    frames = [f for f in frames if start_time <= f["timestamp"] <= actual_end]

    if not frames:
        console.print(f"[red]No frames found in time range {start_time}-{actual_end}s.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Analyzing {len(frames)} frames for '{prompt}'...[/dim]")

    from storage.paths import run_cache_dir
    cache_dir = str(run_cache_dir(config))
    cache = RunCache(cache_dir)
    run_dir = cache.create_run("seg_" + prompt.replace(" ", "_")[:20], video_meta)

    log_vram("before_segmentation")
    sam_config = config.get("models", {}).get("segmentation", {})
    try:
        provider = create_segmenter(sam_config)
    except (ImportError, ValueError) as e:
        console.print(f"[red]Segmentation unavailable:[/red] {e}")
        raise typer.Exit(code=1)

    try:
        result = provider.segment_frames(
            [(f["timestamp"], f["image"]) for f in frames],
            prompt,
            run_dir,
        )
    except ImportError as e:
        console.print(f"[red]Segmentation unavailable:[/red] {e}")
        console.print("Install dependencies: pip install transformers ultralytics")
        raise typer.Exit(code=1)
    finally:
        if config.get("gpu_policy", {}).get("unload_after_job", True):
            provider.unload()

    log_vram("after_segmentation")

    objects = []
    for obj_data in result.get("objects", []):
        obj = SegmentObject(
            object_id=f"{prompt.replace(' ', '_')}_{obj_data.get('time_seconds', 0):.0f}",
            timestamps=[obj_data.get("timestamp", "00:00")],
            mask_paths=[obj_data.get("mask_path", "")] if obj_data.get("mask_path") else [],
            bbox=obj_data.get("bbox", []),
        )
        objects.append(obj)

    output = SegmentResult(
        prompt=prompt,
        frames_processed=result.get("frames_processed", 0),
        masks_dir=str(run_dir / "masks"),
        objects=objects,
    )

    cache.save_artifact(run_dir, "segmentation.json", output.model_dump())

    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_probe({"type": "segmentation", "objects_found": len(objects)})
        for obj in objects:
            writer.emit("object", obj.model_dump())
        writer.emit_done({"objects": len(objects), "frames": output.frames_processed})
        writer.flush()

    if json_output:
        typer.echo(json.dumps(output.model_dump(), indent=2))
    elif not jsonl:
        _display_segment_result(output)


def _display_segment_result(result: SegmentResult):
    """Display segmentation result."""
    console.print(Panel(
        f"Processed {result.frames_processed} frames for '[bold]{result.prompt}[/bold]'",
        title="[bold cyan]Segmentation Complete[/bold cyan]",
    ))

    table = Table()
    table.add_column("Object ID", style="cyan")
    table.add_column("Timestamps", style="yellow")
    table.add_column("Masks", style="green")

    for obj in result.objects:
        ts = ", ".join(obj.timestamps[:3])
        if len(obj.timestamps) > 3:
            ts += f" ... (+{len(obj.timestamps) - 3})"
        masks = str(len(obj.mask_paths))
        table.add_row(obj.object_id[:25], ts, masks)

    console.print(table)
    console.print(f"\n[dim]Masks saved to: {result.masks_dir}[/dim]")
