"""openvision probe <path> - Extract video metadata."""
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from core.video import probe, estimate_processing_cost, detect_vfr, detect_color_range
from models import ProbeResult

console = Console()


def probe_cmd(
    path: str = typer.Argument(..., help="Path to video file", exists=True),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Streaming JSONL output"),
    deep: bool = typer.Option(False, "--deep", help="Run VFR detection and color range analysis"),
):
    """
    Extract metadata from a video file without processing it.
    
    Returns duration, resolution, FPS, codec, and estimated processing cost.
    With --deep, also detects VFR mode and color range.
    """
    global console
    console = Console(stderr=json_output or jsonl)
    # Validate file exists
    video_path = Path(path)
    if not video_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)
    
    # Probe the video
    try:
        metadata = probe(str(video_path))
    except Exception as e:
        console.print(f"[red]Error probing video:[/red] {e}")
        raise typer.Exit(code=1)
    
    # Deep analysis: VFR detection + color range
    if deep:
        console.print("[dim]Running deep analysis (VFR + color range)...[/dim]")
        vfr_info = detect_vfr(str(video_path))
        color_info = detect_color_range(str(video_path))
        metadata["vfr_mode"] = vfr_info.get("mode", metadata.get("vfr_mode", "unknown"))
        metadata["vfr_variable_frames"] = vfr_info.get("variable_frames", 0)
        metadata["vfr_constant_frames"] = vfr_info.get("constant_frames", 0)
        metadata["vfr_ratio"] = vfr_info.get("vfr_ratio", 0.0)
        metadata["color_range"] = color_info.get("range_type", "unknown")
    
    # Estimate processing cost
    cost = estimate_processing_cost(metadata)
    metadata["estimated_processing_cost"] = cost
    
    result = ProbeResult(**metadata)

    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_probe(result.model_dump())
        writer.emit_done({"type": "probe"})
        writer.flush()

    if json_output:
        typer.echo(json.dumps(result.model_dump(), indent=2))
    elif not jsonl:
        _display_probe_table(result, str(video_path), deep)


def _display_probe_table(result: ProbeResult, path: str, deep: bool = False):
    """Display probe results in a rich table."""
    table = Table(title=f"Video Probe: {path}", title_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Duration", f"{result.duration_seconds:.1f}s ({_format_duration(result.duration_seconds)})")
    table.add_row("Resolution", f"{result.width}x{result.height}")
    table.add_row("FPS", f"{result.fps:.2f}")
    table.add_row("Codec", result.codec)
    table.add_row("Frame Count", str(result.frame_count))

    # VFR info
    vfr_mode = getattr(result, "vfr_mode", "unknown")
    if vfr_mode != "unknown":
        vfr_color = "yellow" if vfr_mode == "vfr" else "green"
        table.add_row("Frame Rate Mode", f"[{vfr_color}]{vfr_mode.upper()}[/{vfr_color}]")
        if deep:
            vfr_ratio = getattr(result, "vfr_ratio", 0.0)
            v_var = getattr(result, "vfr_variable_frames", 0)
            v_const = getattr(result, "vfr_constant_frames", 0)
            table.add_row("VFR Analysis", f"{v_var} variable / {v_const} constant ({vfr_ratio:.1%} VFR)")

    # Color range
    color_range = getattr(result, "color_range", "unknown")
    if color_range != "unknown":
        table.add_row("Color Range", color_range)

    table.add_row("Has Audio", "Yes" if getattr(result, "has_audio", False) else "No")
    table.add_row("Estimated Cost", result.estimated_processing_cost.upper())
    
    console.print(table)
    
    # Color-coded cost warning
    cost_colors = {"low": "green", "medium": "yellow", "high": "red"}
    color = cost_colors.get(result.estimated_processing_cost, "white")
    console.print(f"\nEstimated processing cost: [{color}]{result.estimated_processing_cost.upper()}[/{color}]")

    # VFR optimization tips
    if vfr_mode == "vfr":
        console.print("\n[yellow]VFR detected![/yellow] Frame timestamps may be irregular.")
        console.print("[dim]Tip: Open Vision uses ffmpeg timestamp-based extraction for VFR videos.[/dim]")


def _format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"
