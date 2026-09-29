"""openvision ask <path> <question> - Ask a specific question about a video."""
import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from core.video import probe
from core.sampling import smart_sample, find_keyframes_for_question
from providers.registry import ProviderRegistry
from providers.base import TokenUsage
from storage.config import load_config, get_mode_config
from openvision.models import AskResult

console = Console()


def _fmt_tokens(tokens: TokenUsage) -> str:
    """Format token usage for display."""
    parts = [f"{tokens.prompt_tokens} prompt"]
    if tokens.reasoning_tokens > 0:
        parts.append(f"{tokens.completion_tokens} completion ({tokens.reasoning_tokens} reasoning)")
    else:
        parts.append(f"{tokens.completion_tokens} completion")
    return f"{tokens.total_tokens} total | " + " + ".join(parts)


def ask_cmd(
    path: str = typer.Argument(..., help="Path to video file"),
    question: str = typer.Argument(..., help="Question to answer about the video"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Search mode: keyframes, scene_changes, dense"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Override max tokens for VLM response"),
    max_frames: int = typer.Option(None, "--max-frames", min=1, max=300, help="Maximum frames to inspect (overrides the mode budget)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Streaming JSONL output"),
    provider_name: str = typer.Option(None, "--provider", help="VLM provider: auto, lmstudio, ollama, llamacpp, openai, groq, together, vllm, localai"),
    raw_frames: bool = typer.Option(False, "--raw-frames", help="Skip VLM — return raw frame paths for vision-capable agents"),
):
    """Ask a targeted question about video content."""
    global console
    console = Console(stderr=json_output or jsonl)
    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_start(path, mode=question[:80])

    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    if not question.strip():
        console.print("[red]Error:[/red] Question cannot be empty.")
        raise typer.Exit(code=1)

    # Load config
    config = load_config()
    mode_config = get_mode_config(config, mode)

    fps = mode_config.get("fps", 0.5)
    max_frames = max_frames or mode_config.get("max_frames", 60)
    resolution = mode_config.get("resolution", 768)

    # Determine search mode
    search_mode = None
    if mode == "dense":
        search_mode = "dense"
    elif mode == "scene_changes":
        search_mode = "scenes"
    else:
        search_mode = "keyframes"  # default

    # Adjust sampling based on search mode
    if search_mode == "dense":
        dense_fps = min(fps * 3, 2.0)  # Max 2 FPS for dense
        frames = smart_sample(str(input_path), fps=dense_fps, max_frames=max_frames * 2, max_resolution=resolution)
    elif search_mode == "scenes":
        frames = smart_sample(str(input_path), fps=fps, max_frames=max_frames, max_resolution=resolution, scene_threshold=15.0)  # More sensitive
    else:
        frames = smart_sample(str(input_path), fps=fps, max_frames=max_frames, max_resolution=resolution)

    if not frames:
        console.print("[red]No frames extracted from video.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Analyzing {len(frames)} frames to answer: {question}[/dim]")

    # --- raw-frames mode: skip VLM, return frame paths ---
    if raw_frames:
        import tempfile
        from core.image import save_frame
        from openvision.models import RawFrame

        tmp_dir = Path(tempfile.mkdtemp(prefix="openvision_raw_"))
        raw = []
        for i, frame in enumerate(frames):
            fname = f"frame_{i:04d}.jpg"
            fpath = tmp_dir / fname
            save_frame(frame["image"], str(fpath))
            raw.append(RawFrame(index=i, timestamp=frame["timestamp"], path=str(fpath)))

        result = {
            "question": question,
            "raw_frames": [r.model_dump() for r in raw],
            "frame_dir": str(tmp_dir),
            "total_frames": len(raw),
        }

        if json_output:
            typer.echo(json.dumps(result, indent=2))
        elif not jsonl:
            console.print(f"[green]Extracted {len(raw)} raw frames to {tmp_dir}[/green]")
            console.print("[dim]Ready for vision-capable agent processing.[/dim]")
        if writer:
            for frame in raw:
                writer.emit_frame(frame.index, frame.timestamp, frame.path)
            writer.emit_done({"raw_frames": len(raw), "frame_dir": str(tmp_dir)})
            writer.flush()
        return

    # Query VLM
    vlm_config = config.get("models", {}).get("vlm", {})
    if max_tokens:
        vlm_config = {**vlm_config, "max_tokens": max_tokens}
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

    evidence = []
    total_tokens = TokenUsage()

    for frame in frames:
        try:
            resp = provider.describe_image(frame["image"], question)
            answer = resp.content
            total_tokens.prompt_tokens += resp.tokens.prompt_tokens
            total_tokens.completion_tokens += resp.tokens.completion_tokens
            total_tokens.reasoning_tokens += resp.tokens.reasoning_tokens
            total_tokens.total_tokens += resp.tokens.total_tokens
        except Exception as e:
            console.print(f"[dim]Error on frame {frame['timestamp_str']}: {e}[/dim]")
            continue

        # Only include frames that provide useful answers (not empty, not "I don't know")
        if answer and len(answer) > 5 and not _is_uncertain(answer):
            evidence.append({
                "timestamp": frame["timestamp_str"],
                "time_seconds": frame["timestamp"],
                "answer": answer,
                "confidence": "medium",
            })

        # Stop early if we found strong evidence
        if len(evidence) >= 3:
            console.print(f"[dim]Found sufficient evidence, stopping early.[/dim]")
            break

    provider.close()

    # Compile final answer from evidence
    if evidence:
        # Use the most detailed answer as primary
        best = max(evidence, key=lambda e: len(e["answer"]))
        final_answer = best["answer"]
        confidence = "high" if len(evidence) >= 2 else "medium"

        # Add synthesis if multiple evidence points
        if len(evidence) > 1:
            timestamps = [e["timestamp"] for e in evidence]
            final_answer += f" (observed at {', '.join(timestamps)})"
    else:
        final_answer = f"Could not find evidence in the video to answer: {question}"
        confidence = "low"

    result = AskResult(
        answer=final_answer,
        evidence=evidence,
        confidence=confidence,
        tokens={"prompt_tokens": total_tokens.prompt_tokens, "completion_tokens": total_tokens.completion_tokens, "reasoning_tokens": total_tokens.reasoning_tokens, "total_tokens": total_tokens.total_tokens},
    )

    if writer:
        writer.emit_summary(final_answer, [e.get("description", "") for e in evidence])
        writer.emit_done({"confidence": confidence, "evidence_count": len(evidence)})
        writer.flush()

    if json_output:
        typer.echo(json.dumps(result.model_dump(), indent=2))
    elif not jsonl:
        _display_ask_result(result, question)


def _is_uncertain(answer: str) -> bool:
    """Check if a VLM response indicates uncertainty."""
    uncertain_phrases = [
        "i cannot", "i can't", "unable to", "no information",
        "not visible", "cannot determine", "not clear",
        "i don't know", "there is no", "does not appear",
        "insufficient", "unclear", "not enough",
        "could not connect", "timed out after", "returned http ",
        "unexpected error contacting",
    ]
    lower = answer.lower()
    return any(phrase in lower for phrase in uncertain_phrases)


def _display_ask_result(result: AskResult, question: str):
    """Display ask result in rich format."""
    # Color by confidence
    colors = {"high": "green", "medium": "yellow", "low": "red"}
    color = colors.get(result.confidence, "white")

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[bold]Confidence:[/bold] [{color}]{result.confidence.upper()}[/{color}]")
    console.print(Panel(result.answer, title="[bold cyan]Answer[/bold cyan]"))

    if result.evidence:
        console.print("\n[bold]Evidence:[/bold]")
        for ev in result.evidence:
            ts = ev.get("timestamp", "??:??")
            ans = ev.get("answer", "")
            if len(ans) > 120:
                ans = ans[:120] + "..."
            console.print(f"  [dim][{ts}][/dim] {ans}")

    if result.tokens:
        console.print(f"\n[dim]Tokens: {_fmt_tokens(result.tokens)}[/dim]")
