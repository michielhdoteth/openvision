"""openvision ask <path> <question> - Ask a specific question about a video."""
import typer
import json
from pathlib import Path
from typing import Optional, cast
from rich.console import Console
from rich.panel import Panel

from core.video import probe
from core.download import is_url
from core.sampling import smart_sample, find_keyframes_for_question
from core.errors import OpenVisionError
from providers.decisions import DecisionProvider
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
    tier: str = typer.Option("auto", "--tier", help="Who answers: auto (bounded answer space -> decision model), vlm, decision"),
    decision: Optional[str] = typer.Option(None, "--decision", "--decide", help="Decision answer shape: noul (true/false probability), choice (candidate labels), score (ordered levels)"),
    choices: Optional[str] = typer.Option(None, "--choices", help="Comma-separated candidate answers for a choice decision"),
    levels: Optional[str] = typer.Option(None, "--levels", help="Comma-separated ordered levels for a score decision"),
    min_confidence: float = typer.Option(0.6, "--min-confidence", min=0.0, max=1.0, help="Probability bar for a confident decision answer"),
    provider_name: str = typer.Option(None, "--provider", help="Provider: auto, lmstudio, llamacpp, gemini, openai, groq, together, vllm, localai, or a decision model (luna-decisions, clef, clef-flash, clef-omni, jev, decisions)"),
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

    config = load_config()
    mode_config = get_mode_config(config, mode)

    try:
        decision_kind = _normalize_decision_kind(decision, choices, levels)
    except OpenVisionError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(code=1)

    # A still image has no frame track to sample: answer over the single frame
    # instead of routing it through video extraction.
    from core.image import is_image_file
    if not is_url(path) and is_image_file(input_path):
        if decision_kind is not None:
            console.print("[yellow]Note:[/yellow] --decision uses frames; falling back to the VLM for a still image.")
        import cli.commands.observe as observe_cmd
        observe_cmd._handle_image(
            input_path, question, None, json_output, config,
            mode_config.get("resolution", 768), max_tokens,
            provider_name, jsonl, False,
        )
        return

    fps = mode_config.get("fps", 0.5)
    max_frames = max_frames or mode_config.get("max_frames", 60)
    resolution = mode_config.get("resolution", 768)

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

    resolved_tier = (tier or "auto").strip().lower()
    if resolved_tier not in ("auto", "vlm", "decision"):
        console.print("[red]Error:[/red] --tier must be auto, vlm, or decision.")
        raise typer.Exit(code=1)

    use_decision = resolved_tier == "decision" or (
        resolved_tier == "auto" and decision_kind is not None
    )
    if use_decision:
        try:
            result = _run_decision(
                frames,
                question,
                decision_kind or "noul",
                choices,
                levels,
                min_confidence,
                config,
                provider_name,
            )
        except OpenVisionError as e:
            console.print(f"[red]Error: {e.message}[/red]")
            raise typer.Exit(code=1)

        if writer:
            writer.emit_summary(result.answer, [e.get("answer", "") for e in result.evidence])
            writer.emit_done({
                "confidence": result.confidence,
                "evidence_count": len(result.evidence),
                "tier": "decision",
            })
            writer.flush()

        if json_output:
            typer.echo(json.dumps(result.model_dump(), indent=2))
        elif not jsonl:
            _display_ask_result(result, question)
        return

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

    if evidence:
        best = max(evidence, key=lambda e: len(e["answer"]))
        final_answer = best["answer"]
        confidence = "high" if len(evidence) >= 2 else "medium"

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


def _split_csv(value: Optional[str]) -> list[str]:
    """Split a comma or pipe separated option into trimmed labels."""
    if not value:
        return []
    return [part.strip() for part in value.replace("|", ",").split(",") if part.strip()]


def _normalize_decision_kind(
    decision: Optional[str], choices: Optional[str], levels: Optional[str]
) -> Optional[str]:
    """Resolve --decision to noul, choice, or score (None = VLM path)."""
    aliases = {
        "label": "choice",
        "labels": "choice",
        "choices": "choice",
        "pick": "choice",
        "bool": "noul",
        "boolean": "noul",
        "yes/no": "noul",
        "predicate": "noul",
        "rating": "score",
        "scale": "score",
    }
    kind = (decision or "").strip().lower()
    kind = aliases.get(kind, kind)

    if not kind:
        if choices:
            return "choice"
        if levels:
            return "score"
        return None

    if kind not in ("noul", "choice", "score"):
        raise OpenVisionError(
            "DECISION_BAD_KIND",
            f"Unknown --decision '{decision}'. Use noul, choice, or score.",
        )
    return kind


def _evenly(frames: list, limit: int) -> list:
    """Pick at most `limit` frames spread across the whole timeline."""
    limit = max(int(limit or 1), 1)
    if len(frames) <= limit:
        return list(frames)
    step = len(frames) / limit
    picked = [frames[min(int(index * step), len(frames) - 1)] for index in range(limit)]
    if picked[-1] is not frames[-1]:
        picked[-1] = frames[-1]
    return picked


def _run_decision(
    frames: list,
    question: str,
    kind: str,
    choices: Optional[str],
    levels: Optional[str],
    min_confidence: float,
    config: dict,
    provider_name: Optional[str],
) -> AskResult:
    """Answer one typed question by sending the sampled frames in a single request."""
    decision_config = dict((config.get("models", {}) or {}).get("decisions", {}) or {})
    if provider_name:
        decision_config["provider"] = provider_name
        # A named provider alias carries its own model id.
        decision_config.pop("model", None)
    resolved_name = decision_config.pop("provider", "decisions")

    try:
        provider = ProviderRegistry.create({"provider": resolved_name, **decision_config})
    except ValueError as e:
        raise OpenVisionError("DECISION_UNKNOWN_PROVIDER", str(e)) from e

    try:
        if getattr(provider, "kind", "chat") != "decision":
            raise OpenVisionError(
                "DECISION_WRONG_PROVIDER",
                f"'{resolved_name}' is a chat model and cannot answer typed questions. "
                "Use a decision provider: decisions, luna-decisions, clef, clef-flash, "
                "clef-omni, jev, or openai-decisions.",
            )

        criteria: Optional[list[str]] = None
        if kind == "choice":
            criteria = _split_csv(choices)
            if len(criteria) < 2:
                raise OpenVisionError(
                    "DECISION_NEEDS_CHOICES",
                    "A choice decision needs candidate answers: --choices a,b,c.",
                    {"flag": "--choices"},
                )
        elif kind == "score":
            criteria = _split_csv(levels)
            if len(criteria) < 2:
                raise OpenVisionError(
                    "DECISION_NEEDS_LEVELS",
                    "A score decision needs ordered levels: --levels low,medium,high.",
                    {"flag": "--levels"},
                )

        if frames and not getattr(provider, "supports_images", True):
            raise OpenVisionError(
                "DECISION_TEXT_ONLY",
                f"{getattr(provider, 'preset', 'this')} decisions are text-only: "
                "frames would be billed as text and ignored. Use --provider jev "
                "(OpenRouter) or Jev's own API when the frames must be seen.",
                {"provider": "openrouter"},
            )
        used = _evenly(frames, getattr(provider, "max_images", 8))
        decision = cast(DecisionProvider, provider)
        response = decision.decide(
            [frame["image"] for frame in used],
            question,
            kind=kind,
            criteria=criteria,
        )
    finally:
        provider.close()

    answer = response.first()
    text = answer.as_text()
    agreement = answer.agreement()
    if agreement >= max(0.85, min_confidence):
        confidence = "high"
    elif agreement >= min_confidence:
        confidence = "medium"
    else:
        confidence = "low"

    if len(used) > 1:
        span = f"{used[0]['timestamp_str']} - {used[-1]['timestamp_str']}"
    else:
        span = used[0]["timestamp_str"]

    evidence = [{
        "timestamp": span,
        "time_seconds": used[0]["timestamp"],
        "answer": text,
        "confidence": confidence,
        "frames": len(used),
        "model": response.model,
        "provider": response.provider,
    }]

    return AskResult(
        answer=text,
        evidence=evidence,
        confidence=confidence,
        tokens=response.tokens,
    )


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
