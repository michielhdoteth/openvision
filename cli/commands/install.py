"""Install and manage STT models for Open Vision."""
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.models import (
    list_all_models,
    check_model_status,
    install_model,
    models_dir,
)

console = Console()
app = typer.Typer(help="Install and manage STT models")


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1_000:
        return f"{size_bytes} B"
    elif size_bytes < 1_000_000:
        return f"{size_bytes / 1_000:.1f} KB"
    elif size_bytes < 1_000_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    else:
        return f"{size_bytes / 1_000_000_000:.1f} GB"


@app.command()
def list_models(
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """List all available models and their install status."""
    models = list_all_models()

    if json_output:
        import json
        typer.echo(json.dumps([{
            "name": m.name, "provider": m.provider, "description": m.description,
            "installed": m.installed, "size_bytes": m.size_bytes,
            "path": str(m.path) if m.path else None,
        } for m in models], indent=2))
        return

    table = Table(title="STT Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Description")

    for m in models:
        status = "[green]Installed[/green]" if m.installed else "[red]Not installed[/red]"
        size = _format_size(m.size_bytes)
        table.add_row(m.name, m.provider, size, status, m.description)

    console.print(table)
    console.print(f"\nModels directory: {models_dir()}")


@app.command()
def get(
    provider: str = typer.Argument(
        help="Provider: parakeet or whisper",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name (e.g., base, small, large-v3 for whisper)",
    ),
    all_models: bool = typer.Option(
        False, "--all", "-a",
        help="Install all models (parakeet + whisper variants)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Install STT models for Open Vision.

    Examples:

        openvision install get parakeet                        # Install parakeet (default)
        openvision install get whisper --model base            # Install whisper base
        openvision install get parakeet --all                  # Install all models

    Shorthands (same behavior, shorter for agents):
        openvision install parakeet
        openvision install whisper --model base
    """
    if all_models:
        _install_all()
        return

    if provider == "parakeet":
        _install_st("parakeet", model, json_output)
    elif provider == "whisper":
        _install_st("whisper", model, json_output)
    else:
        message = f"Unknown provider: {provider}. Use 'parakeet' or 'whisper'."
        if json_output:
            import json
            typer.echo(json.dumps({"error": "UNKNOWN_PROVIDER", "message": message}, indent=2))
        else:
            console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@app.command()
def parakeet(json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON")):
    """Install the default Parakeet STT model (shorthand for `install get parakeet`)."""
    _install_st("parakeet", None, json_output)


@app.command()
def whisper(
    model_size: Optional[str] = typer.Argument(
        None, help="Model size: tiny, base, small, medium, large-v3",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model size (alternative to the positional argument)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Install a Whisper STT model (shorthand for `install get whisper --model SIZE`)."""
    size = model or model_size
    if size is None:
        message = "Whisper requires a model size. Use: openvision install whisper base"
        if json_output:
            import json
            typer.echo(json.dumps({"error": "MISSING_MODEL_SIZE", "message": message}, indent=2))
        else:
            console.print(f"[yellow]{message}[/yellow]")
        raise typer.Exit(1)
    _install_st("whisper", size, json_output)


def _install_st(provider: str, model: Optional[str], json_output: bool = False) -> None:
    """Shared installer for one provider/model pair (machine or human output)."""
    import json

    if provider == "parakeet":
        provider_key, model_key = "parakeet", "parakeet-tdt-0.6b-v3-int8"
    elif provider == "whisper":
        provider_key, model_key = "whisper", f"whisper-{model}"
    else:
        message = f"Unknown provider: {provider}. Use 'parakeet' or 'whisper'."
        if json_output:
            typer.echo(json.dumps({"error": "UNKNOWN_PROVIDER", "message": message}, indent=2))
        else:
            console.print(f"[red]{message}[/red]")
        raise typer.Exit(code=1)

    status = check_model_status(provider_key, model_key)
    if status.installed:
        if json_output:
            typer.echo(json.dumps({
                "model": status.name, "installed": True, "path": str(status.path),
                "message": "already installed",
            }, indent=2))
        else:
            console.print(f"[green]{status.name} model already installed at {status.path}[/green]")
        return

    try:
        path = install_model(provider_key, model_key)
        if json_output:
            typer.echo(json.dumps({"model": status.name, "installed": True, "path": str(path)}, indent=2))
        else:
            console.print(f"[green]{status.name} model installed to {path}[/green]")
    except Exception as e:
        message = f"Failed to install {status.name}: {e}"
        if json_output:
            typer.echo(json.dumps({"error": "INSTALL_FAILED", "message": message}, indent=2))
        else:
            console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


def _install_all():
    """Install all models."""
    console.print(Panel("Installing all STT models", title="Open Vision Install"))

    console.print("\n[bold]1/6 Parakeet TDT 0.6B v3[/bold]")
    _install_st("parakeet", None)

    whisper_sizes = ["tiny", "base", "small", "medium", "large-v3"]
    for i, size in enumerate(whisper_sizes, 2):
        console.print(f"\n[bold]{i}/6 Whisper {size}[/bold]")
        _install_st("whisper", size)

    console.print("\n[green bold]All models installed.[/green bold]")
