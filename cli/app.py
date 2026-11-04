"""Open Vision CLI application."""
import typer
import sys
from pathlib import Path
from typing import Optional
from cli.commands import probe, observe, ask, segment, track, status, observations, transcribe, ground, skill, install, memory
from core.errors import OpenVisionError, handle_cli_error


def _force_utf8_console() -> None:
    """Force UTF-8 on the console so Rich does not fall back to ASCII boxes.

    This lives here, not in the root ``__main__.py`` shim, because
    ``[project.scripts] openvision = "cli.app:app"`` imports this module directly
    and would otherwise bypass the reconfigure, rendering mangled box drawing
    and ellipsis characters in the console script.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_console()

app = typer.Typer(
    name="openvision",
    help="Self-hosted, privacy-first AI vision tool for CLI and agents",
    no_args_is_help=True,
)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("openvision")
    except Exception:
        return "1.0.0"


def version_callback(value: bool):
    if value:
        from storage.paths import get_home, observations_dir, downloads_dir

        typer.echo(f"openvision {_package_version()}")
        typer.echo(f"home: {get_home()}")
        typer.echo(f"observations: {observations_dir()}")
        typer.echo(f"downloads: {downloads_dir()}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and data paths",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Self-hosted, privacy-first AI vision tool for CLI and agents."""
    pass


app.command(name="probe")(probe.probe_cmd)
app.command(name="observe")(observe.observe_cmd)
app.command(name="ask")(ask.ask_cmd)
app.command(name="segment")(segment.segment_cmd)
app.command(name="track")(track.track_cmd)
app.command(name="status")(status.status_cmd)
app.command(name="observations")(observations.observations_cmd)
app.command(name="transcribe")(transcribe.transcribe_cmd)
app.command(name="ground")(ground.ground_cmd)
app.command(name="skill")(skill.skill_cmd)
app.add_typer(memory.app, name="memory")
app.add_typer(install.app, name="install")


def load_config() -> dict:
    """Load config from default.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Entry point for the CLI."""
    try:
        app()
    except OpenVisionError as e:
        error_info = e.to_dict()
        if "--json" in sys.argv:
            import json
            from rich.console import Console
            console = Console()
            console.print(json.dumps(error_info, indent=2))
        else:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel(
                f"[red]{e.code}[/red]\n{e.message}",
                title="Error",
                border_style="red",
            ))
            if e.suggested_retry:
                console.print("[yellow]Suggested retry:[/yellow]")
                import json as _json
                console.print(_json.dumps(e.suggested_retry, indent=2))
        sys.exit(1)
    except Exception as e:
        from rich.console import Console
        console = Console()
        error_info = handle_cli_error(e)
        console.print(f"[red]Error:[/red] {error_info.get('message', str(e))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
