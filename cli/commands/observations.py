"""openvision observations - Manage saved video observations."""
import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from datetime import datetime

from storage.paths import observations_dir
from storage.config import load_config

console = Console()


def _obs_dir() -> Path:
    """Resolve stable observations directory (not CWD-relative)."""
    try:
        config = load_config()
    except Exception:
        config = None
    return observations_dir(config)


def observations_cmd(
    action: str = typer.Argument("list", help="Action: list, view, delete"),
    positional_name: Optional[str] = typer.Argument(
        None, help="Observation name (alternative to --name)",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Observation name to view/delete"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Manage saved video observations."""
    name = name or positional_name
    if action == "list":
        _list_observations(json_output)
    elif action in ("view", "delete"):
        if not name:
            console.print(f"[red]Error:[/red] pass the observation name (--name or positional) for the {action} action")
            raise typer.Exit(code=1)
        if action == "view":
            _view_observation(name)
        else:
            _delete_observation(name)
    else:
        console.print(f"[red]Unknown action: {action}[/red] (use: list, view, delete)")
        raise typer.Exit(code=1)


def _list_observations(json_output: bool = False):
    """List all saved observation markdown notes."""
    obs = _obs_dir()
    observations = sorted(obs.glob("*.md"), reverse=True)

    if not observations:
        console.print(
            f"[dim]No saved observations yet in {obs}. "
            "Use --save-observations with observe.[/dim]"
        )
        return

    if json_output:
        result = []
        for m in observations:
            result.append(
                {
                    "name": m.stem,
                    "path": str(m),
                    "size": m.stat().st_size,
                    "modified": datetime.fromtimestamp(m.stat().st_mtime).isoformat(),
                }
            )
        console.print(json.dumps(result, indent=2))
        return

    table = Table(title=f"Saved Observations ({len(observations)})")
    table.add_column("Name", style="cyan")
    table.add_column("Date", style="yellow")
    table.add_column("Size", style="green")
    table.add_column("Path", style="dim")

    for m in observations:
        modified = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size = m.stat().st_size
        table.add_row(m.stem, modified, f"{size} B", str(m))

    console.print(table)
    console.print(f"[dim]Observations dir: {obs}[/dim]")


def _view_observation(name: str):
    """View a specific observation file."""
    obs = _obs_dir()
    obs_file = _resolve_observation(obs, name)
    if not obs_file.exists():
        matches = list(obs.glob(f"*{name}*.md"))
        if not matches:
            console.print(f"[red]Observation not found: {name}[/red]")
            console.print(f"[dim]Looked in: {obs}[/dim]")
            raise typer.Exit(code=1)
        obs_file = matches[0]

    content = obs_file.read_text(encoding="utf-8")
    console.print(Markdown(content))


def _delete_observation(name: str):
    """Delete an observation file."""
    obs = _obs_dir()
    obs_file = _resolve_observation(obs, name)
    if not obs_file.exists():
        matches = list(obs.glob(f"*{name}*.md"))
        if not matches:
            console.print(f"[red]Observation not found: {name}[/red]")
            raise typer.Exit(code=1)
        obs_file = matches[0]

    obs_file.unlink()
    console.print(f"[green]Deleted:[/green] {obs_file.name}")


def _resolve_observation(obs: Path, name: str) -> Path:
    """Resolve an observation name without permitting path or glob traversal."""
    if not name or Path(name).name != name or name in {".", ".."} or any(c in name for c in "*?[]"):
        console.print("[red]Observation name must be a plain filename, without path or wildcard characters.[/red]")
        raise typer.Exit(code=1)
    candidate = (obs / f"{name}.md").resolve()
    if candidate.parent != obs.resolve():
        console.print("[red]Observation path is outside the observations directory.[/red]")
        raise typer.Exit(code=1)
    if candidate.exists():
        return candidate
    matches = list(obs.glob(f"*{name}*.md"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        console.print(f"[red]Ambiguous observation name: {name}. Use the exact filename stem.[/red]")
        raise typer.Exit(code=1)
    return candidate
