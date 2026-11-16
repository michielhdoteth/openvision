"""Print the LLM agent skill guide."""
import typer
from pathlib import Path
import sys


def skill_cmd():
    """Print the full LLM agent skill guide for openvision.

    Shows all commands, flags, workflows, and backend selection.
    """
    source_tree_skill = Path(__file__).resolve().parents[3] / "SKILL.md"
    installed_skill = Path(sys.prefix) / "share" / "openvision" / "SKILL.md"
    skill_path = source_tree_skill if source_tree_skill.is_file() else installed_skill
    if skill_path.exists():
        typer.echo(skill_path.read_text())
    else:
        typer.echo("SKILL.md not found in project root", err=True)
        raise typer.Exit(1)
