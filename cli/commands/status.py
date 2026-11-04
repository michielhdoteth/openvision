"""openvision status - Show system status and run diagnostic checks."""
import time
import sys
import importlib.util
import concurrent.futures
import typer
import subprocess
import shutil
from typing import Any, Iterator, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from core.gpu import get_vram_info, get_gpu_name, check_safe_mode
from storage.cache import RunCache
from storage.config import load_config
from openvision.models import StatusResult
from providers.registry import ProviderRegistry

console = Console()


def status_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    doctor: bool = typer.Option(False, "--doctor", help="Run full diagnostic checks"),
    timeout: int = typer.Option(60, "--timeout", help="Seconds to spend on --doctor checks (0 = no limit)"),
):
    """Show openvision system status, GPU state, cache info. Use --doctor for diagnostics."""
    global console
    console = Console(stderr=json_output)
    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    # Probe providers once; --doctor reuses the same result instead of re-probing.
    try:
        detected = ProviderRegistry.detect()
    except Exception:
        detected = []

    gpu_name = get_gpu_name()
    vram = get_vram_info()
    safety = check_safe_mode(config)

    # Paths + cache (stable under openvision_HOME)
    from storage.paths import get_home, observations_dir, downloads_dir, run_cache_dir

    home = get_home(config)
    obs = observations_dir(config)
    dls = downloads_dir(config)
    runs_path = run_cache_dir(config)
    run_cache = RunCache(str(runs_path))
    runs = run_cache.list_runs()

    result = StatusResult(
        gpu=gpu_name or "unknown",
        vram_total_gb=round(vram["total_gb"], 1) if vram["total_gb"] > 0 else 0.0,
        vram_free_gb=round(vram["free_gb"], 1) if vram["free_gb"] > 0 else 0.0,
        loaded_models=_get_loaded_models(config),
        safe_mode=safety.get("safe_mode", True),
        queue=0,
    )

    if json_output:
        payload = result.model_dump()
        payload["paths"] = {
            "home": str(home),
            "observations": str(obs),
            "downloads": str(dls),
            "runs": str(runs_path),
        }
        payload["providers"] = detected
        payload["models"] = _get_model_status_list()
        if doctor:
            payload["diagnostics"] = list(_iter_diagnostics(config, detected, timeout))
        typer.echo(json.dumps(payload, indent=2))
    else:
        _display_status(result, runs, safety)
        _display_provider_health(detected)
        _display_model_status()
        path_table = Table(show_header=False, box=None, title="Data Paths")
        path_table.add_column("Key", style="cyan")
        path_table.add_column("Path")
        path_table.add_row("Home", str(home))
        path_table.add_row("Observations", str(obs))
        path_table.add_row("Downloads", str(dls))
        path_table.add_row("Runs", str(runs_path))
        console.print(path_table)
        obs_count = len(list(obs.glob("*.md")))
        console.print(f"[dim]Saved observations: {obs_count}[/dim]")

        if doctor:
            console.print()
            _display_diagnostics(config, detected, timeout)


def _get_loaded_models(config: dict) -> list[str]:
    """Get list of configured models."""
    models = []
    vlm = config.get("models", {}).get("vlm", {})
    if vlm.get("provider"):
        models.append(f"{vlm['provider']}:{vlm.get('model', 'unknown')}")
    return models


def _display_status(result: StatusResult, runs: list[dict], safety: dict):
    """Display status in rich format."""
    gpu_table = Table(show_header=False, box=None)
    gpu_table.add_column("Property", style="cyan")
    gpu_table.add_column("Value")

    gpu_table.add_row("GPU", result.gpu)

    free_gb = safety.get("free_gb", 0)
    total_gb = safety.get("total_gb", 0)
    used = total_gb - free_gb
    gpu_table.add_row("VRAM Used", f"{used:.1f} / {total_gb:.1f} GB" if total_gb > 0 else "Unknown")
    gpu_table.add_row("VRAM Free", f"{free_gb:.1f} GB" if free_gb > 0 else "Unknown")

    safe_color = "green" if not result.safe_mode else "red"
    gpu_table.add_row("Safe Mode", f"[{safe_color}]{'ON' if result.safe_mode else 'OFF'}[/{safe_color}]")
    gpu_table.add_row("Safety", safety.get("reason", "Unknown"))

    if result.loaded_models:
        gpu_table.add_row("Configured Models", ", ".join(result.loaded_models))

    console.print(Panel(gpu_table, title="[bold cyan]System Status[/bold cyan]"))

    if runs:
        cache_table = Table(title=f"Cached Runs ({len(runs)})")
        cache_table.add_column("Run", style="cyan")
        cache_table.add_column("Date")
        cache_table.add_column("Path")

        for run in runs[:10]:
            from datetime import datetime
            created = datetime.fromtimestamp(run["created"]).strftime("%Y-%m-%d %H:%M")
            cache_table.add_row(run["name"][:30], created, run["path"])

        console.print(cache_table)
    else:
        console.print("[dim]No cached runs yet.[/dim]")


def _display_provider_health(detected: Optional[list[dict]] = None):
    """Display health status of all VLM providers."""
    if detected is None:
        try:
            detected = ProviderRegistry.detect()
        except Exception as e:
            console.print(f"[dim]Could not probe providers: {e}[/dim]")
            return

    if not detected:
        console.print("[dim]No VLM providers detected.[/dim]")
        return

    provider_table = Table(title="VLM Providers")
    provider_table.add_column("Provider", style="cyan")
    provider_table.add_column("Status")
    provider_table.add_column("URL")
    provider_table.add_column("Models")

    for info in detected:
        name = info.get("name", "unknown")
        status = info.get("status", "unknown")
        url = info.get("url", "")
        models = info.get("models", [])
        error = info.get("error", "")

        if status == "available":
            status_display = "[green]Available[/green]"
            models_display = ", ".join(models[:3]) if models else "none"
        elif status == "timeout":
            status_display = "[yellow]Timeout[/yellow]"
            models_display = error
        else:
            status_display = "[red]Unavailable[/red]"
            models_display = error

        provider_table.add_row(name, status_display, url, models_display)

    console.print(provider_table)



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


def _get_model_status_list() -> list[dict]:
    """Get model status as a list of dicts (for JSON output)."""
    try:
        from core.models import list_all_models
        return [
            {
                "name": m.name,
                "provider": m.provider,
                "description": m.description,
                "installed": m.installed,
                "size_bytes": m.size_bytes,
                "path": str(m.path) if m.path else None,
            }
            for m in list_all_models()
        ]
    except Exception:
        return []


def _display_model_status():
    """Display installed STT models."""
    try:
        from core.models import list_all_models
        models = list_all_models()
    except Exception:
        return

    table = Table(title="STT Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Status")

    for m in models:
        status = "[green]Installed[/green]" if m.installed else "[red]Not installed[/red]"
        size = _format_size(m.size_bytes)
        table.add_row(m.name, m.provider, size, status)

    console.print(table)



def _check_binary(name: str) -> dict:
    """Check if a binary is available on PATH."""
    path = shutil.which(name)
    version = None
    if path:
        try:
            r = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=10)
            output = (r.stdout + r.stderr).strip().split("\n")[0][:120]
            version = output
        except Exception:
            version = "installed (version unknown)"
    return {"name": name, "installed": path is not None, "path": path, "version": version}


def _check_python_package(pkg: str) -> dict:
    """Check if a Python package is installed, without importing it.

    Importing STT backends here costs tens of seconds (sherpa_onnx plus
    faster_whisper alone measured 95s cold) and loads their runtimes into
    this process for no reason. find_spec answers the presence question
    instantly and reports versions only for packages already imported.
    """
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return {"name": pkg, "installed": False, "version": None}
    version = "installed"
    module = sys.modules.get(pkg)
    if module is not None:
        version = str(getattr(module, "__version__", "installed"))
    return {"name": pkg, "installed": True, "version": version}


def _check_gpu_compute() -> dict:
    """Check GPU compute capability.

    Deliberately does NOT call torch.cuda.*: initializing a CUDA context in
    this process measured 26.7s cold and stalls far longer when the GPU is
    busy, which is what made --doctor unusable. nvidia-smi answers the same
    question in about a second.
    """
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 2:
                try:
                    vram_gb = round(float(parts[1].split()[0]) / 1024, 1)
                except ValueError:
                    vram_gb = 0
                cap = parts[2] if len(parts) > 2 else "unknown"
                return {"ok": True, "name": parts[0], "capability": cap, "vram_gb": vram_gb}
    except Exception:
        pass
    return {"ok": False, "name": "none", "capability": "unknown", "vram_gb": 0}


def _check_parakeet_model(config: dict) -> dict:
    """Check if Parakeet model files exist (shared resolution chain)."""
    from core.models import resolve_model_dir, _expected_files
    name = "parakeet-tdt-0.6b-v3-int8"
    target, origin = resolve_model_dir("parakeet", name, config)
    required = _expected_files("parakeet", name)
    present = [f.name for f in target.iterdir()] if target.exists() else []
    missing = [f for f in required if f not in present]
    return {
        "name": name,
        "path": str(target),
        "origin": origin,
        "installed": origin != "missing",
        "present_files": present,
        "missing_files": missing,
    }


def _iter_diagnostics(config: dict, detected: Optional[list[dict]] = None,
                      timeout: int = 60) -> Iterator[dict]:
    """Yield diagnostic checks one at a time, so output survives a slow check.

    A generator rather than a list: results stream to the display as each
    check finishes. Checks over the remaining time budget are marked skipped
    instead of blocking the whole run.
    """
    if detected is None:
        detected = []

    deadline = time.time() + timeout if timeout else None

    def budget_left() -> bool:
        return deadline is not None and time.time() > deadline

    def mark(check: dict) -> dict:
        if check.get("category") == "vlm":
            return check
        if budget_left():
            return {**check, "skipped": True,
                    "version": f"skipped: over {timeout}s budget"}
        return check

    # 1. Binaries (probed concurrently; yt-dlp --version alone costs ~4s on Windows)
    binaries = ["ffmpeg", "ffprobe", "yt-dlp"]
    probe_results: list[Optional[dict]] = [None] * len(binaries)
    with ThreadPoolExecutor(max_workers=len(binaries)) as pool:
        futures = {pool.submit(_check_binary, name): i for i, name in enumerate(binaries)}
        for future in concurrent.futures.as_completed(futures):
            try:
                probe_results[futures[future]] = future.result()
            except Exception as e:
                probe_results[futures[future]] = {"name": binaries[futures[future]],
                                                 "installed": False, "error": str(e)[:80]}
    for index, check in enumerate(probe_results):
        yield mark({"category": "binary", **(check or {"name": binaries[index], "installed": False})})

    # 2. GPU
    yield mark({"category": "gpu", **_check_gpu_compute()})

    # 3. Python packages (presence only, no imports)
    for pkg in ["cv2", "torch", "numpy", "typer", "rich"]:
        yield mark({"category": "python", **_check_python_package(pkg)})

    # 4. STT providers
    for pkg, label in [("sherpa_onnx", "parakeet"), ("faster_whisper", "faster-whisper"), ("whisper", "openai-whisper")]:
        yield mark({"category": "stt", "name": label, **_check_python_package(pkg)})

    # 5. Parakeet model
    yield mark({"category": "model", **_check_parakeet_model(config)})

    # 6. VLM providers (already probed by the caller)
    if detected:
        for p in detected:
            yield {
                "category": "vlm",
                "name": p.get("name", "?"),
                "installed": p.get("status") == "available",
                "version": ", ".join(p.get("models", [])[:3]),
            }


def _run_diagnostics(config: dict, detected: Optional[list[dict]] = None,
                     timeout: int = 60) -> list[dict]:
    """Materialize all diagnostic checks (kept for callers that want a list)."""
    return list(_iter_diagnostics(config, detected, timeout))


def _display_diagnostics(config: dict, detected: Optional[list[dict]] = None,
                         timeout: int = 60):
    """Display doctor-style diagnostic checks, one category at a time."""
    checks = []
    current_cat = None
    items: list[dict] = []

    def flush():
        if current_cat and items:
            table = Table(title=current_cat.upper(), show_lines=False)
            table.add_column("Check", style="cyan", min_width=24)
            table.add_column("Status")
            table.add_column("Details")
            for item in items:
                name = item.get("name", "?")
                installed = item.get("installed", item.get("ok", False))
                version = item.get("version", "")
                path_val = item.get("path", "")
                missing = item.get("missing_files", [])

                if item.get("skipped"):
                    status = "[yellow]SKIP[/yellow]"
                elif installed:
                    status = "[green]OK[/green]"
                else:
                    status = "[red]MISSING[/red]"

                details = version or path_val or ""
                if missing:
                    details += f" (missing: {', '.join(missing)})"
                table.add_row(name, status, details)
            console.print(table)

    for check in _iter_diagnostics(config, detected, timeout):
        cat = check.get("category", "other")
        if cat != current_cat:
            flush()
            items = []
            current_cat = cat
        items.append(check)
        checks.append(check)
    flush()

    total = len(checks)
    skipped = sum(1 for c in checks if c.get("skipped"))
    passed = sum(1 for c in checks
                 if not c.get("skipped") and c.get("installed", c.get("ok", False)))
    if passed + skipped == total:
        console.print(
            f"\n[green bold]All {passed} checks passed"
            + (f" ({skipped} skipped).[/green bold]" if skipped else f".[/green bold]"))
    else:
        console.print(
            f"\n[yellow]{passed}/{total - skipped} checks passed, "
            f"{total - passed - skipped} issues found.[/yellow]")
