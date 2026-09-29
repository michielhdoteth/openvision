"""openvision status - Show system status and run diagnostic checks."""
import typer
import subprocess
import shutil
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
):
    """Show openvision system status, GPU state, cache info. Use --doctor for diagnostics."""
    global console
    console = Console(stderr=json_output)
    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    # GPU info
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
        try:
            detected = ProviderRegistry.detect()
            payload["providers"] = detected
        except Exception:
            payload["providers"] = []
        payload["models"] = _get_model_status_list()
        if doctor:
            payload["diagnostics"] = _run_diagnostics(config)
        typer.echo(json.dumps(payload, indent=2))
    else:
        _display_status(result, runs, safety)
        _display_provider_health()
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
            _display_diagnostics(config)


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
    gpu_table.add_row("VRAM", f"{result.vram_used_gb:.1f} / {result.vram_total_gb:.1f} GB"
                      if hasattr(result, 'vram_used_gb') else
                      f"{result.vram_free_gb:.1f} GB free")

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


def _display_provider_health():
    """Display health status of all VLM providers."""
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


# ---------------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Doctor diagnostics
# ---------------------------------------------------------------------------

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
    """Check if a Python package is importable."""
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        return {"name": pkg, "installed": True, "version": ver}
    except ImportError:
        return {"name": pkg, "installed": False, "version": None}


def _check_gpu_compute() -> dict:
    """Check GPU compute capability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return {"ok": True, "name": name, "capability": f"{cap[0]}.{cap[1]}", "vram_gb": round(vram, 1)}
    except Exception:
        pass
    # Fallback: try nvidia-smi
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return {"ok": True, "name": parts[0].strip(), "vram_gb": round(float(parts[1].strip().split()[0]) / 1024, 1)}
    except Exception:
        pass
    return {"ok": False, "name": "none", "vram_gb": 0}


def _check_parakeet_model(config: dict) -> dict:
    """Check if Parakeet model files exist."""
    paths = config.get("paths") or {}
    models_dir = Path(paths.get("models_dir") or "models")
    if not models_dir.is_absolute():
        from storage.paths import get_home
        home = get_home(config)
        models_dir = home / "models"

    target = models_dir / "parakeet-tdt-0.6b-v3-int8"
    required = ["model.onnx", "tokens.txt"]
    present = [f.name for f in target.iterdir()] if target.exists() else []
    missing = [f for f in required if f not in present]
    return {
        "name": "parakeet-tdt-0.6b-v3-int8",
        "path": str(target),
        "installed": target.exists() and len(missing) == 0,
        "present_files": present,
        "missing_files": missing,
    }


def _run_diagnostics(config: dict) -> list[dict]:
    """Run all diagnostic checks and return results."""
    checks = []

    # 1. Binaries
    for binary in ["ffmpeg", "ffprobe", "yt-dlp"]:
        checks.append({"category": "binary", **_check_binary(binary)})

    # 2. GPU
    checks.append({"category": "gpu", **_check_gpu_compute()})

    # 3. Python packages
    for pkg in ["cv2", "torch", "numpy", "typer", "rich"]:
        checks.append({"category": "python", **_check_python_package(pkg)})

    # 4. OpenCV 5 engine check
    try:
        import cv2
        ver = cv2.__version__
        has_new_engine = hasattr(cv2, 'dnn') and hasattr(cv2.dnn, 'ENGINE_AUTO')
        cuda_count = 0
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            pass
        checks.append({
            "category": "opencv",
            "name": f"opencv {ver}",
            "installed": True,
            "version": f"{'new DNN engine' if has_new_engine else 'classic engine'} | CUDA devices: {cuda_count}",
        })
    except Exception:
        checks.append({"category": "opencv", "name": "opencv", "installed": False, "version": "not installed"})

    # 4. STT providers
    for pkg, label in [("sherpa_onnx", "parakeet"), ("faster_whisper", "faster-whisper"), ("whisper", "openai-whisper")]:
        checks.append({"category": "stt", "name": label, **_check_python_package(pkg)})

    # 5. Parakeet model
    checks.append({"category": "model", **_check_parakeet_model(config)})

    # 6. VLM providers
    try:
        detected = ProviderRegistry.detect()
        for p in detected:
            checks.append({
                "category": "vlm",
                "name": p.get("name", "?"),
                "installed": p.get("status") == "available",
                "version": ", ".join(p.get("models", [])[:3]),
            })
    except Exception as e:
        checks.append({"category": "vlm", "name": "registry", "installed": False, "version": str(e)[:80]})

    return checks


def _display_diagnostics(config: dict):
    """Display doctor-style diagnostic checks."""
    checks = _run_diagnostics(config)

    # Group by category
    categories = {}
    for c in checks:
        cat = c.pop("category", "other")
        categories.setdefault(cat, []).append(c)

    for cat, items in categories.items():
        table = Table(title=cat.upper(), show_lines=False)
        table.add_column("Check", style="cyan", min_width=24)
        table.add_column("Status")
        table.add_column("Details")

        for item in items:
            name = item.get("name", "?")
            installed = item.get("installed", item.get("ok", False))
            version = item.get("version", "")
            path_val = item.get("path", "")
            missing = item.get("missing_files", [])

            if installed:
                status = "[green]OK[/green]"
            else:
                status = "[red]MISSING[/red]"

            details = version or path_val or ""
            if missing:
                details += f" (missing: {', '.join(missing)})"

            table.add_row(name, status, details)

        console.print(table)

    # Summary
    total = len(checks)
    passed = sum(1 for c in checks if c.get("installed", c.get("ok", False)))
    if passed == total:
        console.print(f"\n[green bold]All {total} checks passed.[/green bold]")
    else:
        console.print(f"\n[yellow]{passed}/{total} checks passed, {total - passed} issues found.[/yellow]")
