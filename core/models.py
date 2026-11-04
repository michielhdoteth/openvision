"""Model registry and download management for Open Vision.

Centralizes model metadata (URLs, sizes, cache paths) and provides
download/install functionality for STT backends.
"""
import hashlib
import logging
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from storage.paths import get_home

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Metadata for a downloadable model."""
    name: str
    provider: str  # "parakeet" or "whisper"
    description: str
    url: str
    size_bytes: int
    compressed_size_bytes: int = 0  # 0 = same as size_bytes (no compression)
    files: list[str] = None  # expected files after extraction

    def __post_init__(self):
        if self.files is None:
            self.files = []


# Parakeet TDT 0.6B v3 INT8 -- the default STT model
PARAKEET_MODEL = ModelEntry(
    name="parakeet-tdt-0.6b-v3-int8",
    provider="parakeet",
    description="NVIDIA Parakeet TDT 0.6B v3 (INT8, 25 languages, ~640MB)",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
    size_bytes=670_000_000,
    compressed_size_bytes=350_000_000,
    files=["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
)

# Whisper models via faster-whisper (Hugging Face Hub)
WHISPER_MODELS = {
    "tiny": ModelEntry(
        name="tiny",
        provider="whisper",
        description="Whisper tiny (39M params, ~75MB)",
        url="https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin",
        size_bytes=75_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "base": ModelEntry(
        name="base",
        provider="whisper",
        description="Whisper base (74M params, ~140MB)",
        url="https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
        size_bytes=140_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "small": ModelEntry(
        name="small",
        provider="whisper",
        description="Whisper small (244M params, ~460MB)",
        url="https://huggingface.co/Systran/faster-whisper-small/resolve/main/model.bin",
        size_bytes=460_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "medium": ModelEntry(
        name="medium",
        provider="whisper",
        description="Whisper medium (769M params, ~1.5GB)",
        url="https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin",
        size_bytes=1_500_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "large-v3": ModelEntry(
        name="large-v3",
        provider="whisper",
        description="Whisper large-v3 (1550M params, ~3.1GB)",
        url="https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin",
        size_bytes=3_100_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
}

# All models indexed by (provider, name)
ALL_MODELS: dict[tuple[str, str], ModelEntry] = {
    ("parakeet", "parakeet-tdt-0.6b-v3-int8"): PARAKEET_MODEL,
}
ALL_MODELS.update({("whisper", k): v for k, v in WHISPER_MODELS.items()})



def models_dir(config: Optional[dict] = None) -> Path:
    """Central model cache directory: ~/.openvision/models/

    Overridable via OPENVISION_MODELS_DIR (env) or paths.models (config,
    absolute or relative to the openvision home).
    """
    import os
    env = os.environ.get("OPENVISION_MODELS_DIR")
    if env:
        d = Path(env).expanduser()
    else:
        configured = ((config or {}).get("paths") or {}).get("models")
        if configured:
            candidate = Path(configured).expanduser()
            d = candidate if candidate.is_absolute() else get_home(config) / candidate
        else:
            d = get_home(config) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parakeet_model_dir() -> Path:
    """Where the parakeet model files live."""
    return models_dir() / "parakeet-tdt-0.6b-v3-int8"


def whisper_model_dir(model_size: str) -> Path:
    """Where the whisper model files live."""
    return models_dir() / f"whisper-{model_size}"


def repo_local_model_dir(name: str) -> Path:
    """Repo-local fallback used by dev checkouts: <project>/models/<name>."""
    return Path(__file__).resolve().parent.parent / "models" / name


def _expected_files(provider: str, name: str) -> list[str]:
    """Files that must exist for a model to count as installed.

    Whisper accepts either the HF-legacy vocabulary.json or the newer
    vocabulary.txt + tokenizer.json layout (faster-whisper writes the latter).
    """
    if provider == "parakeet":
        return ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
    if provider == "whisper":
        return ["model.bin", "config.json"]
    return []


def _files_present(provider: str, name: str, directory: Path) -> bool:
    """True when every required file (or an accepted alternative) is present."""
    required = _expected_files(provider, name)
    if not required or not directory.is_dir():
        return False
    for f in required:
        if not (directory / f).exists():
            return False
    if provider == "whisper":
        has_vocab = (directory / "vocabulary.json").exists() or (
            (directory / "vocabulary.txt").exists() and (directory / "tokenizer.json").exists()
        )
        if not has_vocab:
            return False
    return True


def resolve_model_dir(provider: str, name: str, config: Optional[dict] = None) -> tuple[Path, str]:
    """Resolve where a model actually lives so runtime, install and status agree.

    Order: explicit config override -> home cache (~/.openvision/models or
    OPENVISION_MODELS_DIR) -> repo-local dev checkout -> missing (returns the
    home-cache path as the install target).

    Returns (path, origin) with origin in override|home|repo|missing.
    """
    override = ((config or {}).get("models") or {}).get(provider, {}).get("model_dir")
    if override:
        candidate = Path(override).expanduser()
        if _files_present(provider, name, candidate):
            return candidate, "override"
    home = models_dir(config) / name
    if _files_present(provider, name, home):
        return home, "home"
    repo = repo_local_model_dir(name)
    if _files_present(provider, name, repo):
        return repo, "repo"
    return home, "missing"


def adopt_repo_model(provider: str, name: str, config: Optional[dict] = None) -> Optional[Path]:
    """Copy a complete repo-local model into the home cache so every project
    (and a clean reinstall) reuses it. Returns the new path, or None when the
    model is not duplicated in a repo checkout. Never deletes the source.
    """
    path, origin = resolve_model_dir(provider, name, config)
    if origin != "repo":
        return None
    destination = models_dir(config) / name
    destination.mkdir(parents=True, exist_ok=True)
    required = _expected_files(provider, name)
    alternatives = ["vocabulary.txt", "tokenizer.json"] if provider == "whisper" else []
    for filename in required + alternatives:
        src = path / filename
        if src.is_file():
            shutil.copy2(src, destination / filename)
    return destination


def cleanup_partial_downloads(config: Optional[dict] = None) -> list[Path]:
    """Remove leftover partial downloads from interrupted installs."""
    base = models_dir(config)
    removed: list[Path] = []
    for pattern in ("*.part", "parakeet-download.tar.bz2"):
        for stale in base.glob(pattern):
            stale.unlink(missing_ok=True)
            removed.append(stale)
    return removed



@dataclass
class ModelStatus:
    """Status of a model in the cache."""
    name: str
    provider: str
    description: str
    installed: bool
    size_bytes: int
    path: Optional[Path] = None


def check_model_status(provider: str, name: str, config: Optional[dict] = None) -> ModelStatus:
    """Check if a model is installed and its size on disk."""
    entry = ALL_MODELS.get((provider, name))
    if not entry:
        return ModelStatus(
            name=name, provider=provider, description="Unknown model",
            installed=False, size_bytes=0,
        )

    if provider in ("parakeet", "whisper"):
        # Whisper directories are stored as whisper-<size>.
        dir_name = name if provider == "parakeet" else f"whisper-{name}"
        model_path, origin = resolve_model_dir(provider, dir_name, config)
    else:
        return ModelStatus(
            name=name, provider=provider, description=entry.description,
            installed=False, size_bytes=0,
        )

    installed = origin != "missing"

    if installed and model_path.is_dir():
        size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    else:
        size = entry.size_bytes  # estimated download size

    return ModelStatus(
        name=name,
        provider=provider,
        description=entry.description,
        installed=installed,
        size_bytes=size,
        path=model_path if installed else None,
    )


def list_all_models() -> list[ModelStatus]:
    """Return status of all known models."""
    statuses = []
    statuses.append(check_model_status("parakeet", "parakeet-tdt-0.6b-v3-int8"))
    for name in WHISPER_MODELS:
        statuses.append(check_model_status("whisper", name))
    return statuses



def _progress_hook(block_num: int, block_size: int, total_size: int, callback=None):
    """urllib download progress hook."""
    downloaded = block_num * block_size
    if total_size > 0 and callback:
        pct = min(100.0, downloaded * 100.0 / total_size)
        callback(pct, downloaded, total_size)
    elif total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / 1_000_000
        mb_total = total_size / 1_000_000
        print(f"\r  Downloading: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)


def download_file(url: str, dest: Path, callback=None) -> Path:
    """Download a file atomically with progress reporting.

    Writes to ``<dest>.part`` first and renames on success, so an interrupted
    download never leaves a file that looks installed. The ``.part`` file is
    removed on failure (call ``cleanup_partial_downloads`` to sweep older ones).
    """
    logger.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    def hook(block_num, block_size, total_size):
        _progress_hook(block_num, block_size, total_size, callback)

    try:
        urllib.request.urlretrieve(url, str(part_path), reporthook=hook)
        if not part_path.is_file() or part_path.stat().st_size == 0:
            raise RuntimeError(f"download produced no data: {url}")
        part_path.replace(dest)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    print()  # newline after progress
    return dest


def download_and_extract_parakeet(callback=None) -> Path:
    """Download and extract the Parakeet model.

    Downloads the official sherpa-onnx tarball and extracts the
    INT8 quantized model files to ~/.openvision/models/parakeet-tdt-0.6b-v3-int8/

    Returns:
        Path to the extracted model directory
    """
    target_dir = parakeet_model_dir()

    # Already installed in the home cache (or a repo checkout — adopt it so
    # ~/.openvision/models becomes the single cache every caller uses).
    resolved, origin = resolve_model_dir("parakeet", PARAKEET_MODEL.name)
    if origin == "home":
        logger.info(f"Parakeet model already installed at {resolved}")
        return resolved
    if origin == "repo":
        adopted = adopt_repo_model("parakeet", PARAKEET_MODEL.name)
        if adopted:
            logger.info(f"Adopted repo-local Parakeet model into {adopted}")
            return adopted

    # Download tarball (atomic: interrupts leave no fake-installed file)
    tarball_path = target_dir / "parakeet-download.tar.bz2"
    url = PARAKEET_MODEL.url

    print(f"Downloading Parakeet TDT 0.6B v3 (~350MB compressed)...")
    download_file(url, tarball_path, callback=callback)

    print("Extracting model files...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(tarball_path), "r:bz2") as tar:
        members = tar.getmembers()
        for member in members:
            # The tarball has a nested directory structure; we want specific files
            basename = member.name.split("/")[-1]
            if basename in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]:
                member.name = basename
                tar.extract(member, str(target_dir))

    for f in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]:
        if not (target_dir / f).exists():
            raise RuntimeError(f"Extraction failed: {f} not found in {target_dir}")

    tarball_path.unlink(missing_ok=True)
    print(f"Parakeet model installed to {target_dir}")
    return target_dir


def download_whisper_model(model_size: str, callback=None) -> Path:
    """Download a Whisper model via faster-whisper.

    Uses faster-whisper's built-in Hugging Face Hub download.

    Returns:
        Path to the downloaded model directory
    """
    if model_size not in WHISPER_MODELS:
        raise ValueError(f"Unknown whisper model: {model_size}. Choose from: {', '.join(WHISPER_MODELS.keys())}")

    target_dir = whisper_model_dir(model_size)

    # Already installed anywhere on the resolution chain (home cache or repo
    # checkout) — accept the faster-whisper vocabulary.txt+tokenizer.json
    # layout so a complete download is never fetched twice.
    resolved, origin = resolve_model_dir("whisper", f"whisper-{model_size}")
    if origin in ("home", "repo", "override"):
        if origin == "repo":
            adopted = adopt_repo_model("whisper", f"whisper-{model_size}")
            if adopted:
                return adopted
        logger.info(f"Whisper {model_size} already installed at {resolved}")
        return resolved

    try:
        from faster_whisper import download_model
        print(f"Downloading Whisper {model_size} via faster-whisper...")
        download_model(model_size, str(target_dir))
        print(f"Whisper {model_size} installed to {target_dir}")
        return target_dir
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )


def install_model(provider: str, name: str, callback=None) -> Path:
    """Install a model by provider and name.

    Args:
        provider: "parakeet" or "whisper"
        name: Model name (e.g., "parakeet-tdt-0.6b-v3-int8" or "base")
        callback: Optional progress callback

    Returns:
        Path to installed model directory
    """
    if provider == "parakeet":
        return download_and_extract_parakeet(callback=callback)
    elif provider == "whisper":
        return download_whisper_model(name, callback=callback)
    else:
        raise ValueError(f"Unknown provider: {provider}")
