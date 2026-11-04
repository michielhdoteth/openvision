# Changelog

All notable changes to Open Vision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Transcript persistence**: `transcribe` now writes every run into the local run cache and video memory, so a transcript survives stdout truncation and becomes searchable for its video. `--no-run` opts out; `--save-to <path>` writes plain text to an exact path (instead of always beside the input)
- `VideoMemoryStore.add_transcript` merges speech into an existing video hierarchy without dropping visual, segmentation, or agent evidence, and updates audio text in place instead of duplicating it

- **Gemini provider** (native `generateContent` REST API): the first backend that accepts whole-video input via `fileData`; select with `--provider gemini`, key via `GEMINI_API_KEY`/`GOOGLE_API_KEY`
- **Decision (chat) tier** (`models.decision` in config): `memory ask` and the `--memory` profile synthesis now run on the cheap backend (local llama.cpp by default) instead of the observe-time VLM
- **Typed decision models** (`providers/decisions.py`, `models.decisions`): `ask --decision yesno|choice|score` answers with probabilities instead of generated text, one request per video over the sampled frames. `--choices`/`--levels` declare the answer space, so free-form questions work as labels; `--tier auto|vlm|decision` routes, and `--provider` accepts `decisions`, `luna-decisions`, `clef`, `clef-flash`, `clef-omni`, `jev`, `openai-decisions` (OpenRouter by default, native OpenAI `/v1/decisions` as the second preset)
- **OpenCode Zen decision route** (`zen`/`jev-zen`, `jev-1.13-free`): Jev at `POST /zen/v1/systemone` with `OPENCODE_API_KEY`. Text-only — Zen tokenizes image parts as text, so frames are rejected with `DECISION_TEXT_ONLY` rather than silently costing tokens
- **Bidirectional memory**: `openvision memory note <video> "..." --at 1:23 [--frame path]` writes agent observations back into video memory as searchable `agent` evidence; `memory notes` lists them
- `openvision memory import` re-indexes saved observation runs into memory without re-analysis
- `--json` output for `install get`, `install parakeet`, `install whisper`, and `install list-models`
- `OPENVISION_MODELS_DIR` env var / `paths.models` config to relocate the model cache

### Changed

- **`status --doctor` is usable again**: package presence is checked with `find_spec` instead of importing STT backends (sherpa_onnx + faster_whisper alone cost 95 s cold), GPU compute comes from `nvidia-smi` instead of initializing a CUDA context (26.7 s cold), `core.gpu` imports torch lazily (~12 s saved on every `status`), and provider detection and binary probes run concurrently. Checks stream as they complete and respect a `--timeout` budget (`--timeout 0` disables). Measured locally: `--doctor` from over 120 s to roughly 18-20 s, plain `status` to roughly 11-15 s
- **SKILL.md now states what `transcribe` actually persists** and that run-cache auto-import covers `observe`/`ask`/`transcribe` artifacts only, closing the gap that made agents hunt runs.db for transcripts that were never written
- **One model cache**: install, status, and runtime now resolve models through one chain (config `model_dir` override -> `~/.openvision/models` -> models found beside the CLI). Existing models are adopted into the home cache, never re-downloaded. Downloads are atomic (`.part` then rename; interrupted downloads leave no partial files)
- Whisper accepts the faster-whisper `vocabulary.txt` + `tokenizer.json` layout, so a complete download is never fetched twice
- `--memory` enforces a coverage floor of one frame per 30 s even when an explicit small `--max-frames` is passed, and reports the bump
- Raw-frames runs index frame paths as visual evidence so speech can be matched to frames
- `--memory` runs build a `video_profile.json` global picture (deterministic digest, upgraded by the decision tier) that seeds the memory root summary
- `observations view` accepts the name positionally (`view <name>`) as well as `--name`
- `openvision install parakeet` / `install whisper [size]` shorthand commands resolve to the same installer as `install get`
- Transcript re-parsing understands HH:MM:SS, fixing wrong timestamps for videos over an hour

### Removed

- **Ollama provider**: use llama.cpp as the local backend. Provider registry, config, docs, and tests updated

### Fixed

- `status --doctor` no longer checks for a nonexistent `model.onnx`; it reports the resolved Parakeet path and origin
- Missing STT models raise a structured `STT_MODEL_UNAVAILABLE` error with an install retry hint

## [1.0.0] - 2026-09-08

### Added

- **Parakeet TDT 0.6B v3** as default STT provider (INT8 quantized via sherpa-onnx, ~640MB, 10x faster than Whisper on CPU)
- **Provider selection** for STT: `--provider parakeet|whisper` on transcribe and observe commands
- **Python API**: `from openvision import process, transcribe` for programmatic access
- **Streaming JSONL output**: `--jsonl` flag on observe, transcribe, probe, segment, ask commands
- **SQLite run storage**: Replaced flat JSON with SQLite database (`runs.db`) for run metadata, artifacts, frames, and transcripts
- **Cross-video search index**: `core.memory.VideoMemory` for searching frames and transcripts across all processed videos
- **Contact sheets**: `--grid` flag generates 3x3 keyframe grids
- **Text anchors**: `--text-anchors` forces frame extraction at subtitle-cue timestamps (parses .srt/.vtt)
- **Local HTML viewer**: `--viewer` generates self-contained HTML with base64-embedded keyframes and transcript
- **Keep/drop report**: `--report` visualizes frame selection decisions as HTML timeline
- **Time windowing**: `--from` and `--to` flags for processing video segments
- **Cookie support**: `--cookies` and `--cookies-from-browser` for authenticated video downloads
- **Keep audio**: `--keep-audio` saves full soundtrack as audio.m4a
- **OpenCV 5 DNN provider**: `providers.opencv_dnn.OpenCVDNNProvider` for running ONNX models via OpenCV 5's rewritten DNN engine (ENGINE_AUTO/NEW/CLASSIC)
- **FFmpeg ONNX Runtime DNN filter**: `core.dnn_filter.DNNFilterPipeline` for inline ONNX model inference during frame extraction, with `--dnn-model` flag on observe
- **FFmpeg 9.0 CUDA transpose**: GPU-accelerated frame orientation correction with automatic CPU fallback
- **OpenCV 5 upgrade**: `pyproject.toml` bumped to `opencv-python>=5.0`
- **yt-dlp upgrade**: Updated to 2026.8.19 for YouTube download support
- **Doctor diagnostics**: `openvision status --doctor` checks ffmpeg, yt-dlp, GPU, OpenCV 5 engine, providers, parakeet model, and Python packages
- **155 new unit tests** covering stream, memory, contact_sheet, report, viewer, parakeet, opencv_dnn, dnn_filter, cache (SQLite), and api modules

### Changed

- `whisper_device` renamed to `stt_device` throughout the codebase
- `storage.cache.RunCache` rewritten to use SQLite with backward-compatible auto-import of existing JSON runs
- All commands now use `--provider` / `--stt-provider` flags instead of hardcoded whisper references
- README rewritten to be product-focused (less technical, more use-case driven)

### Removed

- All external project references (Agent-Reach)
- MCP adapter (not planned -- CLI access is cleaner)

## [0.9.0] - 2026-07-30

### Added

- Initial release with 10 commands (observe, ask, transcribe, segment, track, probe, ground, status, observations)
- Multi-provider VLM architecture (LM Studio, Ollama, llama.cpp, OpenAI, Groq, Together AI, vLLM)
- Auto-detection of running backends
- Processing modes (fast, balanced, detailed)
- Cross-modal timeline (visual + audio)
- Raw frames mode for vision-capable agents
- Speaker diarization support (pyannote.audio)
- SAM segmentation (ultralytics)
- Object tracking
- Observation ledgers
