# openvision -- Video Perception CLI for AI Agents

Open Vision gives AI agents local video/audio perception and time-grounded retrieval. Local inference requires no API keys; optional hosted providers or remote source URLs use the network only when selected.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `openvision probe <video>` | Video metadata (duration, fps, codec, resolution) |
| `openvision observe <video>` | Visual analysis + cross-modal timeline |
| `openvision ask <video> "question"` | Targeted visual Q&A |
| `openvision transcribe <video>` | Speech-to-text (Parakeet or Whisper) |
| `openvision segment <video> "prompt"` | Text-prompt object segmentation (SAM) |
| `openvision track <video> "prompt"` | Object tracking across frames |
| `openvision status` | GPU, VRAM, provider health, cache stats |
| `openvision status --doctor` | Full diagnostic checks |
| `openvision install parakeet` | Install Parakeet STT model (~350MB) |
| `openvision install whisper --model base` | Install Whisper model |
| `openvision install list-models` | Show all models and install status |
| `openvision observations list` | Saved observation ledgers |
| `openvision observations view <id>` | View a saved observation |
| `openvision memory sources` | List videos indexed in the shared local database |
| `openvision memory status` | Show indexed source and evidence counts |
| `openvision memory search "query"` | Search across all indexed videos |
| `openvision memory outline --video <path>` | Inspect a video's time hierarchy |
| `openvision memory ask <path> "question"` | Retrieve and VLM-check evidence for one video |

## Global Flags

| Flag | Purpose |
|------|---------|
| `--json` | Output structured JSON (for agent consumption) |
| `--jsonl` | Streaming JSONL output -- one event per line |
| `--provider <name>` | VLM backend: `auto`, `lmstudio`, `ollama`, `llamacpp`, `openai`, `groq`, `together` |
| `--stt-provider <name>` | STT backend: `parakeet` (default), `whisper` |
| `--mode <mode>` | Processing mode: `fast`, `balanced`, `detailed` |
| `--start <time>` | Start time (MM:SS, HH:MM:SS, or seconds) |
| `--end <time>` | End time |
| `--transcribe` / `-t` | Transcribe audio |
| `--diarize` | Add speaker labels (requires pyannote.audio) |
| `--raw-frames` | Skip VLM -- return frame paths for vision-capable agents |
| `--grid` | Generate 3x3 contact sheet |
| `--text-anchors` | Force frames at subtitle-cue timestamps |
| `--viewer` | Generate self-contained HTML viewer |
| `--report` | Generate keep/drop visualization HTML |
| `--keep-audio` | Save full soundtrack as audio.m4a |
| `--cookies <file>` | Netscape cookie file for authenticated videos |
| `--cookies-from-browser <name>` | Extract cookies from browser (chrome, firefox, etc.) |
| `--dnn-model <path>` | ONNX model for inline DNN inference during extraction |
| `--save-observations` | Save observation ledger to ~/.openvision/observations/ |
| `--memory` | Increase long-video coverage; transcribe, diarize, and sample segmentation |

## Backend Selection

### VLM Backends (vision analysis)

Auto-detection order: LM Studio -> Ollama -> llama.cpp -> cloud APIs

| Backend | Endpoint | Notes |
|---------|----------|-------|
| LM Studio | `localhost:1234/v1` | Default local |
| Ollama | `localhost:11434` | Native API |
| llama.cpp | `localhost:8080/v1` | Direct GGUF |
| OpenAI | `api.openai.com/v1` | GPT-4V, GPT-4o |
| Groq | `api.groq.com/openai/v1` | Fast cloud |
| Together AI | `api.together.xyz/v1` | Open-source VLMs |

### STT Backends (speech-to-text)

| Backend | Default | Notes |
|---------|---------|-------|
| Parakeet TDT 0.6B v3 | Yes | INT8 quantized via sherpa-onnx, ~640MB, CPU |
| Whisper | No | Requires `faster-whisper` package |

## Workflows

### Basic video analysis

```bash
openvision observe video.mp4
openvision observe video.mp4 --json
```

### Vision-capable agent (recommended)

```bash
openvision observe video.mp4 --raw-frames --transcribe --json
```

Returns frame paths + audio timeline. Your agent reads the frames directly.

### Text-only agent

```bash
openvision observe video.mp4 --json
```

Open Vision runs a local VLM and returns structured observations.

### Targeted Q&A

```bash
openvision ask video.mp4 "what color is the car?"
openvision ask video.mp4 "is there a dog in this video?" --json
```

### Transcription

```bash
openvision transcribe video.mp4
openvision transcribe video.mp4 --diarize
openvision transcribe video.mp4 --from 1:30 --to 2:00
```

### Contact sheet

```bash
openvision observe video.mp4 --grid
openvision observe video.mp4 --grid
```

### HTML viewer

```bash
openvision observe video.mp4 --viewer --transcribe
```

Opens in any browser -- no server needed.

### Streaming for pipelines

```bash
openvision observe video.mp4 --jsonl --transcribe
```

Each event is a single JSON line: start, probe, frame, transcript, summary, done.

### YouTube videos

```bash
openvision observe "https://youtube.com/watch?v=..."
openvision observe "https://youtube.com/watch?v=..." --cookies-from-browser chrome
```

### Cross-video search

```bash
# Successful observe runs are automatically indexed in the local SQLite database.
openvision observe video1.mp4 --json
openvision observe video2.mp4 --json
openvision memory sources
openvision memory search "the red dial"
openvision memory search "the red dial" --video video1.mp4
openvision memory outline --video video1.mp4
openvision memory ask video1.mp4 "What happens when the dial turns?"
```

`--save-observations` writes a readable Markdown ledger; it is separate from the SQLite run and memory database. Use `observe --memory` for denser long-video coverage. Raw-frame mode is an agent handoff: transcript evidence may be indexed, but the frames are not searchable visually until a vision-capable agent analyzes them.

### Diagnose issues

```bash
openvision status --doctor
```

## Data Paths

| Path | Default |
|------|---------|
| Home | `~/.openvision` (`OPENVISION_HOME` env var) |
| Observations | `~/.openvision/observations/videos` |
| Downloads | `~/.openvision/downloads` |
| Runs and video memory (SQLite) | `~/.openvision/runs/runs.db` (or configured cache directory) |

## Safety

1. **Provider must be running.** Start your VLM backend first (LM Studio, Ollama, etc.) or use `--raw-frames` to skip the VLM.
2. **Parakeet model downloads on first use.** ~640MB cached in `~/.openvision/models/`.
3. **YouTube cookies.** Some videos require authentication. Use `--cookies-from-browser chrome`.
4. **`--jsonl` is streaming.** Events emit as they happen. Pipe to `jq` for pretty-printing.
5. **`--raw-frames` skips the VLM.** Fastest mode for vision-capable agents.
6. **Time windowing is explicit.** Use `observe --start 1:30 --end 2:00` to focus an interval.
7. **SQLite cache auto-imports.** Legacy JSON runs migrate on first access.
