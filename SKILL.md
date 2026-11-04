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
| `openvision status --doctor` | Full diagnostic checks (slow cold start, see Safety) |
| `openvision install parakeet` | Install Parakeet STT model (~350MB) |
| `openvision install whisper --model base` | Install Whisper model |
| `openvision install list-models` | Show all models and install status |
| `openvision observations list` | Saved observation ledgers |
| `openvision observations view --name <n>` | View a saved observation |
| `openvision memory sources` | List videos indexed in the shared local database |
| `openvision memory status` | Show indexed source and evidence counts |
| `openvision memory search "query"` | Search across all indexed videos |
| `openvision memory outline --video <path>` | Inspect a video's time hierarchy |
| `openvision memory ask <path> "question"` | Retrieve and VLM-check evidence for one video |
| `openvision memory import` | Re-index saved runs into memory (no re-analysis) |
| `openvision memory note <video> "what you saw" --at 1:23` | Write your own observation back into video memory |

## Global Flags

| Flag | Purpose |
|------|---------|
| `--json` | Output structured JSON (for agent consumption) |
| `--jsonl` | Streaming JSONL output -- one event per line |
| `--provider <name>` | VLM backend: `auto`, `lmstudio`, `llamacpp`, `gemini`, `openai`, `groq`, `together` |
| `--stt-provider <name>` | STT backend: `parakeet` (default), `whisper` |
| `--mode <mode>` | Processing mode: `fast`, `balanced`, `detailed` |
| `--start <time>` | Start time (MM:SS, HH:MM:SS, or seconds) |
| `--end <time>` | End time |
| `--transcribe` / `-t` | Transcribe audio (on by default; use `--no-transcribe` to skip) |
| `--diarize` | Add speaker labels (requires pyannote.audio) |
| `--raw-frames` | Skip VLM -- return frame paths for vision-capable agents |
| `--grid` | Generate a contact sheet from keyframes |
| `--grid-size <n>` | Contact sheet grid dimensions, n x n (1-12, default 3) |
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

Auto-detection order: LM Studio -> llama.cpp -> (no local server found) remote providers

| Backend | Endpoint | Notes |
|---------|----------|-------|
| LM Studio | `localhost:1234/v1` | Default local |
| llama.cpp | `localhost:8080/v1` | Direct GGUF, local default |
| Gemini | `generativelanguage.googleapis.com` | **Also accepts whole-video input**; key via `GEMINI_API_KEY` |
| OpenAI | `api.openai.com/v1` | Vision reasoning models |
| Groq | `api.groq.com/openai/v1` | Fast cloud |
| Together AI | `api.together.xyz/v1` | Open-source VLMs |

Gemini is selected explicitly (`--provider gemini`) because it is keyed and
remote; it is not probed by auto-detection. Its image/frame path works like
any other backend; a `describe_video` API path accepts a video URI directly.

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

### Typed decisions (cheap, no output tokens)

Decision models (Luna Decisions, Jev, Clef) return probabilities instead of
generated text. Your question can still be free-form because the frames travel in
the request `state`; only the *answer space* has to be declared:

```bash
# truth claim -> probability
openvision ask video.mp4 "is a person wearing a helmet?" --decision noul

# any open question, as candidate answers
openvision ask video.mp4 "what is the person doing?" \
  --decision choice --choices "walking,running,sitting,standing"

# ordered severity / intensity
openvision ask video.mp4 "how bad is the panel damage?" \
  --decision score --levels "none,mild,moderate,severe"

# explicit backend; --tier auto already picks the decision tier when
# --decision/--choices/--levels is present
openvision ask video.mp4 "is the door open?" --decision noul --provider clef
```

- Answer shapes: `noul`, `choice` (labels), `score` (ordered levels). `--choices`
  or `--levels` alone imply the shape.
- Output is one line of text plus a probability/confidence, e.g.
  `walking (p=0.86, confidence 0.72)`. `--json` returns the same `AskResult`.
- `--tier auto|vlm|decision`: `auto` (default) uses a decision model when an answer
  space is given and the VLM otherwise; `vlm` forces chat; `decision` forces typed.
- `--min-confidence` (default 0.6) sets the bar for a `medium`/`high` confidence label.
- One request covers all sampled frames, so cost is a single input bill. Frame caps
  are per model (Clef and Jev 4, Luna 128); extra sampled frames are dropped evenly.
- Decision models cannot write prose. For "describe what happens", use the default
  VLM path or `observe`. For open questions phrased as labels, `--decision choice`
  is usually cheaper than a VLM.
- Backends are aliases on the OpenRouter decisions endpoint:
  `decisions`/`luna-decisions` (default), `clef`, `clef-flash`, `clef-omni`, `jev`,
  or `openai-decisions` for native `POST /v1/decisions`. Keys:
  `OPENROUTER_API_KEY`, or `OPENAI_API_KEY` for the native preset.
- `zen`/`jev-zen` route Jev through OpenCode Zen (`POST /zen/v1/systemone`,
  `OPENCODE_API_KEY`, `jev-1.13-free` bills at cost 0). Text-only: it tokenizes
  image parts as text, so `ask --decision` refuses frames with `DECISION_TEXT_ONLY`.
  Use it to judge a transcript or caption; use `jev` on OpenRouter for frames.

### Transcription

```bash
openvision transcribe video.mp4
openvision transcribe video.mp4 --diarize
openvision transcribe video.mp4 --from 1:30 --to 2:00
```

Transcripts are written to the run cache (runs.db) on every run, so they
survive the process and are recoverable with `openvision memory search` or a
direct query against the transcripts table. Use `--no-run` to skip that write.

**Redirect, do not over-read stdout.** `--jsonl` emits one JSON event per line
as it happens, which survives a harness that truncates the middle of a long
print. Plain output and `--json` are one large block; a truncated middle
cannot be recovered from the run cache if `--no-run` was passed.

```bash
openvision transcribe video.mp4 --jsonl > .workspace/transcript.jsonl
```

`--save <file>` writes a plain-text transcript to the exact path you give.
Without a path it lands next to the input file. `--keep-audio` saves the
soundtrack. Diarization (`--diarize`) adds speaker labels and persists with the
segments.

### Contact sheet

```bash
openvision observe video.mp4 --grid
openvision observe video.mp4 --grid --grid-size 4
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
# transcribe runs are persisted to the same runs.db (see Transcription above),
# and only become searchable memory rows after `openvision memory import`.
openvision observe video1.mp4 --json
openvision observe video2.mp4 --json
openvision memory sources
openvision memory search "the red dial"
openvision memory search "the red dial" --video video1.mp4
openvision memory outline --video video1.mp4
openvision memory ask video1.mp4 "What happens when the dial turns?"
```

`--save-observations` writes a readable Markdown ledger; it is separate from the SQLite run and memory database. Use `observe --memory` for denser long-video coverage. Raw-frame mode is an agent handoff: transcript evidence may be indexed, but the frames are not searchable visually until a vision-capable agent analyzes them.

### Connecting to a non-default VLM endpoint

LM Studio (or any OpenAI-compatible server) may run on a port other than 1234.
`openvision` finds it automatically — no URL knowledge required:

```bash
# Just use the provider name. Auto-detection scans common ports (1234, 1235, 8080).
openvision observe image.png --provider lmstudio
openvision observe image.png --provider auto
```

For a non-standard port, configure it once in settings:

```yaml
# ~/.openvision/config.yaml
models:
  vlm:
    provider: lmstudio
    lmstudio:
      base_url: http://127.0.0.1:9999/v1
    model: google/gemma-4-e2b
```

Power users and CI can also set `OPENVISION_VLM_BASE_URL` and `OPENVISION_VLM_MODEL`.
The agent should never need to know or pass a URL.

### Images and text-only models

`observe` and `ask` accept still images (`.png`, `.jpg`, …) directly; no
video probe runs on them, so there is no decoder noise. A model without a
multimodal projector (a text-only GGUF loaded without its `mmproj`) is
detected before any inference and fails fast with an actionable message
instead of injecting an HTTP 500 into an observation:

```
Model qwen2.5-0.5b-instruct at http://127.0.0.1:1235/v1 does not accept
image input. Load a vision-capable model with its projector (mmproj), or
run with --raw-frames so a vision-capable agent can read the frames.
```

`--raw-frames` skips the VLM entirely and returns frame paths, which is the
right path for agents that have their own vision (or none): the agent reads
the images itself and writes back what it saw with `memory note`.

### Diagnose issues

```bash
openvision status --doctor
```

## Data Paths

| Path | Default |
|------|---------|
| Home | `~/.openvision` (`OPENVISION_HOME` env var) |
| Models | `~/.openvision/models/` (`OPENVISION_MODELS_DIR` env var, or `paths.models` in config) |
| Observations | `~/.openvision/observations/videos` |
| Downloads | `~/.openvision/downloads` |
| Runs and video memory (SQLite) | `~/.openvision/runs/runs.db` (or configured cache directory) |

### Model cache and bring-your-own models

All models (Parakeet, Whisper) resolve through one cache in this order:
explicit `model_dir` in config -> home cache -> a model found beside the CLI.
Nothing is re-downloaded when the files already exist. Point at your own
model directory to skip downloads entirely:

```yaml
# ~/.openvision/config.yaml
models:
  parakeet:
    model_dir: "/absolute/path/to/parakeet-tdt-0.6b-v3-int8"
  whisper:
    model_size: "base"
    model_dir: "/absolute/path/to/faster-whisper-base"
```

Diagnostics (`status --doctor`, `install list-models`, install output) all
report the resolved path and where it came from.

### Model tiers

Two tiers exist and they are not interchangeable:

```yaml
models:
  # Chat tier: copies text answers. Used by memory ask and profile synthesis.
  decision:
    provider: "llamacpp"     # local default
    # provider: "gemini"     # hosted, video-capable
    # provider: "openai"     # hosted vision reasoning
    model: "auto"

  # Typed tier: returns probabilities, no output tokens. Used by ask --decision.
  decisions:
    preset: "openrouter"                  # openrouter | openai | zen
    model: "openai/gpt-6-luna-decisions"  # or cloudflare/clef, typesafe/jev-1.14,
                                          # jev-1.13-free (zen, text-only)
    api_key_env: "OPENROUTER_API_KEY"     # OPENAI_API_KEY / OPENCODE_API_KEY
    timeout: 90
    # max_images: 16                      # defaults to the model's cap
```

`models.decision` must point at a chat provider. A typed decision model there is
rejected with an explanation rather than failing on a missing text method.

## Safety

1. **Provider must be running.** Start your VLM backend first (LM Studio, llama.cpp, ...), set a hosted provider key, or use `--raw-frames` to skip the VLM.
2. **Models are cached once.** Parakeet/Whisper download into `~/.openvision/models/` and are reused (interrupted downloads leave no partial files).
3. **YouTube cookies.** Some videos require authentication. Use `--cookies-from-browser chrome`.
4. **`--jsonl` is streaming.** Events emit as they happen. Pipe to `jq` for pretty-printing.
5. **`--raw-frames` skips the VLM.** Fastest mode for vision-capable agents -- and you can write what you see back with `memory note`.
6. **Time windowing is explicit.** Use `observe --start 1:30 --end 2:00` to focus an interval.
7. **SQLite cache auto-imports.** Legacy JSON runs migrate on first access; `memory import` re-indexes saved runs on demand. Only artifacts written by `observe`, `ask`, and `transcribe` are in that path — a bare `transcribe --jsonl` stream is not a run artifact.
8. **`status --doctor` is a cold diagnostic.** It probes models and backend imports, so the first run takes tens of seconds even though `status` alone is fast. It is not a health probe for a timeout-budgeted pipeline — use plain `status`.

## Bidirectional memory (agents with their own vision)

If your agent can see frames itself, use `--raw-frames` and hand back what it
observed so it joins the searchable global picture instead of staying in chat:

```bash
openvision observe video.mp4 --raw-frames --json
openvision memory note video.mp4 "red dial on the left panel" --at 1:23 --frame frames/f_0042.jpg
openvision memory search "red dial"
openvision memory note video.mp4 "operator turns it clockwise" --at 2:10
```

Notes land in the matching 30-second chunk as agent evidence, are full-text
searchable, survive re-observation, and feed the `--memory` root summary.
`memory notes --video <path>` lists them.
