# Open Vision

![Open Vision - local vision for AI agents](.github/banner.png)

Open Vision is a local-first video and image analysis CLI for AI agents. Give it a video file, URL, or still image to extract frames, transcribe audio, and return timestamped evidence as JSON, JSONL, and files. Vision-capable agents can inspect the frames directly; other agents can use results from a configured vision model. Transcripts are persisted to a local SQLite index and become searchable for their video.

The flow is simple: **video or URL → Open Vision → visual and audio evidence → JSON, JSONL, and files → agent.** The CLI works with any agent that can run a command and consume its output. No framework-specific SDK is required.

Open Vision is local-first. Local video, local models, and the local SQLite index stay on your machine. Nothing is uploaded unless you use a remote source such as a video URL or configure a hosted model provider. Local providers need no API key or cloud account.

```
openvision observe video.mp4
openvision observe screenshot.png
openvision ask video.mp4 "what color is the car?"
openvision ask screenshot.png "what text is on this form?"
openvision transcribe video.mp4
openvision ask video.mp4 "is there a person?" --decision noul
```

## For Agents

Choose the handoff that matches your agent. Vision-capable agents can inspect the original frame images; text-only agents can use observations from a configured local or hosted VLM.

### Vision-capable agents

```bash
openvision observe video.mp4 --raw-frames --transcribe --json
```

This skips local VLM analysis and returns frame paths with the audio timeline, so your agent can do the visual reasoning. Use `--jsonl` for incremental events in streaming pipelines:

```bash
openvision observe video.mp4 --raw-frames --transcribe --jsonl
```

### Text-only agents

```bash
openvision observe video.mp4 --json
```

Open Vision analyzes the sampled frames with a configured vision model and returns structured observations and time-grounded evidence. Machine-readable output can be saved, passed to another tool, or consumed directly by the agent.

### Video memory for agents

Open Vision can turn separate observations into a persistent, searchable timeline. Successful `observe` runs index available evidence in a local SQLite database. `transcribe` runs are also persisted and merged into the same index, so speech becomes searchable alongside visual evidence. Agents can search across videos, inspect a video's time hierarchy, or ask a question about one source and have retrieved intervals checked against stored evidence frames.

This gives agents a way to return to specific moments across long recordings and multiple videos, instead of relying on a single summary or reprocessing the source from scratch. Speech, visual notes, speakers, and frame or mask references remain distinct and time-grounded. Speaker labels are associated with nearby visual events by time; they do not identify a visible person.

```bash
openvision memory sources
openvision memory search "when does the speaker demonstrate the setup?"
openvision memory ask lecture.mp4 "What happens during the setup demonstration?"
```

The index uses SQLite full-text search, not embeddings. `memory ask` uses the configured VLM to verify retrieved intervals against stored evidence frames.

## Install

```bash
pip install -e .
```

That's it. yt-dlp is included for YouTube URLs. Speech-model weights are installed separately. No GPU required (but it helps with vision and diarization).

**Optional extras:**

```bash
uv sync --extra diarize     # Local pyannote Community-1 + CUDA PyTorch (Windows/Linux)
uv sync --extra segment     # Ultralytics segmentation support
```

Transcription uses a locally installed Parakeet or Whisper model. Install Parakeet with `openvision install parakeet`. Optional pyannote Community-1 diarization requires the `diarize` extra, Hugging Face access approval, and `hf auth login`. Use `--no-transcribe` or `--no-diarize` to skip those stages. Segmentation is on demand with `--segment` (or follows the selected processing profile); SAM 3 requires its package and gated weights. The CLI reports unavailable stages instead of pretending they ran.

## 30-Second Start

```bash
# Check that everything works
openvision status

# Analyze a video
openvision observe video.mp4

# Ask a specific question
openvision ask video.mp4 "is there a dog in this video?"

# Typed decision (cheaper, returns probability)
openvision ask video.mp4 "is there a dog?" --decision noul --provider clef

# Transcribe with speaker labels
openvision transcribe video.mp4 --diarize

# Works with YouTube too
openvision observe "https://youtube.com/watch?v=..."
```

## What It Does

**Observe:** Extracts keyframes from video or analyzes a still image, runs a vision model, and builds a cross-modal timeline (visual + audio). Returns a structured summary. Use `observe --read-text` to ask the VLM to quote clearly legible text.

**Ask:** Targeted question about video or image content. "What color is the car?" returns a direct answer with evidence frames. Use `--decision noul|choice|score` for typed answers (probabilities instead of prose) — cheaper and faster for classification. Decision models include Luna (OpenAI/OpenRouter), Clef, Clef Flash, Clef Omni, Jev (OpenRouter), and Jev via OpenCode Zen (text-only, free). Select with `--provider clef`, `--provider jev`, `--provider zen`, or `--provider luna-decisions`.

**Transcribe:** Speech-to-text using Parakeet (default, fast, runs on CPU) or Whisper. Supports speaker diarization. Every run is persisted to the local SQLite index and becomes searchable for its video.

**Segment:** Text-prompt object segmentation. "person" returns masked regions with bounding boxes.

**Video memory:** Successful video `observe` runs automatically index their available evidence into the local SQLite database and report the indexing result in `stages.memory`. `--memory` requests denser long-video coverage (about one visual sample per 30 seconds, capped at 300 frames), transcription/diarization, and sampled segmentation. The database keeps a whole-video root, five-minute sections, and thirty-second evidence windows.

**Track:** Frame-by-frame object tracking across a video.

**Probe:** Video metadata: duration, resolution, FPS, codec, estimated processing cost.

## Features

### Long-video memory details

```bash
# Run a higher-coverage analysis. Its transcript and visual evidence join the
# same local index used by every other successful observe run.
openvision observe lecture.mp4 --memory

# Discover indexed videos, inspect the hierarchy, search across all sources,
# or ask about one source video.
openvision memory status
openvision memory sources
openvision memory outline --video lecture.mp4
openvision memory search "when does the speaker demonstrate the setup?"
openvision memory search "red dial" --video lecture.mp4
openvision memory ask lecture.mp4 "What happens during the setup demonstration?"
```

`observe` saves run artifacts and indexes the evidence it produced in `~/.openvision/runs/runs.db` (or the configured cache directory). `--save-observations` is separate: it writes a human-readable Markdown note under `~/.openvision/observations/videos/`; it is not required for database search. `memory search` works across all indexed sources; add `--video` to filter. `memory sources` lists what is available, and `memory outline` shows its time hierarchy. `memory ask VIDEO QUESTION` retrieves matching intervals and asks the configured VLM to verify them against stored evidence frames; it needs a reachable VLM. Retrieval uses SQLite full-text search, not embeddings.

Raw-frame mode is an agent handoff: its transcript can be indexed, but its images have no searchable visual descriptions until a vision-capable agent analyzes them. Use normal `observe` (or `observe --memory`) when you want visual evidence indexed for later retrieval. Re-running an unchanged observation uses its cache and refreshes its memory index; `--force` recomputes it.

| Feature | Command |
|---------|---------|
| Visual analysis | `observe video.mp4` or `observe image.png` |
| Targeted Q&A | `ask video.mp4 "question"` or `ask image.png "question"` |
| Typed decisions (noul/choice/score) | `ask video.mp4 "question" --decision noul --provider clef` |
| Read visible text with the VLM | `observe video.mp4 --read-text` |
| Speech-to-text (persisted) | `transcribe video.mp4` |
| Speaker labels | `transcribe video.mp4 --diarize` |
| Automatic memory indexing | `observe video.mp4` |
| Higher-coverage long-video memory | `observe video.mp4 --memory` |
| Browse indexed videos | `memory sources` |
| Memory retrieval | `memory search "query"` / `memory ask video.mp4 "question"` |
| Object segmentation | `segment video.mp4 "prompt"` |
| Object tracking | `track video.mp4 "prompt"` |
| YouTube support | `observe "https://..."` |
| Time windowing | `observe video.mp4 --start 1:30 --end 2:00` |
| Contact sheets | `observe video.mp4 --grid` |
| HTML viewer | `observe video.mp4 --viewer` |
| Frame selection report | `observe video.mp4 --report` |
| Keep audio | `observe video.mp4 --keep-audio` |
| Cookie auth | `observe "url" --cookies-from-browser chrome` |
| Inline DNN inference | `observe video.mp4 --dnn-model yolov8n.onnx` |
| Streaming JSONL | `observe video.mp4 --jsonl` |
## Agent integration

The CLI is the stable, framework-independent agent interface. Any agent that can invoke a command and read standard output can integrate through JSON for one-shot calls or JSONL for incremental events:

```bash
openvision observe video.mp4 --json
openvision memory search "the presenter changes the setting" --json
```

The automated tests exercise both machine-output protocols and check that JSONL output contains only parseable events. This validates the integration contract without tying Open Vision to a particular agent framework.

## Backends

Open Vision auto-detects what's running. No config needed.

**Vision models:** LM Studio, llama.cpp, Gemini (video-capable), OpenAI, Groq, Together AI, vLLM. Auto-detects what's running — no port or URL configuration needed. For a non-standard port, set `models.vlm.<provider>.base_url` once in `~/.openvision/config.yaml`.

**Decision models:** Luna (OpenAI/OpenRouter), Clef, Clef Flash, Clef Omni, Jev (OpenRouter), Jev via OpenCode Zen (text-only, free). Typed answers (noul/choice/score) return probabilities instead of prose — cheaper and faster for classification. Use `--provider clef`, `--provider jev`, `--provider zen`, or `--provider luna-decisions`. Zen is text-only (frames rejected with `DECISION_TEXT_ONLY`).

**Speech-to-text:** Parakeet TDT 0.6B v3 (default, INT8, ~640MB, CPU), Whisper (optional). Every transcript is persisted to the local SQLite index and becomes searchable.

**Hardware:** Works on CPU. GPU (CUDA) is faster for VLM inference. Parakeet runs efficiently on CPU. Text-only models are detected before inference and fail fast with an actionable message.

## Configuration

Defaults work out of the box. Customize via `~/.openvision/config.yaml` or environment variables:

```bash
export OPENVISION_CONFIG=my_config.yaml
export OPENVISION_HOME=/path/to/data
export OPENVISION_VLM_BASE_URL=http://127.0.0.1:9999/v1
export OPENVISION_VLM_MODEL=google/gemma-4-e2b
```

Three processing modes:

| Mode | Best for |
|------|----------|
| `fast` | Quick inspection, low resource usage |
| `balanced` (default) | Most workflows |
| `detailed` | High-value clips, maximum detail |

All data lives in `~/.openvision` (override with `OPENVISION_HOME`).

## Contributing

```bash
git clone https://github.com/michielhdoteth/openvision.git
cd openvision
pip install -e ".[test]"
pytest tests/ -v
```

Architecture: `openvision/` (public Python API and schemas), `cli/` (Typer commands), `core/` (video, audio, sampling, streaming), `providers/` (VLM and STT backends), and `storage/` (SQLite cache and paths). The command-line interface and its JSON and JSONL output are the agent integration contract.

If Open Vision makes video useful to your agent, [give the project a star](https://github.com/michielhdoteth/openvision). It helps more people discover the project and tells us what to keep building.

## License

MIT
