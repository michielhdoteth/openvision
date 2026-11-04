# TRACES.md — How agents use OpenVision + enhancement plan

Status: research + planning only. No code changes have been made.
Captured: 2026-10-30. Sources: agent session logs only (read-only analysis).

## Data sources

1. `C:/Users/michi/.local/share/opencode/opencode.db` (~44 GB SQLite, read-only).
   Queried `part`, `message`, `session`, `project` tables for rows containing `openvision`.
2. `C:/Users/michi/.codex/sessions/**/*.jsonl` (73 files, 4,956 mentions).

Scan scripts (throwaway, in temp): `db_schema.py`, `db_cols.py`, `db_scan.py`,
`db_scan2.py`, `db_errors.py`, `scan_codex.py`, `codex_verbs.py` in
`C:/Users/michi/AppData/Local/Temp/opencode/`.

## 1. How agents actually use OpenVision

### Volume (opencode.db)

| Signal | Value |
|---|---|
| part rows mentioning openvision | 2,256 |
| message rows mentioning openvision | 12 |
| distinct sessions | 175 |
| heavy sessions | ses_f7cbaff1fffe (489), ses_f0b7df83dffe (312), ses_f48f5a152ffe (122) |
| projects (worktree) | `/` (root, top sessions) and `Companies OS/DynaMind` |

### Volume (codex jsonl)

4,956 mentions across 73 files. Top 4 sessions carry ~80% of all mentions:
1,577 / 1,268 / 871 / 253 (dated 10-26, 10-29, 10-22, 10-14) — repeated long
build sessions, not one-off queries.

### Command shapes (verb frequency)

| openvision.db | codex | Command |
|---|---:|---|
| 631 | 80 | `observe` |
| 166 | 6 | `status` |
| 163 | 12 | `install` |
| 158 | 15 | `transcribe` |
| 116 | 9 | `ask` |
| 101 | 2 | `probe` |
| 87 | — | `observations` |
| 67 | 9 | `import` (command does not exist) |
| 53 | — | `track` |
| 47 | — | `--version` (used as a health probe) |
| 32 | 24 | `memory` |
| 28 | 2 | `segment` |
| 22 | — | `--help` |
| 19 | — | `skill` / `ground` |

Reading: `observe` is ~90% of real codex usage — the observe happy path is
the product. `status`/`--version`/`--help` (~15% of invocations) are health
probing before real work. Agents are also **builders of the tool itself**
(install, skill, package/repo chatter), not only consumers.

### Error signals in agent runs (opencode.db, near openvision text)

| Signal | Count |
|---|---|
| generic "Error" context | 333 |
| "failed" | 146 |
| Traceback | 30 |
| "is not installed" | 29 |
| ModuleNotFoundError | 24 |
| "not recognized" (command not found) | 3 |
| "No such file" / "command not found" | 1 each |

One targeted error-extraction run (filtering Traceback / ModuleNotFoundError /
not-recognized / No-such-file chunks) produced no captured sample rows in the
emitted buffer — treat the raw counts above as the evidence; samples unverified.

## 2. Defects verified against the code (each reproduces)

Ordered by impact. All verified by running the installed CLI and reading source.

### D1 — Documented install syntax does not exist (P0)

SKILL.md lines 17–19 and README.md line 70 document:

```
openvision install parakeet
openvision install whisper --model base
```

Real CLI (`openvision install --help`, run 2026-10-30) has subcommands
`list-models` and `get` only:

```
openvision install get parakeet
openvision install get whisper --model base
openvision install list-models
```

Evidence match: install is the #3 most-invoked command (163 in opencode.db,
12 in codex) and "not recognized"/No-such-command errors appear in the logs.
Agents follow the SKILL doc and fail.

### D2 — `observations view <id>` syntax does not exist (P0)

SKILL.md line 21 documents `openvision observations view <id>`. Real command is
`observations [action] --name <n>`; a positional id fails with
"Got unexpected extra argument(s)".

### D3 — `openvision import` does not exist (P1)

67 invocations in opencode.db + 9 in codex, documented nowhere. Agents expect
an import path (probably: index existing runs/observations into the memory DB).
Either implement a documented `openvision memory import` or stop the demand by
documenting that `observe` auto-indexes. Demand signal is real; decide, don't ignore.

### D4 — Model cache is split across two locations (P0, user-flagged)

This is the issue you named: downloads must land in and be used from
`~/.openvision/models`.

Current state (verified on this machine):

- Installer/auto-download target — `core/models.py`:
  `models_dir()` = `get_home()/models` = `~/.openvision/models`
  (parakeet → `models/parakeet-tdt-0.6b-v3-int8`, whisper → `models/whisper-<size>`).
- Runtime default — `providers/parakeet.py:19`:
  `_DEFAULT_MODEL_DIR = <repo>/models/parakeet-tdt-0.6b-v3-int8` (project-local!).
- `status --doctor` — `status.py:295-299` checks the **home** cache path.

Resulting split on this machine:

- Repo-local `openvision/models/parakeet-tdt-0.6b-v3-int8/` holds the ONNX
  weights (~750 MB: encoder/decoder/joiner + tokens) → transcribe works, but
  only inside this checkout.
- `~/.openvision/models/` holds `whisper-base/` and a dangling 16 MB
  `parakeet-download.tar.bz2` (interrupted download, never cleaned).
- So `status --doctor` reports parakeet NOT installed (checks home) while
  `transcribe` actually works (loads repo). Conversely, a fresh machine that
  ran `openvision install get parakeet` has the model in home cache while the
  provider default looks in the repo. Two divergent answers = the "is not
  installed" / install-loop failures in the logs.

Sub-defects in the same flow:

- **D4a — whisper install detection mismatch.** `check_model_status` expects
  `model.bin, config.json, vocabulary.json`, but faster-whisper's
  `download_model()` writes `model.bin, config.json, tokenizer.json,
  vocabulary.txt`. Verified: `~/.openvision/models/whisper-base/` has no
  `vocabulary.json`, so `install list-models` / status report whisper-base as
  NOT installed despite a complete 145 MB download → re-download each time.
- **D4b — no resume/verify/cleanup for interrupted downloads.** `urlretrieve`
  writes straight into the cache; the 16 MB `parakeet-download.tar.bz2` is the
  residue. Failure path leaves junk behind and matches the "failed" signals.

### D5 — Parakeet registry entry: name/description vs URL (P1, UNVERIFIED)

`core/models.py` PARAKEET_MODEL is named/described "Parakeet TDT 0.6B v3 INT8,
25 languages, ~640MB" but `url` points at
`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2`
(a 2023 bilingual streaming zipformer, not the 0.6B v3 multilingual release).
The repo-local weights that do exist load fine, so the installed model ≠ what
the registry advertises. Needs verification against the sherpa-onnx release
assets before rewriting (cannot confirm offline which asset name corresponds
to the v3 INT8 multilingual model — flagged **unverified**).

## 3. Enhancement plan

Sequenced so the top command (`observe`, ~90% of usage) is never destabilized.
Add a test first, then the fix, for each item.

### Phase 1 — Fix the agent-facing surface (docs + command syntax). No model-path refactor.

1. **SKILL.md + README.md**: replace `install parakeet` / `install whisper --model base`
   with `install get ...`; replace `observations view <id>` with
   `observations view --name <n>`; add the `import` story (see Phase 2 item 3).
   Verify: re-run each documented command shape live; done when every command in
   SKILL.md's Quick Reference executes or prints a structured error.
2. Add `--json` to `install get` and `install list-models` (agents run
   `--json` everywhere else; today install is the only install-path command
   with no machine-readable output).
3. Add a test (`tests/test_skill_docs.py` or equivalent) that parses SKILL.md
   command shapes and asserts each shape resolves in the typer app — prevents
   future doc drift. This is the highest-leverage fix: every future doc edit
   that breaks agents fails CI.

### Phase 2 — Model cache convergence (the user-flagged change)

1. **One cache root.** Keep `models_dir()` as the canonical `~/.openvision/models`;
   add resolution order `config paths.models` → `OPENVISION_MODELS_DIR` env →
   home cache. No new default behavior; just configurability.
2. **One resolution chain used by everyone.** New helper (e.g.
   `resolve_model_dir(provider, name)` in `core/models.py`) that returns the
   first existing candidate in order: explicit config `model_dir` →
   home cache → repo-local `models/<name>` (fallback, logs which was used).
   Wire it into `providers/parakeet.py` (replacing `_DEFAULT_MODEL_DIR` as sole
   default), `check_model_status`, and `status --doctor` so runtime, installer,
   and status all answer identically.
3. **Adopt, don't duplicate.** First resolve of a repo-local model copies it into
   the home cache (or hardlinks/symlinks where the platform allows) so other
   projects/agents and a clean reinstall reuse it; repo copy is never deleted.
   Feature-flag it (`OPENVISION_ADOPT_LOCAL_MODELS=1` default on) and log it.
4. **Import command.** Implement `openvision memory import` (index legacy runs /
   observation ledgers into the memory DB — the 76 agent invocations suggest
   this is the intended import path), or explicitly document `observe`
   auto-indexing in SKILL.md. Pick one; the demand won't go away.
5. **Download hygiene.** Atomic downloads: fetch to `.part` in the cache dir,
   verify size against the registry entry, rename on success, delete `.part`
   on failure; verify the extracted file set instead of trusting the tarball;
   clean the existing 16 MB dangling `parakeet-download.tar.bz2` (manual /
   first-run sweep).
6. **Fix whisper detection (D4a):** expected files = model.bin + config.json +
   (vocabulary.json OR vocabulary.txt) + tokenizer.json; mirror what
   faster-whisper actually writes.

### Phase 3 — Sherpa/parakeet registry correctness (D5)

1. Verify the correct sherpa-onnx release asset for Parakeet TDT 0.6B v3 INT8
   (external check, requires network). If the current URL is wrong, point
   PARAKEET_MODEL at the real asset and add the SHA256 to the entry.
2. If the URL stays, rename the entry to match what it actually downloads so
   `status`/`install` stop lying about which model is present.
3. Add `check_model_status` verification of `tokens.txt` sanity (line count) so
   a half-extracted model is reported as broken, not installed.

### Phase 4 — Agent ergonomics seen in the logs (only after 1–3 land)

1. Health probes (`--version` 47, `status` 166) — make
   `openvision --version --json` (or `status --json` one-liner) the documented
   pre-flight check; keep output small.
2. Model-missing error: raise the structured `OpenVisionError` with
   `suggested_retry: {"command": "openvision install get parakeet"}` instead of
   a raw `FileNotFoundError` from the provider (matches D1's new syntax).
3. `--help` (22 hits) suggests agents guess flags; the doc-drift test in
   Phase 1 is the guard.

### Non-goals / risk notes

- Observe internals: untouched. Any perf or pipeline change waits until the
  model-path refactor is verified.
- Repo has uncommitted work already (`README.md`, `SKILL.md`, `cli/app.py`,
  `cli/commands/observe.py`, `core/sampling.py`, `openvision/models.py`,
  `pyproject.toml` modified). Phase 1 doc edits land in that same dirty
  tree — rebase the diff onto the working copy, don't clobber.
- Repo-local `openvision/models/` is git-ignored (commit 997277f) so adopting
  into `~/.openvision/models` does not add binaries to git.

## 4. Verification checklist per phase

| Phase | Check |
|---|---|
| 1 | `openvision install get parakeet --json` exits 0; `openvision observations view --name X --json` behavior documented; SKILL-doc test green in pytest |
| 2 | `openvision status --json` and live `observe`/`transcribe` report the same model path; `OPENVISION_MODELS_DIR=/tmp/x openvision install get whisper --model base` lands under /tmp/x; interrupted download leaves no residue; re-run install twice = zero bytes re-downloaded |
| 3 | Registry URL/asset verified externally; mismatched name fixed; `tests/test_status.py` still green |
| 4 | Pre-flight one-liner documented; missing-model error yields suggested retry |

## 5. Open questions

- Parakeet asset: which sherpa-onnx release matches "TDT 0.6B v3 INT8, 25
  languages" — unverified, blocks Phase 3.
- `import`: implement vs document-away — needs a decision before Phase 2 item 4.
- Adopt strategy on non-Windows: copy vs symlink — platform check needed.

---

## SECOND PASS (2026-10-30) — tool-call reality, memory gaps, decision models

### 6. What agents actually pass (measured, not guessed)

Flag frequency in openvision.db part rows:

| Flag | Count | Note |
|---|---:|---|
| `--json` | 865 | agents want machine output |
| `--raw-frames` | 151 | vision-capable-agent handoff |
| `--grid` | 112 | contact sheets |
| `--provider` | 77 | explicit provider picks |
| `--diarize` | 72 | |
| `--jsonl` | 72 | streaming pipelines |
| time window (`--start/--from`) | 62 | |
| `--stt-provider` | 62 | |
| `--no-transcribe` | 55 | |
| `--cookies-from-browser` | 54 | |
| `--mode` | 46 | |
| `--memory` | 43 | |
| `--max-frames` | 42 | |

`--max-frames` values passed: **8 (×28), 5 (×6), 4 (×6), 30 (×2)** — confirms
the complaint: agents request 4–8 frames on long-form videos. With
`--memory` the coverage floor is only applied `if memory_build and not
max_frames` (observe.py:482-487), and `actual_max = min(max_frames, ...)`
hard-caps (observe.py:858) — an explicit `--max-frames 8` on a 30-minute
video yields 8 frames for the whole memory build. Memory then has enormous
empty intervals (buckets are skipped when no evidence lands in them,
video_memory.py:140), which is invisible to the agent.

Model-download signals: 268 download-mention events (opvision.db) and 116
(codex) vs exactly **1** "already-installed" confirmation in the logs.
218 caption/subs mentions in codex (148 in openvision.db) — the
subtitle-first path dominates remote videos.

### 7. Defects found in this pass (verified in code)

- **D6 — Re-downloading "installed" models (root cause confirmed).** Two
  independent causes: (a) `check_model_status`/`download_whisper_model`
  early-return require `vocabulary.json` (models.py:152, 289) but
  faster-whisper writes `vocabulary.txt` + `tokenizer.json` — the
  already-present `~/.openvision/models/whisper-base/model.bin` (145 MB)
  never satisfies the check, so every install re-downloads; (b) the home
  cache has no extracted parakeet (only the 16 MB dangling tarball), so
  parakeet also re-downloads while a fully installed copy sits in the
  repo (`models/parakeet-tdt-0.6b-v3-int8/`) that the check never looks at.
- **D7 — Raw-frames memory is speech-only.** In `--raw-frames` (observe.py:1113+)
  the result summary is literally `"Raw frames extracted: N frames from X"`,
  and the memory timeline is built from transcript segments only — frame
  paths never enter `video_memory_nodes` as visual evidence. So the
  "global picture" node is that string, and caption text lands in memory
  with no link to any frame. This is the user's complaint, confirmed:
  subtitles index, but nothing matches them to frames and there is no
  whole-video synthesis.
- **D8 — Root "global picture" is inherited, not synthesized.** The root
  node summary is whatever `observe` printed (video_memory.py:206-212:
  `add_node(root_id, ..., summary, ...)`); section/chunk summaries are
  naive 500-char truncations (`_chunk_summary`, video_memory.py:291-297).
  There is no whole-video topic/chapter/entity snapshot, so `memory ask`
  and `outline` reconstruct the big picture by concatenating crumbs.
- **D9 — transcript.txt fallback breaks >1h videos.** `_index_video_memory`
  re-parses saved transcripts with `^\[(\d{1,2}):(\d{2})\]`
  (observe.py:265-277) — matches MM:SS only. Transcripts saved in
  HH:MM:SS form (they are: `_fmt_time` emits hours when present) parse to
  minutes:seconds and land at wildly wrong times in memory for any video
  over an hour, exactly the class of video agents observe with 8 frames.
- **D10 — No decision-model tier.** `memory ask` / verification reuse the
  same VLM config as observe (memory.py:104-109, registry:76-88). Logs
  show agents forcing `--provider` 77 times (openai ×9, anthropic ×4,
  ollama ×1 sightings) to get a cheap/strong-enough answering model, with
  no config tier for it. The registry has no Gemini entry
  (`_PROVIDERS` = lmstudio, ollama, llamacpp + CloudProvider aliases;
  CloudProvider reads only `OPENAI_API_KEY`, cloud.py:30). Any Gemini
  endpoint claim is unverified — Google's OpenAI-compatible endpoint needs
  a URL/key check before wiring.

### 8. Plan additions (fold into the phases above)

**Phase 2 additions (memory quality — the "global picture" work):**
1. Coverage floor: when `--memory` is set, enforce at least 1 frame / 30 s
   (max 300) *even if* `--max-frames` was passed; emit the existing
   `coverage_percent` line prominently and warn when the agent's budget is
   below the floor. Kills the silent-8-frames case.
2. Visual evidence must survive `--raw-frames`: add the extracted frames to
   the memory timeline as `modality="visual"` evidence entries carrying
   `frame_path` and a `described: false` marker, so caption speech can be
   time-matched to frames (and a later VLM pass can upgrade them).
3. Real root synthesis: a second cheap-VLM pass that writes a video profile
   snapshot (`video_profile.json` + root node summary): topics, chapters,
   entities, speakers, key moments, duration/fps metadata — replacing D8's
   inherited string. Section/chunk summaries keep the truncated text but
   gain the profile as context for `memory ask`.
4. Caption→frame matching: when captions/subtitles supplied the transcript
   and `--memory` is on, automatically apply the `--text-anchors` cue
   timestamps so speech evidence and visual evidence co-locate.
5. Fix D9: parse HH:MM:SS and MM:SS in the transcript fallback (and prefer
   saving segment times as JSON so parsing back is never lossy).
6. Fix D6 (belongs here, not just install): expected-file sets per provider
   that match what the downloader actually writes; `check_model_status`
   uses the shared resolution chain from Phase 2.

**Phase 3 additions (decision-model tiers):**
1. New config section `models.decision` (provider/model/base_url/api_key)
   used by `memory ask`'s verification pass and the Phase-2 root synthesis;
   `observe`/`transcribe` keep their own `models.vlm`. Defaults to the
   cheapest local (ollama `llava`/`qwen2-vl`) so ask doesn't burn the same
   budget as observe.
2. Add Gemini as a registry entry via the OpenAI-compatible surface
   (new env `GEMINI_API_KEY`, explicit `--provider gemini`); verify the
   base URL and auth against Google's current docs BEFORE wiring —
   unverified today. Keep `openai`/`groq`/`together` as-is for the cost tier.
3. `memory ask --provider <name>` override so agents stop guessing with the
   observe-time flag (D10 log evidence).
