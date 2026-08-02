# ✂️ Clipperboy

Upload a full-length podcast. Get back short, vertical, captioned clips of the
best moments — cropped to keep the speaker in frame and ready to post to
Reels, TikTok or Shorts.

Everything runs **on your own machine**. No account, no API key, no upload
limits, nothing sent to a third party. Every component is free and open source.

---

## What it does

1. **Takes any podcast file** — video (`.mp4 .mov .mkv .webm .avi`) or audio-only
   (`.mp3 .m4a .wav .flac .ogg`).
2. **Compresses the source** if it's oversized, so a 4 GB export doesn't slow the
   rest of the pipeline down (automatic, or force it on/off).
3. **Transcribes the whole episode** locally with OpenAI Whisper (via
   `faster-whisper`), down to word-level timings.
4. **Finds the clip-worthy moments** by scoring every sentence-aligned window on
   hooks, questions, insights, emotion, storytelling, vocal energy and
   self-containment — then picking the best non-overlapping set. Sponsor reads,
   intros and outros are actively penalised.
5. **Edits each clip**: crops to **9:16** with the speaker tracked by face
   detection, and burns in **word-by-word captions**. Audio-only episodes get an
   animated waveform card instead.
6. **Shows you everything**: title, summary, why it was picked, topics, exact
   source timecode, and the full transcript of the clip — with one-click
   download.

Audio is loudness-normalised to −14 LUFS so clips don't arrive quiet.

---

## Get it running

### Option A — Docker (recommended, nothing else to install)

```bash
git clone https://github.com/ahroonsanthosh/clipperboy.git
cd clipperboy
docker compose up --build
```

Then open **<http://localhost:8000>**.

### Option B — Python directly

You need Python 3.10+ and FFmpeg.

```bash
# install ffmpeg first:
#   macOS          brew install ffmpeg
#   Debian/Ubuntu  sudo apt install ffmpeg
#   Windows        winget install Gyan.FFmpeg

git clone https://github.com/ahroonsanthosh/clipperboy.git
cd clipperboy
./run.sh
```

`run.sh` creates a virtualenv and installs dependencies on first run, then
serves the app at **<http://localhost:8000>**.

> **First run downloads the speech model** (~150 MB for `base`) from
> huggingface.co. It's cached afterwards, and everything else works offline.

---

## Using it

1. Drag your episode onto the drop zone (or click to browse).
2. Pick the aspect ratio, how many clips you want, and whether to compress.
3. Press **Find my clips** and leave the tab open.
4. As each clip finishes it appears in the grid — preview it, read what's in it,
   and **Download**. Or grab **Download all clips (.zip)**, which also includes
   `.srt` subtitle files and a `clip-notes.txt` with timecodes and reasoning.

Any clip can be **re-exported** in a different aspect ratio from its card
without re-running the analysis.

### How long does it take?

Roughly **⅕ to ⅓ of the episode length** on a modern laptop CPU — so a 60-minute
episode lands around 10–20 minutes. Transcription is the slow part; it's much
faster on a machine with an NVIDIA GPU (detected automatically).

---

## Settings

Set these as environment variables (or in `docker-compose.yml`):

| Variable | Default | What it does |
| --- | --- | --- |
| `CLIPPERBOY_WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`. Bigger = more accurate, slower. |
| `CLIPPERBOY_CLIP_COUNT` | `8` | Default number of clips to find. |
| `CLIPPERBOY_ASPECT` | `9:16` | Default aspect ratio. |
| `CLIPPERBOY_MIN_CLIP` / `CLIPPERBOY_MAX_CLIP` | `18` / `75` | Clip length bounds, in seconds. |
| `CLIPPERBOY_COMPRESS_MB` | `800` | Above this file size, "automatic" compression kicks in. |
| `CLIPPERBOY_DATA` | `./data` | Where uploads, clips and models are stored. |
| `CLIPPERBOY_USE_OLLAMA` | `auto` | `off` to disable the optional local-LLM title polish. |
| `OLLAMA_MODEL` / `OLLAMA_HOST` | `llama3.2` / `localhost:11434` | Optional: if you run [Ollama](https://ollama.com), titles and summaries are written by a local model. Purely optional — it never blocks a run. |

---

## Accessibility

The interface is built to be usable without a mouse or without sight:

- Full keyboard operation, including the drop zone and the preview dialog, with
  a visible focus ring on every control and a skip-to-content link.
- Progress and status changes are announced via ARIA live regions.
- Semantic headings, labelled form controls, and `aria-label`s that name the
  clip a button acts on.
- WCAG AA contrast in both light and dark colour schemes, which follow your
  system setting.
- Respects `prefers-reduced-motion`.
- Every clip ships with an `.srt` subtitle file alongside the burned-in captions,
  and the full transcript is downloadable as JSON.

---

## Development

```bash
./.venv/bin/python -m pytest tests -q
```

The suite generates its own synthetic podcast fixture with espeak-ng + FFmpeg
(`tests/make_fixture.py`) and stubs the speech model, so it runs offline and
needs no copyrighted media. It covers probing, compression decisions, scoring,
caption generation (including that captions are genuinely burned into the
picture), crop planning, every aspect ratio, and the full HTTP API.

Layout:

```
backend/
  app.py              FastAPI routes
  jobs.py             job store + background worker
  config.py           env-driven settings
  pipeline/
    media.py          ffprobe/ffmpeg, compression, audio envelope
    transcribe.py     faster-whisper wrapper
    score.py          clip-worthiness scoring & selection
    speaker.py        face-tracked crop planning
    captions.py       ASS (burned-in) + SRT generation
    render.py         final clip render
frontend/             plain HTML/CSS/JS, no build step
tests/
```

## Licence & credits

MIT. Built on [FFmpeg](https://ffmpeg.org),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[OpenCV](https://opencv.org) and [FastAPI](https://fastapi.tiangolo.com) — all free
and open source.
