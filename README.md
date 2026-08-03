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
2. **Downscales the source** if it's above 1080p, so a 4K export doesn't slow the
   rest of the pipeline down (automatic, or force it on/off).
3. **Transcribes the whole episode** locally with OpenAI Whisper (via
   `faster-whisper`), down to word-level timings.
4. **Finds the clip-worthy moments** by scoring every sentence-aligned window on
   hooks, questions, insights, emotion, storytelling, vocal energy and
   self-containment — then picking the best non-overlapping set. Sponsor reads,
   intros and outros are actively penalised.
5. **Edits each clip**: crops to **9:16** with the camera *locked* on the
   speaker — still within a shot, hard-cutting when the speaker changes, never
   drifting — using MediaPipe's BlazeFace detector, and burns in **word-by-word captions** — kept inside the safe area
   so TikTok and Reels chrome never covers them. Audio-only episodes get an
   animated audiogram card with the clip title instead.
6. **Shows you everything**: title, summary, why it was picked, topics, exact
   source timecode, and the full transcript of the clip — with one-click
   download.

Audio is loudness-normalised to −14 LUFS so clips don't arrive quiet.

**Export settings**: H.264 high profile, CRF 20, lanczos scaling, a keyframe
every 2 seconds, AAC 192 kbps at 48 kHz, `+faststart`. The **source frame rate
is preserved** rather than forced — resampling 25fps to 30 duplicates every
fifth frame, which reads as judder on any movement.

---

## Get it running

### Option A — Docker (recommended, nothing else to install)

```bash
git clone https://github.com/ahroonsanthosh/clipperboy.git
cd clipperboy
docker compose up --build
```

Then open **<http://localhost:8000>**.

### Option B — Python directly (macOS / Linux)

You need Python 3.10–3.12 and FFmpeg.

```bash
# install ffmpeg first:
#   macOS          brew install ffmpeg
#   Debian/Ubuntu  sudo apt install ffmpeg
#   Windows        winget install Gyan.FFmpeg

git clone https://github.com/ahroonsanthosh/clipperboy.git
cd clipperboy
./run.sh
```

On **Windows**, install [Python](https://python.org) and FFmpeg
(`winget install Gyan.FFmpeg`), then double-click **`run.bat`** — or just use
Docker, which needs neither.

Either script creates a virtualenv and installs dependencies on first run, then
serves the app at **<http://localhost:8000>**.

> **A long episode needs headroom.** Transcription is memory-hungry, and an
> hour-plus recording on a laptop with other apps open can be killed outright by
> the operating system. Clipperboy sizes its batch to your RAM automatically; if
> a long episode still dies partway, set `CLIPPERBOY_WHISPER_BATCHED=0` to use
> the slower, leanest path. `/api/health` reports the RAM and batch size in use.

> **The first upload pauses to download the speech model** (~150 MB for `base`)
> from huggingface.co, and the progress bar sits on "Loading speech model" for a
> few minutes while it does. It is cached in `./data/models` afterwards, and
> everything else works offline. If huggingface.co is unreachable, Clipperboy
> says so rather than failing silently.

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

For a 60-minute episode, budget roughly **10–15 minutes** on a recent laptop.
Transcription is now the bulk of that, and it's dramatically faster on a machine
with an NVIDIA GPU — detected and used automatically.

**The Speed setting is the big lever.** Transcription is most of the wait, and
it scales with model size:

| Speed | Model | Relative to Balanced |
| --- | --- | --- |
| Fast | `tiny` | ~3× quicker, slightly rougher wording |
| Balanced | `base` | the default |
| Most accurate | `small` | ~2.5× slower |

On a long episode, or on an older machine, start with **Fast**. The clip-finding
and editing are unaffected — only how carefully the speech is transcribed.
Asking for fewer clips and unticking **Burn captions into the video** also help.

Earlier versions took 15–40 minutes. Most of that was a full re-encode of the
whole episode that fired on any file over 800 MB — measured at ~22 minutes for a
60-minute episode, to save under a minute of decoding later. It now only
downscales genuinely oversized (above-1080p) sources. Crop planning was rebuilt
to walk each clip once instead of seeking per sample (14.8s → 2.8s per clip),
clips render in parallel, and transcription uses batched inference.

---

## Settings

Set these as environment variables (or in `docker-compose.yml`):

| Variable | Default | What it does |
| --- | --- | --- |
| `CLIPPERBOY_SPEED` | `balanced` | Which Speed the upload form starts on: `fast`, `balanced`, `accurate`. |
| `CLIPPERBOY_WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`. Bigger = more accurate, slower. |
| `CLIPPERBOY_CLIP_COUNT` | `8` | Default number of clips to find. |
| `CLIPPERBOY_ASPECT` | `9:16` | Default aspect ratio. |
| `CLIPPERBOY_MIN_CLIP` / `CLIPPERBOY_MAX_CLIP` | `18` / `60` | Clip length bounds, in seconds. The maximum is a hard cap — the lead-in/out padding is trimmed rather than allowed to overshoot it. |
| `CLIPPERBOY_WHISPER_BATCHED` | `1` | Batched transcription — much faster. `0` for the sequential path. |
| `CLIPPERBOY_WHISPER_BATCH` | auto | Batch size, chosen from your RAM (2 under 6 GB, 4 under 12 GB, else 8). Raise it on a GPU; lower it if a long episode is being killed. |
| `CLIPPERBOY_DATA` | `./data` | Where uploads, clips and models are stored. |
| `CLIPPERBOY_USE_OLLAMA` | `auto` | `off` to disable the optional local-LLM title polish. |
| `CLIPPERBOY_OLLAMA_MODEL` / `OLLAMA_HOST` | `llama3.2` / `http://127.0.0.1:11434` | Optional: if you run [Ollama](https://ollama.com), titles and summaries are written by a local model. Purely optional — it never blocks a run. |

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
  and the full episode transcript downloads as timestamped plain text.
- Touch targets are at least 44 px, and results refresh in place without
  stealing focus or collapsing anything you have open.

---

## Development

```bash
./run.sh                                   # creates .venv on first run, then Ctrl+C
./.venv/bin/pip install -r requirements-dev.txt
sudo apt install espeak-ng                 # macOS: brew install espeak-ng
./.venv/bin/python -m pytest tests -q      # ~6 minutes; it renders real video
```

### The face detector, and why it installs oddly

`run.sh`, `run.bat` and the Dockerfile all install the detector as a second,
separate step:

```bash
pip install --no-deps -r requirements-detect.txt
```

The `--no-deps` is load-bearing. MediaPipe ships one wheel for all of its
solutions, so it *declares* jax, jaxlib, matplotlib, sounddevice and
opencv-contrib-python even though face detection imports none of them. Letting
pip resolve that installs opencv-contrib-python, which overwrites the `cv2`
module provided by opencv-python-headless with a GUI build that needs an X
display — breaking OpenCV on any server or container. Installing with
`--no-deps` (66 MB, plus four small libraries pinned in the same file) avoids
this entirely. **Do not add `mediapipe` to `requirements.txt`.**

The detector is optional. If no wheel exists for your Python (it covers 3.9-3.12
on Windows, macOS and Linux) or the install fails for any other reason, the app
still runs: it logs one line and falls back to OpenCV's bundled Haar cascades,
which frame noticeably less well. Both detectors are fully offline — BlazeFace's
weights are inside the wheel, so nothing is ever downloaded.

Measured on a real single-speaker podcast clip (52 samples, 1080x1920):

| detector | finds the face | wrong boxes | ms/frame | crop planning, 40s clip |
|---|---|---|---|---|
| BlazeFace (default) | 98% | 0 | 8 | 0.6s |
| Haar cascades (fallback) | 20% | 0 | 155 | 12.4s |
| Haar, before this change | 40% | 53% | 98 | 7.8s |

The old ladder's "wrong boxes" were not noise: it locked onto a sponsor sign on
the wall behind the speaker, which is why crops used to land on blank wall. Its
median horizontal error was 273px on a 1080px-wide frame. The fallback now
applies a skin-colour test to every cascade box, which trades recall for
precision — the right trade here, because the camera is locked, so a *missing*
detection simply holds the shot while a *wrong* one moves the camera to the wall.

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
    speaker.py        face detection + locked-camera crop planning
    captions.py       ASS (burned-in) + SRT generation
    render.py         final clip render
frontend/             plain HTML/CSS/JS, no build step
assets/fonts/         bundled DejaVu, so captions look the same on every OS
tests/
```

## Licence & credits

MIT. Built on [FFmpeg](https://ffmpeg.org),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[OpenCV](https://opencv.org), [MediaPipe](https://github.com/google-ai-edge/mediapipe)
and [FastAPI](https://fastapi.tiangolo.com) — all free and open source.
