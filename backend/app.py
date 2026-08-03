"""Clipperboy — FastAPI app: upload a podcast, get social-ready clips."""
from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import config, jobs

app = FastAPI(title="Clipperboy", docs_url="/api/docs", openapi_url="/api/openapi.json")

FRONTEND_DIR = config.ROOT / "frontend"
ALLOWED_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma",
}


# --- helpers --------------------------------------------------------------

def _get_job(job_id: str) -> jobs.Job:
    job = jobs.STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _get_clip(job: jobs.Job, clip_id: str) -> jobs.Clip:
    clip = next((c for c in job.clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, "Clip not found")
    return clip


def _clip_file(job: jobs.Job, clip: jobs.Clip, suffix: str) -> Path:
    path = config.CLIP_DIR / job.id / f"clip-{clip.id}{suffix}"
    if not path.exists():
        raise HTTPException(404, "File not ready yet")
    return path


def _safe_slug(text: str, fallback: str = "clip") -> str:
    keep = "".join(ch if ch.isalnum() or ch in " -_" else "" for ch in text).strip()
    keep = "-".join(keep.split())[:60]
    return keep or fallback


# --- API ------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    import shutil as sh

    return {
        "ok": True,
        "ffmpeg": bool(sh.which(config.FFMPEG)),
        "whisper_model": config.WHISPER_MODEL,
        "aspects": list(config.ASPECTS),
        "default_aspect": config.DEFAULT_ASPECT,
        "max_clip_seconds": config.MAX_CLIP_SEC,
        "ram_gb": config.TOTAL_RAM_GB,
        "whisper_batch_size": config.WHISPER_BATCH_SIZE,
        "speeds": config.SPEED_PRESETS,
        "qualities": {k: f"{p}/crf{c}" for k, (p, c) in config.QUALITY_PRESETS.items()},
        "default_quality": config.DEFAULT_QUALITY,
    }


@app.post("/api/jobs")
def create_job(
    file: UploadFile,
    aspect: str = Form(config.DEFAULT_ASPECT),
    clip_count: int = Form(config.DEFAULT_CLIP_COUNT),
    compress: str = Form("auto"),
    captions: bool = Form(True),
    speed: str = Form(config.DEFAULT_SPEED),
    quality: str = Form(config.DEFAULT_QUALITY),
) -> JSONResponse:
    if aspect not in config.ASPECTS:
        raise HTTPException(400, f"Unsupported aspect ratio. Choose one of {list(config.ASPECTS)}.")
    if compress not in {"auto", "always", "never"}:
        raise HTTPException(400, "compress must be auto, always or never")
    if speed not in config.SPEED_PRESETS:
        raise HTTPException(400, f"speed must be one of {list(config.SPEED_PRESETS)}")
    if quality not in config.QUALITY_PRESETS:
        raise HTTPException(400, f"quality must be one of {list(config.QUALITY_PRESETS)}")
    clip_count = max(1, min(20, int(clip_count)))

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"'{suffix or 'this file type'}' isn't supported. Upload an audio or video "
            f"file such as .mp4, .mov, .mp3 or .m4a.",
        )

    job = jobs.STORE.create(file.filename or "episode", aspect, clip_count, captions, speed, quality)
    dest = config.UPLOAD_DIR / f"{job.id}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
    finally:
        file.file.close()

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        jobs.STORE.delete(job.id)
        raise HTTPException(400, "That file appears to be empty.")

    jobs.start(job, dest, compress)
    return JSONResponse({"job": job.to_dict()}, status_code=201)


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in jobs.STORE.list()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return {"job": _get_job(job_id).to_dict()}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not jobs.STORE.delete(job_id):
        raise HTTPException(404, "Job not found")
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    path = config.CLIP_DIR / job.id / "transcript.json"
    if not path.exists():
        raise HTTPException(404, "Transcript not ready yet")
    return FileResponse(path, media_type="application/json",
                        filename=f"{_safe_slug(job.filename, 'episode')}-transcript.json")


@app.get("/api/jobs/{job_id}/transcript.txt")
def get_transcript_text(job_id: str) -> Response:
    """The episode transcript as readable, timestamped plain text."""
    job = _get_job(job_id)
    path = config.CLIP_DIR / job.id / "transcript.json"
    if not path.exists():
        raise HTTPException(404, "Transcript not ready yet")

    data = json.loads(path.read_text())
    lines = [f"Transcript — {job.filename}", ""]
    for seg in data.get("segments", []):
        stamp = int(seg["start"])
        h, rem = divmod(stamp, 3600)
        m, s = divmod(rem, 60)
        clock = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        lines.append(f"[{clock}] {seg['text'].strip()}")

    name = f"{_safe_slug(job.filename, 'episode')}-transcript.txt"
    return Response(
        "\n".join(lines),
        media_type="text/plain; charset=utf-8",
        headers={"content-disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/jobs/{job_id}/clips/{clip_id}/video")
def clip_video(job_id: str, clip_id: str, download: bool = False) -> FileResponse:
    job = _get_job(job_id)
    clip = _get_clip(job, clip_id)
    path = _clip_file(job, clip, ".mp4")
    name = f"{_safe_slug(clip.title, 'clip')}-{clip.aspect.replace(':', 'x')}.mp4"
    return FileResponse(
        path, media_type="video/mp4",
        filename=name if download else None,
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/jobs/{job_id}/clips/{clip_id}/thumbnail")
def clip_thumbnail(job_id: str, clip_id: str) -> FileResponse:
    job = _get_job(job_id)
    clip = _get_clip(job, clip_id)
    return FileResponse(_clip_file(job, clip, ".jpg"), media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/clips/{clip_id}/subtitles")
def clip_subtitles(job_id: str, clip_id: str) -> FileResponse:
    job = _get_job(job_id)
    clip = _get_clip(job, clip_id)
    return FileResponse(
        _clip_file(job, clip, ".srt"), media_type="application/x-subrip",
        filename=f"{_safe_slug(clip.title, 'clip')}.srt",
    )


@app.post("/api/jobs/{job_id}/clips/{clip_id}/render")
def rerender(job_id: str, clip_id: str, payload: dict) -> dict:
    job = _get_job(job_id)
    clip = _get_clip(job, clip_id)
    aspect = payload.get("aspect", clip.aspect)
    if aspect not in config.ASPECTS:
        raise HTTPException(400, "Unsupported aspect ratio")
    try:
        jobs.rerender_clip(job, clip, aspect, bool(payload.get("captions", clip.captions)))
    except FileNotFoundError as exc:
        raise HTTPException(410, str(exc))
    except Exception:
        raise HTTPException(
            500,
            "This clip couldn't be re-exported. This usually means the machine ran "
            "short of memory — close other apps and try again.",
        )
    return {"clip": json.loads(json.dumps(clip, default=lambda o: o.__dict__))}


@app.get("/api/jobs/{job_id}/download-all")
def download_all(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    ready = [c for c in job.clips if c.status == "ready"]
    if not ready:
        raise HTTPException(404, "No finished clips to download yet")

    bundle = config.CLIP_DIR / job.id / "all-clips.zip"
    staging = bundle.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as zf:
        for clip in ready:
            base = f"{clip.index:02d}-{_safe_slug(clip.title, f'clip-{clip.index}')}"
            for suffix, ext in ((".mp4", "mp4"), (".srt", "srt")):
                path = config.CLIP_DIR / job.id / f"clip-{clip.id}{suffix}"
                if path.exists():
                    zf.write(path, f"{base}.{ext}")
        notes = "\n\n".join(
            f"[{c.index}] {c.title}\n"
            f"Source timecode: {c.start:.1f}s – {c.end:.1f}s ({c.duration:.0f}s)\n"
            f"Why it works: {'; '.join(c.reasons)}\n"
            f"Transcript: {c.transcript}"
            for c in ready
        )
        zf.writestr("clip-notes.txt", notes)

    staging.replace(bundle)
    return FileResponse(bundle, media_type="application/zip",
                        filename=f"{_safe_slug(job.filename, 'podcast')}-clips.zip")


# --- static frontend (mounted last so /api wins) --------------------------
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
