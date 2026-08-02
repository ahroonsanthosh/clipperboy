"""Job orchestration: one background worker thread per upload."""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline import media, render, score
from backend.pipeline.transcribe import Transcript, transcribe

STAGES = [
    ("upload", "Uploading"),
    ("probe", "Inspecting file"),
    ("compress", "Compressing source"),
    ("audio", "Extracting audio"),
    ("transcribe", "Transcribing speech"),
    ("analyse", "Finding clip-worthy moments"),
    ("render", "Editing & rendering clips"),
    ("done", "Ready"),
]


@dataclass
class Clip:
    id: str
    index: int
    title: str
    summary: str
    start: float
    end: float
    duration: float
    score: float
    reasons: list[str]
    keywords: list[str]
    features: dict
    transcript: str
    aspect: str
    crop_mode: str = ""
    faces_found: int = 0
    video: str = ""
    thumbnail: str = ""
    srt: str = ""
    status: str = "pending"     # pending | rendering | ready | failed
    error: str = ""


@dataclass
class Job:
    id: str
    filename: str
    created_at: str
    status: str = "queued"       # queued | running | done | failed
    stage: str = "upload"
    stage_label: str = "Uploading"
    progress: float = 0.0
    message: str = ""
    error: str = ""
    duration: float = 0.0
    has_video: bool = True
    compressed: bool = False
    source_size_mb: float = 0.0
    working_size_mb: float = 0.0
    aspect: str = config.DEFAULT_ASPECT
    clip_count: int = config.DEFAULT_CLIP_COUNT
    clips: list[Clip] = field(default_factory=list)
    transcript_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = [{"key": k, "label": v} for k, v in STAGES]
        return d


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._load_existing()

    def cancel(self, job_id: str) -> None:
        """Ask a running worker to stop at its next checkpoint."""
        with self._lock:
            self._cancelled.add(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def discard_cancelled(self, job_id: str) -> None:
        """Called by a worker once it has actually stopped."""
        with self._lock:
            self._cancelled.discard(job_id)

    # --- persistence ---------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        d = config.CLIP_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_existing(self) -> None:
        for state in sorted(config.CLIP_DIR.glob("*/job.json")):
            try:
                data = json.loads(state.read_text())
                clips = [Clip(**c) for c in data.pop("clips", [])]
                data.pop("stages", None)
                job = Job(**data, clips=clips)
                if job.status == "running":       # interrupted by a restart
                    job.status, job.error = "failed", "Interrupted — please re-upload."
                self._jobs[job.id] = job
            except Exception:
                continue

    def persist(self, job: Job) -> None:
        # A cancelled job's directory is being torn down; writing would recreate it.
        if job.id not in self._jobs or job.id in self._cancelled:
            return
        try:
            target = self._job_dir(job.id) / "job.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(job.to_dict(), indent=2))
            tmp.replace(target)   # atomic: a crash mid-write can't corrupt the job
        except Exception:
            pass

    # --- accessors -----------------------------------------------------
    def create(self, filename: str, aspect: str, clip_count: int) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            filename=filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            aspect=aspect,
            clip_count=clip_count,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._job_dir(job.id)
        self.persist(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        import shutil

        with self._lock:
            job = self._jobs.pop(job_id, None)
            self._cancelled.add(job_id)
        if not job:
            self._cancelled.discard(job_id)
            return False
        for path in (config.CLIP_DIR / job_id, config.WORK_DIR / job_id):
            shutil.rmtree(path, ignore_errors=True)
        for leftover in config.UPLOAD_DIR.glob(f"{job_id}.*"):
            leftover.unlink(missing_ok=True)
        if job.status != "running":
            # Nothing is going to call discard_cancelled for us.
            self.discard_cancelled(job_id)
        return True


STORE = JobStore()


# --- worker -------------------------------------------------------------

class Cancelled(RuntimeError):
    pass


def _check_cancelled(job: Job) -> None:
    if STORE.is_cancelled(job.id):
        raise Cancelled("Cancelled by the user")


def _set_stage(job: Job, key: str, progress: float, message: str = "") -> None:
    _check_cancelled(job)
    job.stage = key
    job.stage_label = dict(STAGES).get(key, key)
    job.progress = round(max(0.0, min(1.0, progress)), 4)
    if message:
        job.message = message
    STORE.persist(job)


# Fractions of overall progress owned by each stage.
_SPAN = {
    "probe": (0.01, 0.03),
    "compress": (0.03, 0.14),
    "audio": (0.14, 0.18),
    "transcribe": (0.18, 0.60),
    "analyse": (0.60, 0.66),
    "render": (0.66, 1.00),
}


def process(job_id: str, source_path: Path, compress_mode: str) -> None:
    job = STORE.get(job_id)
    if job is None:
        return
    work = config.WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    clip_dir = config.CLIP_DIR / job_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    try:
        job.status = "running"
        _set_stage(job, "probe", _SPAN["probe"][0], "Inspecting file")
        info = media.probe(source_path)
        job.duration = info.duration
        job.has_video = info.has_video
        job.source_size_mb = round(info.size_bytes / 1024 / 1024, 1)
        job.working_size_mb = job.source_size_mb
        _set_stage(job, "probe", _SPAN["probe"][1])

        working = source_path
        if media.should_compress(info, compress_mode):
            _set_stage(job, "compress", _SPAN["compress"][0], "Compressing source video")
            working = media.compress(source_path, work / "source.mp4")
            info = media.probe(working)
            job.compressed = True
            job.working_size_mb = round(info.size_bytes / 1024 / 1024, 1)
        _set_stage(job, "compress", _SPAN["compress"][1])

        _set_stage(job, "audio", _SPAN["audio"][0], "Extracting audio")
        wav = media.extract_audio(working, work / "audio.wav")
        envelope, hz = media.load_wav_envelope(wav)
        _set_stage(job, "audio", _SPAN["audio"][1])

        lo, hi = _SPAN["transcribe"]
        _set_stage(job, "transcribe", lo, "Loading speech model (first run downloads it once)")

        def on_transcribe(frac: float, msg: str) -> None:
            _set_stage(job, "transcribe", lo + (hi - lo) * frac, msg)

        tr: Transcript = transcribe(wav, info.duration, on_transcribe)
        (clip_dir / "transcript.json").write_text(json.dumps(tr.to_dict()))
        job.transcript_available = True
        _set_stage(job, "transcribe", hi)

        lo, hi = _SPAN["analyse"]
        _set_stage(job, "analyse", lo, "Scoring moments across the episode")
        candidates = score.find_clips(
            tr, envelope, hz, job.clip_count,
            progress=lambda m: _set_stage(job, "analyse", (lo + hi) / 2, m),
            scan_progress=lambda frac, m: _set_stage(job, "analyse", lo + (hi - lo) * frac, m),
        )
        _set_stage(job, "analyse", hi, f"Found {len(candidates)} strong moments")

        if not candidates:
            raise RuntimeError("No speech was detected in this file.")

        job.clips = [
            Clip(
                id=f"{i + 1:02d}", index=i + 1, title=c.title, summary=c.summary,
                start=round(c.start, 2), end=round(c.end, 2), duration=round(c.duration, 2),
                score=round(c.score, 2), reasons=c.reasons, keywords=c.keywords,
                features=c.features, transcript=c.text, aspect=job.aspect,
            )
            for i, c in enumerate(candidates)
        ]
        STORE.persist(job)

        lo, hi = _SPAN["render"]
        words = tr.words
        for i, clip in enumerate(job.clips):
            _set_stage(job, "render", lo + (hi - lo) * (i / len(job.clips)),
                       f"Rendering clip {i + 1} of {len(job.clips)} — {clip.title}")
            clip.status = "rendering"
            STORE.persist(job)
            try:
                result = render.render_clip(
                    source=working, info=info, start=clip.start, end=clip.end, words=words,
                    aspect=clip.aspect,
                    out_video=clip_dir / f"clip-{clip.id}.mp4",
                    out_thumb=clip_dir / f"clip-{clip.id}.jpg",
                    out_srt=clip_dir / f"clip-{clip.id}.srt",
                    work_dir=work,
                    title=clip.title,
                )
                clip.video = f"/api/jobs/{job.id}/clips/{clip.id}/video"
                clip.thumbnail = f"/api/jobs/{job.id}/clips/{clip.id}/thumbnail"
                clip.srt = f"/api/jobs/{job.id}/clips/{clip.id}/subtitles"
                clip.crop_mode = result.crop_mode
                clip.faces_found = result.faces_found
                clip.status = "ready"
            except Exception as exc:  # one bad clip shouldn't sink the batch
                clip.status = "failed"
                clip.error = str(exc)[:400]
            STORE.persist(job)

        ready = sum(1 for c in job.clips if c.status == "ready")
        job.status = "done"
        _set_stage(job, "done", 1.0, f"{ready} clip{'s' if ready != 1 else ''} ready to download")

    except Cancelled:
        job.status = "failed"
        job.error = "Cancelled."
    except Exception as exc:
        job.status = "failed"
        job.error = f"{exc}"[:800]
        job.message = "Something went wrong"
        traceback.print_exc()
        STORE.persist(job)
    finally:
        _cleanup_work(job_id, keep_source=not STORE.is_cancelled(job_id))
        STORE.discard_cancelled(job_id)


def _cleanup_work(job_id: str, keep_source: bool = True) -> None:
    """Drop the large intermediates. `source.mp4` is kept so a clip can be
    re-exported in another aspect ratio without re-uploading the episode."""
    import shutil

    work = config.WORK_DIR / job_id
    if not keep_source:
        # The job was cancelled or deleted: sweep up anything the worker wrote
        # after the delete request came in.
        for path in (work, config.CLIP_DIR / job_id):
            shutil.rmtree(path, ignore_errors=True)
        for leftover in config.UPLOAD_DIR.glob(f"{job_id}.*"):
            leftover.unlink(missing_ok=True)
        return
    if not work.exists():
        return
    for pattern in ("*.wav", "*.ass", "*.title.txt", "*.part.mp4"):
        for tmp in work.glob(pattern):
            tmp.unlink(missing_ok=True)


def rerender_clip(job: Job, clip: Clip, aspect: str, burn_captions: bool = True) -> None:
    """Re-export a single clip in a different aspect ratio."""
    work = config.WORK_DIR / job.id
    work.mkdir(parents=True, exist_ok=True)
    clip_dir = config.CLIP_DIR / job.id
    source = work / "source.mp4"
    if not source.exists():
        matches = list(config.UPLOAD_DIR.glob(f"{job.id}.*"))
        if not matches:
            raise FileNotFoundError("The original upload is no longer available on disk.")
        source = matches[0]

    tr = Transcript.from_dict(json.loads((clip_dir / "transcript.json").read_text()))
    info = media.probe(source)

    clip.status = "rendering"
    STORE.persist(job)
    try:
        result = render.render_clip(
            source=source, info=info, start=clip.start, end=clip.end, words=tr.words,
            aspect=aspect,
            out_video=clip_dir / f"clip-{clip.id}.mp4",
            out_thumb=clip_dir / f"clip-{clip.id}.jpg",
            out_srt=clip_dir / f"clip-{clip.id}.srt",
            work_dir=work,
            burn_captions=burn_captions,
            title=clip.title,
        )
        clip.aspect = aspect
        clip.crop_mode = result.crop_mode
        clip.faces_found = result.faces_found
        clip.status = "ready"
        clip.error = ""
    except Exception as exc:
        clip.status = "failed"
        clip.error = str(exc)[:400]
        raise
    finally:
        STORE.persist(job)


def start(job: Job, source_path: Path, compress_mode: str) -> None:
    threading.Thread(
        target=process, args=(job.id, source_path, compress_mode), daemon=True,
        name=f"clipperboy-{job.id}",
    ).start()
