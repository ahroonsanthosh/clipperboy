"""ffmpeg/ffprobe helpers: probing, optional compression, audio extraction."""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import config


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], desc: str = "") -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise MediaError(f"{desc or cmd[0]} failed:\n" + "\n".join(tail))
    return proc.stdout


@dataclass
class MediaInfo:
    duration: float
    has_video: bool
    width: int
    height: int
    fps: float
    size_bytes: int

    @property
    def is_audio_only(self) -> bool:
        return not self.has_video


def probe(path: Path) -> MediaInfo:
    out = run(
        [config.FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        "ffprobe",
    )
    data = json.loads(out)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or 0.0)

    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    # Cover art is a video stream with a single frame; don't mistake it for video.
    if video is not None and video.get("disposition", {}).get("attached_pic"):
        video = None

    width = int(video.get("width", 0)) if video else 0
    height = int(video.get("height", 0)) if video else 0
    fps = 30.0
    if video and video.get("avg_frame_rate", "0/0") not in ("0/0", "0/1"):
        num, _, den = video["avg_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = 30.0

    if duration <= 0:
        raise MediaError("Could not read a duration from this file — is it a valid media file?")

    return MediaInfo(
        duration=duration,
        has_video=video is not None,
        width=width,
        height=height,
        fps=fps or 30.0,
        size_bytes=path.stat().st_size,
    )


def should_compress(info: MediaInfo, mode: str) -> bool:
    """mode: auto | always | never

    "auto" never compresses video, and that is deliberate.

    The old rule downscaled anything above 1080p, on the theory that every
    later pass then decodes a quarter of the pixels. But every later pass is
    building a *vertical crop*: 9:16 out of a frame keeps only ~32% of its
    width. Pre-downscaling a 4K episode to 1080p throws away three quarters of
    the pixels and then the render upscales the 608px-wide survivor back to
    1080 — measured at a fraction of the detail of cropping 1215x2160 straight
    out of the original and scaling that *down* to 1080x1920. The decode time
    saved is minutes; the quality lost is the entire product.

    Compressing merely because a file is *large* was already a bad trade:
    re-encoding a 60-minute episode costs ~22 minutes to save well under one.

    "always" remains for people who genuinely need the disk back; renders still
    prefer the original file when it is still on disk (see `record_origin`).
    """
    if mode == "never" or info.is_audio_only:
        return False
    return mode == "always"


ORIGIN_SUFFIX = ".origin"


def origin_sidecar(working: Path) -> Path:
    return working.with_name(working.name + ORIGIN_SUFFIX)


def record_origin(working: Path, original: Path) -> None:
    """Note which file a working copy was derived from.

    A working copy is a downscale, so anything that cares about picture quality
    (i.e. the renderer) should go back to the original if it is still there.
    """
    try:
        origin_sidecar(working).write_text(str(Path(original).resolve()), encoding="utf-8")
    except OSError:
        pass


def origin_of(working: Path) -> Path | None:
    """The original this working copy came from, if it is still on disk."""
    try:
        text = origin_sidecar(working).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.is_file() else None


def compress(src: Path, dst: Path, progress=None) -> Path:
    """Transcode to a 1080p-tall working copy. Only used when the operator asks
    for it explicitly — see `should_compress`. The provenance sidecar lets the
    renderer keep using the full-resolution original for the actual clips."""
    cmd = [
        config.FFMPEG, "-y", "-i", str(src),
        "-vf", "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    if progress:
        progress("Compressing source video")
    run(cmd, "compression")
    record_origin(dst, src)
    return dst


def extract_audio(src: Path, dst: Path) -> Path:
    """16 kHz mono WAV — what Whisper wants, and cheap to analyse with numpy."""
    run([
        config.FFMPEG, "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
    ], "audio extraction")
    return dst


def load_wav_envelope(wav_path: Path, hz: int = 20) -> tuple[np.ndarray, int]:
    """Return a coarse RMS-loudness envelope sampled at `hz` samples/second.

    Used by the scorer to spot laughter, emphasis and energy spikes without
    loading the whole podcast into memory as floats.
    """
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        chunk = max(1, rate // hz)
        env: list[float] = []
        remaining = n_frames
        while remaining > 0:
            raw = wf.readframes(min(chunk, remaining))
            if not raw:
                break
            remaining -= len(raw) // 2
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size:
                env.append(float(np.sqrt(np.mean(samples * samples))))
    return np.asarray(env, dtype=np.float32), hz
