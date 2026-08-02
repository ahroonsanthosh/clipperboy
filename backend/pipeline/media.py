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
    """mode: auto | always | never"""
    if mode == "never" or info.is_audio_only:
        return False
    if mode == "always":
        return True
    over_size = info.size_bytes > config.COMPRESS_THRESHOLD_MB * 1024 * 1024
    over_res = max(info.width, info.height) > 1920
    return over_size or over_res


def compress(src: Path, dst: Path, progress=None) -> Path:
    """Transcode to a 1080p-tall working copy. Keeps quality high enough for
    vertical crops while cutting decode cost for the rest of the pipeline."""
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
