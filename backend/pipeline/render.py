"""Render a scored moment into a finished, caption-burned social clip."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend import config
from backend.pipeline import captions as captions_mod
from backend.pipeline import speaker
from backend.pipeline.media import MediaInfo, run
from backend.pipeline.transcribe import Word

PAD_BEFORE = 0.25   # a breath of lead-in so the first word isn't clipped
PAD_AFTER = 0.35


@dataclass
class RenderResult:
    video_path: Path
    thumbnail_path: Path
    srt_path: Path
    crop_mode: str
    faces_found: int
    width: int
    height: int


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    text = str(path)
    for ch in ("\\", ":", "'", "[", "]", ",", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


def _audio_only_filter(width: int, height: int) -> str:
    """A clean animated waveform card for audio-only podcasts."""
    return (
        f"[0:a]showwaves=s={width}x{int(height * 0.28)}:mode=cline:rate=25:"
        f"colors=0x22E0FF|0x7A5CFF,format=yuv420p[wave];"
        f"color=c=0x0E1116:s={width}x{height}:r=25[bg];"
        f"[bg][wave]overlay=(W-w)/2:(H-h)/2:shortest=1[v]"
    )


def render_clip(
    source: Path,
    info: MediaInfo,
    start: float,
    end: float,
    words: list[Word],
    aspect: str,
    out_video: Path,
    out_thumb: Path,
    out_srt: Path,
    work_dir: Path,
    burn_captions: bool = True,
) -> RenderResult:
    width, height = config.ASPECTS.get(aspect, config.ASPECTS[config.DEFAULT_ASPECT])
    target_ratio = width / height

    start = max(0.0, start - PAD_BEFORE)
    end = min(info.duration, end + PAD_AFTER)
    duration = max(0.5, end - start)

    clip_words = [w for w in words if w.end > start and w.start < end]

    ass_path = work_dir / f"{out_video.stem}.ass"
    captions_mod.build_srt(clip_words, start, end, out_srt)
    if burn_captions and clip_words:
        captions_mod.build_ass(clip_words, start, end, width, height, ass_path)

    crop_mode = "audio-waveform"
    faces_found = 0

    if info.has_video:
        plan = speaker.plan_crop(source, start, end, info.width, info.height, target_ratio)
        crop_mode, faces_found = plan.mode, plan.faces_found
        chain = ["setpts=PTS-STARTPTS"]
        if plan.mode != "full-frame":
            chain.append(f"crop={plan.crop_w}:{plan.crop_h}:x='{plan.expression()}':y=(ih-oh)/2")
        chain.append(
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        chain.append("setsar=1")
        if burn_captions and clip_words:
            chain.append(f"subtitles='{_escape_filter_path(ass_path)}'")
        vf = ",".join(chain)
        cmd = [
            config.FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{duration:.3f}", "-vf", vf,
        ]
    else:
        filter_complex = _audio_only_filter(width, height)
        if burn_captions and clip_words:
            filter_complex = filter_complex.replace(
                "[v]", "[vraw];[vraw]subtitles='" + _escape_filter_path(ass_path) + "'[v]"
            )
        cmd = [
            config.FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{duration:.3f}", "-filter_complex", filter_complex, "-map", "[v]", "-map", "0:a",
        ]

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-r", "30", "-g", "60",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-movflags", "+faststart",
        str(out_video),
    ]
    run(cmd, "clip render")

    run([
        config.FFMPEG, "-y", "-ss", f"{min(1.0, duration / 3):.2f}", "-i", str(out_video),
        "-frames:v", "1", "-q:v", "3", str(out_thumb),
    ], "thumbnail")

    return RenderResult(
        video_path=out_video, thumbnail_path=out_thumb, srt_path=out_srt,
        crop_mode=crop_mode, faces_found=faces_found, width=width, height=height,
    )
