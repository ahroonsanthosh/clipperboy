"""Render a scored moment into a finished, caption-burned social clip."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend import config
from backend.pipeline import captions as captions_mod
from backend.pipeline import speaker
from backend.pipeline import media as media_mod
from backend.pipeline.media import MediaInfo, run
from backend.pipeline.transcribe import Word

PAD_BEFORE = 0.25   # a breath of lead-in so the first word isn't clipped
PAD_AFTER = 0.35

# Sharpening to pay back an upscale. A 9:16 crop out of 1080p is only 608px
# wide and has to reach 1080 — a 1.78x upscale no resampler can make crisp.
# Amplitude 0.8/0.4 was chosen by looking at frames, not by maximising a
# sharpness number: 1.0 and 1.2 score higher but put a visible bright rim
# around dark-on-light text and around hair against a pale wall, which is
# exactly the artefact that makes a clip look cheap.
UNSHARP = "unsharp=5:5:0.8:3:3:0.4"
# Don't sharpen a downscale. Below this ratio the scale step is shrinking the
# picture, which is already sharpening it; unsharp on top only adds ringing.
UNSHARP_MIN_SCALE = 1.05


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


def _subtitles_filter(ass_path: Path) -> str:
    """Burn an ASS file in, using the bundled font so output matches everywhere."""
    args = f"subtitles='{_escape_filter_path(ass_path)}'"
    if config.FONTS_DIR.is_dir():
        args += f":fontsdir='{_escape_filter_path(config.FONTS_DIR)}'"
    return args


def _audio_only_filter(width: int, height: int, title_file: Path | None) -> str:
    """An animated audiogram card for audio-only podcasts.

    A live, mirrored spectrum sits on a dark card with the clip title above it.
    The spectrum is colour-keyed to transparency before compositing — it renders
    on opaque black, which would otherwise show as a hard band across the card.
    """
    half = max(60, int(height * 0.14))
    title = ""
    font = config.FONTS_DIR / "DejaVuSans-Bold.ttf"
    if title_file is not None and font.exists():
        title = (
            f",drawtext=fontfile='{_escape_filter_path(font)}'"
            f":textfile='{_escape_filter_path(title_file)}'"
            f":fontcolor=0xB8C6D6:fontsize={int(min(width, height) * 0.042)}"
            f":line_spacing={int(min(width, height) * 0.016)}"
            f":x=(w-text_w)/2:y=h*0.14"
        )
    return (
        f"[0:a]showfreqs=s={width}x{half}:mode=bar:ascale=cbrt:fscale=log:"
        f"win_size=1024:colors=0x22E0FF,format=rgba,"
        f"colorkey=0x000000:0.30:0.10,split[top][low];[low]vflip[bottom];"
        f"[top][bottom]vstack=2[wave];"
        f"color=c=0x0E1116:s={width}x{height}:r=30,format=rgba[bg];"
        f"[bg][wave]overlay=(W-w)/2:(H-h)/2-{int(height * 0.06)}:shortest=1"
        f"{title},format=yuv420p{{extra}}[v]"
    )


def _write_title_card(title: str, path: Path, max_chars: int = 24) -> Path | None:
    """Word-wrap a clip title to at most two lines for the audiogram card."""
    words = (title or "").split()
    if not words:
        return None
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][:max_chars - 1].rstrip() + "…"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _best_source(source: Path, info: MediaInfo,
                 original: Path | None = None) -> tuple[Path, MediaInfo]:
    """Pick the file to render from, and the MediaInfo that describes *it*.

    Prefers an explicitly supplied original, then one recorded by
    `media.compress`. Falls back to `source` unless the candidate probes clean,
    still has video, covers the same timeline (so timestamps still mean the
    same thing) and is genuinely no smaller than what we were handed.
    """
    candidate = original or media_mod.origin_of(source)
    if candidate is None or Path(candidate) == Path(source):
        return source, info
    try:
        cand_info = media_mod.probe(Path(candidate))
    except Exception:
        return source, info
    if not cand_info.has_video or not info.has_video:
        return source, info
    if abs(cand_info.duration - info.duration) > 1.0:
        return source, info
    if cand_info.width < info.width or cand_info.height < info.height:
        return source, info
    return Path(candidate), cand_info


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
    title: str = "",
    original: Path | None = None,
    quality: str = "",
) -> RenderResult:
    width, height = config.ASPECTS.get(aspect, config.ASPECTS[config.DEFAULT_ASPECT])
    target_ratio = width / height

    # Always render from the highest-resolution copy we still have. `source`
    # may be a downscaled working copy, and cropping 9:16 out of a downscale
    # and upscaling the result is the single biggest quality loss in the
    # pipeline. `info` must describe whatever we actually render, or the crop
    # geometry silently refers to the wrong frame size.
    source, info = _best_source(source, info, original)

    start = max(0.0, start - PAD_BEFORE)
    end = min(info.duration, end + PAD_AFTER)
    # The lead-in/out padding must not push a clip past the length limit — the
    # cap is what the user asked for, not a target to overshoot.
    end = min(end, start + config.MAX_CLIP_SEC)
    duration = max(0.5, end - start)

    clip_words = [w for w in words if w.end > start and w.start < end]

    ass_path = work_dir / f"{out_video.stem}.ass"
    captions_mod.build_srt(clip_words, start, end, out_srt)
    if burn_captions and clip_words:
        captions_mod.build_ass(clip_words, start, end, width, height, ass_path,
                               font=config.CAPTION_FONT)

    crop_mode = "audio-waveform"
    faces_found = 0

    # Match the source frame rate rather than forcing one. Resampling 25fps to
    # 30 inserts a duplicate every fifth frame — measured 150 frames becoming
    # 180 — which reads as judder on any camera or hand movement. Only cap
    # above 60, which no platform wants anyway.
    src_fps = info.fps if info.has_video and 1 < info.fps <= 240 else 30.0
    target_fps = min(src_fps, 60.0)
    rate_args = ["-r", f"{target_fps:.3f}"] if src_fps > 60.0 else []

    x264_preset, crf = config.QUALITY_PRESETS.get(
        quality or config.DEFAULT_QUALITY, config.QUALITY_PRESETS[config.DEFAULT_QUALITY])

    encode_args = rate_args + [
        # See config.QUALITY_PRESETS for the measurements behind these. The
        # short version: "superfast" was wasting about half its bitrate, and on
        # "faster" the encode time is flat across crf, so quality here costs
        # file size rather than minutes. Slower presets than "faster" bought
        # nothing measurable for a lot of seconds.
        "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        # A keyframe every 2 seconds: what the platforms' own re-encoders expect,
        # and what makes scrubbing in the preview responsive.
        "-g", str(max(2, int(round(target_fps * 2)))),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-movflags", "+faststart",
    ]
    # Render to a temp file and swap it in, so a viewer streaming the previous
    # version of this clip never sees a half-written file.
    staging = out_video.with_name(out_video.name + ".part.mp4")

    if info.has_video:
        # The clip's words tell the planner when anybody is speaking at all,
        # which gates its lip-motion evidence and stops silence from voting.
        plan = speaker.plan_crop(source, start, end, info.width, info.height,
                                 target_ratio, words=clip_words)
        crop_mode, faces_found = plan.mode, plan.faces_found

        def build(active: speaker.CropPlan) -> list[str]:
            chain = ["setpts=PTS-STARTPTS"]
            if active.mode != "full-frame":
                chain.append(
                    f"crop={active.crop_w}:{active.crop_h}:"
                    f"x='{active.expression()}':y=(ih-oh)/2"
                )
            # Lanczos holds detail and caption edges far better than the
            # default bicubic, at negligible cost.
            chain.append(
                f"scale={width}:{height}:force_original_aspect_ratio=increase"
                f":flags=lanczos,crop={width}:{height}"
            )
            src_w = active.crop_w if active.mode != "full-frame" else info.width
            src_h = active.crop_h if active.mode != "full-frame" else info.height
            factor = max(width / max(1, src_w), height / max(1, src_h))
            if factor >= UNSHARP_MIN_SCALE:
                # Note the ordering: sharpening goes *before* the captions are
                # burned in, so our own text is composited crisp at full size
                # and never picks up ringing from the filter.
                chain.append(UNSHARP)
            chain.append("setsar=1")
            if burn_captions and clip_words:
                chain.append(_subtitles_filter(ass_path))
            return [
                config.FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{duration:.3f}", "-vf", ",".join(chain),
            ] + encode_args + [str(staging)]

        try:
            run(build(plan), "clip render")
        except Exception:
            if plan.mode != "speaker-tracked":
                raise
            # A tracked camera path can defeat ffmpeg's expression parser on
            # pathological input; a static centre crop always works.
            fallback = speaker.CropPlan(
                plan.crop_w, plan.crop_h,
                [(0.0, max(0, (info.width - plan.crop_w) // 2))], plan.faces_found, "centred",
            )
            run(build(fallback), "clip render")
            crop_mode = "centred"
    else:
        title_file = _write_title_card(title, work_dir / f"{out_video.stem}.title.txt")
        extra = f",{_subtitles_filter(ass_path)}" if (burn_captions and clip_words) else ""
        filter_complex = _audio_only_filter(width, height, title_file).format(extra=extra)
        run([
            config.FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{duration:.3f}", "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a",
        ] + encode_args + [str(staging)], "clip render")

    staging.replace(out_video)

    run([
        config.FFMPEG, "-y", "-ss", f"{min(1.0, duration / 3):.2f}", "-i", str(out_video),
        "-frames:v", "1", "-q:v", "3", str(out_thumb),
    ], "thumbnail")
    ass_path.unlink(missing_ok=True)

    return RenderResult(
        video_path=out_video, thumbnail_path=out_thumb, srt_path=out_srt,
        crop_mode=crop_mode, faces_found=faces_found, width=width, height=height,
    )
