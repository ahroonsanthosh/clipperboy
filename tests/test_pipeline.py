"""Unit + integration tests for the clip pipeline (no model download needed)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.pipeline import captions, media, render, score, speaker
from backend.pipeline.transcribe import Word


# --- media ---------------------------------------------------------------

def test_probe_reads_video(video_fixture: Path):
    info = media.probe(video_fixture)
    assert info.has_video
    assert info.width == 1280 and info.height == 720
    assert info.duration > 30


def test_probe_marks_audio_only(audio_fixture: Path):
    info = media.probe(audio_fixture)
    assert info.is_audio_only
    assert info.duration > 30


def test_probe_rejects_garbage(tmp_path: Path):
    junk = tmp_path / "not-media.mp4"
    junk.write_bytes(b"definitely not a video")
    with pytest.raises(media.MediaError):
        media.probe(junk)


def test_audio_extraction_and_envelope(video_fixture: Path, tmp_path: Path):
    wav = media.extract_audio(video_fixture, tmp_path / "a.wav")
    assert wav.exists() and wav.stat().st_size > 1000
    env, hz = media.load_wav_envelope(wav)
    assert hz == 20
    assert env.size > 100
    assert float(env.max()) > 0  # the fixture actually contains sound


def test_compress_decision(video_fixture: Path):
    info = media.probe(video_fixture)
    assert media.should_compress(info, "always") is True
    assert media.should_compress(info, "never") is False
    assert media.should_compress(info, "auto") is False  # small 720p file


# --- scoring -------------------------------------------------------------

def test_sentences_are_split(fake_transcript):
    tr = fake_transcript(50.0)
    sents = score.to_sentences(tr)
    assert len(sents) >= 4
    assert all(s.end > s.start for s in sents)


def test_scoring_prefers_hooks_over_sponsor_reads(fake_transcript):
    import numpy as np

    tr = fake_transcript(50.0)
    sents = score.to_sentences(tr)
    env = np.full(1000, 0.3, dtype=np.float32)

    hook = next(s for s in sents if "nobody talks about" in s.text)
    ad = next(s for s in sents if "sponsor" in s.text.lower())

    hook_score = score.score_window([hook], env, 20, 50.0).score
    ad_score = score.score_window([ad], env, 20, 50.0).score
    assert hook_score > ad_score


def test_selection_returns_non_overlapping_clips(fake_transcript, monkeypatch):
    import numpy as np

    monkeypatch.setattr(score.config, "MIN_CLIP_SEC", 6)
    monkeypatch.setattr(score.config, "MAX_CLIP_SEC", 25)
    tr = fake_transcript(60.0)
    env = np.abs(np.sin(np.linspace(0, 40, 1200))).astype(np.float32)

    clips = score.find_clips(tr, env, 20, 3)
    assert 1 <= len(clips) <= 3
    for a, b in zip(clips, clips[1:]):
        assert a.end <= b.start + 1e-6
    for c in clips:
        assert c.title and c.summary
        assert 6 <= c.duration <= 25
        assert c.reasons


def test_titles_stay_short(fake_transcript):
    tr = fake_transcript(50.0)
    cand = score.score_window(score.to_sentences(tr), __import__("numpy").zeros(10, dtype="float32"), 20, 50.0)
    title, summary = score.local_title_and_summary(cand)
    assert 0 < len(title) <= 60
    assert summary


def test_keywords_exclude_stopwords():
    kws = score.keywords_for("the focus and the output and the focus of the work")
    assert "focus" in kws
    assert "the" not in kws and "and" not in kws


# --- captions ------------------------------------------------------------

def _words(pairs):
    return [Word(start=s, end=e, text=t) for s, e, t in pairs]


def test_word_grouping_breaks_on_pauses_and_punctuation():
    words = _words([
        (0.0, 0.3, "Hello"), (0.3, 0.6, "there"), (0.6, 0.9, "friend."),
        (3.0, 3.3, "New"), (3.3, 3.6, "line"),
    ])
    lines = captions.group_words(words)
    assert lines[0][-1].text == "friend."
    assert lines[1][0].text == "New"


def test_ass_file_has_events_and_valid_timing(tmp_path: Path):
    words = _words([(10.0, 10.4, "one"), (10.4, 10.9, "two"), (10.9, 11.5, "three.")])
    out = captions.build_ass(words, 10.0, 12.0, 1080, 1920, tmp_path / "c.ass")
    text = out.read_text()
    assert "[Events]" in text and "Dialogue:" in text
    # Timings are clip-relative, so nothing should start beyond the clip length.
    assert "0:00:0" in text
    assert text.count("Dialogue:") == 3   # one event per word


def test_srt_is_written(tmp_path: Path):
    words = _words([(5.0, 5.4, "hello"), (5.4, 6.0, "world.")])
    out = captions.build_srt(words, 5.0, 7.0, tmp_path / "c.srt")
    body = out.read_text()
    assert "-->" in body and "hello world." in body


def test_ass_escapes_brace_characters(tmp_path: Path):
    words = _words([(0.0, 0.5, "{tricky}")])
    out = captions.build_ass(words, 0.0, 1.0, 1080, 1920, tmp_path / "c.ass")
    line = [l for l in out.read_text().splitlines() if l.startswith("Dialogue:")][0]
    assert "{tricky}" not in line     # would otherwise be read as an ASS override tag


# --- speaker crop --------------------------------------------------------

def test_crop_plan_for_wide_source(video_fixture: Path):
    info = media.probe(video_fixture)
    plan = speaker.plan_crop(video_fixture, 5.0, 12.0, info.width, info.height, 9 / 16)
    assert plan.mode in {"speaker-tracked", "centred"}
    assert plan.crop_h == 720
    assert plan.crop_w == pytest.approx(720 * 9 / 16, abs=2)
    assert plan.keyframes
    for _, x in plan.keyframes:
        assert 0 <= x <= info.width - plan.crop_w


def test_crop_skipped_when_source_is_already_narrow():
    plan = speaker.plan_crop(Path("/nonexistent.mp4"), 0, 5, 1080, 1920, 9 / 16)
    assert plan.mode == "full-frame"


def test_crop_expression_is_valid_ffmpeg(video_fixture: Path, tmp_path: Path):
    info = media.probe(video_fixture)
    plan = speaker.plan_crop(video_fixture, 2.0, 8.0, info.width, info.height, 9 / 16)
    out = tmp_path / "crop.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-ss", "2", "-i", str(video_fixture), "-t", "2",
        "-vf", f"setpts=PTS-STARTPTS,crop={plan.crop_w}:{plan.crop_h}:x='{plan.expression()}':y=(ih-oh)/2",
        "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out),
    ], check=True, capture_output=True)
    assert media.probe(out).width == plan.crop_w


# --- rendering ------------------------------------------------------------

@pytest.mark.parametrize("aspect,expect", [("9:16", (1080, 1920)), ("1:1", (1080, 1080)), ("16:9", (1920, 1080))])
def test_render_produces_correct_dimensions(video_fixture, tmp_path, fake_transcript, aspect, expect):
    info = media.probe(video_fixture)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=video_fixture, info=info, start=6.0, end=18.0, words=tr.words, aspect=aspect,
        out_video=tmp_path / "clip.mp4", out_thumb=tmp_path / "clip.jpg",
        out_srt=tmp_path / "clip.srt", work_dir=tmp_path,
    )
    out_info = media.probe(result.video_path)
    assert (out_info.width, out_info.height) == expect
    assert 11 < out_info.duration < 14          # includes the small pads
    assert result.thumbnail_path.exists()
    assert result.srt_path.read_text().strip()


def test_render_audio_only_makes_a_waveform_video(audio_fixture, tmp_path, fake_transcript):
    info = media.probe(audio_fixture)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=audio_fixture, info=info, start=4.0, end=14.0, words=tr.words, aspect="9:16",
        out_video=tmp_path / "a.mp4", out_thumb=tmp_path / "a.jpg",
        out_srt=tmp_path / "a.srt", work_dir=tmp_path,
    )
    out_info = media.probe(result.video_path)
    assert out_info.has_video and (out_info.width, out_info.height) == (1080, 1920)
    assert result.crop_mode == "audio-waveform"


def test_render_clamps_to_source_end(video_fixture, tmp_path, fake_transcript):
    info = media.probe(video_fixture)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=video_fixture, info=info, start=info.duration - 5, end=info.duration + 30,
        words=tr.words, aspect="9:16",
        out_video=tmp_path / "tail.mp4", out_thumb=tmp_path / "tail.jpg",
        out_srt=tmp_path / "tail.srt", work_dir=tmp_path,
    )
    assert media.probe(result.video_path).duration <= 6.5


def test_captions_are_actually_burned_into_the_picture(video_fixture, tmp_path, fake_transcript):
    """Compare the caption band of a captioned render against an uncaptioned one."""
    import numpy as np

    info = media.probe(video_fixture)
    tr = fake_transcript(info.duration)

    def frame_band(burn: bool, name: str) -> np.ndarray:
        result = render.render_clip(
            source=video_fixture, info=info, start=8.0, end=20.0, words=tr.words,
            aspect="9:16", out_video=tmp_path / f"{name}.mp4",
            out_thumb=tmp_path / f"{name}.jpg", out_srt=tmp_path / f"{name}.srt",
            work_dir=tmp_path, burn_captions=burn,
        )
        png = tmp_path / f"{name}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", "3", "-i", str(result.video_path),
            "-frames:v", "1", str(png),
        ], check=True, capture_output=True)
        import cv2
        img = cv2.imread(str(png))
        return img[int(img.shape[0] * 0.7):int(img.shape[0] * 0.95), :, :]

    with_caps = frame_band(True, "caps")
    without = frame_band(False, "nocaps")
    assert with_caps.shape == without.shape
    changed = float(np.mean(np.abs(with_caps.astype(float) - without.astype(float))))
    assert changed > 1.0, "caption band looks identical — captions were not burned in"


# --- regressions ----------------------------------------------------------

def test_long_crop_expression_is_accepted_by_ffmpeg(video_fixture, tmp_path):
    """ffmpeg's expression parser rejects >100 nested calls; a long tracked
    clip used to produce exactly that and fail the whole render."""
    keyframes = [(i * 0.5, 100 + (i % 40)) for i in range(151)]
    plan = speaker.CropPlan(404, 720, keyframes, faces_found=151, mode="speaker-tracked")
    assert "if(" not in plan.expression()

    out = tmp_path / "long.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-ss", "2", "-i", str(video_fixture), "-t", "2",
        "-vf", f"setpts=PTS-STARTPTS,crop=404:720:x='{plan.expression()}':y=(ih-oh)/2",
        "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out),
    ], check=True, capture_output=True)
    assert media.probe(out).width == 404


def test_crop_expression_holds_the_first_keyframe(video_fixture):
    """The opening position must be honoured, not skipped."""
    plan = speaker.CropPlan(400, 720, [(0.0, 111), (5.0, 222)], 2, "speaker-tracked")
    expr = plan.expression()
    assert "111" in expr and "222" in expr
    assert expr.startswith("111*lt(t,5.00)")


def test_plan_crop_caps_keyframe_count(video_fixture):
    info = media.probe(video_fixture)
    plan = speaker.plan_crop(video_fixture, 0.0, info.duration, info.width, info.height, 9 / 16)
    assert len(plan.keyframes) <= speaker.MAX_KEYFRAMES


def test_render_falls_back_to_centre_crop_when_tracking_fails(
        video_fixture, tmp_path, fake_transcript, monkeypatch):
    info = media.probe(video_fixture)
    tr = fake_transcript(info.duration)

    # A plan whose expression ffmpeg cannot evaluate at all.
    broken = speaker.CropPlan(405, 720, [(0.0, 0)], 5, "speaker-tracked")
    monkeypatch.setattr(broken, "expression", lambda: "this-is-not-an-expression")
    monkeypatch.setattr(speaker, "plan_crop", lambda *a, **k: broken)

    result = render.render_clip(
        source=video_fixture, info=info, start=4.0, end=14.0, words=tr.words,
        aspect="9:16", out_video=tmp_path / "fb.mp4", out_thumb=tmp_path / "fb.jpg",
        out_srt=tmp_path / "fb.srt", work_dir=tmp_path,
    )
    assert result.crop_mode == "centred"
    assert media.probe(result.video_path).width == 1080


def test_audio_card_has_no_black_band(audio_fixture, tmp_path, fake_transcript):
    """showwaves/showfreqs render on opaque black; without colour-keying, the
    card showed a hard black rectangle across the middle."""
    import cv2
    import numpy as np

    info = media.probe(audio_fixture)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=audio_fixture, info=info, start=8.0, end=20.0, words=tr.words,
        aspect="9:16", out_video=tmp_path / "card.mp4", out_thumb=tmp_path / "card.jpg",
        out_srt=tmp_path / "card.srt", work_dir=tmp_path, title="A test clip title",
    )
    png = tmp_path / "card.png"
    subprocess.run(["ffmpeg", "-y", "-ss", "4", "-i", str(result.video_path),
                    "-frames:v", "1", str(png)], check=True, capture_output=True)
    img = cv2.imread(str(png))

    # The card background is 0x0E1116; pure black anywhere means the band is back.
    band = img[int(img.shape[0] * 0.35):int(img.shape[0] * 0.65), :, :]
    pure_black = float(np.mean(np.all(band < 4, axis=2)))
    assert pure_black < 0.02, f"{pure_black:.1%} of the card centre is pure black"


def test_audio_card_draws_the_clip_title(audio_fixture, tmp_path, fake_transcript):
    import cv2
    import numpy as np

    info = media.probe(audio_fixture)
    tr = fake_transcript(info.duration)

    def top_band(title: str, name: str):
        result = render.render_clip(
            source=audio_fixture, info=info, start=8.0, end=16.0, words=tr.words,
            aspect="9:16", out_video=tmp_path / f"{name}.mp4",
            out_thumb=tmp_path / f"{name}.jpg", out_srt=tmp_path / f"{name}.srt",
            work_dir=tmp_path, title=title,
        )
        png = tmp_path / f"{name}.png"
        subprocess.run(["ffmpeg", "-y", "-ss", "2", "-i", str(result.video_path),
                        "-frames:v", "1", str(png)], check=True, capture_output=True)
        img = cv2.imread(str(png))
        return img[int(img.shape[0] * 0.10):int(img.shape[0] * 0.22), :, :].astype(float)

    assert float(np.mean(np.abs(top_band("Focus beats hours", "titled") - top_band("", "plain")))) > 1.0
