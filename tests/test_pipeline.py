"""Unit + integration tests for the clip pipeline (no model download needed)."""
from __future__ import annotations

import re
import subprocess
import sys
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
    import dataclasses

    info = media.probe(video_fixture)
    assert media.should_compress(info, "always") is True
    assert media.should_compress(info, "never") is False
    assert media.should_compress(info, "auto") is False  # 720p — nothing to gain

    # A big 1080p episode must NOT be re-encoded on "auto": that costs ~22
    # minutes for a 60-minute file and saves under a minute of decode.
    big_1080p = dataclasses.replace(info, width=1920, height=1080,
                                    size_bytes=3 * 1024 ** 3, duration=3600)
    assert media.should_compress(big_1080p, "auto") is False

    # A 4K source must NOT be downscaled either. Every clip is a vertical crop
    # of the frame, so pre-downscaling to 1080p discards exactly the pixels the
    # crop was going to keep and forces the render to upscale what's left.
    uhd = dataclasses.replace(info, width=3840, height=2160)
    assert media.should_compress(uhd, "auto") is False

    # Audio-only has no video to compress.
    assert media.should_compress(dataclasses.replace(info, has_video=False), "always") is False


def test_compress_records_its_origin(video_fixture: Path, tmp_path):
    """A working copy remembers where it came from, so the renderer can go back
    to the full-resolution file instead of cropping a downscale."""
    dst = tmp_path / "source.mp4"
    media.compress(video_fixture, dst)
    assert media.origin_of(dst) == video_fixture.resolve()

    # Nothing recorded, or the original since deleted: no claim is made.
    assert media.origin_of(tmp_path / "unrelated.mp4") is None
    media.record_origin(dst, tmp_path / "gone.mp4")
    assert media.origin_of(dst) is None


def test_render_prefers_the_original_over_a_downscaled_copy(video_fixture: Path, tmp_path):
    """The crop must be taken from the biggest frame available, and the
    MediaInfo used for the crop maths must describe that same file."""
    small = tmp_path / "small.mp4"
    media.run([
        "ffmpeg", "-y", "-i", str(video_fixture),
        "-vf", "scale=640:360", "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "30", str(small),
    ], "downscale")
    media.record_origin(small, video_fixture)
    small_info = media.probe(small)

    chosen, chosen_info = render._best_source(small, small_info)
    assert chosen == video_fixture.resolve()
    assert (chosen_info.width, chosen_info.height) == (small_info.width * 2,
                                                       small_info.height * 2)

    # An explicit original wins, and a bogus one is ignored rather than fatal.
    assert render._best_source(small, small_info, video_fixture)[0] == video_fixture
    assert render._best_source(small, small_info, tmp_path / "nope.mp4")[0] == small

    # A candidate that is *smaller* than what we were handed is never used.
    tiny = tmp_path / "tiny.mp4"
    media.run([
        "ffmpeg", "-y", "-i", str(small), "-vf", "scale=320:180",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", str(tiny),
    ], "downscale")
    assert render._best_source(small, small_info, tiny)[0] == small


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
    assert expr.startswith("111*lt(t,5.000)")


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


def test_over_long_words_are_hyphenated_for_burned_captions(tmp_path):
    """libass only wraps at spaces, so a 46-character compound used to be
    sliced off both frame edges."""
    long_word = "Verkehrsinfrastrukturfinanzierungsgesellschaft"
    words = _words([(0.0, 1.2, long_word), (1.2, 1.6, "ist"), (1.6, 2.0, "lang.")])

    lines = captions.group_words(words)
    pieces = [w.text for line in lines for w in line]
    assert long_word not in pieces
    assert any(p.endswith("-") for p in pieces)
    assert max(len(p) for p in pieces) <= captions.HARD_BREAK_CHARS
    # The word is preserved, just broken.
    assert "".join(p.rstrip("-") for p in pieces if p not in ("ist", "lang.")) == long_word

    # The .srt sidecar has no width limit, so it keeps the real word.
    srt = captions.build_srt(words, 0.0, 3.0, tmp_path / "w.srt").read_text()
    assert long_word in srt


def test_wide_lines_are_scaled_down_to_fit_the_margins(tmp_path, monkeypatch):
    """Belt-and-braces: a line that would still overrun the safe margins is
    scaled down rather than sliced. With normal settings the line-length cap
    keeps this from ever firing, so the cap is raised to exercise it."""
    line = _words([(0.0, 0.4, "abcdefghijklmnopqrst"), (0.4, 0.8, "uvwxyzabcdefghij")])

    normal = captions.build_ass(line, 0.0, 1.0, 1080, 1920, tmp_path / "n.ass").read_text()
    assert {int(m) for m in re.findall(r"\\fscx(\d+)", normal)} == {100, 108}

    monkeypatch.setattr(captions, "MAX_LINE_CHARS", 80)
    monkeypatch.setattr(captions, "WORDS_PER_LINE", 8)
    wide = captions.build_ass(line, 0.0, 1.0, 1080, 1920, tmp_path / "w.ass").read_text()
    scales = {int(m) for m in re.findall(r"\\fscx(\d+)", wide)}
    assert min(scales) < 100
    assert min(scales) >= captions.MIN_LINE_SCALE


def test_long_words_stay_inside_the_frame(tmp_path):
    """End-to-end proof: burn a brutal line and check no pixel touches an edge."""
    import cv2
    import numpy as np

    words = _words([(0.0, 1.0, "Die"), (1.0, 2.4, "Verkehrsinfrastrukturfinanzierungsgesellschaft"),
                    (2.4, 3.0, "ist"), (3.0, 3.6, "kompliziert.")])
    ass = captions.build_ass(words, 0.0, 4.0, 1080, 1920, tmp_path / "edge.ass")
    out = tmp_path / "edge.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x203040:s=1080x1920:d=4:r=8",
        "-vf", f"subtitles='{ass}':fontsdir='{Path('assets/fonts').resolve()}'",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out),
    ], check=True, capture_output=True)

    cap = cv2.VideoCapture(str(out))
    left, right, seen = 1.0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cols = np.where((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 200).any(axis=0))[0]
        if cols.size:
            left = min(left, cols.min() / 1080)
            right = max(right, cols.max() / 1080)
            seen += 1
    cap.release()

    assert seen > 5, "no captions were rendered at all"
    assert left > 0.05, f"caption reaches the left edge ({left:.1%})"
    assert right < 0.95, f"caption reaches the right edge ({right:.1%})"


def test_face_sampling_walks_the_clip_once(video_fixture, monkeypatch):
    """Seeking per sample forced a keyframe re-decode each time and dominated
    this stage; the sampler must seek at most once."""
    info = media.probe(video_fixture)
    seeks = []
    real_set = speaker.cv2.VideoCapture.set

    def counting_set(self, prop, value):
        seeks.append(prop)
        return real_set(self, prop, value)

    monkeypatch.setattr(speaker.cv2.VideoCapture, "set", counting_set)
    times, centres = speaker._sample_faces(video_fixture, 5.0, 20.0)

    assert len(seeks) <= 1
    assert len(times) == len(centres)
    expected = 20 * speaker.SAMPLE_HZ
    assert expected * 0.75 <= len(times) <= expected * 1.3
    assert times == sorted(times)
    assert max(times) <= 20.5


def _skin_frame(box=(100, 50, 60, 60), size=(1080, 1920)):
    """A source frame that is blank except for a skin-toned patch at `box`.

    `box` is in 640px-wide detection coordinates, which is where the cascades
    report. The patch is BGR (90, 130, 190) — a mid-brown that sits inside the
    YCrCb skin locus — so a box drawn around it survives the colour test.
    """
    import numpy as np

    frame = np.zeros((*size, 3), dtype=np.uint8)
    scale = size[1] / speaker.DETECT_WIDTH
    x, y, w, h = (int(v * scale) for v in box)
    frame[y:y + h, x:x + w] = (90, 130, 190)
    return frame


def test_detection_reports_centres_in_source_pixels(monkeypatch):
    """The Haar fallback detects on a 640px copy; centres must be scaled back."""
    import numpy as np

    seen = {}

    def fake_run(name, gray, flip):
        seen["width"] = gray.shape[1]
        seen["name"] = name
        return np.array([[100, 50, 60, 60]])      # a face at x=100 on the small frame

    # BlazeFace is the primary detector and would answer first (with nothing, on
    # a synthetic frame), so this pins the fallback specifically.
    monkeypatch.setattr(speaker, "_mediapipe", lambda: None)
    monkeypatch.setattr(speaker, "_run_cascade", fake_run)
    centre = speaker._detect_centre(_skin_frame())

    assert seen["width"] == speaker.DETECT_WIDTH
    assert centre == pytest.approx((100 + 30) * (1920 / 640))


def test_haar_fallback_rejects_boxes_with_no_skin_in_them(monkeypatch):
    """The wall-sign false positives are what made real clips crop onto wall.

    On the reference podcast clip the cascades fired on 83% of samples and every
    box was the sponsor sign behind the speaker, never the speaker. The colour
    test is what removes them, so a box over blank background must find nothing.
    """
    import numpy as np

    monkeypatch.setattr(speaker, "_mediapipe", lambda: None)
    # Only the *face* cascades answer. The upper-body cascade is deliberately
    # exempt from the colour test (a torso is clothing, not skin), so letting it
    # fire here would test nothing.
    monkeypatch.setattr(
        speaker, "_run_cascade",
        lambda name, gray, flip: (np.empty((0, 4), int) if name == "upperbody"
                                  else np.array([[100, 50, 60, 60]])))

    # Same box, same cascade output — only the pixels underneath differ.
    assert speaker._detect_centre(_skin_frame()) is not None
    assert speaker._detect_centre(np.zeros((1080, 1920, 3), np.uint8)) is None


def test_blazeface_boxes_are_scaled_back_to_source_pixels(monkeypatch):
    """BlazeFace reports relative coordinates on the same 640px copy."""
    import numpy as np

    class _Box:
        xmin, ymin, width, height = 0.25, 0.10, 0.10, 0.20

    class _Det:
        score = [0.9]
        location_data = type("L", (), {"relative_bounding_box": _Box()})()

    class _FakeGraph:
        def process(self, rgb):
            assert rgb.shape[1] == speaker.DETECT_WIDTH, "must run on the small copy"
            return type("R", (), {"detections": [_Det()]})()

        def close(self):
            pass

    monkeypatch.setattr(speaker, "_mediapipe",
                        lambda: type("M", (), {"FaceDetection": lambda **kw: _FakeGraph()}))
    centre = speaker._detect_centre(np.zeros((1080, 1920, 3), np.uint8))

    scale = 1920 / speaker.DETECT_WIDTH
    expected = (0.25 + 0.10 / 2) * speaker.DETECT_WIDTH * scale
    assert centre == pytest.approx(expected, abs=scale)


def test_low_confidence_blazeface_detections_are_discarded(monkeypatch):
    """A locked camera should not be pointed on the strength of a maybe."""
    import numpy as np

    class _Box:
        xmin, ymin, width, height = 0.25, 0.10, 0.10, 0.20

    def graph_scoring(score):
        det = type("D", (), {
            "score": [score],
            "location_data": type("L", (), {"relative_bounding_box": _Box()})(),
        })()
        return type("G", (), {
            "process": lambda self, rgb: type("R", (), {"detections": [det]})(),
            "close": lambda self: None,
        })()

    frame = np.zeros((1080, 1920, 3), np.uint8)
    for score, expected in ((0.95, True), (speaker.FACE_MIN_CONF - 0.05, False)):
        monkeypatch.setattr(
            speaker, "_mediapipe",
            lambda s=score: type("M", (), {"FaceDetection": lambda **kw: graph_scoring(s)}))
        # With mediapipe declining, the (unmocked) cascades see a blank frame.
        assert (speaker._detect_centre(frame) is not None) is expected


def test_a_broken_mediapipe_never_fails_a_job(monkeypatch, caplog):
    """A detector import error is a downgrade to Haar, not an error."""
    import numpy as np

    monkeypatch.setattr(speaker, "_mp_face_detection", None)
    monkeypatch.setattr(
        speaker, "_run_cascade",
        lambda name, gray, flip: np.array([[100, 50, 60, 60]]))

    # Simulate mediapipe not being installed. Every cached `mediapipe.*` entry
    # has to go: a `from mediapipe.python.solutions import ...` is served
    # straight out of sys.modules under its full dotted name, so stubbing only
    # the top-level package would leave the import quietly succeeding.
    for name in [n for n in sys.modules if n == "mediapipe"
                 or n.startswith("mediapipe.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mediapipe", None)   # force the import to fail

    with caplog.at_level("WARNING"):
        centre = speaker._detect_centre(_skin_frame())

    assert centre == pytest.approx((100 + 30) * (1920 / 640)), \
        "the Haar fallback must still produce a framing decision"
    assert any("Haar" in r.getMessage() for r in caplog.records), \
        "the downgrade must be logged once"


def test_detector_learns_which_cascade_suits_the_footage(monkeypatch):
    """A full ladder pass is ~230ms; only the first samples may pay for it."""
    import numpy as np

    calls = []
    winner = ("profileface", False)

    def fake_run(name, gray, flip):
        calls.append((name, flip))
        if (name, flip) == winner:
            return np.array([[10, 10, 50, 50]])
        return np.empty((0, 4), int)

    monkeypatch.setattr(speaker, "_run_cascade", fake_run)
    det = speaker._Detector()
    gray = np.zeros((360, 640), dtype=np.uint8)

    for _ in range(speaker.PROBE_SAMPLES):
        boxes, is_face = det.detect(gray)
        assert len(boxes) and is_face
    probe_calls = len(calls)
    assert det.preferred == winner

    calls.clear()
    for _ in range(10):
        assert len(det.detect(gray)[0])
    assert len(calls) == 10, "the learned cascade should be tried alone"
    assert set(calls) == {winner}
    assert probe_calls > 10, "the probe phase is what pays for the search"


def test_upper_body_never_becomes_the_preferred_detector(monkeypatch):
    """Its box centre is the torso, so it must stay a last resort."""
    import numpy as np

    def fake_run(name, gray, flip):
        return np.array([[10, 10, 80, 120]]) if name == "upperbody" else np.empty((0, 4), int)

    monkeypatch.setattr(speaker, "_run_cascade", fake_run)
    det = speaker._Detector()
    boxes, is_face = det.detect(np.zeros((360, 640), dtype=np.uint8))

    assert len(boxes) == 1
    assert is_face is False
    assert det.preferred is None


def test_transcribe_falls_back_when_batching_is_unavailable(monkeypatch):
    """A batched-pipeline incompatibility surfaces during iteration. It must
    never sink the job — the sequential path has to take over."""
    from backend.pipeline import transcribe as tr

    calls = []

    class FakeModel:
        def transcribe(self, path, **kw):
            calls.append("sequential")
            seg = type("S", (), {"start": 0.0, "end": 1.0, "text": "hello there.",
                                 "words": []})()
            return iter([seg]), type("I", (), {"language": "en", "duration": 1.0})()

    def exploding(*a, **k):
        def gen():
            raise RuntimeError("batched not supported here")
            yield
        return gen(), None

    class FakePipeline:
        def __init__(self, model):
            pass
        transcribe = staticmethod(exploding)

    monkeypatch.setattr(tr, "_load_model", lambda *a, **k: FakeModel())
    import faster_whisper
    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", FakePipeline)

    result = tr.transcribe(Path("ignored.wav"), 1.0)
    assert calls == ["sequential"]
    assert result.segments[0].text == "hello there."


def test_no_clip_is_longer_than_the_limit(video_fixture, tmp_path, fake_transcript):
    """The cap is a hard limit: the lead-in/out padding must not overshoot it."""
    info = media.probe(video_fixture)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=video_fixture, info=info,
        start=2.0, end=2.0 + media.config.MAX_CLIP_SEC,     # already at the limit
        words=tr.words, aspect="9:16",
        out_video=tmp_path / "cap.mp4", out_thumb=tmp_path / "cap.jpg",
        out_srt=tmp_path / "cap.srt", work_dir=tmp_path,
    )
    assert media.probe(result.video_path).duration <= media.config.MAX_CLIP_SEC + 0.15


def test_output_keeps_the_source_frame_rate(tmp_path, fake_transcript):
    """Forcing 30fps on a 25fps source duplicates every fifth frame, which reads
    as judder on any movement."""
    src = tmp_path / "25fps.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=8",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src),
    ], check=True, capture_output=True)

    info = media.probe(src)
    assert round(info.fps) == 25
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=src, info=info, start=1.0, end=6.0, words=tr.words, aspect="9:16",
        out_video=tmp_path / "r.mp4", out_thumb=tmp_path / "r.jpg",
        out_srt=tmp_path / "r.srt", work_dir=tmp_path,
    )

    out = media.probe(result.video_path)
    assert round(out.fps) == 25, f"frame rate was resampled to {out.fps}"

    # And no duplicated frames were introduced.
    frames = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(result.video_path),
    ], check=True, capture_output=True, text=True).stdout.strip()
    assert abs(int(frames) - round(out.duration * 25)) <= 3


# --- source metadata that changes the render ------------------------------

def _rotated(tmp_path, degrees):
    flat = tmp_path / "flat.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(flat),
    ], check=True, capture_output=True)
    out = tmp_path / f"rot{degrees}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-display_rotation", str(degrees), "-i", str(flat),
        "-c", "copy", str(out),
    ], check=True, capture_output=True)
    return out


def test_probe_reports_the_rotated_dimensions(tmp_path):
    """ffprobe gives the CODED size but ffmpeg's decoder auto-rotates, so the
    filter graph sees the frame turned. Taking the coded size meant a portrait
    phone clip had a 608x1080 crop applied to a 1080-wide frame."""
    info = media.probe(_rotated(tmp_path, 90))
    assert info.rotation == 90
    assert (info.width, info.height) == (1080, 1920)

    # ...and that matches what ffmpeg actually hands the filters.
    transcoded = tmp_path / "re.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(_rotated(tmp_path, 90)), "-frames:v", "2",
        "-c:v", "libx264", "-preset", "ultrafast", str(transcoded),
    ], check=True, capture_output=True)
    seen = media.probe(transcoded)
    assert (seen.width, seen.height) == (info.width, info.height)


def test_a_rotated_portrait_source_is_not_cropped(tmp_path, fake_transcript):
    """A 1080x1920 phone clip already is 9:16 — it must pass through whole, not
    have a third of it cut out and blown up."""
    src = _rotated(tmp_path, 90)
    info = media.probe(src)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=src, info=info, start=0.2, end=1.8, words=tr.words, aspect="9:16",
        out_video=tmp_path / "r.mp4", out_thumb=tmp_path / "r.jpg",
        out_srt=tmp_path / "r.srt", work_dir=tmp_path,
    )
    assert result.crop_mode == "full-frame", "a portrait source was cropped anyway"
    assert (media.probe(result.video_path).width,
            media.probe(result.video_path).height) == (1080, 1920)


def test_output_is_tagged_bt709(tmp_path, video_fixture, fake_transcript):
    """An untagged file gets guessed at BT.601 by some players — the classic
    'skin tones look wrong' bug."""
    info = media.probe(video_fixture)
    tr = fake_transcript(info.duration)
    result = render.render_clip(
        source=video_fixture, info=info, start=2.0, end=8.0, words=tr.words,
        aspect="9:16", out_video=tmp_path / "c.mp4", out_thumb=tmp_path / "c.jpg",
        out_srt=tmp_path / "c.srt", work_dir=tmp_path,
    )
    out = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
        "stream=color_space,color_transfer,color_primaries,color_range",
        "-of", "default=nw=1", str(result.video_path),
    ], check=True, capture_output=True, text=True).stdout
    for field in ("color_space=bt709", "color_transfer=bt709",
                  "color_primaries=bt709", "color_range=tv"):
        assert field in out, f"{field} missing from:\n{out}"


def test_interlaced_sources_are_detected(tmp_path):
    src = tmp_path / "int.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=720x576:rate=25:duration=2",
        "-vf", "interlace", "-c:v", "libx264", "-preset", "ultrafast",
        "-flags", "+ilme+ildct", "-top", "1", "-pix_fmt", "yuv420p", str(src),
    ], check=True, capture_output=True)
    assert media.probe(src).interlaced is True

    progressive = tmp_path / "prog.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=720x576:rate=25:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(progressive),
    ], check=True, capture_output=True)
    assert media.probe(progressive).interlaced is False


@pytest.mark.parametrize("seconds, expected", [
    (0.0, "00:00:00,000"),
    (-1.0, "00:00:00,000"),
    (1.25, "00:00:01,250"),
    (1.9996, "00:00:02,000"),
    (59.9999, "00:01:00,000"),
    (3599.9999, "01:00:00,000"),
])
def test_srt_time_carries_rounding(seconds, expected):
    assert captions._srt_time(seconds) == expected


@pytest.mark.parametrize("seconds, expected", [
    (0.0, "0:00:00.00"),
    (12.34, "0:00:12.34"),
    (59.996, "0:01:00.00"),
    (3599.999, "1:00:00.00"),
])
def test_ass_time_carries_rounding(seconds, expected):
    assert captions._ass_time(seconds) == expected
