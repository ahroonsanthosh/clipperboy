"""Locked-camera framing: the frame must sit still and cut, never drift.

OpenCV's Haar cascade does not fire on drawn faces, so real video can't be
synthesised to exercise detection offline. These tests inject known face
observations directly into the planner instead, which is where all the new
behaviour lives.
"""
from __future__ import annotations

import pytest

from backend.pipeline import speaker
from backend.pipeline.speaker import Face


def _plan(monkeypatch, samples, times=None, words=None,
          src_w=1920, src_h=1080, ratio=9 / 16):
    times = times or [i / speaker.SAMPLE_HZ for i in range(len(samples))]
    monkeypatch.setattr(speaker, "_sample_faces", lambda *a, **k: (times, samples))
    return speaker.plan_crop("ignored.mp4", 0.0, times[-1] + 1, src_w, src_h, ratio, words=words)


def _face(cx, lip=0.0):
    return Face(cx=cx, cy=520.0, w=220.0, h=260.0, lip=lip)


class _Word:
    def __init__(self, start, end):
        self.start, self.end = start, end


def test_single_speaker_never_moves(monkeypatch):
    """A lone speaker with ordinary detector jitter must produce a dead-still frame."""
    jitter = [0, 6, -5, 9, -8, 4, -3, 7, -6, 2] * 6
    samples = [[_face(900 + j)] for j in jitter]
    plan = _plan(monkeypatch, samples)

    assert plan.mode == "speaker-tracked"
    assert plan.switches == 0, "camera moved on detector jitter alone"
    assert len({x for _t, x in plan.keyframes}) == 1


def test_biggest_face_does_not_win_when_the_other_one_is_talking(monkeypatch):
    """On a two-shot the camera must follow the voice, not whoever is nearest."""
    near = Face(cx=500.0, cy=520.0, w=400.0, h=460.0, lip=0.0)   # big, silent
    far = Face(cx=1500.0, cy=520.0, w=180.0, h=210.0, lip=9.0)   # small, talking
    samples = [[near, far] for _ in range(40)]
    words = [_Word(0.0, 40.0)]
    plan = _plan(monkeypatch, samples, words=words)

    held = {x for _t, x in plan.keyframes}
    assert len(held) == 1
    x = held.pop()
    # The framed window must contain the talker, not the nearer silent guest.
    assert x <= 1500 <= x + plan.crop_w, f"framed {x}..{x + plan.crop_w}, talker at 1500"


def test_speaker_change_produces_a_cut_not_a_drift(monkeypatch):
    """Two turns must give exactly one instant switch between two held positions."""
    def turn(active_cx, other_cx, n):
        return [[_face(active_cx, lip=9.0), _face(other_cx, lip=0.2)] for _ in range(n)]

    samples = turn(500, 1500, 24) + turn(1500, 500, 24)
    plan = _plan(monkeypatch, samples, words=[_Word(0.0, 60.0)])

    xs = [x for _t, x in plan.keyframes]
    assert plan.switches == 1, f"expected one cut, got {plan.switches}: {plan.keyframes}"
    assert len(set(xs)) == 2
    # Instant, not eased: no intermediate positions between the two held values.
    assert set(xs) == {xs[0], xs[-1]}


def test_rapid_back_and_forth_cannot_ping_pong(monkeypatch):
    """Fast dialogue must not make the camera oscillate — dwell time forbids it."""
    samples = []
    for i in range(60):
        a_talks = (i // 2) % 2 == 0          # swaps every ~1s at 2Hz
        samples.append([_face(500, lip=9.0 if a_talks else 0.2),
                        _face(1500, lip=0.2 if a_talks else 9.0)])
    plan = _plan(monkeypatch, samples, words=[_Word(0.0, 60.0)])

    span = plan.keyframes[-1][0] - plan.keyframes[0][0]
    assert plan.switches <= max(1, int(span / speaker.MIN_DWELL_S)), \
        f"{plan.switches} switches over {span:.0f}s breaks the {speaker.MIN_DWELL_S}s dwell"

    cut_times = [t for i, (t, x) in enumerate(plan.keyframes)
                 if i and x != plan.keyframes[i - 1][1]]
    for earlier, later in zip(cut_times, cut_times[1:]):
        assert later - earlier >= speaker.MIN_DWELL_S - 1e-6


def test_silence_does_not_move_the_camera(monkeypatch):
    """Lip-ish motion while nobody speaks — chewing, nodding — must not vote."""
    samples = ([[_face(500, lip=9.0), _face(1500, lip=0.1)] for _ in range(20)]
               + [[_face(500, lip=0.1), _face(1500, lip=9.0)] for _ in range(20)])
    # Speech covers only the first turn; the second is entirely silent.
    plan = _plan(monkeypatch, samples, words=[_Word(0.0, 9.0)])
    assert plan.switches == 0


def test_keyframes_stay_within_frame_and_under_the_cap(monkeypatch):
    samples = [[_face(300 + (i * 37) % 1400, lip=float(i % 5))] for i in range(400)]
    plan = _plan(monkeypatch, samples, words=[_Word(0.0, 400.0)])

    assert len(plan.keyframes) <= speaker.MAX_KEYFRAMES
    for _t, x in plan.keyframes:
        assert 0 <= x <= 1920 - plan.crop_w
    assert "if(" not in plan.expression()          # flat gated sum, not nested


def test_no_faces_falls_back_to_a_static_centre_crop(monkeypatch):
    plan = _plan(monkeypatch, [[] for _ in range(30)])
    assert plan.mode == "centred"
    assert plan.switches == 0


def test_narrow_source_is_left_alone(monkeypatch):
    plan = _plan(monkeypatch, [[_face(540)] for _ in range(10)], src_w=1080, src_h=1920)
    assert plan.mode == "full-frame"
    assert plan.switches == 0


def test_a_single_frame_of_motion_cannot_cut_the_camera(monkeypatch):
    """A cough or a laugh is one spike, not a turn. It must not move the frame."""
    samples = []
    for i in range(40):
        blip = 40.0 if i == 20 else 0.1        # one huge isolated spike
        samples.append([_face(500, lip=9.0), _face(1500, lip=blip)])
    plan = _plan(monkeypatch, samples, words=[_Word(0.0, 40.0)])
    assert plan.switches == 0, f"a single spike moved the camera: {plan.keyframes}"


# --- detector loading: concurrency and fallbacks --------------------------

def test_mediapipe_import_is_locked_against_the_render_pool(monkeypatch):
    """Clips are planned in parallel. Without a lock the thread that installs the
    matplotlib stub wins and the rest silently fall back to Haar — measured at 7
    of 8 clips on the first job of a fresh install."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import time

    calls = []

    def slow_import():
        calls.append(1)
        time.sleep(0.25)     # a real mediapipe import is far slower than this
        return object()

    monkeypatch.setattr(speaker, "_mp_face_detection", None)
    monkeypatch.setattr(speaker, "_import_mediapipe", slow_import)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: speaker._mediapipe() is not None, range(8)))

    assert len(calls) == 1, f"imported {len(calls)} times — the lock is not holding"
    assert all(results), "some threads were denied the detector and fell back"


def test_matplotlib_shim_is_reentrant_and_leaves_sys_modules_clean(monkeypatch):
    """find_spec raises ValueError once a stub with no __spec__ is installed, so
    a nested use must not try to install a second one."""
    import sys

    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    monkeypatch.setattr(speaker.importlib.util, "find_spec", lambda name: None)

    with speaker._without_matplotlib():
        assert "matplotlib" in sys.modules
        with speaker._without_matplotlib():        # must not raise
            assert "matplotlib" in sys.modules

    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules


def test_a_broken_detector_never_fails_a_job(monkeypatch):
    """Any import failure is a downgrade to Haar, never an exception."""
    def explode():
        raise RuntimeError("simulated broken install")

    monkeypatch.setattr(speaker, "_mp_face_detection", None)
    monkeypatch.setattr(speaker, "_import_mediapipe", explode)
    try:
        speaker._mediapipe()
    except Exception as exc:                       # noqa: BLE001
        raise AssertionError(f"detector failure escaped: {exc}") from exc


def test_haar_cascades_are_per_thread(monkeypatch):
    """cv2.CascadeClassifier carries detection state; sharing one across the
    render pool returned 5 different crop positions for identical input."""
    from concurrent.futures import ThreadPoolExecutor

    seen = []

    def grab(_):
        c = speaker._cascade("frontalface_alt2")
        seen.append(id(c))
        return id(c)

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(grab, range(4)))

    assert len(set(ids)) == len(ids), "threads shared a cascade instance"
    # ...but within one thread it is still cached rather than rebuilt each time.
    assert speaker._cascade("frontalface_alt2") is speaker._cascade("frontalface_alt2")
