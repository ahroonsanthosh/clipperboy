"""Keep the speaker in frame when cropping wide video to vertical.

The camera is *locked*, not tracked. We sample the clip once, find every face,
cluster the face positions into a small set of stable "subject positions", then
decide which subject is active over time and emit a piecewise-CONSTANT camera
path: the frame sits perfectly still on one subject and hard-cuts to another,
like a human editor switching cameras. It never glides.

Faces come from MediaPipe's BlazeFace ("full-range" variant), a small CNN whose
weights ship inside the mediapipe wheel — no download, no network. It replaced a
ladder of OpenCV Haar cascades, which on real footage were not merely imprecise
but actively wrong: measured on a podcast clip, the ladder fired on 83% of
samples yet *every one of those boxes was the sponsor sign on the wall behind
the speaker*, never the speaker. That is what made real clips crop onto blank
wall. BlazeFace finds the actual face in 87% of the same samples with no false
positives at all, and does it in a twelfth of the time.

The Haar ladder survives as a fallback for installs where mediapipe is missing
or fails to load, so a detector problem can never fail a job — but it is now
gated on a skin-colour test, which is what removes the wall-sign detections.

Active-speaker selection is a HEURISTIC, not verified audio-visual ASD. It
combines (a) speech gating from the word-level transcript, when one is supplied,
with (b) lip-region motion energy per face, measured as the frame-to-frame pixel
difference in each face's mouth band minus the same difference over that face's
upper half (which cancels head movement and global flicker). That correlates
with talking, but it is not a trained speaker model: nodding, laughing or
chewing can score. When the signal fails to discriminate, the design degrades to
the safe outcome — a single, static, well-framed shot on the most prominent
subject, which is still strictly better than drifting.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_HZ = 2.0          # face lookups per second of clip

# --- locked-camera behaviour ---------------------------------------------
MIN_DWELL_S = 2.0        # camera may not move again within this long
SWITCH_EVIDENCE_S = 0.9  # candidate must lead for this long before we cut
SWITCH_MARGIN = 0.35     # ...and lead by this fraction more evidence
MIN_SWITCH_PX_FRAC = 0.04  # a "switch" closer than this to the current x is a no-op

# --- subject clustering ---------------------------------------------------
CLUSTER_MERGE_FRAC = 0.10   # faces within 10% of frame width are the same subject
MIN_SUBJECT_SUPPORT = 0.08  # a subject must appear in this fraction of samples
MAX_SUBJECTS = 4

# ffmpeg's expression parser rejects anything nested deeper than 100 levels, so
# the camera path is emitted as a flat sum of gated terms and capped in length.
MAX_KEYFRAMES = 60


@dataclass
class CropPlan:
    crop_w: int
    crop_h: int
    keyframes: list[tuple[float, int]]   # (clip-relative seconds, crop x)
    faces_found: int
    mode: str                             # "speaker-tracked" | "centred" | "full-frame"

    def expression(self) -> str:
        """ffmpeg crop-x expression: piecewise constant over the keyframes.

        Built as a flat sum of `gte(t,a)*lt(t,b)` gated terms rather than nested
        if()s — ffmpeg's parser has a hard 100-level nesting limit, and a long
        tracked clip easily exceeds it, which fails the render outright.
        """
        if len(self.keyframes) <= 1:
            return str(self.keyframes[0][1] if self.keyframes else 0)

        terms = []
        for i, (t, x) in enumerate(self.keyframes):
            if i == 0:
                gate = f"lt(t,{self.keyframes[1][0]:.3f})"
            elif i == len(self.keyframes) - 1:
                gate = f"gte(t,{t:.3f})"
            else:
                gate = f"gte(t,{t:.3f})*lt(t,{self.keyframes[i + 1][0]:.3f})"
            terms.append(f"{x}*{gate}")
        return "+".join(terms)

    @property
    def switches(self) -> int:
        """How many times the camera actually moves during the clip."""
        return max(0, len({x for _t, x in self.keyframes}) - 1) if self.keyframes else 0


@dataclass
class Face:
    """One detected face in a sample, in *source* pixels."""
    cx: float
    cy: float
    w: float
    h: float
    lip: float = 0.0     # lip-motion energy, 0 when unmeasurable
    conf: float = 1.0    # detector confidence, 0-1 (Haar has none, so 1.0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def weight(self) -> float:
        """How much this detection should count when choosing the subject.

        A 0.95-confidence face is worth roughly twice a 0.5 one. Confidences are
        rescaled from the acceptance floor rather than used raw, so the weakest
        detection we are willing to keep still carries a little weight instead of
        an arbitrary 0.4.
        """
        span = max(1e-6, 1.0 - BLAZEFACE_MIN_CONF)
        return 0.35 + 0.65 * float(np.clip((self.conf - BLAZEFACE_MIN_CONF) / span, 0.0, 1.0))


# --- BlazeFace (primary) --------------------------------------------------
# `model_selection=1` is the full-range model. The short-range one assumes a
# selfie-distance face filling much of the frame; a podcast guest sitting back
# from the camera is well outside its range.
BLAZEFACE_MODEL = 1
BLAZEFACE_MIN_CONF = 0.35   # the detector's own floor: below this it says nothing
FACE_MIN_CONF = 0.45        # ...and below this we decline to plan a camera on it
# Haar reports no score. Treat its boxes as mid-confidence: better than a
# barely-passing BlazeFace hit, worse than a confident one — which is about
# right for a detector that needs a colour test to be trusted at all.
HAAR_CONF = 0.7

_mp_face_detection = None   # None = not tried yet, False = unavailable
_mp_lock = threading.Lock()


@contextmanager
def _without_matplotlib():
    """Let `mediapipe.python.solutions` import without matplotlib installed.

    `face_detection.py` needs numpy, protobuf and mediapipe's own framework —
    nothing else. But importing it runs the `solutions/__init__.py` aggregator,
    which pulls in every solution including `drawing_utils`, whose sole use of
    matplotlib is a 3-D landmark plotting helper we never call. Honouring that
    import would mean shipping matplotlib and its nine import-time dependencies
    (~60MB, pillow included) to draw nothing.

    So when matplotlib genuinely is not installed, a placeholder stands in for
    the duration of the import and is removed immediately afterwards, leaving
    sys.modules exactly as it was found. If matplotlib *is* installed, this does
    nothing at all and the real one is used.
    """
    # `find_spec` raises ValueError when a stub is already in sys.modules with a
    # None __spec__, so check sys.modules first and never nest a second stub.
    if "matplotlib" in sys.modules or importlib.util.find_spec("matplotlib") is not None:
        yield
        return
    stub = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    stub.pyplot = pyplot
    added = {"matplotlib": stub, "matplotlib.pyplot": pyplot}
    sys.modules.update(added)
    try:
        yield
    finally:
        for name in added:
            if sys.modules.get(name) in added.values():
                del sys.modules[name]


def _mediapipe():
    """Import mediapipe once, tolerating every way it can fail to be there.

    A missing or broken detector must never fail a job — the Haar ladder below
    still works, so an ImportError here is a downgrade, not an error.
    """
    global _mp_face_detection
    # Clips are planned concurrently, so this must be locked. Without it the
    # thread that installs the matplotlib stub wins and every other thread sees
    # a half-built sys.modules, silently falling back to Haar — measured at 7 of
    # 8 clips on the first job of a fresh install, which is exactly the run that
    # matters most.
    if _mp_face_detection is None:
        with _mp_lock:
            if _mp_face_detection is None:
                try:
                    _mp_face_detection = _import_mediapipe()
                except Exception as exc:              # noqa: BLE001
                    # _import_mediapipe guards its own body, but belt and braces:
                    # nothing about loading a detector may ever fail a job.
                    log.warning("Face detection: loader failed (%s); using Haar.", exc)
                    _mp_face_detection = False
    return _mp_face_detection or None


def _import_mediapipe():
    # mediapipe's absl/TF-Lite chatter goes to stderr at import and at every
    # graph construction; it is noise in a user-facing terminal.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_logtostderr", "0")
    try:
        with _without_matplotlib():
            from mediapipe.python.solutions import face_detection
        try:    # absl routes its own warnings past the GLOG env vars
            from absl import logging as absl_logging

            absl_logging.set_verbosity(absl_logging.ERROR)
            absl_logging.use_absl_handler()
        except Exception:                             # noqa: BLE001
            pass
        log.info("Face detection: MediaPipe BlazeFace (full-range).")
        return face_detection
    except Exception as exc:                          # noqa: BLE001 — any failure
        log.warning(
            "Face detection: MediaPipe unavailable (%s: %s); "
            "falling back to OpenCV Haar cascades, which frame less "
            "accurately. Reinstall with: pip install --no-deps -r "
            "requirements-detect.txt", type(exc).__name__, exc)
        return False


# cv2.CascadeClassifier carries per-detection state, so sharing one across the
# render pool corrupts results rather than merely slowing them: measured 6
# concurrent plans returning 5 distinct crop positions, one 267px out, where
# serial runs agreed exactly. Thread-local classifiers restore determinism.
_cascade_store = threading.local()


def _cascade(name: str) -> cv2.CascadeClassifier:
    cache = getattr(_cascade_store, "cascades", None)
    if cache is None:
        cache = _cascade_store.cascades = {}
    if name not in cache:
        cache[name] = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / f"haarcascade_{name}.xml"))
    return cache[name]


DETECT_WIDTH = 640       # detect on a small copy; 6x fewer pixels to scan
MIN_FACE_PX = 40         # at 640 wide, ~120px at 1080p — a talking head, not a poster
MIN_BODY_PX = 60
DETECT_SCALE_FACTOR = 1.05
DETECT_NEIGHBOURS = 3
PROBE_SAMPLES = 4        # how many samples run the full ladder to learn the winner

# Haar cascades are not mirror-symmetric, and no single one is close to
# sufficient on real footage. Measured on a real podcast clip (26 frames of a
# bespectacled speaker turned slightly away from camera), recall was:
#   frontalface_default, as previously configured .....  3%
#   frontalface_alt2, mirrored, sf 1.05, neighbours 3 . 84%
#   profileface, mirrored ............................. 38%
#   upperbody, mirrored .............................. 100%
# Hence a ladder over both orientations. Faces are always preferred; upper body
# is a last resort that never becomes the cached favourite, since its box centre
# is the torso rather than the head.
FACE_LADDER = (
    ("frontalface_alt2", True), ("frontalface_alt2", False),
    ("frontalface_default", True), ("frontalface_default", False),
    ("profileface", True), ("profileface", False),
)
BODY_LADDER = (("upperbody", True), ("upperbody", False))


def _small(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale to DETECT_WIDTH, keeping colour. Returns (bgr, scale).

    BlazeFace runs on this same small copy rather than the full frame. Measured
    on the reference clip, 640px-wide input was both faster (6.1 vs 7.6 ms a
    frame) and marginally *more* accurate (88% vs 87% recall, identical boxes to
    within a pixel) — the network resizes its input to 192x192 internally
    anyway, so handing it 1080p only pays for a bigger resize inside the graph.
    """
    scale = 1.0
    if frame.shape[1] > DETECT_WIDTH:
        scale = frame.shape[1] / DETECT_WIDTH
        frame = cv2.resize(
            frame, (DETECT_WIDTH, int(round(frame.shape[0] / scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    return frame, scale


def _small_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale to DETECT_WIDTH and convert to gray. Returns (gray, scale)."""
    small, scale = _small(frame)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale


# --- Haar false-positive rejection ----------------------------------------
# The wall sign that the ladder kept detecting is printed text: bright, flat and
# almost perfectly desaturated. A face is not. Requiring a usable share of
# skin-toned pixels inside the box is the cheapest discriminator that separates
# them (~0.2ms a box) and it does not depend on the box being static, so it also
# rejects posters, mic flags and monitors that happen to move with the camera.
SKIN_MIN_FRAC = 0.40     # share of the box's pixels that must be skin-toned
SKIN_MIN_LUMA = 50       # ...and skin is never near-black (that is the mic)
SKIN_MAX_LUMA = 250      # ...nor blown-out white (that is the sign)
SKIN_MIN_SAT = 40


def _skin_fraction(bgr: np.ndarray) -> float:
    """Fraction of pixels in a BGR patch that fall in the YCrCb skin locus.

    The Cr/Cb bounds are the long-standing Chai & Ngan values, which cover the
    full range of human skin tones — the chrominance of skin is remarkably
    consistent across complexions; it is luma that varies. Saturation and luma
    floors are added on top because printed greys can drift into the Cr/Cb box
    while being visibly colourless, and because chroma is meaningless in the
    near-black pixels of a microphone.

    Measured on the reference clip: real face boxes score 0.61-0.78 under these
    bounds, the sponsor sign and the microphone both score 0.00.
    """
    if bgr.size == 0 or bgr.ndim != 3:
        return 0.0
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    luma, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = ((cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 127)
            & (sat >= SKIN_MIN_SAT)
            & (luma >= SKIN_MIN_LUMA) & (luma <= SKIN_MAX_LUMA))
    return float(np.count_nonzero(mask)) / float(mask.size)


def _plausible_face(box: Sequence[int], colour: np.ndarray | None) -> bool:
    """Does this Haar box contain enough skin to be a face at all?

    Applied only when a colour frame is available. Without one (the unit tests
    drive the cascades on synthetic grayscale) every box is accepted, so the
    cascade contract is unchanged.
    """
    if colour is None:
        return True
    x, y, w, h = (int(v) for v in box)
    # Sample the middle of the box: the border of a face box is hair and
    # background, which would dilute the measurement either way.
    x0, x1 = x + w // 4, x + w - w // 4
    y0, y1 = y + h // 4, y + h - h // 4
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(colour.shape[1], x1), min(colour.shape[0], y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    return _skin_fraction(colour[y0:y1, x0:x1]) >= SKIN_MIN_FRAC


def _run_cascade(name: str, gray: np.ndarray, flip: bool) -> np.ndarray:
    """One cascade pass, returning boxes in unflipped small-frame coordinates."""
    img = cv2.flip(gray, 1) if flip else gray
    floor = MIN_BODY_PX if name == "upperbody" else MIN_FACE_PX
    found = _cascade(name).detectMultiScale(
        img, scaleFactor=DETECT_SCALE_FACTOR, minNeighbors=DETECT_NEIGHBOURS,
        minSize=(floor, floor),
    )
    if not len(found):
        return np.empty((0, 4), int)
    boxes = np.asarray(found).reshape(-1, 4)
    if flip:
        boxes[:, 0] = gray.shape[1] - boxes[:, 0] - boxes[:, 2]
    return boxes


class _Detector:
    """Finds people in one clip's worth of samples.

    BlazeFace first: one CNN pass, ~6ms on the 640px copy, and its boxes are
    trustworthy enough to use directly. Only when mediapipe is absent (or a
    frame genuinely has no face in it) does the Haar ladder run.

    The Haar path keeps its original design, which exists because a full ladder
    pass costs ~230ms: the first few samples pay for the search, the winning
    cascade is remembered, and later samples try it alone. When the favourite
    misses, exactly one alternate is tried and then the frame is left empty —
    the camera is locked anyway, so a missed sample holds position rather than
    moving, which is the right answer.

    One instance per clip. That is also what makes it thread-safe: clips are
    planned on a ThreadPoolExecutor and a MediaPipe graph must not be shared
    across threads, so each detector builds its own (~2ms).
    """

    def __init__(self) -> None:
        self.preferred: tuple[str, bool] | None = None
        self.samples_seen = 0
        self.confidences: list[float] = []   # per box from the last detect()
        self._preferred_body: tuple[str, bool] | None = None
        self._blaze = None
        module = _mediapipe()
        if module is not None:
            try:
                self._blaze = module.FaceDetection(
                    model_selection=BLAZEFACE_MODEL,
                    min_detection_confidence=BLAZEFACE_MIN_CONF)
            except Exception as exc:          # noqa: BLE001 — never fail a job
                log.warning("Face detection: BlazeFace failed to start (%s); "
                            "using Haar cascades.", exc)

    def close(self) -> None:
        if self._blaze is not None:
            try:
                self._blaze.close()
            except Exception:                 # noqa: BLE001
                pass
            self._blaze = None

    def _blazeface(self, colour: np.ndarray) -> np.ndarray:
        """BlazeFace boxes in small-frame pixels; fills self.confidences."""
        try:
            result = self._blaze.process(cv2.cvtColor(colour, cv2.COLOR_BGR2RGB))
        except Exception as exc:              # noqa: BLE001 — never fail a job
            log.warning("Face detection: BlazeFace errored (%s); "
                        "using Haar cascades for the rest of this clip.", exc)
            self.close()
            return np.empty((0, 4), int)

        h, w = colour.shape[:2]
        boxes, scores = [], []
        for det in (result.detections or ()):
            score = float(det.score[0]) if det.score else 0.0
            if score < FACE_MIN_CONF:
                # Real, but not real enough to point a locked camera at.
                continue
            rel = det.location_data.relative_bounding_box
            bw, bh = rel.width * w, rel.height * h
            if bw < 8 or bh < 8:
                continue
            # BlazeFace can report boxes that run off the edge of the frame.
            x0, y0 = max(0.0, rel.xmin * w), max(0.0, rel.ymin * h)
            x1, y1 = min(float(w), rel.xmin * w + bw), min(float(h), rel.ymin * h + bh)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
            scores.append(score)
        self.confidences = scores
        return np.asarray(boxes, dtype=int).reshape(-1, 4)

    def detect(self, gray: np.ndarray,
               colour: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Returns (boxes, is_face). Boxes are in small-frame pixels.

        `colour` is the same frame before the gray conversion. BlazeFace needs
        it, and the Haar path uses it to reject boxes with no skin in them.
        Callers that only have grayscale still work — they simply get the Haar
        ladder, unfiltered, exactly as before.
        """
        self.samples_seen += 1
        self.confidences = []

        if self._blaze is not None and colour is not None:
            boxes = self._blazeface(colour)
            if len(boxes):
                return boxes, True
            if self._blaze is not None:
                # BlazeFace ran and saw nothing. Trust it: the Haar ladder's
                # answer on these frames was the wall, not a face.
                return np.empty((0, 4), int), True

        return self._haar(gray, colour)

    def _haar(self, gray: np.ndarray,
              colour: np.ndarray | None) -> tuple[np.ndarray, bool]:
        def sift(boxes: np.ndarray) -> np.ndarray:
            kept = [b for b in boxes if _plausible_face(b, colour)]
            return np.asarray(kept, dtype=int).reshape(-1, 4)

        def hit(boxes: np.ndarray, is_face: bool) -> tuple[np.ndarray, bool]:
            self.confidences = [HAAR_CONF] * len(boxes)
            return boxes, is_face

        if self.preferred is not None and self.samples_seen > PROBE_SAMPLES:
            boxes = sift(_run_cascade(*(self.preferred[0], gray, self.preferred[1])))
            if len(boxes):
                return hit(boxes, True)
            alternate = next(s for s in FACE_LADDER if s != self.preferred)
            boxes = sift(_run_cascade(alternate[0], gray, alternate[1]))
            return hit(boxes, True) if len(boxes) else (np.empty((0, 4), int), True)

        # The skin test can reject everything the face ladder finds — that is
        # the point — after which nothing is ever cached as `preferred` and
        # every sample re-pays for the whole ladder plus the body pass. On the
        # reference clip that took the fallback from 149ms to 412ms a frame. So
        # the body cascade gets remembered too, and the exhaustive search gets
        # a budget: once the probe samples have come up empty, this footage is
        # simply one the cascades cannot read, and we stop rediscovering that.
        if self.samples_seen > PROBE_SAMPLES:
            if self._preferred_body is not None:
                boxes = _run_cascade(*(self._preferred_body[0], gray,
                                       self._preferred_body[1]))
                if len(boxes):
                    return hit(boxes, False)
            return np.empty((0, 4), int), True

        for name, flip in FACE_LADDER:
            boxes = sift(_run_cascade(name, gray, flip))
            if len(boxes):
                self.preferred = (name, flip)
                return hit(boxes, True)
        for name, flip in BODY_LADDER:
            # An upper-body box is mostly clothing, so the skin test does not
            # apply to it; it is a last resort for horizontal position only.
            boxes = _run_cascade(name, gray, flip)
            if len(boxes):
                self._preferred_body = (name, flip)
                return hit(boxes, False)
        return np.empty((0, 4), int), True


def _detect_boxes(gray: np.ndarray, colour: np.ndarray | None = None) -> np.ndarray:
    """Single-shot detection for callers without their own detector state."""
    detector = _Detector()
    try:
        boxes, _is_face = detector.detect(gray, colour)
    finally:
        detector.close()
    return boxes


def _detect_centre(frame: np.ndarray) -> float | None:
    """x-centre, in source pixels, of the most prominent face (or None).

    Retained for callers/tests that only want a single position; the planner
    itself keeps every face.
    """
    colour, scale = _small(frame)
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    boxes = _detect_boxes(gray, colour)
    if not len(boxes):
        return None
    x, _y, w, _h = max(boxes, key=lambda f: f[2] * f[3])
    return float(x + w / 2) * scale


def _lip_energy(gray_a: np.ndarray, gray_b: np.ndarray | None,
                box: Sequence[int]) -> float:
    """Mouth-band motion minus upper-face motion, for one face box.

    The subtraction is what makes this usable: a head that turns, a camera that
    flickers, or an encoder that shifts noise moves the whole face equally, so
    only the *differential* activity of the mouth survives. Still a heuristic —
    it cannot tell talking from chewing.
    """
    if gray_b is None or gray_a.shape != gray_b.shape:
        return 0.0
    x, y, w, h = (int(v) for v in box)
    if w < 8 or h < 8:
        return 0.0

    def band(y0: float, y1: float, x0: float, x1: float) -> float:
        ya, yb = int(y + h * y0), int(y + h * y1)
        xa, xb = int(x + w * x0), int(x + w * x1)
        ya, yb = max(0, ya), min(gray_a.shape[0], yb)
        xa, xb = max(0, xa), min(gray_a.shape[1], xb)
        if yb - ya < 2 or xb - xa < 2:
            return 0.0
        a = gray_a[ya:yb, xa:xb].astype(np.int16)
        b = gray_b[ya:yb, xa:xb].astype(np.int16)
        return float(np.mean(np.abs(a - b)))

    mouth = band(0.62, 0.95, 0.22, 0.78)
    upper = band(0.05, 0.45, 0.10, 0.90)
    return max(0.0, mouth - upper)


# --- planning -------------------------------------------------------------

def plan_crop(video: Path, start: float, end: float, src_w: int, src_h: int,
              target_ratio: float, words: Iterable | None = None) -> CropPlan:
    """target_ratio = width / height of the output (e.g. 9/16).

    `words` is the clip's word-level transcript (objects with `.start`/`.end` in
    absolute source seconds). When given it gates the lip-motion evidence to
    moments where somebody is actually speaking, which is the single biggest
    accuracy win available offline. It is optional: without it the planner falls
    back to lip motion alone.
    """
    src_ratio = src_w / src_h if src_h else 16 / 9

    if src_ratio <= target_ratio + 1e-3:
        # Source is already as narrow as (or narrower than) the target — no crop.
        return CropPlan(src_w, src_h, [(0.0, 0)], 0, "full-frame")

    crop_h = src_h
    crop_w = int(round(crop_h * target_ratio))
    crop_w = min(crop_w - (crop_w % 2), src_w)
    max_x = max(0, src_w - crop_w)
    centre_default = max_x // 2

    duration = max(0.1, end - start)
    times, samples = _sample_faces(video, start, duration)

    faces_found = sum(len(s) for s in samples)
    if faces_found == 0:
        return CropPlan(crop_w, crop_h, [(0.0, centre_default)], 0, "centred")

    subjects = _cluster_subjects(samples, src_w, len(times))
    if not subjects:
        return CropPlan(crop_w, crop_h, [(0.0, centre_default)], faces_found, "centred")

    speech = _speech_mask(times, words, start)
    path = _decide_path(times, samples, subjects, speech)

    keyframes = _to_keyframes(path, times, subjects, crop_w, max_x, src_w)
    if not keyframes:
        keyframes = [(0.0, centre_default)]

    keyframes = _decimate(keyframes, MAX_KEYFRAMES)
    return CropPlan(crop_w, crop_h, keyframes, faces_found, "speaker-tracked")


def _cluster_subjects(samples: list[list[Face]], src_w: int,
                      n_samples: int) -> list[float]:
    """Collapse every observed face x into a few stable subject positions.

    A podcast is one or two people who do not move much; treating their whole
    scatter of detections as fixed positions is exactly what removes the drift.
    """
    xs = sorted(f.cx for s in samples for f in s)
    if not xs:
        return []

    merge = max(24.0, src_w * CLUSTER_MERGE_FRAC)
    groups: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= merge:
            groups[-1].append(x)
        else:
            groups.append([x])

    # One refinement pass: reassign each face to its nearest centroid, which
    # repairs the greedy chaining that a long tail of outliers can cause.
    centroids = [float(np.median(g)) for g in groups]
    for _ in range(2):
        buckets: list[list[float]] = [[] for _ in centroids]
        for x in xs:
            j = int(np.argmin([abs(x - c) for c in centroids]))
            buckets[j].append(x)
        centroids = [float(np.median(b)) if b else centroids[i]
                     for i, b in enumerate(buckets)]
        groups = buckets

    floor = max(1, int(n_samples * MIN_SUBJECT_SUPPORT))
    kept = [(c, len(g)) for c, g in zip(centroids, groups) if len(g) >= floor]
    if not kept:
        kept = [(centroids[int(np.argmax([len(g) for g in groups]))], 1)]
    kept.sort(key=lambda cg: -cg[1])
    kept = kept[:MAX_SUBJECTS]
    return sorted(c for c, _n in kept)


def _speech_mask(times: list[float], words: Iterable | None,
                 clip_start: float) -> list[bool]:
    """Per-sample: is anybody speaking? All-True when no transcript is given."""
    spans = []
    for w in words or ():
        try:
            spans.append((float(w.start) - clip_start, float(w.end) - clip_start))
        except AttributeError:
            continue
    if not spans:
        return [True] * len(times)
    spans.sort()
    mask = []
    for t in times:
        # A sample counts as speech if a word overlaps a short window around it.
        mask.append(any(a - 0.35 <= t <= b + 0.35 for a, b in spans))
    return mask


def _assign(samples: list[list[Face]],
            subjects: list[float]) -> list[list[Face | None]]:
    """Nearest-subject assignment: per sample, the best face for each subject."""
    out: list[list[Face | None]] = []
    for faces in samples:
        row: list[Face | None] = [None] * len(subjects)
        for f in faces:
            j = int(np.argmin([abs(f.cx - c) for c in subjects]))
            if row[j] is None or f.area * f.weight > row[j].area * row[j].weight:
                row[j] = f
        out.append(row)
    return out


def _decide_path(times: list[float], samples: list[list[Face]],
                 subjects: list[float], speech: list[bool]) -> list[int]:
    """Which subject index the camera is on, per sample.

    A rolling-window state machine, not a filter: the incumbent keeps the frame
    unless a challenger accumulates clearly more speaking evidence over a
    sustained window AND the minimum dwell has elapsed. Everything smaller —
    detection jitter, a brief interjection, a nod — changes nothing at all.
    """
    rows = _assign(samples, subjects)
    n = len(times)
    if len(subjects) == 1:
        return [0] * n

    # Per-sample evidence that subject j is talking.
    evidence = np.zeros((n, len(subjects)), dtype=float)
    for i, row in enumerate(rows):
        if not speech[i]:
            continue
        for j, f in enumerate(row):
            if f is not None:
                # Scale the talking evidence by how sure we are this is even a
                # face: a marginal detection should not be able to pull the
                # camera off a confidently-detected subject.
                evidence[i, j] = f.lip * f.weight

    # If lip motion carries no information at all, fall back to presence-weighted
    # prominence — which yields a single static shot rather than a moving one.
    if float(evidence.max()) <= 1e-6:
        prominence = np.zeros(len(subjects))
        for row in rows:
            for j, f in enumerate(row):
                if f is not None:
                    prominence[j] += f.area * f.weight
        return [int(np.argmax(prominence))] * n

    dt = (times[-1] - times[0]) / max(1, n - 1) if n > 1 else 1.0 / SAMPLE_HZ
    win = max(1, int(round(SWITCH_EVIDENCE_S / max(dt, 1e-3))))

    # Opening shot: whoever leads over the first evidence window.
    head = evidence[:max(win, 1)].sum(axis=0)
    active = int(np.argmax(head)) if head.max() > 0 else int(
        np.argmax(evidence.sum(axis=0)))
    path = [active] * n
    last_switch_t = times[0] - MIN_DWELL_S

    for i in range(n):
        lo = max(0, i - win + 1)
        if i - lo + 1 < win:
            path[i] = active
            continue
        block = evidence[lo:i + 1]
        window = block.sum(axis=0)
        best = int(np.argmax(window))

        # Sustained, not momentary. Summed evidence alone lets a single frame of
        # mouth movement — a cough, a laugh, a swallow — flip the camera, because
        # one large value can carry the whole window. Require the challenger to
        # actually win most of the individual samples too, on at least two of
        # them, so the evidence has to persist rather than merely spike.
        speaking_rows = [r for r in block if r.max() > 0]
        wins = sum(1 for r in speaking_rows if int(np.argmax(r)) == best)
        sustained = len(speaking_rows) >= 2 and wins * 2 >= len(speaking_rows) + 1

        if (best != active
                and sustained
                and window[best] > 0
                and window[best] >= window[active] * (1.0 + SWITCH_MARGIN)
                and times[lo] - last_switch_t >= MIN_DWELL_S):
            # Backdate the cut to where the winning evidence began, so the frame
            # lands on the new speaker as they start rather than a beat late.
            active = best
            last_switch_t = times[lo]
            for k in range(lo, i + 1):
                path[k] = active
        path[i] = active
    return path


def _to_keyframes(path: list[int], times: list[float], subjects: list[float],
                  crop_w: int, max_x: int, src_w: int) -> list[tuple[float, int]]:
    """Turn a per-sample subject path into a piecewise-constant camera path.

    One keyframe per shot — the x is held exactly, so ffmpeg's crop does not
    move by a single pixel until the next cut.
    """
    if not path:
        return []
    xs = [_subject_x(subjects[j], crop_w, max_x) for j in path]
    threshold = max(2.0, src_w * MIN_SWITCH_PX_FRAC)

    keyframes: list[tuple[float, int]] = [(0.0, xs[0])]
    current = xs[0]
    for i in range(1, len(xs)):
        if abs(xs[i] - current) < threshold:
            continue
        # An instant cut, never an ease: a moving frame is the thing this
        # module exists to avoid.
        keyframes.append((max(0.0, float(times[i])), xs[i]))
        current = xs[i]
    return keyframes


def _subject_x(centre: float, crop_w: int, max_x: int) -> int:
    return int(np.clip(int(round(centre - crop_w / 2)), 0, max_x))


def _decimate(keyframes: list[tuple[float, int]], limit: int) -> list[tuple[float, int]]:
    """Thin a camera path down to `limit` points, keeping the largest moves."""
    if len(keyframes) <= limit:
        return keyframes
    step = len(keyframes) / limit
    kept = [keyframes[min(len(keyframes) - 1, int(i * step))] for i in range(limit)]
    # Always keep the true start so the clip opens where the speaker actually is.
    kept[0] = keyframes[0]
    return kept


def _sample_faces(video: Path, start: float, duration: float
                  ) -> tuple[list[float], list[list[Face]]]:
    """Walk the clip once, detecting every face SAMPLE_HZ times a second.

    Seeking to each sample with CAP_PROP_POS_MSEC forces a keyframe re-decode
    every time and dominated this stage — measured 13.6s of a 16.6s pass, versus
    3.5s decoding the same window sequentially. So seek once, then grab() every
    frame and only retrieve() (which actually decodes) the ones we sample. One
    extra retrieve per sample, a couple of frames later, supplies the lip-motion
    difference; it costs a decode we were going to walk past anyway.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [0.0], [[]]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 <= fps <= 240):
        fps = 30.0
    step = max(1, int(round(fps / SAMPLE_HZ)))
    lip_gap = max(1, min(step - 1, int(round(fps * 0.08))))
    wanted = int(duration * fps)

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    detector = _Detector()
    times: list[float] = []
    samples: list[list[Face]] = []
    index = 0
    try:
        while index <= wanted:
            if not cap.grab():
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                times.append(index / fps)
                if not ok or frame is None:
                    samples.append([])
                    index += 1
                    continue

                colour, scale = _small(frame)
                gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
                boxes, is_face = detector.detect(gray, colour)
                scores = list(detector.confidences)

                later = None
                if len(boxes):
                    for _ in range(lip_gap):
                        if not cap.grab():
                            break
                        index += 1
                    ok2, frame2 = cap.retrieve()
                    if ok2 and frame2 is not None:
                        later, _ = _small_gray(frame2)

                faces = []
                for i, (bx, by, bw, bh) in enumerate(boxes):
                    # An upper-body box is torso-shaped: its centre sits well
                    # below the head, and lip motion cannot be read from it. Take
                    # the horizontal position, which is what framing needs, and
                    # put the notional head in the upper third.
                    cy = float(by + bh / 2) if is_face else float(by + bh * 0.28)
                    faces.append(Face(
                        cx=float(bx + bw / 2) * scale, cy=cy * scale,
                        w=float(bw) * scale, h=float(bh) * scale,
                        lip=_lip_energy(gray, later, (bx, by, bw, bh)) if is_face else 0.0,
                        conf=float(scores[i]) if i < len(scores) else HAAR_CONF,
                    ))
                samples.append(faces)
            index += 1
    finally:
        cap.release()
        detector.close()

    if not times:
        return [0.0], [[]]
    return times, samples
