"""Keep the speaker in frame when cropping wide video to vertical.

The camera is *locked*, not tracked. We sample the clip once, find every face
with OpenCV's bundled Haar cascade (no model download, no network), cluster the
face positions into a small set of stable "subject positions", then decide which
subject is active over time and emit a piecewise-CONSTANT camera path: the frame
sits perfectly still on one subject and hard-cuts to another, like a human editor
switching cameras. It never glides.

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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

SAMPLE_HZ = 2.0          # face lookups per second of clip

# --- locked-camera behaviour ---------------------------------------------
MIN_DWELL_S = 2.0        # camera may not move again within this long
SWITCH_EVIDENCE_S = 0.9  # candidate must lead for this long before we cut
SWITCH_MARGIN = 0.35     # ...and lead by this fraction more evidence
MIN_SWITCH_PX_FRAC = 0.04  # a "switch" closer than this to the current x is a no-op
SWITCH_EASE_S = 0.0      # 0.0 = hard cut. >0 emits a short ease (kept under 0.4s)
EASE_STEPS = 4

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

    @property
    def area(self) -> float:
        return self.w * self.h


_cascades: dict[str, cv2.CascadeClassifier] = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _cascades:
        _cascades[name] = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / f"haarcascade_{name}.xml"))
    return _cascades[name]


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


def _small_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale to DETECT_WIDTH and convert to gray. Returns (gray, scale)."""
    scale = 1.0
    if frame.shape[1] > DETECT_WIDTH:
        scale = frame.shape[1] / DETECT_WIDTH
        frame = cv2.resize(
            frame, (DETECT_WIDTH, int(round(frame.shape[0] / scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale


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
    """Finds people, cheaply, by learning which cascade suits this footage.

    A full ladder pass costs ~230ms a frame, which is far too slow to run on
    every sample. So the first few samples pay for the search, the winner is
    remembered, and later samples try it alone — 1.19 cascade calls per frame
    measured, 102ms. When the favourite misses, exactly one alternate is tried
    and then the frame is simply left empty: the camera is locked anyway, so a
    missed sample holds position rather than moving, which is the right answer.
    """

    def __init__(self) -> None:
        self.preferred: tuple[str, bool] | None = None
        self.samples_seen = 0

    def detect(self, gray: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns (boxes, is_face). Boxes are in small-frame pixels."""
        self.samples_seen += 1

        if self.preferred is not None and self.samples_seen > PROBE_SAMPLES:
            boxes = _run_cascade(*(self.preferred[0], gray, self.preferred[1]))
            if len(boxes):
                return boxes, True
            alternate = next(s for s in FACE_LADDER if s != self.preferred)
            boxes = _run_cascade(alternate[0], gray, alternate[1])
            return (boxes, True) if len(boxes) else (np.empty((0, 4), int), True)

        for name, flip in FACE_LADDER:
            boxes = _run_cascade(name, gray, flip)
            if len(boxes):
                self.preferred = (name, flip)
                return boxes, True
        for name, flip in BODY_LADDER:
            boxes = _run_cascade(name, gray, flip)
            if len(boxes):
                return boxes, False
        return np.empty((0, 4), int), True


def _detect_boxes(gray: np.ndarray) -> np.ndarray:
    """Single-shot detection for callers without their own detector state."""
    boxes, _is_face = _Detector().detect(gray)
    return boxes


def _detect_centre(frame: np.ndarray) -> float | None:
    """x-centre, in source pixels, of the most prominent face (or None).

    Retained for callers/tests that only want a single position; the planner
    itself keeps every face.
    """
    gray, scale = _small_gray(frame)
    boxes = _detect_boxes(gray)
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
            if row[j] is None or f.area > row[j].area:
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
                evidence[i, j] = f.lip

    # If lip motion carries no information at all, fall back to presence-weighted
    # prominence — which yields a single static shot rather than a moving one.
    if float(evidence.max()) <= 1e-6:
        prominence = np.zeros(len(subjects))
        for row in rows:
            for j, f in enumerate(row):
                if f is not None:
                    prominence[j] += f.area
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
        t = max(0.0, float(times[i]))
        if SWITCH_EASE_S > 0:
            # A deliberate short move rather than a cut, still finishing well
            # inside 0.4s and still piecewise-constant between steps.
            span = min(SWITCH_EASE_S, 0.4)
            for s in range(1, EASE_STEPS):
                frac = s / EASE_STEPS
                eased = 0.5 - 0.5 * np.cos(np.pi * frac)   # smoothstep
                keyframes.append((t + span * frac,
                                  int(round(current + (xs[i] - current) * eased))))
            t = t + span
        keyframes.append((t, xs[i]))
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

            gray, scale = _small_gray(frame)
            boxes, is_face = detector.detect(gray)

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
            for bx, by, bw, bh in boxes:
                # An upper-body box is torso-shaped: its centre sits well below
                # the head, and lip motion cannot be read from it. Take the
                # horizontal position, which is what framing needs, and put the
                # notional head in the upper third.
                cy = float(by + bh / 2) if is_face else float(by + bh * 0.28)
                faces.append(Face(
                    cx=float(bx + bw / 2) * scale, cy=cy * scale,
                    w=float(bw) * scale, h=float(bh) * scale,
                    lip=_lip_energy(gray, later, (bx, by, bw, bh)) if is_face else 0.0,
                ))
            samples.append(faces)
        index += 1
    cap.release()

    if not times:
        return [0.0], [[]]
    return times, samples
