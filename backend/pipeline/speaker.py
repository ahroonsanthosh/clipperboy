"""Keep the speaker in frame when cropping wide video to vertical.

Samples frames across the clip, finds faces with OpenCV's bundled Haar cascade
(no model download, no network), picks the most likely active speaker, then
smooths the horizontal centre into a stable camera path. The result is a
piecewise ffmpeg crop expression.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SAMPLE_HZ = 1.0          # face lookups per second of clip
SMOOTH_ALPHA = 0.30      # exponential smoothing on the camera path
DEADZONE_FRAC = 0.04     # ignore jitter smaller than this fraction of width
MAX_STEP_FRAC = 0.05     # cap camera speed per sample to avoid whip pans

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
                gate = f"lt(t,{self.keyframes[1][0]:.2f})"
            elif i == len(self.keyframes) - 1:
                gate = f"gte(t,{t:.2f})"
            else:
                gate = f"gte(t,{t:.2f})*lt(t,{self.keyframes[i + 1][0]:.2f})"
            terms.append(f"{x}*{gate}")
        return "+".join(terms)


_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(str(path))
    return _cascade


def _detect_centre(frame: np.ndarray) -> float | None:
    """Return the x-centre (px) of the most prominent face, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6,
        minSize=(max(24, frame.shape[1] // 24), max(24, frame.shape[1] // 24)),
    )
    if len(faces) == 0:
        return None
    # Biggest face wins; on a two-shot that's usually whoever is closest to camera.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return float(x + w / 2)


def plan_crop(video: Path, start: float, end: float, src_w: int, src_h: int,
              target_ratio: float) -> CropPlan:
    """target_ratio = width / height of the output (e.g. 9/16)."""
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
    n_samples = max(2, int(duration * SAMPLE_HZ))
    times = np.linspace(0, duration, n_samples)

    cap = cv2.VideoCapture(str(video))
    centres: list[float | None] = []
    if cap.isOpened():
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + float(t)) * 1000.0)
            ok, frame = cap.read()
            centres.append(_detect_centre(frame) if ok and frame is not None else None)
        cap.release()
    else:
        centres = [None] * n_samples

    faces_found = sum(1 for c in centres if c is not None)
    if faces_found == 0:
        return CropPlan(crop_w, crop_h, [(0.0, centre_default)], 0, "centred")

    # Fill gaps by holding the last known position (then back-fill the head).
    filled: list[float] = []
    last = next(c for c in centres if c is not None)
    for c in centres:
        if c is not None:
            last = c
        filled.append(last)

    # Reject outliers: a detection far from the running median is usually a
    # false positive on a background object.
    median = float(np.median(filled))
    spread = max(src_w * 0.18, float(np.std(filled)) * 2.0)
    filled = [c if abs(c - median) <= spread else median for c in filled]

    keyframes: list[tuple[float, int]] = []
    cam = filled[0]
    deadzone = src_w * DEADZONE_FRAC
    max_step = src_w * MAX_STEP_FRAC
    prev_x: int | None = None

    for t, target in zip(times, filled):
        if abs(target - cam) > deadzone:
            step = (target - cam) * SMOOTH_ALPHA
            step = float(np.clip(step, -max_step, max_step))
            cam += step
        x = int(round(cam - crop_w / 2))
        x = int(np.clip(x, 0, max_x))
        if prev_x is None or abs(x - prev_x) >= 2:
            keyframes.append((float(t), x))
            prev_x = x

    if not keyframes:
        keyframes = [(0.0, centre_default)]
    if keyframes[0][0] > 0:
        keyframes.insert(0, (0.0, keyframes[0][1]))

    keyframes = _decimate(keyframes, MAX_KEYFRAMES)
    return CropPlan(crop_w, crop_h, keyframes, faces_found, "speaker-tracked")


def _decimate(keyframes: list[tuple[float, int]], limit: int) -> list[tuple[float, int]]:
    """Thin a camera path down to `limit` points, keeping the largest moves."""
    if len(keyframes) <= limit:
        return keyframes
    step = len(keyframes) / limit
    kept = [keyframes[min(len(keyframes) - 1, int(i * step))] for i in range(limit)]
    # Always keep the true start so the clip opens where the speaker actually is.
    kept[0] = keyframes[0]
    return kept
