"""Synthesise a two-shot 'podcast' with faces the Haar cascade really detects.

The repo's main fixture is ffmpeg `testsrc2` and contains no faces at all, so it
cannot say anything about framing quality. This module draws crude but genuinely
detectable frontal faces (verified against the bundled cascade — see
`assert_detectable`) at known x positions and animates the mouth of whichever
subject is "speaking" on a known schedule. That gives ground truth: the camera
must sit at exactly one x per turn and cut once per real speaker change.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 1920, 1080, 25
FACE_SIZE = 300


@dataclass
class Turn:
    start: float
    end: float
    speaker: int      # index into the face x positions


def draw_face(canvas: np.ndarray, cx: int, cy: int, size: int,
              mouth_open: float) -> None:
    """Draw one frontal face. Geometry is tuned so the cascade fires on it."""
    r = size // 2
    cv2.ellipse(canvas, (cx, cy), (int(size * 0.32), int(size * 0.42)), 0, 0, 360,
                (190, 195, 200), -1)
    ey = cy - r + int(size * 0.42)
    for dx in (-1, 1):
        ex = cx + dx * int(size * 0.14)
        cv2.ellipse(canvas, (ex, ey), (int(size * 0.07), int(size * 0.045)), 0, 0, 360,
                    (40, 40, 45), -1)
        cv2.rectangle(canvas, (ex - int(size * 0.09), ey - int(size * 0.12)),
                      (ex + int(size * 0.09), ey - int(size * 0.09)), (35, 35, 40), -1)
    cv2.ellipse(canvas, (cx, cy - r + int(size * 0.55)),
                (int(size * 0.05), int(size * 0.09)), 0, 0, 360, (150, 155, 160), -1)
    mh = int(size * 0.02 + size * 0.09 * mouth_open)
    cv2.ellipse(canvas, (cx, cy - r + int(size * 0.70)),
                (int(size * 0.12), max(2, mh)), 0, 0, 360, (60, 45, 50), -1)


def render_frame(t: float, positions: list[int], speaking: int | None,
                 rng: np.random.Generator) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 30, np.uint8)
    for i, cx in enumerate(positions):
        # Everyone gets the same low-level noise, so a mouth-only signal is the
        # only thing that can distinguish the talker.
        open_amt = 0.0
        if i == speaking:
            open_amt = 0.5 + 0.5 * float(np.sin(t * 2 * np.pi * 4.0))
        draw_face(frame, cx, HEIGHT // 2, FACE_SIZE, open_amt)
    noise = rng.integers(0, 6, size=(HEIGHT, WIDTH, 1), dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def assert_detectable(positions: list[int]) -> None:
    """Fail loudly if the cascade cannot see these faces — the tests below are
    worthless otherwise, so this is checked rather than assumed."""
    from backend.pipeline import speaker

    frame = render_frame(0.0, positions, 0, np.random.default_rng(0))
    gray, scale = speaker._small_gray(frame)
    boxes = speaker._detect_boxes(gray)
    found = sorted(float(b[0] + b[2] / 2) * scale for b in boxes)
    assert len(found) == len(positions), (
        f"cascade found {len(found)} of {len(positions)} synthetic faces")
    for want, got in zip(sorted(positions), found):
        assert abs(want - got) < FACE_SIZE * 0.25, (want, got)


def build(out: Path, positions: list[int], turns: list[Turn],
          duration: float, seed: int = 7) -> Path:
    """Write an mp4 of `duration` seconds with the given speaking schedule."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(out),
    ], stdin=subprocess.PIPE)

    for n in range(int(duration * FPS)):
        t = n / FPS
        speaking = next((tn.speaker for tn in turns if tn.start <= t < tn.end), None)
        proc.stdin.write(render_frame(t, positions, speaking, rng).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed writing the synthetic face fixture")
    return out
