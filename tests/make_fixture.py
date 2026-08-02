"""Generate a synthetic 'podcast' fixture for end-to-end testing.

Uses espeak-ng for speech and ffmpeg for a talking-head-ish video, so the test
suite never needs a real copyrighted podcast checked into the repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = [
    "Welcome back to the show. Today we are talking about focus.",
    "Here's the thing nobody talks about: the biggest mistake people make is measuring hours instead of output.",
    "I'll never forget the first time I tried it. I was working fourteen hours a day and shipping absolutely nothing.",
    "Why does that happen? Because busy work feels incredible in the moment and it is completely useless.",
    "The truth is, I realised that three focused hours beat twelve distracted ones, every single time.",
    "This episode is brought to you by our sponsor. Use code PODCAST for twenty percent off.",
    "So the lesson is simple. Pick one thing, protect two hours, and let the rest of the day be messy.",
    "That is all we have time for. Thanks for listening and we will see you next week.",
]


def build(out_dir: Path, with_video: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "speech.wav"

    text = " ".join(SCRIPT)
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(wav), text],
        check=True, capture_output=True,
    )

    if not with_video:
        mp3 = out_dir / "fixture.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "96k", str(mp3)],
            check=True, capture_output=True,
        )
        return mp3

    out = out_dir / "fixture.mp4"
    # A 1280x720 "two-shot": moving colour blocks stand in for speakers, which
    # exercises the crop path without shipping a real person's likeness.
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25",
        "-i", str(wav),
        "-shortest",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ], check=True, capture_output=True)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
    print(build(target))
    print(build(target, with_video=False))
