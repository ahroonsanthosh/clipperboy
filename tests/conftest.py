import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep test artefacts out of the user's real ./data directory. Must happen
# before backend.config is imported anywhere.
os.environ.setdefault("CLIPPERBOY_DATA", str(ROOT / "tests" / ".tmp-data"))
os.environ.setdefault("CLIPPERBOY_USE_OLLAMA", "off")

FIXTURE_DIR = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def video_fixture() -> Path:
    from tests.make_fixture import build

    path = FIXTURE_DIR / "fixture.mp4"
    if not path.exists():
        build(FIXTURE_DIR, with_video=True)
    return path


@pytest.fixture(scope="session")
def audio_fixture() -> Path:
    from tests.make_fixture import build

    path = FIXTURE_DIR / "fixture.mp3"
    if not path.exists():
        build(FIXTURE_DIR, with_video=False)
    return path


@pytest.fixture(scope="session")
def fake_transcript():
    """A transcript matching the fixture script, with word timings spread evenly.

    Lets the whole pipeline be tested without downloading a Whisper model.
    """
    from backend.pipeline.transcribe import Segment, Transcript, Word
    from tests.make_fixture import SCRIPT

    def build(duration: float) -> Transcript:
        total_words = sum(len(s.split()) for s in SCRIPT)
        per_word = duration / max(1, total_words)
        segments, t = [], 0.0
        for line in SCRIPT:
            words, start = [], t
            for token in line.split():
                words.append(Word(start=t, end=t + per_word * 0.9, text=token))
                t += per_word
            segments.append(Segment(start=start, end=t, text=line, words=words))
        return Transcript(segments=segments, language="en", duration=duration)

    return build
