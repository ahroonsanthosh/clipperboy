"""Local speech-to-text with faster-whisper. No API key, runs offline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from backend import config

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    segments: list[Segment]
    language: str = "en"
    duration: float = 0.0

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def text_between(self, start: float, end: float) -> str:
        parts = [s.text.strip() for s in self.segments if s.end > start and s.start < end]
        return " ".join(p for p in parts if p).strip()

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "start": s.start, "end": s.end, "text": s.text,
                    "words": [{"start": w.start, "end": w.end, "text": w.text} for w in s.words],
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segs = [
            Segment(
                start=s["start"], end=s["end"], text=s["text"],
                words=[Word(w["start"], w["end"], w["text"]) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return cls(segments=segs, language=d.get("language", "en"), duration=d.get("duration", 0.0))


class ModelUnavailable(RuntimeError):
    pass


def _load_model():
    key = (config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        device = config.WHISPER_DEVICE
        compute = config.WHISPER_COMPUTE
        if device == "auto":
            try:  # prefer GPU when the machine has one, fall back silently
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
            if device == "cuda" and compute == "int8":
                compute = "float16"
        try:
            # CTranslate2 defaults to 4 intra-op threads regardless of the
            # machine, which leaves most of a bigger CPU idle.
            threads = 0 if device == "cuda" else (os.cpu_count() or 4)
            _MODEL_CACHE[key] = WhisperModel(
                config.WHISPER_MODEL, device=device, compute_type=compute,
                cpu_threads=threads,
            )
        except Exception as exc:
            raise ModelUnavailable(
                f"Could not load the '{config.WHISPER_MODEL}' speech model ({exc}). "
                "The first run downloads it from huggingface.co — check your internet "
                "connection, or pre-download the model and point CLIPPERBOY_WHISPER_MODEL "
                "at the local folder."
            ) from exc
    return _MODEL_CACHE[key]


def transcribe(audio_path: Path, duration: float, progress=None) -> Transcript:
    """Word-level transcription. `progress(fraction, message)` is called as it goes."""
    model = _load_model()

    # Shared settings. temperature=0 and no prompt conditioning matter for speed:
    # the default temperature ladder re-decodes any segment that trips Whisper's
    # logprob/compression thresholds, at best_of=5 — so beam_size=1 does not
    # actually bound the work on difficult audio.
    common = dict(
        word_timestamps=True,          # the karaoke captions need word timings
        vad_filter=True,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    def collect(seg_iter, info) -> Transcript:
        total = duration or getattr(info, "duration", 0.0) or 1.0
        segments: list[Segment] = []
        for s in seg_iter:
            words = [
                Word(start=float(w.start), end=float(w.end), text=w.word.strip())
                for w in (s.words or [])
                if w.start is not None and w.end is not None and w.word.strip()
            ]
            text = s.text.strip()
            if not text:
                continue
            segments.append(Segment(start=float(s.start), end=float(s.end),
                                    text=text, words=words))
            if progress:
                progress(min(0.99, float(s.end) / total),
                         f"Transcribing… {_clock(s.end)} / {_clock(total)}")
        return Transcript(segments=segments,
                          language=getattr(info, "language", "en"), duration=total)

    if config.WHISPER_BATCHED:
        try:
            from faster_whisper import BatchedInferencePipeline

            # Note the whole consumption is inside the try: transcribe() returns
            # a generator, so an incompatibility surfaces during iteration, not
            # at the call. A failure here must never sink the job.
            return collect(*BatchedInferencePipeline(model=model).transcribe(
                str(audio_path), batch_size=config.WHISPER_BATCH_SIZE, **common
            ))
        except Exception:
            if progress:
                progress(0.0, "Transcribing (standard mode)…")

    return collect(*model.transcribe(
        str(audio_path),
        vad_parameters={"min_silence_duration_ms": 400},
        **common,
    ))


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
