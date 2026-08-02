"""Burned-in captions with word-by-word highlighting (ASS subtitle format).

Words are grouped into short lines; one ASS event is emitted per word so the
spoken word is highlighted in the accent colour as it is said — the style that
performs on Reels/TikTok. Also emits a plain .srt alongside for accessibility.
"""
from __future__ import annotations

from pathlib import Path

from backend.pipeline.transcribe import Word

WORDS_PER_LINE = 4
MAX_LINE_CHARS = 20
GAP_BREAK = 0.7          # a pause longer than this starts a new caption line

# Social platforms overlay their own UI on top of the video. Keep captions
# inside the area that stays visible on TikTok, Reels and Shorts.
SIDE_MARGIN_FRAC = 0.10   # 10% each side clears the right-hand action rail
BOTTOM_MARGIN_FRAC = 0.22  # clears the caption/username chrome along the bottom
FONT_FRAC = 0.062         # of the *shorter* edge, so every aspect gets sane text

# ASS colours are &HAABBGGRR (alpha, blue, green, red).
COLOUR_BASE = "&H00FFFFFF"    # white
COLOUR_ACTIVE = "&H0022E0FF"  # amber
COLOUR_OUTLINE = "&HC8000000"


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


# A single token longer than this cannot fit a caption line at any sane size,
# so it is split across lines rather than sliced off by the frame edge.
HARD_BREAK_CHARS = 22
# Never shrink a line below this much of the normal size — past it, readability
# is worse than a hyphenated break.
MIN_LINE_SCALE = 62
# Rough advance width of DejaVu Sans Bold, as a fraction of the font size.
CHAR_WIDTH_FACTOR = 0.47


def _hard_break(word: Word) -> list[Word]:
    """Split an over-long token into hyphenated pieces, sharing its timing."""
    text = word.text
    if len(text) <= HARD_BREAK_CHARS:
        return [word]
    chunk = HARD_BREAK_CHARS - 1     # leave room for the hyphen
    pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    span = max(0.01, word.end - word.start) / len(pieces)
    out: list[Word] = []
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1
        out.append(Word(
            start=word.start + i * span,
            end=word.start + (i + 1) * span,
            text=piece if last else piece + "-",
        ))
    return out


def group_words(words: list[Word], split_long: bool = True) -> list[list[Word]]:
    """Chunk words into caption lines on length, char count and natural pauses.

    `split_long` hyphenates over-long tokens to fit the frame — wanted for
    burned-in captions, not for the .srt sidecar, which has no width limit.
    """
    lines: list[list[Word]] = []
    current: list[Word] = []
    if split_long:
        words = [piece for w in words for piece in _hard_break(w)]
    for w in words:
        if current:
            gap = w.start - current[-1].end
            projected = len(" ".join(x.text for x in current)) + 1 + len(w.text)
            if gap > GAP_BREAK or len(current) >= WORDS_PER_LINE or projected > MAX_LINE_CHARS:
                lines.append(current)
                current = []
        current.append(w)
        if w.text.endswith((".", "!", "?")):
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    return lines


def build_ass(words: list[Word], clip_start: float, clip_end: float,
              width: int, height: int, out_path: Path,
              font: str = "DejaVu Sans") -> Path:
    """Write an ASS file with clip-relative timings."""
    font_size = max(28, int(min(width, height) * FONT_FRAC))
    margin_v = int(height * BOTTOM_MARGIN_FRAC)
    margin_h = int(width * SIDE_MARGIN_FRAC)
    outline = max(2, int(font_size * 0.10))

    # WrapStyle 0 lets libass wrap anything that would otherwise overflow the
    # margins onto a second line — without it, long lines are sliced off at the
    # frame edge. group_words() keeps lines short enough that this rarely fires.
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{font_size},{COLOUR_BASE},{COLOUR_BASE},{COLOUR_OUTLINE},{COLOUR_OUTLINE},-1,0,0,0,100,100,0,0,1,{outline},0,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    available = width - 2 * margin_h

    events: list[str] = []
    for line in group_words(words):
        line = [w for w in line if w.end > clip_start and w.start < clip_end]
        if not line:
            continue

        # libass only wraps at spaces, so a long token would still be sliced by
        # the frame edge. Scale the whole line down until it fits the margins.
        text_len = len(" ".join(w.text for w in line))
        estimated = text_len * font_size * CHAR_WIDTH_FACTOR * 1.08  # active-word bump
        scale = 100 if estimated <= available else max(
            MIN_LINE_SCALE, int(100 * available / estimated))
        active_scale = int(scale * 1.08)

        for idx, active in enumerate(line):
            start = max(0.0, active.start - clip_start)
            end = max(start + 0.08, min(active.end, clip_end) - clip_start)
            # Hold the last word of a line on screen until the next line starts.
            if idx == len(line) - 1:
                end = max(end, min(active.end + 0.12, clip_end) - clip_start)
            parts = []
            for j, w in enumerate(line):
                text = _escape(w.text)
                if j == idx:
                    parts.append(
                        f"{{\\c{COLOUR_ACTIVE}\\fscx{active_scale}\\fscy{active_scale}}}"
                        f"{text}"
                        f"{{\\c{COLOUR_BASE}\\fscx{scale}\\fscy{scale}}}"
                    )
                else:
                    parts.append(text)
            body = f"{{\\fscx{scale}\\fscy{scale}}}" + " ".join(parts)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{body}"
            )

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path


def build_srt(words: list[Word], clip_start: float, clip_end: float, out_path: Path) -> Path:
    """Plain sidecar subtitles — useful for platforms that prefer real caption
    tracks over burned-in text, and for screen-reader-friendly review."""
    entries: list[str] = []
    index = 1
    for line in group_words(words, split_long=False):
        line = [w for w in line if w.end > clip_start and w.start < clip_end]
        if not line:
            continue
        start = max(0.0, line[0].start - clip_start)
        end = max(start + 0.3, min(line[-1].end, clip_end) - clip_start)
        text = " ".join(w.text for w in line)
        entries.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
        index += 1
    out_path.write_text("\n".join(entries), encoding="utf-8")
    return out_path
