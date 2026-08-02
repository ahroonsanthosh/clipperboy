"""Burned-in captions with word-by-word highlighting (ASS subtitle format).

Words are grouped into short lines; one ASS event is emitted per word so the
spoken word is highlighted in the accent colour as it is said — the style that
performs on Reels/TikTok. Also emits a plain .srt alongside for accessibility.
"""
from __future__ import annotations

from pathlib import Path

from backend.pipeline.transcribe import Word

WORDS_PER_LINE = 4
MAX_LINE_CHARS = 26
GAP_BREAK = 0.7          # a pause longer than this starts a new caption line

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


def group_words(words: list[Word]) -> list[list[Word]]:
    """Chunk words into caption lines on length, char count and natural pauses."""
    lines: list[list[Word]] = []
    current: list[Word] = []
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
    font_size = max(28, int(height * 0.055))
    margin_v = int(height * 0.14)
    outline = max(2, int(font_size * 0.10))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{font_size},{COLOUR_BASE},{COLOUR_BASE},{COLOUR_OUTLINE},{COLOUR_OUTLINE},-1,0,0,0,100,100,0,0,1,{outline},0,2,{int(width * 0.06)},{int(width * 0.06)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for line in group_words(words):
        line = [w for w in line if w.end > clip_start and w.start < clip_end]
        if not line:
            continue
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
                    parts.append(f"{{\\c{COLOUR_ACTIVE}\\fscx108\\fscy108}}{text}{{\\c{COLOUR_BASE}\\fscx100\\fscy100}}")
                else:
                    parts.append(text)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{' '.join(parts)}"
            )

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path


def build_srt(words: list[Word], clip_start: float, clip_end: float, out_path: Path) -> Path:
    """Plain sidecar subtitles — useful for platforms that prefer real caption
    tracks over burned-in text, and for screen-reader-friendly review."""
    entries: list[str] = []
    index = 1
    for line in group_words(words):
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
