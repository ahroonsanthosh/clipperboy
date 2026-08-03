"""Runtime configuration. Everything here has a working free default."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CLIPPERBOY_DATA", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
WORK_DIR = DATA_DIR / "work"
CLIP_DIR = DATA_DIR / "clips"

for _d in (UPLOAD_DIR, WORK_DIR, CLIP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Whisper. "base" is a good speed/accuracy tradeoff on CPU; override with env.
WHISPER_MODEL = os.environ.get("CLIPPERBOY_WHISPER_MODEL", "base")

# Transcription is the bulk of the wait, and model size is the biggest lever on
# it, so the choice belongs to the person waiting rather than buried in an env
# var. Roughly, relative to "balanced": fast ~3x quicker, accurate ~2.5x slower.
SPEED_PRESETS = {
    "fast": "tiny",
    "balanced": WHISPER_MODEL,
    "accurate": "small",
}
DEFAULT_SPEED = os.environ.get("CLIPPERBOY_SPEED", "balanced")

# Export quality. Measured on real footage against a lossless render of the same
# filter chain, so these compare like with like:
#
#   preset/crf        SSIM       size    encode   bitrate
#   superfast/20   0.99003      17 MB      9.6s   ~7 Mbps   (the old default)
#   faster/18      0.99045      12 MB     14.9s   ~5 Mbps
#   faster/16      0.99200      18 MB     15.6s   ~8 Mbps   <- default now
#   faster/14      0.99337      25 MB     15.7s  ~11 Mbps
#   faster/12      0.99464      36 MB     16.3s  ~15 Mbps
#
# Two things fall out of that. "superfast" wasted about half its bitrate: at the
# SAME file size, "faster" at crf 16 is measurably better. And on "faster",
# encode time is flat across crf — quality costs file size, not minutes — so
# there is no reason to be stingy. Slower presets were rejected: medium/16
# scored 0.99206 against faster/16's 0.99200 for 9 more seconds a clip, and
# slow/16 0.99218 for 27 more.
QUALITY_PRESETS = {
    "balanced": ("faster", 18),
    "high": ("faster", 16),
    "maximum": ("faster", 12),
}
DEFAULT_QUALITY = os.environ.get("CLIPPERBOY_QUALITY", "high")

# The uploaded episode is kept after a job so clips can be re-exported in another
# shape without re-uploading. That is a full copy of every podcast you process,
# so it can be turned off — re-export then needs a fresh upload.
KEEP_SOURCE = os.environ.get("CLIPPERBOY_KEEP_SOURCE", "1") != "0"
WHISPER_DEVICE = os.environ.get("CLIPPERBOY_WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("CLIPPERBOY_WHISPER_COMPUTE", "int8")

# Optional local LLM for smarter clip picking + titles. Never required.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("CLIPPERBOY_OLLAMA_MODEL", "llama3.2")
USE_OLLAMA = os.environ.get("CLIPPERBOY_USE_OLLAMA", "auto")  # auto | on | off

# Clip shape defaults
MIN_CLIP_SEC = float(os.environ.get("CLIPPERBOY_MIN_CLIP", 18))
MAX_CLIP_SEC = float(os.environ.get("CLIPPERBOY_MAX_CLIP", 60))
DEFAULT_CLIP_COUNT = int(os.environ.get("CLIPPERBOY_CLIP_COUNT", 8))

# Speech-to-text tuning. Batched inference is much faster and needs no extra
# setup; it falls back to the sequential path automatically if unavailable.
WHISPER_BATCHED = os.environ.get("CLIPPERBOY_WHISPER_BATCHED", "1") != "0"


def _total_ram_gb() -> float:
    """Physical RAM in GB, or 0.0 if it can't be determined."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
        if os.name == "nt":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _default_batch_size() -> int:
    """Batching is a speed/memory trade. A long episode on a modest laptop can
    be killed outright by the peak, and a job that dies at 35% is far worse than
    one that runs a little slower — so size the batch to the machine."""
    ram = _total_ram_gb()
    if ram and ram < 6:
        return 2
    if ram and ram < 12:
        return 4
    return 8


WHISPER_BATCH_SIZE = int(os.environ.get("CLIPPERBOY_WHISPER_BATCH", 0)) or _default_batch_size()
TOTAL_RAM_GB = round(_total_ram_gb(), 1)

# Captions use a bundled font so clips look identical on macOS, Windows and Linux.
FONTS_DIR = ROOT / "assets" / "fonts"
CAPTION_FONT = os.environ.get("CLIPPERBOY_CAPTION_FONT", "DejaVu Sans")

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

ASPECTS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}
DEFAULT_ASPECT = os.environ.get("CLIPPERBOY_ASPECT", "9:16")
