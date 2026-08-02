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
WHISPER_DEVICE = os.environ.get("CLIPPERBOY_WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("CLIPPERBOY_WHISPER_COMPUTE", "int8")

# Optional local LLM for smarter clip picking + titles. Never required.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("CLIPPERBOY_OLLAMA_MODEL", "llama3.2")
USE_OLLAMA = os.environ.get("CLIPPERBOY_USE_OLLAMA", "auto")  # auto | on | off

# Clip shape defaults
MIN_CLIP_SEC = float(os.environ.get("CLIPPERBOY_MIN_CLIP", 18))
MAX_CLIP_SEC = float(os.environ.get("CLIPPERBOY_MAX_CLIP", 75))
DEFAULT_CLIP_COUNT = int(os.environ.get("CLIPPERBOY_CLIP_COUNT", 8))

# Source compression: transcode oversized sources to speed the pipeline up.
COMPRESS_THRESHOLD_MB = float(os.environ.get("CLIPPERBOY_COMPRESS_MB", 800))

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
