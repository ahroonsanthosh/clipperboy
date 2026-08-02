#!/usr/bin/env bash
# Start Clipperboy without Docker. Creates a virtualenv on first run.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
STAMP=".venv/.installed"

if ! command -v python3 >/dev/null 2>&1; then
  echo "✗ python3 is not installed (Clipperboy needs Python 3.10-3.12)."
  echo "  macOS:          brew install python@3.12"
  echo "  Debian/Ubuntu:  sudo apt install python3 python3-venv"
  echo "  Or use Docker instead:  docker compose up --build"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "✗ ffmpeg is not installed."
  echo "  macOS:          brew install ffmpeg"
  echo "  Debian/Ubuntu:  sudo apt install ffmpeg"
  echo "  Windows:        winget install Gyan.FFmpeg"
  exit 1
fi

# Keep the downloaded speech model next to everything else this app owns.
export HF_HOME="${HF_HOME:-$PWD/data/models}"

# Re-install whenever requirements.txt changes, and never skip a failed install.
WANT="$( (command -v shasum >/dev/null && shasum requirements.txt || sha256sum requirements.txt) | cut -d' ' -f1)"
HAVE="$(cat "$STAMP" 2>/dev/null || true)"

if [ ! -d .venv ]; then
  echo "→ Creating virtual environment (first run only)…"
  python3 -m venv .venv
fi

if [ "$WANT" != "$HAVE" ]; then
  echo "→ Installing dependencies (a few minutes, first run only)…"
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
  echo "$WANT" > "$STAMP"
fi

echo ""
echo "  ✂️  Clipperboy is starting."
echo "  Open  →  http://localhost:${PORT}"
echo "  Stop  →  Ctrl+C"
echo ""

exec ./.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port "${PORT}"
