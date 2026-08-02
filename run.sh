#!/usr/bin/env bash
# Start Clipperboy without Docker. Creates a virtualenv on first run.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8000}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "✗ ffmpeg is not installed."
  echo "  macOS:          brew install ffmpeg"
  echo "  Debian/Ubuntu:  sudo apt install ffmpeg"
  echo "  Windows:        winget install Gyan.FFmpeg"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "→ Creating virtual environment (first run only)…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  echo "→ Installing dependencies (a few minutes, first run only)…"
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

echo ""
echo "  ✂️  Clipperboy is starting."
echo "  Open  →  http://localhost:${PORT}"
echo "  Stop  →  Ctrl+C"
echo ""

exec ./.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port "${PORT}"
