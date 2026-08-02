@echo off
REM Start Clipperboy on Windows. Creates a virtualenv on first run.
setlocal
cd /d "%~dp0"

if "%PORT%"=="" set PORT=8000

where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install Python 3.10-3.12 from https://python.org
  echo or use Docker instead:  docker compose up --build
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg is not installed. Run:  winget install Gyan.FFmpeg
  echo Then close and reopen this window so Windows picks it up.
  exit /b 1
)

set HF_HOME=%CD%\data\models

if not exist .venv (
  echo Creating virtual environment ^(first run only^)...
  python -m venv .venv
)

if not exist .venv\.installed (
  echo Installing dependencies ^(a few minutes, first run only^)...
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\pip install --quiet -r requirements.txt
  if errorlevel 1 exit /b 1
  echo done > .venv\.installed
)

echo.
echo   Clipperboy is starting.
echo   Open  ^-^>  http://localhost:%PORT%
echo   Stop  ^-^>  Ctrl+C
echo.

.venv\Scripts\uvicorn backend.app:app --host 127.0.0.1 --port %PORT%
