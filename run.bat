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

REM Reinstall whenever either requirements file changes, matching run.sh.
for /f %%H in ('certutil -hashfile requirements.txt SHA256 ^| findstr /r "^[0-9a-f]"') do set REQ1=%%H
for /f %%H in ('certutil -hashfile requirements-detect.txt SHA256 ^| findstr /r "^[0-9a-f]"') do set REQ2=%%H
set WANT=%REQ1%%REQ2%
set HAVE=
if exist .venv\.installed-v2 set /p HAVE=<.venv\.installed-v2

if not "%WANT%"=="%HAVE%" (
  echo Installing dependencies ^(a few minutes, first run only^)...
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\pip install --quiet -r requirements.txt
  if errorlevel 1 exit /b 1

  REM The face detector installs separately and with --no-deps ON PURPOSE:
  REM mediapipe declares opencv-contrib-python, which would replace the
  REM headless OpenCV installed above with a GUI build. See
  REM requirements-detect.txt. It is also optional -- without it the app falls
  REM back to OpenCV's Haar cascades, so a failure here must not stop startup.
  .venv\Scripts\pip install --quiet --no-deps -r requirements-detect.txt
  if errorlevel 1 (
    echo Warning: could not install the face detector ^(mediapipe^).
    echo Clipperboy will still run, but vertical crops will be framed less
    echo accurately. Python 3.10-3.12 is needed for the better detector.
  )
  echo %WANT%> .venv\.installed-v2
)

echo.
echo   Clipperboy is starting.
echo   Open  ^-^>  http://localhost:%PORT%
echo   Stop  ^-^>  Ctrl+C
echo.

.venv\Scripts\uvicorn backend.app:app --host 127.0.0.1 --port %PORT%
