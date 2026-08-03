FROM python:3.11-slim

# ffmpeg does the editing; the DejaVu font renders the burned-in captions;
# libGL/libglib are OpenCV's runtime deps for face detection.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-detect.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The face detector is a SEPARATE, --no-deps install on purpose: mediapipe
# declares opencv-contrib-python, which would overwrite the headless OpenCV
# installed above with a GUI build that cannot run in this image. See
# requirements-detect.txt for the full story.
RUN pip install --no-cache-dir --no-deps -r requirements-detect.txt

COPY backend ./backend
COPY frontend ./frontend
COPY assets ./assets

# Fail the build — loudly, here — rather than silently shipping an image that
# falls back to the far less accurate Haar cascades on every clip.
RUN python -c "\
from backend.pipeline import speaker; \
assert speaker._mediapipe() is not None, 'BlazeFace did not load'; \
d = speaker._Detector(); \
assert d._blaze is not None, 'BlazeFace graph did not start'; \
d.close(); print('Face detection: MediaPipe BlazeFace ready')"

RUN mkdir -p /app/data && chmod 777 /app/data

ENV CLIPPERBOY_DATA=/app/data \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
