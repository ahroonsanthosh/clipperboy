"""End-to-end tests through the HTTP API, with the speech model stubbed out."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import jobs
from backend.app import app


@pytest.fixture
def client(monkeypatch, fake_transcript):
    """A client whose jobs run synchronously and never touch a real model."""
    def fake_transcribe(audio_path, duration, progress=None):
        if progress:
            progress(0.5, "Transcribing… test")
        return fake_transcript(duration)

    monkeypatch.setattr(jobs, "transcribe", fake_transcribe)
    monkeypatch.setattr(jobs, "start", lambda job, path, mode: jobs.process(job.id, path, mode))
    monkeypatch.setattr(jobs.config, "MIN_CLIP_SEC", 8)
    monkeypatch.setattr(jobs.config, "MAX_CLIP_SEC", 30)
    monkeypatch.setattr(jobs.score.config, "MIN_CLIP_SEC", 8)
    monkeypatch.setattr(jobs.score.config, "MAX_CLIP_SEC", 30)

    with TestClient(app) as c:
        yield c


def _upload(client, path: Path, **form):
    payload = {"aspect": "9:16", "clip_count": "2", "compress": "never", **form}
    with path.open("rb") as fh:
        return client.post("/api/jobs", files={"file": (path.name, fh, "video/mp4")}, data=payload)


# --- basics ---------------------------------------------------------------

def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["ffmpeg"] is True
    assert body["default_aspect"] == "9:16"


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Clipperboy" in res.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


def test_rejects_unsupported_file_type(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    res = _upload(client, bad)
    assert res.status_code == 400
    assert "isn't supported" in res.json()["detail"]


def test_rejects_bad_aspect(client, video_fixture):
    assert _upload(client, video_fixture, aspect="4:3").status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


# --- full run -------------------------------------------------------------

@pytest.fixture
def finished_job(client, video_fixture):
    res = _upload(client, video_fixture)
    assert res.status_code == 201, res.text
    job_id = res.json()["job"]["id"]
    job = client.get(f"/api/jobs/{job_id}").json()["job"]
    assert job["status"] == "done", job.get("error")
    return client, job


def test_job_completes_with_clips(finished_job):
    _, job = finished_job
    assert job["progress"] == 1.0
    assert job["stage"] == "done"
    assert 1 <= len(job["clips"]) <= 2
    assert job["transcript_available"] is True

    for clip in job["clips"]:
        assert clip["status"] == "ready", clip.get("error")
        assert clip["title"] and clip["summary"] and clip["transcript"]
        assert clip["reasons"] and clip["keywords"]
        assert clip["end"] > clip["start"]
        assert clip["aspect"] == "9:16"
        assert clip["crop_mode"] in {"speaker-tracked", "centred", "full-frame"}


def test_clip_assets_download(finished_job):
    client, job = finished_job
    clip = job["clips"][0]

    video = client.get(f"/api/jobs/{job['id']}/clips/{clip['id']}/video?download=true")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert "attachment" in video.headers["content-disposition"]
    assert len(video.content) > 10_000

    assert client.get(f"/api/jobs/{job['id']}/clips/{clip['id']}/thumbnail").status_code == 200
    srt = client.get(f"/api/jobs/{job['id']}/clips/{clip['id']}/subtitles")
    assert srt.status_code == 200 and "-->" in srt.text


def test_inline_video_is_not_forced_download(finished_job):
    client, job = finished_job
    clip = job["clips"][0]
    res = client.get(f"/api/jobs/{job['id']}/clips/{clip['id']}/video")
    assert "attachment" not in res.headers.get("content-disposition", "")


def test_transcript_endpoint(finished_job):
    client, job = finished_job
    res = client.get(f"/api/jobs/{job['id']}/transcript")
    assert res.status_code == 200
    assert res.json()["segments"]


def test_download_all_zip_contains_clips_and_notes(finished_job):
    client, job = finished_job
    res = client.get(f"/api/jobs/{job['id']}/download-all")
    assert res.status_code == 200

    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = zf.namelist()
        assert "clip-notes.txt" in names
        assert any(n.endswith(".mp4") for n in names)
        assert any(n.endswith(".srt") for n in names)
        notes = zf.read("clip-notes.txt").decode()
        assert "Why it works" in notes


def test_reexport_changes_aspect_ratio(finished_job):
    from backend import config
    from backend.pipeline import media

    client, job = finished_job
    clip = job["clips"][0]
    res = client.post(
        f"/api/jobs/{job['id']}/clips/{clip['id']}/render",
        json={"aspect": "16:9", "captions": True},
    )
    assert res.status_code == 200, res.text
    assert res.json()["clip"]["aspect"] == "16:9"

    rendered = config.CLIP_DIR / job["id"] / f"clip-{clip['id']}.mp4"
    info = media.probe(rendered)
    assert (info.width, info.height) == (1920, 1080)


def test_delete_removes_job_and_files(client, video_fixture):
    from backend import config

    job_id = _upload(client, video_fixture).json()["job"]["id"]
    assert (config.CLIP_DIR / job_id).exists()

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (config.CLIP_DIR / job_id).exists()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_audio_only_episode_produces_video_clips(client, audio_fixture):
    res = _upload(client, audio_fixture, aspect="9:16")
    assert res.status_code == 201, res.text
    job = client.get(f"/api/jobs/{res.json()['job']['id']}").json()["job"]
    assert job["status"] == "done", job.get("error")
    assert job["has_video"] is False
    assert job["clips"]
    assert all(c["status"] == "ready" for c in job["clips"])
    assert all(c["crop_mode"] == "audio-waveform" for c in job["clips"])


def test_job_failure_is_reported_not_crashed(client, tmp_path, monkeypatch):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00" * 4096)
    res = _upload(client, broken)
    assert res.status_code == 201
    job = client.get(f"/api/jobs/{res.json()['job']['id']}").json()["job"]
    assert job["status"] == "failed"
    assert job["error"]


def test_jobs_list_includes_created_jobs(finished_job):
    client, job = finished_job
    ids = [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert job["id"] in ids
