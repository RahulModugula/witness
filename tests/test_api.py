"""HTTP-layer tests using FastAPI's TestClient.

The `client` fixture sets STUB_PIPELINE=1 and RUN_INLINE=1 so each request
finishes synchronously without loading YOLO-World or MediaPipe — that keeps
the suite under a second while still exercising the real lifecycle.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

# Ensure the stub pipeline is on before any app code imports config.
os.environ.setdefault("STUB_PIPELINE", "1")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_upload_creates_task_and_runs_inline(client, sample_video: Path):
    with sample_video.open("rb") as f:
        r = client.post(
            "/tasks",
            files={"file": ("sample.mp4", f, "video/mp4")},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["task_id"]
    # RUN_INLINE=1 means process_video ran before the response came back.
    # The task should already be DONE by the next status call.
    task_id = body["task_id"]

    status = client.get(f"/tasks/{task_id}").json()
    assert status["status"] == "DONE"
    assert status["result_url"] == f"/tasks/{task_id}/result"


def test_unknown_task_returns_404(client):
    r = client.get("/tasks/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "TASK_NOT_FOUND"


def test_invalid_file_extension_rejected(client):
    fake = io.BytesIO(b"not a video")
    r = client.post(
        "/tasks",
        files={"file": ("malware.exe", fake, "application/octet-stream")},
    )
    assert r.status_code == 400
    assert r.json()["code"] == "INVALID_FILE"


def test_empty_upload_rejected(client):
    fake = io.BytesIO(b"")
    r = client.post("/tasks", files={"file": ("empty.mp4", fake, "video/mp4")})
    assert r.status_code == 400


def test_get_result_when_done_returns_schema(client, sample_video: Path):
    with sample_video.open("rb") as f:
        r = client.post("/tasks", files={"file": ("sample.mp4", f, "video/mp4")})
    task_id = r.json()["task_id"]

    rr = client.get(f"/tasks/{task_id}/result")
    assert rr.status_code == 200, rr.text
    payload = rr.json()
    assert "videoMetadata" in payload
    assert "objectsDetected" in payload
    assert payload["videoMetadata"]["frame_count"] > 0


def test_list_tasks_endpoint(client, sample_video: Path):
    # Create two tasks; list should return them newest-first.
    for _ in range(2):
        with sample_video.open("rb") as f:
            client.post("/tasks", files={"file": ("sample.mp4", f, "video/mp4")})
    r = client.get("/tasks?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    assert len(body["items"]) >= 2


def test_keyframe_path_traversal_blocked(client):
    r = client.get("/tasks/any/keyframes/..%2F..%2Fetc%2Fpasswd")
    # Either 400 (path containing /) or 404 (not found) — both are safe.
    assert r.status_code in (400, 404)
