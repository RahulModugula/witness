from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from .config import settings
from .db import session_scope
from .logging_config import get_logger
from .models import Task
from .schemas import ResultPayload
from .storage import result_path, write_json

log = get_logger(__name__)


def _set_status(
    task_id: str,
    status: str,
    *,
    error_message: str | None = None,
    result_file: Path | None = None,
    processing_time_seconds: float | None = None,
) -> None:
    with session_scope() as s:
        task = s.get(Task, task_id)
        if task is None:
            log.warning("set_status_missing_task task_id=%s", task_id)
            return
        task.status = status
        task.updated_at = datetime.utcnow()
        if error_message is not None:
            task.error_message = error_message
        if result_file is not None:
            task.result_path = str(result_file)
        if processing_time_seconds is not None:
            task.processing_time_seconds = processing_time_seconds


def _empty_payload(filename: str, processing_time: float) -> dict:
    """Used when the input is unreadable or has zero frames."""
    return {
        "videoMetadata": {
            "filename": filename,
            "duration_seconds": 0.0,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "codec": "unknown",
            "processing_time_seconds": processing_time,
            "model": settings.detector_model,
            "tracker": "botsort",
            "degraded_interaction": False,
        },
        "objectsDetected": [],
        "keyFrames": [],
    }


def _stub_pipeline_payload(input_path: str, processing_time: float) -> dict:
    """Minimal valid payload that bypasses ML. Used by API tests via STUB_PIPELINE=1."""
    from .pipeline.video import read_metadata

    meta = read_metadata(input_path)
    return {
        "videoMetadata": {
            "filename": meta.filename,
            "duration_seconds": meta.duration_seconds,
            "frame_count": meta.frame_count,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "codec": meta.codec,
            "processing_time_seconds": round(processing_time, 3),
            "model": "stub",
            "tracker": "stub",
            "degraded_interaction": False,
        },
        "objectsDetected": [],
        "keyFrames": [],
    }


def _build_payload(task_id: str, input_path: str, processing_time: float) -> dict:
    if settings.stub_pipeline:
        return _stub_pipeline_payload(input_path, processing_time)

    from .pipeline.assemble import assemble_result
    from .pipeline.detector import get_detector
    from .pipeline.pose import get_pose_extractor
    from .pipeline.video import read_metadata

    meta = read_metadata(input_path)

    t0 = time.time()
    detector = get_detector()
    detections = detector.track_video(input_path)
    t_detect = time.time() - t0

    t0 = time.time()
    pose = get_pose_extractor()
    hands_by_frame = pose.extract_video(input_path)
    t_pose = time.time() - t0

    t0 = time.time()
    payload = assemble_result(
        task_id=task_id,
        video_path=input_path,
        video_meta=meta,
        detections=detections,
        hands_by_frame=hands_by_frame,
        processing_time=processing_time,
    )
    ResultPayload.model_validate(payload)
    t_assemble = time.time() - t0

    log.info(
        "stage_timings task=%s detect=%.2fs pose=%.2fs assemble=%.2fs",
        task_id, t_detect, t_pose, t_assemble,
    )
    payload["videoMetadata"]["stage_timings"] = {
        "detect_track_seconds": round(t_detect, 3),
        "hand_pose_seconds": round(t_pose, 3),
        "assemble_keyframes_seconds": round(t_assemble, 3),
    }
    return payload


def process_video(task_id: str) -> None:
    """Background entry point. Runs the full pipeline."""
    start = time.time()
    try:
        _set_status(task_id, "PROCESSING")
        with session_scope() as s:
            task = s.get(Task, task_id)
            if task is None:
                log.error("process_video_missing_task task_id=%s", task_id)
                return
            input_path = task.input_path
            filename = task.original_filename

        log.info("process_video_start task_id=%s input=%s", task_id, input_path)

        payload = _build_payload(task_id, input_path, time.time() - start)
        payload["videoMetadata"]["processing_time_seconds"] = round(time.time() - start, 3)

        out = result_path(task_id)
        write_json(out, payload)

        _set_status(
            task_id,
            "DONE",
            result_file=out,
            processing_time_seconds=time.time() - start,
        )
        log.info(
            "process_video_done task_id=%s objects=%d elapsed=%.2fs",
            task_id,
            len(payload["objectsDetected"]),
            time.time() - start,
        )
    except Exception as exc:  # noqa: BLE001 — must not propagate from a background task
        log.exception("process_video_failed task_id=%s", task_id)
        _set_status(task_id, "FAILED", error_message=repr(exc))
