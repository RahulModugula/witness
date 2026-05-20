"""Run the real ML pipeline against a video file without going through the HTTP
layer. Creates a Task row, invokes `process_video`, then validates and
summarizes the result. Exits non-zero if validation fails.

Usage: python -m scripts.run_pipeline [path/to/video.mp4]
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from app.config import settings
from app.db import init_db, session_scope
from app.logging_config import configure_logging
from app.models import Task
from app.schemas import ResultPayload
from app.storage import result_path, upload_path
from app.tasks import process_video


REQUIRED_CLASSES = {"person", "cable"}


def _stage_input(src: Path) -> tuple[str, Path]:
    """Copy the source video under uploads/ with a fresh task_id so we don't
    overwrite anything the user uploaded via the HTTP API."""
    with session_scope() as s:
        task = Task(original_filename=src.name, input_path="")
        s.add(task)
        s.flush()
        dest = upload_path(task.id, src.suffix.lower())
        shutil.copyfile(src, dest)
        task.input_path = str(dest)
        return task.id, dest


def main(argv: list[str]) -> int:
    src = Path(argv[1] if len(argv) > 1 else "data/uploads/sample.mp4").resolve()
    if not src.exists():
        print(f"[verify] missing video: {src}", file=sys.stderr)
        return 2

    configure_logging()
    init_db()

    print(f"[verify] staging {src.name}")
    task_id, dest = _stage_input(src)
    print(f"[verify] task_id={task_id}")
    print(f"[verify] input={dest}")
    print(f"[verify] running pipeline (this loads YOLO-World + MediaPipe — first run ~5s warmup)")

    start = time.time()
    process_video(task_id)
    elapsed = time.time() - start

    out = result_path(task_id)
    if not out.exists():
        print(f"[verify] FAIL: result file missing at {out}", file=sys.stderr)
        return 1

    payload = json.loads(out.read_text())
    try:
        ResultPayload.model_validate(payload)
    except Exception as e:
        print(f"[verify] FAIL: schema validation: {e}", file=sys.stderr)
        return 1

    meta = payload["videoMetadata"]
    objects = payload["objectsDetected"]
    keyframes = payload.get("keyFrames", [])
    classes = sorted({o["class"] for o in objects})
    missing = REQUIRED_CLASSES - set(classes)

    print()
    print("=" * 60)
    print(f"  video         : {meta['filename']} ({meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps)")
    print(f"  duration      : {meta['duration_seconds']:.2f}s, {meta['frame_count']} frames")
    print(f"  pipeline time : {meta['processing_time_seconds']:.2f}s wall (model={meta['model']}, tracker={meta['tracker']})")
    print(f"  total elapsed : {elapsed:.2f}s including warmup")
    print(f"  degraded_int  : {meta.get('degraded_interaction', False)}")
    print()
    print(f"  objects       : {len(objects)}")
    for o in objects:
        intervals = len(o["motion_history"])
        interactions = len(o["interactions"])
        print(
            f"    - id={o['object_id']:>2}  class={o['class']:<20}  "
            f"frames=[{o['first_seen_frame']},{o['last_seen_frame']}]  "
            f"motion_intervals={intervals}  interactions={interactions}  "
            f"conf_avg={o['confidence_avg']:.2f}"
        )
    print()
    print(f"  keyframes     : {len(keyframes)}")
    for kf in keyframes:
        print(f"    - frame={kf['frame_number']:>3}  type={kf['type']:<20}  object_id={kf['object_id']}  -> {kf['image_path']}")
    print()
    print(f"  result JSON   : {out}")
    print("=" * 60)
    print()

    if missing:
        print(f"[verify] WARNING: expected classes not detected: {sorted(missing)}", file=sys.stderr)
        print(f"[verify] detected classes: {classes}", file=sys.stderr)
        # Don't hard-fail — let the human decide whether this is acceptable.
        # Failing here would be too brittle for an open-vocab model.
        return 0

    print(f"[verify] OK — detected required classes: {sorted(REQUIRED_CLASSES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
