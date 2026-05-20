"""Schema-level test: synthetic detections + hands → ResultPayload validates."""
from __future__ import annotations

from pathlib import Path

from app.pipeline.assemble import assemble_result
from app.pipeline.detector import FrameDetection
from app.pipeline.pose import HandLandmarks
from app.pipeline.video import VideoMeta
from app.schemas import ResultPayload


SAMPLE_VIDEO = Path(__file__).resolve().parent.parent / "data" / "uploads" / "sample.mp4"


def _hand_at(wrist, fingertips):
    pts = [wrist] * 21
    for idx, ft in zip((4, 8, 12, 16, 20), fingertips):
        pts[idx] = ft
    return HandLandmarks(points=pts)


def test_assembled_payload_passes_pydantic_schema(tmp_path):
    """End-to-end shape check using canned data — no model invocation."""
    meta = VideoMeta(
        filename="canned.mp4",
        duration_seconds=1.0,
        frame_count=10,
        fps=10.0,
        width=200,
        height=200,
        codec="h264",
    )
    # Track 7 = person (full duration); track 9 = cable, stationary first, moving later.
    detections: list[FrameDetection] = []
    for f in range(10):
        detections.append(
            FrameDetection(
                frame_idx=f, track_id=7, class_name="person",
                bbox=(0.0, 0.0, 200.0, 200.0), confidence=0.9,
            )
        )
        # Cable stays put for first 5, then drifts right.
        x_off = 0.0 if f < 5 else (f - 4) * 20.0
        detections.append(
            FrameDetection(
                frame_idx=f, track_id=9, class_name="cable",
                bbox=(40.0 + x_off, 80.0, 100.0 + x_off, 110.0), confidence=0.6,
            )
        )

    hands = {
        f: [_hand_at((50.0, 90.0), [(60.0, 95.0)] * 5)] for f in range(10)
    }

    payload = assemble_result(
        task_id="canned-task",
        video_path=str(SAMPLE_VIDEO) if SAMPLE_VIDEO.exists() else tmp_path / "noop.mp4",
        video_meta=meta,
        detections=detections,
        hands_by_frame=hands,
        processing_time=0.123,
    )

    # The schema is the contract — Pydantic validates aliasing + required fields.
    result = ResultPayload.model_validate(payload)
    assert result.videoMetadata.frame_count == 10
    cls_set = {o.class_ for o in result.objectsDetected}
    assert cls_set == {"person", "cable"}
    cable = next(o for o in result.objectsDetected if o.class_ == "cable")
    # Motion history must cover the full 10 frames with no gaps.
    covered = sum(iv.frame_range[1] - iv.frame_range[0] + 1 for iv in cable.motion_history)
    assert covered == 10
    assert cable.motion_history[0].frame_range[0] == 0
    assert cable.motion_history[-1].frame_range[1] == 9
    # The cable should have at least one interaction (we put a hand on it
    # for the full duration).
    assert len(cable.interactions) >= 1
