from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_config import get_logger
from ..storage import keyframes_dir

log = get_logger(__name__)

Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class KeyframeSpec:
    object_id: int
    frame_number: int
    type: str  # "motion_transition" | "interaction_peak"
    # Annotation overlay fields (optional — frame is still saved without them).
    bbox: Bbox | None = None
    label: str | None = None
    fingertips: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def filename(self) -> str:
        kind = "motion" if self.type == "motion_transition" else "interaction"
        return f"obj{self.object_id}_{kind}_{self.frame_number}.jpg"


def _bbox_at_or_near(
    object_frames: list[tuple[int, Bbox]], frame: int
) -> Bbox | None:
    """Find the object's bbox at `frame` (exact match), or fall back to the
    nearest preceding frame's bbox. Object tracks aren't dense — they may skip
    frames the keyframe happens to land on."""
    if not object_frames:
        return None
    frames_only = [f for f, _ in object_frames]
    i = bisect_left(frames_only, frame)
    if i < len(object_frames) and object_frames[i][0] == frame:
        return object_frames[i][1]
    if i > 0:
        return object_frames[i - 1][1]
    return object_frames[0][1]


def select_motion_transition_frames(
    object_id: int,
    motion_history: list[dict],
    *,
    label: str,
    object_frames: list[tuple[int, Bbox]],
) -> list[KeyframeSpec]:
    """Save the first frame of each new state interval (except the very first
    interval, which is the start of the video). One spec per transition."""
    out: list[KeyframeSpec] = []
    for prior, curr in zip(motion_history, motion_history[1:]):
        f = curr["frame_range"][0]
        out.append(
            KeyframeSpec(
                object_id=object_id,
                frame_number=f,
                type="motion_transition",
                bbox=_bbox_at_or_near(object_frames, f),
                label=f"{label} (id={object_id}) -> {curr['state']}",
            )
        )
    return out


def select_interaction_peak_frames(
    object_id: int,
    interactions: list[dict],
    fingertip_peaks: dict[int, int],
    *,
    label: str,
    object_frames: list[tuple[int, Bbox]],
) -> list[KeyframeSpec]:
    """For each interaction interval, pick the frame inside the interval with
    the highest fingertip-count value (the "peak" interaction moment)."""
    out: list[KeyframeSpec] = []
    for iv in interactions:
        s, e = iv["frame_start"], iv["frame_end"]
        candidates = {f: fingertip_peaks[f] for f in fingertip_peaks if s <= f <= e}
        if not candidates:
            peak_frame = (s + e) // 2
        else:
            peak_frame = max(candidates, key=lambda f: candidates[f])
        pid = iv.get("interacted_by_person", "?")
        out.append(
            KeyframeSpec(
                object_id=object_id,
                frame_number=peak_frame,
                type="interaction_peak",
                bbox=_bbox_at_or_near(object_frames, peak_frame),
                label=f"{label} (id={object_id}) <-> person id={pid}",
            )
        )
    return out


# Distinct colors per keyframe type (BGR for OpenCV).
_COLOR_MOTION = (0, 200, 255)        # amber
_COLOR_INTERACTION = (60, 220, 80)   # green


def _draw_annotation(frame, spec: KeyframeSpec) -> None:
    """Mutate `frame` in-place: draw the bbox + label for this spec."""
    import cv2

    if spec.bbox is None:
        return
    color = _COLOR_INTERACTION if spec.type == "interaction_peak" else _COLOR_MOTION
    h, w = frame.shape[:2]
    x1 = max(0, min(int(spec.bbox[0]), w - 1))
    y1 = max(0, min(int(spec.bbox[1]), h - 1))
    x2 = max(0, min(int(spec.bbox[2]), w - 1))
    y2 = max(0, min(int(spec.bbox[3]), h - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    if spec.label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(spec.label, font, scale, thickness)
        pad = 6
        ty = y1 - pad if y1 - pad - th > 0 else y2 + th + pad
        bg_y1 = ty - th - pad
        bg_y2 = ty + pad
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 2 * pad, bg_y2), color, -1)
        cv2.putText(
            frame, spec.label, (x1 + pad, ty),
            font, scale, (20, 20, 20), thickness, cv2.LINE_AA,
        )


def save_keyframes(
    task_id: str,
    video_path: str | Path,
    specs: list[KeyframeSpec],
    quality: int = 90,
) -> list[KeyframeSpec]:
    """Read frames from the video and write each spec to disk, drawing the
    object bbox + label overlay when the spec carries one. Returns the list of
    specs successfully saved."""
    if not specs:
        return []
    import cv2

    target_dir = keyframes_dir(task_id)
    by_frame: dict[int, list[KeyframeSpec]] = {}
    for s in specs:
        by_frame.setdefault(s.frame_number, []).append(s)

    saved: list[KeyframeSpec] = []
    wanted = set(by_frame)
    if not wanted:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("keyframes_open_failed task=%s", task_id)
        return []
    try:
        idx = 0
        max_wanted = max(wanted)
        while idx <= max_wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx in wanted:
                # Each spec gets its own annotated copy so multiple
                # overlays for the same frame don't stack on disk.
                for spec in by_frame[idx]:
                    annotated = frame.copy()
                    _draw_annotation(annotated, spec)
                    out_path = target_dir / spec.filename()
                    cv2.imwrite(
                        str(out_path), annotated,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
                    )
                    saved.append(spec)
            idx += 1
    finally:
        cap.release()

    log.info("keyframes_saved task=%s saved=%d wanted=%d", task_id, len(saved), len(specs))
    return saved
