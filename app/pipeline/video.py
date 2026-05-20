from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class VideoMeta:
    filename: str
    duration_seconds: float
    frame_count: int
    fps: float
    width: int
    height: int
    codec: str


def _fourcc_to_str(value: int) -> str:
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower() or "unknown"


@contextmanager
def _open_capture(path: str | Path) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


def read_metadata(path: str | Path) -> VideoMeta:
    p = Path(path)
    with _open_capture(p) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        codec = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0))
    duration = (frame_count / fps) if fps > 0 else 0.0
    return VideoMeta(
        filename=p.name,
        duration_seconds=round(duration, 3),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        codec=codec or "unknown",
    )


def iter_frames(path: str | Path):
    """Yields (frame_idx, BGR ndarray). Releases the capture on exit."""
    with _open_capture(path) as cap:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            yield idx, frame
            idx += 1
