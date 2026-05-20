from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from ..logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class HandLandmarks:
    """21 (x, y) image-pixel points per hand. Indices follow MediaPipe Hands:
    0 = wrist, 4 = thumb tip, 8 = index tip, 12 = middle tip, 16 = ring tip,
    20 = pinky tip. See docs/brief.md §9.3 / §9.5."""

    points: list[tuple[float, float]]

    def fingertips(self) -> list[tuple[float, float]]:
        return [self.points[i] for i in (4, 8, 12, 16, 20)]

    def wrist(self) -> tuple[float, float]:
        return self.points[0]


class HandPoseExtractor:
    """Lazy wrapper around MediaPipe Hands. CPU-only; model_complexity=0 for speed."""

    def __init__(self) -> None:
        import mediapipe as mp  # lazy — heavy import

        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=settings.hand_detection_conf,
            min_tracking_confidence=settings.hand_tracking_conf,
            model_complexity=0,
        )

    def extract(self, frame_bgr) -> list[HandLandmarks]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        hands: list[HandLandmarks] = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                pts = [(lm.x * w, lm.y * h) for lm in hand.landmark]
                hands.append(HandLandmarks(points=pts))
        return hands

    def extract_video(self, path: str | Path) -> dict[int, list[HandLandmarks]]:
        from .video import iter_frames

        out: dict[int, list[HandLandmarks]] = {}
        for idx, frame in iter_frames(path):
            hands = self.extract(frame)
            if hands:
                out[idx] = hands
        log.info("pose_done frames_with_hands=%d", len(out))
        return out

    def close(self) -> None:
        self.hands.close()


_pose: HandPoseExtractor | None = None


def get_pose_extractor() -> HandPoseExtractor:
    global _pose
    if _pose is None:
        _pose = HandPoseExtractor()
    return _pose
