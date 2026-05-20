from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import settings
from ..logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FrameDetection:
    frame_idx: int
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class Detector:
    """Wraps YOLO-World v2 + BoT-SORT tracking.

    Why this exists: COCO-trained YOLO has no `cable` / `spectrophotometer` class.
    YOLO-World accepts a runtime vocabulary; we set it ONCE at construction
    (repeated set_classes calls hit a torch version-counter bug in 8.3.x).
    """

    def __init__(
        self,
        model_path: str | None = None,
        class_prompts: Iterable[str] | None = None,
        conf: float | None = None,
        tracker_yaml: str | None = None,
    ) -> None:
        # Imported lazily so the API skeleton can boot without the ML deps.
        from ultralytics import YOLOWorld  # type: ignore

        self.model_path = model_path or settings.detector_model
        self.class_prompts: list[str] = list(class_prompts or settings.class_prompts)
        self.conf = conf if conf is not None else settings.detection_conf
        self.tracker_yaml = tracker_yaml or settings.tracker_yaml

        log.info("detector_init model=%s prompts=%d", self.model_path, len(self.class_prompts))
        self.model = YOLOWorld(self.model_path)
        self.model.set_classes(self.class_prompts)  # call once — see brief §3.1

    def track_video(self, video_path: str | Path) -> list[FrameDetection]:
        """Run tracking across the entire video. Returns flat per-frame detections."""
        results = self.model.track(
            source=str(video_path),
            persist=True,
            stream=True,
            conf=self.conf,
            tracker=self.tracker_yaml,
            imgsz=settings.detection_imgsz,
            verbose=False,
        )

        detections: list[FrameDetection] = []
        for frame_idx, result in enumerate(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                continue
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            for tid, box, cf, cls_idx in zip(ids, xyxy, confs, classes):
                if cls_idx < 0 or cls_idx >= len(self.class_prompts):
                    continue
                raw_label = self.class_prompts[cls_idx]
                canonical = settings.class_aliases.get(raw_label, raw_label)
                detections.append(
                    FrameDetection(
                        frame_idx=frame_idx,
                        track_id=int(tid),
                        class_name=canonical,
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(cf),
                    )
                )

        log.info(
            "detector_done frames=%d detections=%d unique_tracks=%d",
            frame_idx + 1 if detections else 0,
            len(detections),
            len({d.track_id for d in detections}),
        )
        return detections


_detector: Detector | None = None


def get_detector() -> Detector:
    """Lazy singleton — loading YOLO-World takes ~3s; do it once per process."""
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector
