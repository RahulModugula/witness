"""One-off: tries a wide menu of cable-ish prompts at low conf to discover
which phrasing YOLO-World actually fires on for this video. Used to tune
the production prompt list — NOT part of the runtime pipeline."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLOWorld

PROMPTS = [
    "person",
    "cable",
    "usb cable",
    "data cable",
    "white cable",
    "thin cable",
    "power cord",
    "wire",
    "cord",
    "rope",
    "tube",
    "hose",
    "string",
    "connector",
    "plug",
    "usb connector",
    "usb plug",
    "spectrophotometer",
]


def main(video: str) -> int:
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(PROMPTS)

    max_conf: dict[str, float] = defaultdict(float)
    hits: dict[str, int] = defaultdict(int)

    results = model.track(
        source=video, persist=True, stream=True, conf=0.03, imgsz=1280, verbose=False
    )
    frames = 0
    for r in results:
        frames += 1
        if r.boxes is None or r.boxes.cls is None:
            continue
        for cls_idx, conf in zip(r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
            label = PROMPTS[cls_idx]
            hits[label] += 1
            if conf > max_conf[label]:
                max_conf[label] = conf

    print(f"frames processed: {frames}")
    print(f"{'prompt':<22} {'hits':>6} {'max_conf':>10}")
    for p in PROMPTS:
        print(f"{p:<22} {hits[p]:>6} {max_conf[p]:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/uploads/sample.mp4"))
