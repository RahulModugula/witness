"""Side-by-side comparison of S vs M YOLO-World on a final, visually-curated
prompt set. Pick the model that detects person + cable + spectrophotometer +
robotic arm with the best confidence.

One-off; not part of the runtime pipeline."""
from __future__ import annotations

import sys
from collections import defaultdict

from ultralytics import YOLOWorld

# Visually-specific phrasings discovered via probe_cable.py iteration.
PROMPTS = [
    "person",
    "blue usb connector",   # ← canonical label: cable
    "spectrophotometer",
    "white robotic arm",    # ← canonical label: robotic arm
    "computer monitor",     # ← canonical label: monitor
    "test tube rack",
    "bottle",
    "laptop",
]


def run(model_name: str, video: str) -> dict[str, tuple[int, float]]:
    print(f"\n===== model: {model_name} =====")
    model = YOLOWorld(model_name)
    model.set_classes(PROMPTS)
    max_conf: dict[str, float] = defaultdict(float)
    hits: dict[str, int] = defaultdict(int)
    results = model.track(
        source=video, persist=True, stream=True, conf=0.1, imgsz=1280, verbose=False
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
    print(f"frames: {frames}")
    print(f"{'prompt':<24} {'hits':>6} {'max_conf':>10}")
    for p in PROMPTS:
        print(f"{p:<24} {hits[p]:>6} {max_conf[p]:>10.3f}")
    return {p: (hits[p], max_conf[p]) for p in PROMPTS}


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/sample.mp4"
    s_res = run("yolov8s-worldv2.pt", video)
    m_res = run("yolov8m-worldv2.pt", video)
    print("\n===== summary =====")
    print(f"{'prompt':<24} {'S hits':>8} {'S max':>8} {'M hits':>8} {'M max':>8}")
    for p in PROMPTS:
        sh, sc = s_res[p]
        mh, mc = m_res[p]
        print(f"{p:<24} {sh:>8} {sc:>8.3f} {mh:>8} {mc:>8.3f}")
