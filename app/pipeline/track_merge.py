"""Post-process the tracker output to collapse duplicate tracks for the same
physical object.

Why: even with one CLIP prompt per canonical class, the tracker can fragment a
single object's trajectory across multiple track IDs when it's briefly occluded
(e.g. the technician's hand crossing the cable) or when YOLO emits two close
boxes that beat NMS but don't share an ID. We greedily merge same-class tracks
whose mean bboxes have high IoU.

Pure functions, easy to unit-test."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .detector import FrameDetection


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mean_bbox(dets: list[FrameDetection]) -> tuple[float, float, float, float]:
    n = float(len(dets))
    sx1 = sum(d.bbox[0] for d in dets) / n
    sy1 = sum(d.bbox[1] for d in dets) / n
    sx2 = sum(d.bbox[2] for d in dets) / n
    sy2 = sum(d.bbox[3] for d in dets) / n
    return (sx1, sy1, sx2, sy2)


def _majority_class(dets: Iterable[FrameDetection]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for d in dets:
        counts[d.class_name] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def merge_duplicate_tracks(
    detections: list[FrameDetection],
    *,
    iou_threshold: float = 0.3,
) -> list[FrameDetection]:
    """Merge tracks of the same canonical class whose mean bboxes overlap.

    Strategy:
      1. Group detections by track_id.
      2. For each canonical class, compute pairwise IoU on mean bboxes.
      3. Greedy union-find: any pair > iou_threshold gets merged.
      4. Within a merged group, keep the lowest track_id; re-emit all detections
         under that ID. If two source tracks have a detection in the same frame,
         keep the higher-confidence one (NMS-style).

    Returns a new list of FrameDetection; does not mutate the input.
    """
    by_track: dict[int, list[FrameDetection]] = defaultdict(list)
    for d in detections:
        by_track[d.track_id].append(d)

    # Group track IDs by canonical class.
    tracks_by_class: dict[str, list[int]] = defaultdict(list)
    for tid, dets in by_track.items():
        tracks_by_class[_majority_class(dets)].append(tid)

    # Union-find over track IDs.
    parent: dict[int, int] = {tid: tid for tid in by_track}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the smaller ID as root (deterministic + stable in output order).
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for tids in tracks_by_class.values():
        means = {tid: _mean_bbox(by_track[tid]) for tid in tids}
        for i, ti in enumerate(tids):
            for tj in tids[i + 1 :]:
                if _bbox_iou(means[ti], means[tj]) >= iou_threshold:
                    union(ti, tj)

    # Re-emit detections under canonical (root) track IDs, NMS-resolving
    # same-frame collisions by keeping the higher-confidence detection.
    merged: dict[tuple[int, int], FrameDetection] = {}
    for d in detections:
        root = find(d.track_id)
        key = (root, d.frame_idx)
        prev = merged.get(key)
        if prev is None or d.confidence > prev.confidence:
            merged[key] = FrameDetection(
                frame_idx=d.frame_idx,
                track_id=root,
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
            )

    return sorted(merged.values(), key=lambda d: (d.frame_idx, d.track_id))
