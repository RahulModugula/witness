"""Interaction detection — pure math, mirrors motion.py in style.

A frame is "interacting" between a person P and an object O when one of P's
hands rests partly inside O's bounding box. Fingertip landmarks are the
physical contact points, so they're the primary signal. The bbox is expanded
slightly (default 15%) to tolerate near-misses (a grip slightly outside the
detected bbox).

The IoU-fallback path covers the case where MediaPipe finds no hands across
the whole video — extremely rare but possible (e.g. hands always off-screen).
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .pose import HandLandmarks

Bbox = tuple[float, float, float, float]
FingertipsCount = dict[int, int]  # frame_idx -> fingertips inside bbox at peak


def _expand_bbox(bbox: Bbox, factor: float) -> Bbox:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * factor
    h = (y2 - y1) * factor
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def _point_in_bbox(pt: tuple[float, float], bbox: Bbox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def fingertips_in_bbox(hand: HandLandmarks, bbox: Bbox, expand: float = 1.15) -> int:
    """Number of fingertip landmarks inside the expanded bbox."""
    eb = _expand_bbox(bbox, expand)
    return sum(1 for pt in hand.fingertips() if _point_in_bbox(pt, eb))


def iou(a: Bbox, b: Bbox) -> float:
    """Standard axis-aligned IoU."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def hand_belongs_to_person(hand: HandLandmarks, person_bbox: Bbox) -> bool:
    """Wrist-in-bbox check — see brief §16 pitfall #11."""
    return _point_in_bbox(hand.wrist(), person_bbox)


# ---------- pure interval logic (mirrors motion.to_intervals) ----------


def _per_frame_to_intervals(
    present_frames: Sequence[int],
    total_frames: int,
    min_run: int = 3,
) -> list[tuple[int, int]]:
    """Take a sorted list of frame indices where interaction is true, return
    contiguous [start, end] intervals after applying a min-run filter."""
    if not present_frames:
        return []

    runs: list[list[int]] = []  # [start, end]
    for f in present_frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])

    # Drop short runs (likely noise / single-frame brushes).
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_run]
    return [(r[0], r[1]) for r in runs]


# ---------- main entry ----------


def detect_interactions(
    object_dets: Mapping[int, list[tuple[int, Bbox]]],
    person_dets: Mapping[int, list[tuple[int, Bbox]]],
    hands_by_frame: Mapping[int, Sequence[HandLandmarks]],
    total_frames: int,
    *,
    bbox_expand: float = 1.15,
    fingertip_min: int = 1,
    min_run: int = 3,
    iou_fallback_threshold: float = 0.05,
) -> tuple[dict[int, list[dict]], dict[int, FingertipsCount], bool]:
    """For each (object_id, person_id) where a hand belongs to that person and
    touches that object, emit interaction intervals.

    Returns
    -------
    intervals: object_id -> list of {"interacted_by_person": int,
                                     "frame_start": int, "frame_end": int}
    peaks:     object_id -> {frame_idx: fingertips_in_bbox_at_peak}
               (used by keyframe extraction to pick the "peak" frame).
    degraded:  True if we had to fall back to IoU (no hands ever found).
    """
    # Index per-frame lookups for speed.
    object_by_frame: dict[int, dict[int, Bbox]] = {}
    for oid, dets in object_dets.items():
        for frame, bbox in dets:
            object_by_frame.setdefault(frame, {})[oid] = bbox

    person_by_frame: dict[int, dict[int, Bbox]] = {}
    for pid, dets in person_dets.items():
        for frame, bbox in dets:
            person_by_frame.setdefault(frame, {})[pid] = bbox

    has_any_hands = any(len(h) > 0 for h in hands_by_frame.values())

    intervals_out: dict[int, list[dict]] = {oid: [] for oid in object_dets}
    peaks_out: dict[int, FingertipsCount] = {oid: {} for oid in object_dets}

    if has_any_hands:
        # Per (object, person), per frame: max fingertips inside bbox across
        # any hand owned by that person.
        per_pair: dict[tuple[int, int], dict[int, int]] = {}
        for frame, hands in hands_by_frame.items():
            objs = object_by_frame.get(frame, {})
            persons = person_by_frame.get(frame, {})
            if not objs or not persons or not hands:
                continue
            for hand in hands:
                # Find the person this hand belongs to.
                owner_pid = next(
                    (pid for pid, pbb in persons.items() if hand_belongs_to_person(hand, pbb)),
                    None,
                )
                if owner_pid is None:
                    continue
                for oid, obb in objs.items():
                    count = fingertips_in_bbox(hand, obb, expand=bbox_expand)
                    if count < fingertip_min:
                        continue
                    key = (oid, owner_pid)
                    existing = per_pair.setdefault(key, {})
                    if count > existing.get(frame, 0):
                        existing[frame] = count

        for (oid, pid), per_frame in per_pair.items():
            frames = sorted(per_frame)
            runs = _per_frame_to_intervals(frames, total_frames, min_run=min_run)
            for s, e in runs:
                intervals_out.setdefault(oid, []).append(
                    {"interacted_by_person": pid, "frame_start": s, "frame_end": e}
                )
            # Record peaks (highest fingertip count per frame for this object).
            peaks = peaks_out.setdefault(oid, {})
            for f, c in per_frame.items():
                if c > peaks.get(f, 0):
                    peaks[f] = c

        return intervals_out, peaks_out, False

    # Fallback: no hands detected anywhere. Use bbox-IoU as a coarse proxy.
    per_pair_fallback: dict[tuple[int, int], list[int]] = {}
    for frame, objs in object_by_frame.items():
        persons = person_by_frame.get(frame, {})
        if not persons:
            continue
        for oid, obb in objs.items():
            for pid, pbb in persons.items():
                if iou(obb, pbb) > iou_fallback_threshold:
                    per_pair_fallback.setdefault((oid, pid), []).append(frame)

    for (oid, pid), frames in per_pair_fallback.items():
        runs = _per_frame_to_intervals(sorted(frames), total_frames, min_run=min_run)
        for s, e in runs:
            intervals_out.setdefault(oid, []).append(
                {"interacted_by_person": pid, "frame_start": s, "frame_end": e}
            )

    return intervals_out, peaks_out, True
