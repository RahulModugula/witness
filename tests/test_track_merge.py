"""Unit tests for the duplicate-track merging post-processor.

These pin the key invariants:
  - Same-class tracks with high mean-bbox IoU are merged.
  - Different-class tracks are NEVER merged (cable + person at the same spot
    must remain two objects).
  - Same-frame collisions are resolved by NMS (higher conf wins), not by
    duplicating detections under the same track ID.
  - Empty input is handled cleanly.
"""
from __future__ import annotations

from app.pipeline.detector import FrameDetection
from app.pipeline.track_merge import merge_duplicate_tracks


def _det(frame: int, tid: int, cls: str, bbox: tuple, conf: float = 0.7) -> FrameDetection:
    return FrameDetection(frame_idx=frame, track_id=tid, class_name=cls, bbox=bbox, confidence=conf)


def test_empty_input_returns_empty():
    assert merge_duplicate_tracks([]) == []


def test_same_class_overlapping_tracks_merge_into_one():
    """Two cable tracks with near-identical bboxes across distinct frames →
    merged under the lower track_id."""
    a = [_det(f, 5, "cable", (100, 100, 200, 200)) for f in range(0, 10)]
    b = [_det(f, 7, "cable", (102, 101, 198, 199)) for f in range(20, 30)]
    out = merge_duplicate_tracks(a + b)
    track_ids = {d.track_id for d in out}
    assert track_ids == {5}, f"expected single merged id 5, got {track_ids}"
    assert len(out) == 20


def test_different_classes_never_merge_even_when_bboxes_match():
    """A cable bbox right on top of a person bbox must NOT collapse — they
    are distinct objects regardless of geometric overlap."""
    a = [_det(f, 1, "cable", (100, 100, 200, 200)) for f in range(10)]
    b = [_det(f, 2, "person", (100, 100, 200, 200)) for f in range(10)]
    out = merge_duplicate_tracks(a + b)
    track_ids = {d.track_id for d in out}
    assert track_ids == {1, 2}


def test_disjoint_same_class_tracks_do_not_merge():
    """If two same-class tracks have non-overlapping mean bboxes, they're
    two real instances (e.g. two bottles on the bench) and must not merge."""
    left = [_det(f, 1, "bottle", (50, 50, 100, 100)) for f in range(10)]
    right = [_det(f, 2, "bottle", (500, 500, 600, 600)) for f in range(10)]
    out = merge_duplicate_tracks(left + right)
    assert {d.track_id for d in out} == {1, 2}


def test_same_frame_collision_keeps_higher_confidence():
    """When two source tracks both have a detection on the same frame and get
    merged, the higher-confidence bbox survives (NMS-style resolution)."""
    a = _det(5, 1, "cable", (100, 100, 200, 200), conf=0.4)
    b = _det(5, 2, "cable", (101, 101, 199, 199), conf=0.9)  # overlaps a → merge group
    c = _det(6, 2, "cable", (101, 101, 199, 199), conf=0.9)
    d = _det(6, 1, "cable", (100, 100, 200, 200), conf=0.4)
    # Add filler so each track has enough samples to merge confidently.
    a_filler = [_det(f, 1, "cable", (100, 100, 200, 200), conf=0.4) for f in range(10, 15)]
    b_filler = [_det(f, 2, "cable", (101, 101, 199, 199), conf=0.9) for f in range(10, 15)]

    out = merge_duplicate_tracks([a, b, c, d] + a_filler + b_filler)

    # One detection per surviving frame, each at the higher confidence.
    frame_to_conf = {d.frame_idx: d.confidence for d in out}
    assert frame_to_conf[5] == 0.9
    assert frame_to_conf[6] == 0.9
    # Single merged track ID.
    assert {d.track_id for d in out} == {1}


def test_merge_is_deterministic_picks_lower_id_as_root():
    """Track 9 should merge into 3, not the reverse — root must be the
    minimum ID so the output ordering is stable across runs."""
    a = [_det(f, 9, "monitor", (200, 200, 400, 400)) for f in range(8)]
    b = [_det(f, 3, "monitor", (201, 201, 399, 399)) for f in range(8, 16)]
    out = merge_duplicate_tracks(a + b)
    assert {d.track_id for d in out} == {3}
