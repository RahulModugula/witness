from __future__ import annotations

from app.pipeline.interaction import (
    detect_interactions,
    fingertips_in_bbox,
    hand_belongs_to_person,
    iou,
)
from app.pipeline.pose import HandLandmarks


def _hand_at(x: float, y: float) -> HandLandmarks:
    """Build a 21-point hand whose wrist + all fingertips sit at (x, y)."""
    return HandLandmarks(points=[(x, y)] * 21)


def _hand_with_fingers(wrist: tuple[float, float], fingertips: list[tuple[float, float]]) -> HandLandmarks:
    pts: list[tuple[float, float]] = [(0.0, 0.0)] * 21
    pts[0] = wrist
    for idx, ft in zip((4, 8, 12, 16, 20), fingertips):
        pts[idx] = ft
    # fill middle joints with something near the wrist so they don't fall in unrelated bboxes
    for i in range(1, 21):
        if pts[i] == (0.0, 0.0):
            pts[i] = wrist
    return HandLandmarks(points=pts)


# --------- low-level helpers ---------


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_full_overlap_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_fingertips_in_bbox_counts_only_fingers_not_wrist():
    # Wrist far away, fingertips all inside the bbox.
    hand = _hand_with_fingers(wrist=(1000, 1000), fingertips=[(50, 50)] * 5)
    assert fingertips_in_bbox(hand, (0, 0, 100, 100), expand=1.0) == 5


def test_fingertips_in_bbox_zero_when_all_outside():
    hand = _hand_with_fingers(wrist=(50, 50), fingertips=[(500, 500)] * 5)
    assert fingertips_in_bbox(hand, (0, 0, 100, 100), expand=1.0) == 0


def test_hand_belongs_to_person_via_wrist():
    hand = _hand_with_fingers(wrist=(50, 50), fingertips=[(0, 0)] * 5)
    assert hand_belongs_to_person(hand, (0, 0, 100, 100)) is True
    assert hand_belongs_to_person(hand, (200, 200, 300, 300)) is False


# --------- end-to-end interval logic ---------


def test_interaction_full_duration_one_interval():
    """Hand fingertips in bbox for all 10 frames → one interval [0, 9]."""
    object_dets = {1: [(f, (0.0, 0.0, 100.0, 100.0)) for f in range(10)]}
    person_dets = {0: [(f, (0.0, 0.0, 200.0, 200.0)) for f in range(10)]}
    hands = {f: [_hand_with_fingers((50, 50), [(50, 50)] * 5)] for f in range(10)}

    intervals, peaks, degraded = detect_interactions(
        object_dets, person_dets, hands, total_frames=10
    )
    assert degraded is False
    assert intervals[1] == [{"interacted_by_person": 0, "frame_start": 0, "frame_end": 9}]


def test_single_frame_touch_filtered_by_min_run():
    """Only frame 5 has a touch → no interval emitted (min_run=3)."""
    object_dets = {1: [(f, (0.0, 0.0, 100.0, 100.0)) for f in range(10)]}
    person_dets = {0: [(f, (0.0, 0.0, 200.0, 200.0)) for f in range(10)]}
    hands: dict[int, list[HandLandmarks]] = {}
    for f in range(10):
        if f == 5:
            hands[f] = [_hand_with_fingers((50, 50), [(50, 50)] * 5)]
        else:
            hands[f] = [_hand_with_fingers((50, 50), [(500, 500)] * 5)]
    intervals, _, _ = detect_interactions(object_dets, person_dets, hands, total_frames=10)
    assert intervals[1] == []


def test_two_separated_runs_emit_two_intervals():
    """Touches frames 0-4 and 10-14 → two intervals."""
    object_dets = {1: [(f, (0.0, 0.0, 100.0, 100.0)) for f in range(20)]}
    person_dets = {0: [(f, (0.0, 0.0, 200.0, 200.0)) for f in range(20)]}
    hands: dict[int, list[HandLandmarks]] = {}
    for f in range(20):
        if 0 <= f <= 4 or 10 <= f <= 14:
            hands[f] = [_hand_with_fingers((50, 50), [(50, 50)] * 5)]
        else:
            hands[f] = [_hand_with_fingers((50, 50), [(500, 500)] * 5)]
    intervals, _, _ = detect_interactions(object_dets, person_dets, hands, total_frames=20)
    starts = sorted(iv["frame_start"] for iv in intervals[1])
    ends = sorted(iv["frame_end"] for iv in intervals[1])
    assert starts == [0, 10]
    assert ends == [4, 14]


def test_no_hands_anywhere_triggers_iou_fallback():
    """When MediaPipe finds zero hands across the video, fall back to IoU
    overlap between person and object bboxes."""
    object_dets = {1: [(f, (0.0, 0.0, 100.0, 100.0)) for f in range(10)]}
    person_dets = {0: [(f, (50.0, 50.0, 150.0, 150.0)) for f in range(10)]}
    intervals, _, degraded = detect_interactions(
        object_dets, person_dets, hands_by_frame={}, total_frames=10
    )
    assert degraded is True
    assert intervals[1] == [{"interacted_by_person": 0, "frame_start": 0, "frame_end": 9}]


def test_hand_not_attributed_to_distant_person():
    """A hand whose wrist is far from the person bbox should NOT count."""
    object_dets = {1: [(f, (0.0, 0.0, 100.0, 100.0)) for f in range(10)]}
    person_dets = {0: [(f, (500.0, 500.0, 600.0, 600.0)) for f in range(10)]}
    hands = {f: [_hand_with_fingers((50, 50), [(50, 50)] * 5)] for f in range(10)}
    intervals, _, degraded = detect_interactions(
        object_dets, person_dets, hands, total_frames=10
    )
    assert degraded is False
    assert intervals[1] == []
