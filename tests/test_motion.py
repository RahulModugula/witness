from __future__ import annotations

from app.pipeline.motion import (
    classify_motion_per_frame,
    compute_centroids,
    to_intervals,
)


def _centroids_static(n: int = 20, diag: float = 50.0) -> list[tuple[int, float, float, float]]:
    return [(i, 100.0, 100.0, diag) for i in range(n)]


def _centroids_moving(n: int = 20, dx: float = 10.0, diag: float = 50.0):
    return [(i, 100.0 + i * dx, 100.0, diag) for i in range(n)]


def test_compute_centroids_sorted_and_correct():
    bboxes = [(2, (0.0, 0.0, 10.0, 20.0)), (0, (10.0, 10.0, 30.0, 30.0))]
    centroids = compute_centroids(bboxes)
    assert [c[0] for c in centroids] == [0, 2]
    assert centroids[0] == (0, 20.0, 20.0, ((20**2 + 20**2) ** 0.5))


def test_all_stationary_produces_single_interval():
    states = classify_motion_per_frame(_centroids_static(20))
    intervals = to_intervals(states, total_frames=20)
    assert intervals == [{"frame_range": [0, 19], "state": "stationary"}]


def test_all_moving_produces_single_interval():
    states = classify_motion_per_frame(_centroids_moving(20))
    intervals = to_intervals(states, total_frames=20)
    assert intervals == [{"frame_range": [0, 19], "state": "moving"}]


def test_stationary_then_moving_two_intervals():
    static = _centroids_static(10)
    # second half moves rightward
    moving = [(10 + i, 100.0 + i * 10, 100.0, 50.0) for i in range(10)]
    states = classify_motion_per_frame(static + moving)
    intervals = to_intervals(states, total_frames=20)
    assert len(intervals) == 2
    assert intervals[0]["state"] == "stationary"
    assert intervals[1]["state"] == "moving"
    assert intervals[0]["frame_range"][0] == 0
    assert intervals[-1]["frame_range"][1] == 19


def test_short_burst_of_motion_absorbed_by_hysteresis():
    # 9 stationary, 2 nominally-moving frames, 9 stationary. min_run=3 should
    # collapse the 2-frame burst into the surrounding stationary run.
    centroids = (
        _centroids_static(9)
        + [(9, 100.0 + 30.0, 100.0, 50.0), (10, 100.0 + 60.0, 100.0, 50.0)]
        + [(i, 100.0 + 60.0, 100.0, 50.0) for i in range(11, 20)]
    )
    states = classify_motion_per_frame(centroids)
    intervals = to_intervals(states, total_frames=20, min_run=3)
    # The whole thing should be a small number of intervals (≤ 2) without
    # the absorbed flicker creating extra runs.
    assert all(iv["frame_range"][1] - iv["frame_range"][0] + 1 >= 3 for iv in intervals)


def test_frame_gaps_extend_prior_state():
    # Frames 5-10 missing entirely; the surrounding stationary state should
    # fill the gap rather than the output having gaps.
    centroids = [(i, 100.0, 100.0, 50.0) for i in range(5)] + [
        (i, 100.0, 100.0, 50.0) for i in range(11, 20)
    ]
    states = classify_motion_per_frame(centroids)
    intervals = to_intervals(states, total_frames=20)
    assert intervals[0]["frame_range"][0] == 0
    assert intervals[-1]["frame_range"][1] == 19
    # No gaps:
    for a, b in zip(intervals, intervals[1:]):
        assert b["frame_range"][0] == a["frame_range"][1] + 1


def test_intervals_cover_full_range_no_gaps():
    centroids = _centroids_static(15) + _centroids_moving(15)
    # Renumber moving frames to start at 15.
    centroids = _centroids_static(15) + [
        (15 + i, 100.0 + i * 10, 100.0, 50.0) for i in range(15)
    ]
    states = classify_motion_per_frame(centroids)
    intervals = to_intervals(states, total_frames=30)
    covered = sum(iv["frame_range"][1] - iv["frame_range"][0] + 1 for iv in intervals)
    assert covered == 30
    assert intervals[0]["frame_range"][0] == 0
    assert intervals[-1]["frame_range"][1] == 29


def test_empty_input_returns_default_stationary_interval():
    intervals = to_intervals([], total_frames=10)
    assert intervals == [{"frame_range": [0, 9], "state": "stationary"}]


def test_single_observation_returns_stationary():
    states = classify_motion_per_frame([(0, 100.0, 100.0, 50.0)])
    assert states == [(0, "stationary")]
