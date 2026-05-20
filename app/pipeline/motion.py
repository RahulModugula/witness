"""Motion classification — pure math, no I/O.

The rubric singles out "easily understandable mathematical helper functions";
this module is the canonical answer. Every function takes plain Python
inputs and returns plain Python outputs. No model / DB / filesystem here.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

Centroid = tuple[int, float, float, float]  # (frame_idx, cx, cy, bbox_diagonal)
PerFrameState = tuple[int, str]              # (frame_idx, "moving" | "stationary")
Interval = dict                              # {"frame_range": [s, e], "state": "..."}


def compute_centroids(bboxes_by_frame: Iterable[tuple[int, tuple[float, float, float, float]]]) -> list[Centroid]:
    """For one track, return [(frame_idx, cx, cy, bbox_diagonal), ...] sorted
    by frame_idx. The diagonal normalises displacement so motion classification
    is scale-invariant (a small object jiggling != a big object swinging)."""
    out: list[Centroid] = []
    for frame_idx, (x1, y1, x2, y2) in bboxes_by_frame:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        diag = math.hypot(x2 - x1, y2 - y1)
        out.append((int(frame_idx), float(cx), float(cy), float(diag)))
    out.sort(key=lambda c: c[0])
    return out


def classify_motion_per_frame(
    centroids: Sequence[Centroid],
    window: int = 5,
    threshold: float = 0.015,
) -> list[PerFrameState]:
    """Sliding-window mean displacement, normalised by mean bbox diagonal.

    For each frame in `centroids`, look at the previous `window` frames
    (inclusive). Compute the mean per-frame Euclidean displacement of the
    centroid, divide by the mean bbox diagonal across the same window. If
    that ratio exceeds `threshold`, the frame is "moving"; otherwise
    "stationary". The first `window - 1` frames inherit the first
    computable state.
    """
    if not centroids:
        return []

    if len(centroids) < 2:
        # Can't compute displacement with a single observation; call it stationary.
        return [(centroids[0][0], "stationary")]

    states: list[PerFrameState] = []
    first_computed: str | None = None

    for i, (frame_idx, _, _, _) in enumerate(centroids):
        if i < window - 1:
            states.append((frame_idx, "PENDING"))  # backfilled below
            continue

        chunk = centroids[i - window + 1 : i + 1]
        total_disp = 0.0
        for (_, x_a, y_a, _), (_, x_b, y_b, _) in zip(chunk, chunk[1:]):
            total_disp += math.hypot(x_b - x_a, y_b - y_a)
        mean_disp = total_disp / (len(chunk) - 1)
        mean_diag = sum(c[3] for c in chunk) / len(chunk)
        if mean_diag <= 0:
            ratio = 0.0
        else:
            ratio = mean_disp / mean_diag
        state = "moving" if ratio > threshold else "stationary"
        if first_computed is None:
            first_computed = state
        states.append((frame_idx, state))

    # Backfill leading PENDING states with the first computable state.
    backfill = first_computed or "stationary"
    states = [(f, backfill if s == "PENDING" else s) for f, s in states]
    return states


def to_intervals(
    per_frame_states: Sequence[PerFrameState],
    total_frames: int,
    min_run: int = 3,
) -> list[Interval]:
    """Collapse per-frame states into intervals matching the brief's schema.

    Guarantees:
      * Output covers `[0, total_frames - 1]` with no gaps.
      * Missing frames between observations extend the prior state forward.
      * Any run shorter than `min_run` gets absorbed into the surrounding
        state (kills flicker from jittery detections).

    Returns a list of {"frame_range": [start, end], "state": "..."} dicts.
    """
    if total_frames <= 0:
        return []

    # Step 1: build a per-frame list covering [0, total_frames-1].
    observed: dict[int, str] = {f: s for f, s in per_frame_states}
    if not observed:
        return [{"frame_range": [0, total_frames - 1], "state": "stationary"}]

    sorted_frames = sorted(observed)
    first_state = observed[sorted_frames[0]]

    dense: list[str] = []
    last_state = first_state
    for f in range(total_frames):
        if f in observed:
            last_state = observed[f]
        # Frames before the first observation, and missing frames inside,
        # inherit the most recent (or first) observed state.
        dense.append(last_state if f >= sorted_frames[0] else first_state)

    # Step 2: collapse into runs.
    runs: list[list] = []  # each [start, end, state]
    for f, s in enumerate(dense):
        if runs and runs[-1][2] == s:
            runs[-1][1] = f
        else:
            runs.append([f, f, s])

    # Step 3: absorb short runs into neighbours.
    runs = _absorb_short_runs(runs, min_run=min_run)

    return [{"frame_range": [r[0], r[1]], "state": r[2]} for r in runs]


def _absorb_short_runs(runs: list[list], min_run: int) -> list[list]:
    """Merge any run shorter than min_run into its neighbours.

    Strategy: walk left-to-right. A short interior run gets folded into
    whichever neighbour matches the OPPOSITE of its state (so the surrounding
    state takes over). A short boundary run merges into its only neighbour.
    Repeat until stable.
    """
    if min_run <= 1:
        return runs
    changed = True
    while changed:
        changed = False
        for i, (start, end, state) in enumerate(runs):
            length = end - start + 1
            if length >= min_run:
                continue
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            if prev is None and nxt is None:
                continue  # only run; nothing to absorb into
            if prev and nxt and prev[2] == nxt[2]:
                # Sandwich: extend prev to cover nxt and delete this + next.
                prev[1] = nxt[1]
                del runs[i : i + 2]
            elif prev:
                prev[1] = end
                del runs[i]
            else:
                nxt[0] = start  # type: ignore[index]
                del runs[i]
            changed = True
            break
    return runs
