"""Debug: for the latest result task, walk through frames where spec is
tracked AND hands exist AND person is tracked. Report whether the wrist
landmark falls inside the person bbox (which is the gate for hand ownership)."""
from __future__ import annotations

import sys

from app.config import settings
from app.pipeline.detector import get_detector
from app.pipeline.interaction import _point_in_bbox
from app.pipeline.pose import get_pose_extractor
from app.pipeline.track_merge import merge_duplicate_tracks
from app.pipeline.video import read_metadata


def main(video: str) -> int:
    meta = read_metadata(video)
    det = get_detector()
    raw = det.track_video(video)
    detections = merge_duplicate_tracks(raw)

    pose = get_pose_extractor()
    hands_by_frame = pose.extract_video(video)

    # Per frame, what's tracked?
    by_frame: dict[int, list] = {}
    for d in detections:
        by_frame.setdefault(d.frame_idx, []).append(d)

    # Iterate frames where we expect interactions.
    persons_with_wrist_outside = 0
    persons_with_wrist_inside = 0
    interactable_frames = 0
    for f in range(meta.frame_count):
        dets = by_frame.get(f, [])
        persons = [d for d in dets if d.class_name == "person"]
        objects = [d for d in dets if d.class_name != "person"]
        hands = hands_by_frame.get(f, [])
        if not persons or not objects or not hands:
            continue
        interactable_frames += 1
        for hand in hands:
            wrist = hand.wrist()
            inside = any(_point_in_bbox(wrist, p.bbox) for p in persons)
            if inside:
                persons_with_wrist_inside += 1
            else:
                persons_with_wrist_outside += 1

    print(f"frames with hand+person+object: {interactable_frames}")
    print(f"hand-wrist inside any person bbox: {persons_with_wrist_inside}")
    print(f"hand-wrist OUTSIDE all person bboxes: {persons_with_wrist_outside}")
    if persons_with_wrist_outside > persons_with_wrist_inside * 3:
        print("\n>>> wrist-in-bbox check is likely the culprit. The technician's body")
        print(">>> is mostly off-screen, so the person bbox doesn't contain the wrist")
        print(">>> of the reaching hand. Need an alternative assignment heuristic.")

    # Sample one frame to inspect.
    sample = next(
        (f for f in range(meta.frame_count) if by_frame.get(f) and hands_by_frame.get(f) and
         any(d.class_name == "person" for d in by_frame[f]) and
         any(d.class_name != "person" for d in by_frame[f])),
        None,
    )
    if sample is not None:
        print(f"\nsample frame {sample}:")
        for d in by_frame[sample]:
            print(f"  {d.class_name:<20} bbox={tuple(round(v, 0) for v in d.bbox)}")
        for i, h in enumerate(hands_by_frame[sample]):
            print(f"  hand[{i}] wrist={tuple(round(v, 0) for v in h.wrist())} "
                  f"fingertips={[tuple(round(v, 0) for v in pt) for pt in h.fingertips()]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/uploads/sample.mp4"))
