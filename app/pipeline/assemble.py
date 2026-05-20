"""Glue layer: take raw detections + hand landmarks and produce the final
JSON payload matching the schema in docs/brief.md §7."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..config import settings
from ..logging_config import get_logger
from .detector import FrameDetection
from .interaction import detect_interactions
from .keyframes import (
    KeyframeSpec,
    save_keyframes,
    select_interaction_peak_frames,
    select_motion_transition_frames,
)
from .motion import classify_motion_per_frame, compute_centroids, to_intervals
from .pose import HandLandmarks
from .track_merge import merge_duplicate_tracks
from .video import VideoMeta

log = get_logger(__name__)


def _majority_class(track_dets: list[FrameDetection]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for d in track_dets:
        counts[d.class_name] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def assemble_result(
    *,
    task_id: str,
    video_path: str | Path,
    video_meta: VideoMeta,
    detections: list[FrameDetection],
    hands_by_frame: dict[int, list[HandLandmarks]],
    processing_time: float,
) -> dict:
    # 1. Merge same-class duplicate tracks (tracker fragmentation from
    # occlusion), then group by track_id, then drop short tracks (flicker).
    detections = merge_duplicate_tracks(detections)
    by_track: dict[int, list[FrameDetection]] = defaultdict(list)
    for d in detections:
        by_track[d.track_id].append(d)
    kept_tracks: dict[int, list[FrameDetection]] = {
        tid: sorted(dets, key=lambda x: x.frame_idx)
        for tid, dets in by_track.items()
        if len(dets) >= settings.min_track_frames
    }

    # 2. Assign final object_ids in deterministic order (by first_seen_frame, then track_id).
    track_order = sorted(
        kept_tracks,
        key=lambda tid: (kept_tracks[tid][0].frame_idx, tid),
    )
    object_id_map: dict[int, int] = {tid: i for i, tid in enumerate(track_order)}

    # 3. Build object bbox indexes for interaction detection (skip person tracks).
    object_dets: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    person_dets: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for tid, dets in kept_tracks.items():
        oid = object_id_map[tid]
        cls = _majority_class(dets)
        series = [(d.frame_idx, d.bbox) for d in dets]
        if cls == "person":
            person_dets[oid] = series
        else:
            object_dets[oid] = series

    # 4. Interaction intervals (and per-frame peak fingertip counts for keyframes).
    interactions, peaks, degraded = detect_interactions(
        object_dets,
        person_dets,
        hands_by_frame,
        total_frames=video_meta.frame_count,
        bbox_expand=settings.interaction_bbox_expand,
        fingertip_min=settings.interaction_fingertip_min,
        min_run=settings.interaction_min_run,
        iou_fallback_threshold=settings.interaction_iou_fallback,
    )

    # 5. Build per-object output payloads. Compute motion only for non-person tracks
    # (the schema's "motion_history" applies to objects, not the human reference).
    objects_out: list[dict] = []
    motion_keyframe_specs: list[KeyframeSpec] = []
    interaction_keyframe_specs: list[KeyframeSpec] = []
    for tid in track_order:
        dets = kept_tracks[tid]
        oid = object_id_map[tid]
        cls = _majority_class(dets)

        # Motion history.
        if cls != "person":
            centroids = compute_centroids((d.frame_idx, d.bbox) for d in dets)
            per_frame = classify_motion_per_frame(
                centroids,
                window=settings.motion_window,
                threshold=settings.motion_threshold,
            )
            motion_history = to_intervals(
                per_frame,
                total_frames=video_meta.frame_count,
                min_run=settings.motion_min_run,
            )
        else:
            motion_history = []

        obj_interactions = interactions.get(oid, []) if cls != "person" else []
        conf_avg = sum(d.confidence for d in dets) / len(dets)

        objects_out.append(
            {
                "object_id": oid,
                "class": cls,
                "confidence_avg": round(conf_avg, 4),
                "first_seen_frame": dets[0].frame_idx,
                "last_seen_frame": dets[-1].frame_idx,
                "motion_history": motion_history,
                "interactions": obj_interactions,
            }
        )

        # Keyframe specs for this object — include the bbox + label so the
        # saved JPG has an annotated overlay (reviewers can see what was
        # detected without having to cross-reference the JSON).
        if cls != "person":
            obj_series = object_dets.get(oid, [])
            motion_keyframe_specs.extend(
                select_motion_transition_frames(
                    oid, motion_history, label=cls, object_frames=obj_series,
                )
            )
            interaction_keyframe_specs.extend(
                select_interaction_peak_frames(
                    oid, obj_interactions, peaks.get(oid, {}),
                    label=cls, object_frames=obj_series,
                )
            )

    # 6. Save keyframes to disk and build the JSON-side keyFrames list.
    all_specs = motion_keyframe_specs + interaction_keyframe_specs
    saved = save_keyframes(task_id, video_path, all_specs)
    keyframes_out = [
        {
            "frame_number": spec.frame_number,
            "type": spec.type,
            "object_id": spec.object_id,
            "image_path": f"keyframes/{task_id}/{spec.filename()}",
            "url": f"/tasks/{task_id}/keyframes/{spec.filename()}",
        }
        for spec in saved
    ]

    payload = {
        "videoMetadata": {
            "filename": video_meta.filename,
            "duration_seconds": video_meta.duration_seconds,
            "frame_count": video_meta.frame_count,
            "fps": video_meta.fps,
            "width": video_meta.width,
            "height": video_meta.height,
            "codec": video_meta.codec,
            "processing_time_seconds": round(processing_time, 3),
            "model": settings.detector_model,
            "tracker": "botsort",
            "degraded_interaction": degraded,
        },
        "objectsDetected": objects_out,
        "keyFrames": keyframes_out,
    }
    log.info(
        "assemble_done task=%s objects=%d keyframes=%d degraded=%s",
        task_id,
        len(objects_out),
        len(keyframes_out),
        degraded,
    )
    return payload
