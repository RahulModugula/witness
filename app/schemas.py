from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------- task lifecycle ---------

TaskStatus = Literal["PENDING", "PROCESSING", "DONE", "FAILED"]


class TaskCreatedOut(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime


class TaskStatusOut(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class TaskSummary(BaseModel):
    task_id: str
    status: TaskStatus
    original_filename: str
    created_at: datetime
    updated_at: datetime


class TaskListOut(BaseModel):
    items: list[TaskSummary]
    total: int
    limit: int
    offset: int


class ErrorOut(BaseModel):
    detail: str
    code: str | None = None


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    model: str


# --------- result payload (matches docs/brief.md §7 exactly) ---------


class StageTimings(BaseModel):
    detect_track_seconds: float
    hand_pose_seconds: float
    assemble_keyframes_seconds: float


class VideoMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    filename: str
    duration_seconds: float
    frame_count: int
    fps: float
    width: int
    height: int
    codec: str
    processing_time_seconds: float
    model: str
    tracker: str
    degraded_interaction: bool = False
    stage_timings: Optional[StageTimings] = None


class MotionInterval(BaseModel):
    frame_range: tuple[int, int]
    state: Literal["moving", "stationary"]


class InteractionInterval(BaseModel):
    interacted_by_person: int
    frame_start: int
    frame_end: int


class DetectedObject(BaseModel):
    object_id: int
    class_: str = Field(alias="class")
    confidence_avg: float = 0.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    motion_history: list[MotionInterval] = Field(default_factory=list)
    interactions: list[InteractionInterval] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class KeyFrame(BaseModel):
    frame_number: int
    type: Literal["motion_transition", "interaction_peak"]
    object_id: int
    image_path: str
    url: str


class ResultPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    videoMetadata: VideoMetadata
    objectsDetected: list[DetectedObject] = Field(default_factory=list)
    keyFrames: list[KeyFrame] = Field(default_factory=list)


# --------- SOP compliance check ---------


class SOPCheckIn(BaseModel):
    """POST body for /tasks/{id}/check. `rules` is the DSL text."""

    rules: str


class SOPRuleResult(BaseModel):
    rule: str
    op: str
    passed: bool
    detail: str


class SOPCheckOut(BaseModel):
    rules_total: int
    rules_passed: int
    rules_failed: int
    all_passed: bool
    results: list[SOPRuleResult]
