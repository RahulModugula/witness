from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EDREVEL_", extra="ignore")

    # Storage
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    results_dir: Path = PROJECT_ROOT / "data" / "results"
    keyframes_dir: Path = PROJECT_ROOT / "data" / "keyframes"

    # DB
    database_url: str = f"sqlite:///{PROJECT_ROOT}/edrevel.db"

    # Upload limits
    max_upload_bytes: int = 100 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

    # Model. yolov8s-worldv2 picked over -m: S has better spec recall in the
    # early-video frames where the actual interactions happen. Trade-off
    # discussed in the README.
    detector_model: str = "yolov8s-worldv2.pt"
    tracker_yaml: str = str(PROJECT_ROOT / "app" / "pipeline" / "botsort_tuned.yaml")
    detection_conf: float = 0.15
    detection_imgsz: int = 960
    min_track_frames: int = 5
    # Each prompt maps via class_aliases to a canonical label. Synonyms boost
    # recall; track_merge.py collapses any duplicate tracks they produce.
    class_prompts: tuple[str, ...] = (
        "person",
        "cable",
        "usb cable",
        "power cord",
        "wire",
        "spectrophotometer",
        "scientific instrument",
        "white laboratory device",
        "robotic arm",
        "monitor",
        "computer monitor",
        "test tube rack",
        "laptop",
        "bottle",
    )
    class_aliases: dict[str, str] = {
        "usb cable": "cable",
        "power cord": "cable",
        "wire": "cable",
        "scientific instrument": "spectrophotometer",
        "white laboratory device": "spectrophotometer",
        "computer monitor": "monitor",
    }

    # Motion thresholds (see docs/brief.md §9.4)
    motion_window: int = 5
    motion_threshold: float = 0.015
    motion_min_run: int = 3

    # Interaction
    interaction_bbox_expand: float = 1.15
    interaction_fingertip_min: int = 1
    interaction_min_run: int = 3
    interaction_iou_fallback: float = 0.05

    # Hand pose
    hand_detection_conf: float = 0.4
    hand_tracking_conf: float = 0.4

    # Test/runtime knobs
    run_inline: bool = Field(default=False, alias="RUN_INLINE")
    skip_model_warmup: bool = Field(default=False, alias="SKIP_MODEL_WARMUP")
    # When set, process_video writes a minimal valid payload without loading
    # YOLO-World / MediaPipe. Used by the API tests so they finish in <1s.
    stub_pipeline: bool = Field(default=False, alias="STUB_PIPELINE")

    # Versioning
    app_version: str = "1.0.0"


settings = Settings()

for _d in (settings.uploads_dir, settings.results_dir, settings.keyframes_dir):
    _d.mkdir(parents=True, exist_ok=True)
