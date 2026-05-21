from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_session, init_db
from .logging_config import configure_logging, get_logger
from .models import Task
from .schemas import (
    ErrorOut,
    HealthOut,
    ResultPayload,
    SOPCheckIn,
    SOPCheckOut,
    TaskCreatedOut,
    TaskListOut,
    TaskStatusOut,
    TaskSummary,
)
from .sop import evaluate as _sop_evaluate
from .storage import keyframe_path, read_json, result_path, upload_path
from .tasks import process_video

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    if not settings.skip_model_warmup:
        # Phase 1: model warmup is a no-op; Phase 2 will swap in the detector.
        log.info("startup_complete skip_model_warmup=%s", settings.skip_model_warmup)
    yield


app = FastAPI(
    title="witness",
    description=(
        "Watches a procedure video. Detects objects with open-vocab labels, "
        "tracks them, classifies motion, and identifies which objects the "
        "person interacted with."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)


# --------- error handlers ---------


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    code = None
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code")
        detail = detail.get("detail", "")
    return JSONResponse(status_code=exc.status_code, content={"detail": detail, "code": code})


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "code": "VALIDATION_ERROR", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "INTERNAL_ERROR"},
    )


# --------- routes ---------


@app.get("/health", response_model=HealthOut, tags=["meta"])
async def health() -> HealthOut:
    return HealthOut(version=settings.app_version, model=settings.detector_model)


@app.post(
    "/tasks",
    response_model=TaskCreatedOut,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorOut}},
    tags=["tasks"],
)
def create_task(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Video file (.mp4 .mov .avi .mkv).")],
    session: Annotated[Session, Depends(get_session)],
) -> TaskCreatedOut:
    if not file.filename:
        raise HTTPException(400, detail={"detail": "Missing filename", "code": "INVALID_FILE"})

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            400,
            detail={
                "detail": f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}",
                "code": "INVALID_FILE",
            },
        )

    task = Task(
        original_filename=file.filename,
        input_path="",  # filled in after we know the task_id
    )
    session.add(task)
    session.flush()  # populate task.id

    dest = upload_path(task.id, ext)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
    finally:
        file.file.close()

    size = dest.stat().st_size
    if size == 0:
        dest.unlink(missing_ok=True)
        session.rollback()
        raise HTTPException(400, detail={"detail": "Empty upload", "code": "INVALID_FILE"})
    if size > settings.max_upload_bytes:
        dest.unlink(missing_ok=True)
        session.rollback()
        raise HTTPException(
            400,
            detail={
                "detail": f"File too large (limit {settings.max_upload_bytes} bytes)",
                "code": "FILE_TOO_LARGE",
            },
        )

    task.input_path = str(dest)
    task.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(task)

    if settings.run_inline:
        # Synchronous mode for deterministic tests.
        process_video(task.id)
    else:
        background_tasks.add_task(process_video, task.id)

    return TaskCreatedOut(
        task_id=task.id,
        status=task.status,  # type: ignore[arg-type]
        created_at=task.created_at,
    )


def _task_or_404(session: Session, task_id: str) -> Task:
    task = session.get(Task, task_id)
    if task is None:
        raise HTTPException(404, detail={"detail": "Task not found", "code": "TASK_NOT_FOUND"})
    return task


@app.get(
    "/tasks",
    response_model=TaskListOut,
    tags=["tasks"],
)
async def list_tasks(
    session: Annotated[Session, Depends(get_session)],
    limit: int = 20,
    offset: int = 0,
) -> TaskListOut:
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    total = session.scalar(select(func.count(Task.id))) or 0
    rows = session.execute(
        select(Task).order_by(Task.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()
    items = [
        TaskSummary(
            task_id=t.id,
            status=t.status,  # type: ignore[arg-type]
            original_filename=t.original_filename,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in rows
    ]
    return TaskListOut(items=items, total=total, limit=limit, offset=offset)


@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusOut,
    responses={404: {"model": ErrorOut}},
    tags=["tasks"],
)
async def get_task(
    task_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> TaskStatusOut:
    task = _task_or_404(session, task_id)
    return TaskStatusOut(
        task_id=task.id,
        status=task.status,  # type: ignore[arg-type]
        created_at=task.created_at,
        updated_at=task.updated_at,
        error_message=task.error_message,
        processing_time_seconds=task.processing_time_seconds,
        result_url=f"/tasks/{task.id}/result" if task.status == "DONE" else None,
    )


@app.get(
    "/tasks/{task_id}/result",
    response_model=ResultPayload,
    responses={
        404: {"model": ErrorOut},
        409: {"model": ErrorOut},
        500: {"model": ErrorOut},
    },
    tags=["tasks"],
)
async def get_task_result(
    task_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> ResultPayload:
    task = _task_or_404(session, task_id)
    if task.status == "FAILED":
        raise HTTPException(
            500,
            detail={
                "detail": task.error_message or "Task failed",
                "code": "TASK_FAILED",
            },
        )
    if task.status != "DONE":
        raise HTTPException(
            409,
            detail={
                "detail": f"Task is still {task.status}",
                "code": "TASK_NOT_DONE",
            },
        )
    path = result_path(task.id)
    if not path.exists():
        raise HTTPException(
            500,
            detail={"detail": "Result file missing", "code": "RESULT_MISSING"},
        )
    return ResultPayload.model_validate(read_json(path))


@app.post(
    "/tasks/{task_id}/check",
    response_model=SOPCheckOut,
    responses={404: {"model": ErrorOut}, 409: {"model": ErrorOut}, 400: {"model": ErrorOut}},
    tags=["tasks"],
)
async def check_sop_compliance(
    task_id: str,
    body: SOPCheckIn,
    session: Annotated[Session, Depends(get_session)],
) -> SOPCheckOut:
    """Run a set of SOP rules against this task's result trace.

    The body's `rules` field is plain-text DSL (one rule per line). See
    `app/sop.py` docstring for grammar. Returns per-rule pass/fail plus a
    summary. Read-only — does not mutate the task.
    """
    task = _task_or_404(session, task_id)
    if task.status != "DONE":
        raise HTTPException(
            409,
            detail={"detail": f"Task is still {task.status}", "code": "TASK_NOT_DONE"},
        )
    path = result_path(task.id)
    if not path.exists():
        raise HTTPException(500, detail={"detail": "Result file missing", "code": "RESULT_MISSING"})
    try:
        out = _sop_evaluate(body.rules, read_json(path))
    except ValueError as exc:
        raise HTTPException(
            400,
            detail={"detail": f"Rule parse error: {exc}", "code": "INVALID_RULES"},
        ) from None
    return SOPCheckOut.model_validate(out)


@app.get(
    "/tasks/{task_id}/keyframes/{filename}",
    responses={404: {"model": ErrorOut}},
    tags=["tasks"],
)
async def get_keyframe(task_id: str, filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, detail={"detail": "Invalid filename", "code": "INVALID_PATH"})
    path = keyframe_path(task_id, filename)
    if not path.exists():
        raise HTTPException(404, detail={"detail": "Keyframe not found", "code": "NOT_FOUND"})
    return FileResponse(path, media_type="image/jpeg")


# --------- static + index ---------

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
