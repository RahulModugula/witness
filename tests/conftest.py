from __future__ import annotations

import os
from pathlib import Path

import pytest


# Run process_video inline (synchronously inside the request) for deterministic
# end-to-end tests. The API tests need this; pure-math tests don't care.
os.environ.setdefault("RUN_INLINE", "1")
os.environ.setdefault("SKIP_MODEL_WARMUP", "1")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_video(project_root: Path) -> Path:
    p = project_root / "data" / "uploads" / "sample.mp4"
    if not p.exists():
        pytest.skip(f"sample video missing at {p}")
    return p


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Point the app at a throw-away SQLite + temp data dirs.

    Must run before any `from app.main import app` import in the test, otherwise
    the engine binds to the real DB. Used by the API tests that fully spin up
    the application.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("EDREVEL_DATABASE_URL", f"sqlite:///{db_path}")
    yield db_path


@pytest.fixture()
def client(monkeypatch, tmp_path, sample_video):
    """FastAPI TestClient with isolated SQLite + data dirs and RUN_INLINE on.

    We import the app *after* env vars are set so the Settings singleton picks
    them up.
    """
    db_path = tmp_path / "test.db"
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    keyframes = tmp_path / "keyframes"
    for d in (uploads, results, keyframes):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("EDREVEL_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("EDREVEL_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("EDREVEL_RESULTS_DIR", str(results))
    monkeypatch.setenv("EDREVEL_KEYFRAMES_DIR", str(keyframes))
    monkeypatch.setenv("RUN_INLINE", "1")
    monkeypatch.setenv("SKIP_MODEL_WARMUP", "1")

    # Force a fresh import so the new settings + engine bind to the temp paths.
    import importlib
    import sys

    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]
    app_module = importlib.import_module("app.main")

    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c
