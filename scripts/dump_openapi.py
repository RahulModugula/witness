"""Write the live FastAPI app's OpenAPI schema to docs/openapi.json so it can
be reviewed without running the server. Used by `make openapi`."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings


def main() -> int:
    # Skip ML warmup during the OpenAPI dump — we don't need the model loaded.
    settings.skip_model_warmup = True
    from app.main import app

    schema = app.openapi()
    out = Path("docs/openapi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2))
    print(f"wrote {out} ({len(json.dumps(schema))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
