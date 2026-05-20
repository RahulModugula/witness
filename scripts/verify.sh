#!/usr/bin/env bash
# Runs the real ML pipeline against data/uploads/sample.mp4 and validates the
# output schema. Used as a "one command, end-to-end proof" for reviewers and as
# the basis for `make verify`.
set -euo pipefail

SAMPLE="${SAMPLE:-data/uploads/sample.mp4}"

if [[ ! -f "$SAMPLE" ]]; then
  echo "Sample video not found at $SAMPLE" >&2
  exit 1
fi

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

exec python -m scripts.run_pipeline "$SAMPLE"
