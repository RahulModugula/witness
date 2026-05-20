PY ?= python3.11
VENV := .venv
ACT := . $(VENV)/bin/activate

.PHONY: setup run test verify demo openapi clean help

help:
	@echo "Targets:"
	@echo "  setup    - create .venv and install dependencies"
	@echo "  run      - start the FastAPI server on :8000"
	@echo "  test     - run pytest suite"
	@echo "  verify   - run the real ML pipeline against data/uploads/sample.mp4"
	@echo "             (end-to-end proof: detects, tracks, classifies, finds interactions)"
	@echo "  demo     - curl through the HTTP API against sample.mp4 (server must be running)"
	@echo "  openapi  - regenerate docs/openapi.json from the live FastAPI app"
	@echo "  clean    - remove generated data + caches"

setup:
	$(PY) -m venv $(VENV)
	$(ACT) && pip install --upgrade pip
	$(ACT) && pip install -r requirements.txt

run:
	$(ACT) && uvicorn app.main:app --reload --port 8000

test:
	$(ACT) && pytest

verify:
	$(ACT) && ./scripts/verify.sh

demo:
	./scripts/demo.sh

openapi:
	$(ACT) && python -m scripts.dump_openapi

clean:
	rm -rf data/results/* data/keyframes/* *.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
