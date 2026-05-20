# Contributing to witness

Thanks for taking a look — contributions are welcome.

## Dev setup

```
make setup        # creates .venv with the pinned deps
make test         # 34 pytest cases, ~1s
make verify       # runs the real ML pipeline against data/uploads/sample.mp4
make run          # FastAPI on :8000
```

Python 3.11 is required (MediaPipe wheel constraints).

## What kinds of contributions are useful

- **Bug reports** with a reproducer video (or describe the failure mode
  precisely if the video can't be shared).
- **New detectors / tracker backends.** The detector wrapper in
  `app/pipeline/detector.py` is intentionally small; a similar wrapper
  around Grounding DINO, OWLv2, or YOLOv9 would slot in cleanly.
- **Learned hand-object interaction model** to replace the geometric
  heuristic in `app/pipeline/interaction.py`. Stub interface should
  return the same per-frame fingertip-count signal the keyframe
  selector expects.
- **An SOP DSL** that consumes the output JSON and emits compliance
  pass/fail decisions. See README's "What I'd do with more time"
  section for the sketch.

## Code style

- Pure functions in `pipeline/motion.py` and `pipeline/interaction.py`
  should stay pure. They have no I/O on purpose — that's what makes the
  tests fast.
- New thresholds go in `app/config.py` (`pydantic-settings`), not
  hard-coded inline.
- Add a unit test for any math you change. The bar is "would I notice
  if this regressed silently?" — if yes, write the test.

## PR checklist

- [ ] `make test` is green
- [ ] If you touched the pipeline, `make verify` still produces a valid
      result on `data/uploads/sample.mp4` (schema validation will fail
      loudly if not)
- [ ] No new top-level dependencies without a one-line note in the PR
      explaining why

## Reporting a security issue

See [SECURITY.md](SECURITY.md). Please don't open public issues for
security problems.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
