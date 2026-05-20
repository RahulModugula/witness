# witness

A small FastAPI service that watches a procedure video and tells you what
happened: which objects were on screen, which moved, and which the person
actually touched. The output is JSON plus annotated keyframes you can show
a reviewer.

Built as the SDE intern technical assessment for [Edrevel AI](https://edrevel.ai).
Edrevel's product checks whether regulated manufacturers follow installation
SOPs by analyzing recorded procedures, so the framing throughout this repo
is "minimum prototype of that loop" rather than "generic detection demo."

The bundled sample is an 8-second clip of a lab tech plugging a USB cable
into an Agilent Cary 60 spectrophotometer.

## Run it in 30 seconds

```
make setup            # one-time: venv + pip install
make verify           # runs the real ML pipeline on data/uploads/sample.mp4
```

Or drive it through HTTP:

```
make run              # FastAPI on :8000
make demo             # in another shell: curl through the API end-to-end
```

UI at `http://localhost:8000/`, Swagger at `/docs`.

## What you actually get

CPU-only, MacBook Pro M1 Max:

| metric                | value                                       |
|-----------------------|---------------------------------------------|
| pipeline wall time    | 25.0s on the 8s sample (~3x real-time)      |
| objects detected      | 10 (5 distinct classes)                     |
| interactions found    | 5 (technician ↔ spectrophotometer)          |
| keyframes saved       | 24 (motion transitions + interaction peaks) |
| tests                 | 34 pytest cases, ~0.8s                      |

A representative interaction-peak keyframe (frame 119, technician's hand
visibly on the instrument):

![interaction peak frame 119](docs/sample_interaction_frame_119.jpg)

Full annotated set is at `data/keyframes/<task_id>/` after a run. Three hero
frames and the full result JSON are committed under `docs/`.

## Architecture

```
   browser/curl  ─POST /tasks─▶  FastAPI
                                    │ BackgroundTasks
                                    ▼
                  ┌─────────────────────────────────────┐
                  │  YOLO-World v2  +  BoT-SORT tracker │
                  │              │                       │
                  │              ▼                       │
                  │  track_merge (dedupe fragmented IDs) │
                  │              │                       │
                  │              ▼                       │
                  │  MediaPipe Hands (21 landmarks)      │      SQLite
                  │              │                       │   ┌──────────┐
                  │              ▼                       │◀──┤  tasks   │
                  │  motion + interaction (pure fn)      │   └──────────┘
                  │              │                       │
                  │              ▼                       │      data/results/
                  │  assemble + pydantic-validate ───────┼──▶  data/keyframes/
                  └─────────────────────────────────────┘
```

What each module does (`app/`):

| file | role |
|------|------|
| `main.py` | routes, error handlers, lifespan model warmup |
| `tasks.py` | background-task entry, status transitions |
| `db.py`, `models.py` | SQLAlchemy 2.x, single `tasks` table |
| `schemas.py` | Pydantic IO + ResultPayload |
| `pipeline/detector.py` | YOLO-World wrapper, lazy singleton |
| `pipeline/pose.py` | MediaPipe Hands wrapper |
| `pipeline/track_merge.py` | post-process: collapse same-class duplicate tracks |
| `pipeline/motion.py` | pure-fn motion classification |
| `pipeline/interaction.py` | pure-fn interaction detection |
| `pipeline/keyframes.py` | pick + annotate + write JPGs |
| `pipeline/assemble.py` | glue everything into the final JSON |

## API

All responses are JSON. Errors share one shape: `{"detail": "...", "code": "..."}`.

| method | path | what it does |
|--------|------|--------------|
| POST | `/tasks` | upload a video (multipart `file`), returns 201 with task_id |
| GET | `/tasks` | paginated list (`?limit=20&offset=0`) |
| GET | `/tasks/{id}` | task status (PENDING / PROCESSING / DONE / FAILED) |
| GET | `/tasks/{id}/result` | full result. 409 if not done, 500 if failed |
| GET | `/tasks/{id}/keyframes/{filename}` | serves a saved JPG |
| GET | `/health` | `{status, version, model}` |
| GET | `/docs` | auto Swagger UI |
| GET | `/` | the upload UI |

OpenAPI spec is committed at [`docs/openapi.json`](docs/openapi.json) so you
can read the contract without booting the server.

Curl example:

```
curl -X POST -F "file=@data/uploads/sample.mp4" http://localhost:8000/tasks
# {"task_id":"<uuid>","status":"PENDING",...}

curl http://localhost:8000/tasks/<uuid>/result | jq .
```

## Result schema

Matches the brief's required fields, plus a few additive ones I needed for
the keyframe URLs and the latency story:

```json
{
  "videoMetadata": {
    "filename": "sample.mp4",
    "duration_seconds": 8.0, "frame_count": 192, "fps": 24.0,
    "width": 1280, "height": 720, "codec": "h264",
    "processing_time_seconds": 25.0,
    "model": "yolov8s-worldv2.pt", "tracker": "botsort",
    "degraded_interaction": false
  },
  "objectsDetected": [
    {
      "object_id": 2,
      "class": "spectrophotometer",
      "confidence_avg": 0.47,
      "first_seen_frame": 0, "last_seen_frame": 191,
      "motion_history": [
        {"frame_range": [0, 140], "state": "stationary"},
        {"frame_range": [141, 150], "state": "moving"},
        {"frame_range": [151, 191], "state": "stationary"}
      ],
      "interactions": [
        {"interacted_by_person": 0, "frame_start": 12, "frame_end": 18},
        {"interacted_by_person": 0, "frame_start": 113, "frame_end": 124}
      ]
    }
  ],
  "keyFrames": [
    {
      "frame_number": 119, "type": "interaction_peak", "object_id": 2,
      "image_path": "keyframes/<task_id>/obj2_interaction_119.jpg",
      "url": "/tasks/<task_id>/keyframes/obj2_interaction_119.jpg"
    }
  ]
}
```

Schema validation happens on every write (`app/schemas.py::ResultPayload`).

## How it works

### Detection: YOLO-World v2

COCO doesn't have `spectrophotometer` or `cable` as classes. YOLO-World lets
you pass a list of plain-English class prompts at runtime and matches CLIP
embeddings to image regions. I use `yolov8s-worldv2.pt` (the v2 small
weights). Classes get set once at construction; calling `set_classes`
repeatedly during inference hits a torch version-counter bug in ultralytics
8.3.x.

### Tracking: BoT-SORT

Ultralytics' default since YOLO11. The tuned config is at
`app/pipeline/botsort_tuned.yaml`:
- `track_buffer: 60` (default 30) so a brief hand occlusion doesn't kill the track
- `match_thresh: 0.7` (default 0.8) for easier re-ID after that occlusion

`persist=True` keeps IDs alive across the streamed inference.

### Dedupe fragmented tracks

Open-vocab prompting often produces two slightly different bboxes for the
same object when you list synonyms (`spectrophotometer` and `scientific
instrument`). BoT-SORT then gives them separate IDs. `track_merge.py` walks
the tracker output and unions any same-class tracks whose mean bboxes have
IoU >= 0.3, using the lower track ID as root. Same-frame collisions get
NMS-resolved. Six tests pin the invariants.

This cuts the raw 16+ tracks on the sample down to 10 stable objects.

### Motion classification

`pipeline/motion.py` is the rubric's "easily understandable mathematical
helper functions" line item. Per track:

1. Compute per-frame centroid + bbox diagonal
2. Mean centroid displacement over a 5-frame window, normalized by bbox
   diagonal (makes it scale-invariant)
3. Compare to threshold 0.015 → per-frame moving / stationary
4. Collapse into intervals with hysteresis `min_run=3`: any state change
   that doesn't persist for >=3 frames gets absorbed into its neighbor.
   This kills flicker.
5. Gap-fill: missing frames inherit the prior state, so the output covers
   `[0, frame_count-1]` exactly with no holes

All knobs live in `app/config.py`.

### Interaction detection

MediaPipe Hands returns 21 landmarks per detected hand. Fingertips
(indices 4, 8, 12, 16, 20) are the physical contact points, so that's the
primary signal. A frame counts as interacting when:

1. At least one fingertip lies inside the object's expanded bbox
   (expand=1.15 for a small near-miss tolerance)
2. The hand's wrist landmark lies inside the person's bbox — this is the
   ownership check, prevents one person's hand getting attributed to another

Same hysteresis as motion. Per-frame fingertip counts also drive keyframe
selection.

Fallback: if MediaPipe finds zero hands across the whole video (rare),
drop to `IoU(person, object) > 0.05` and set `degraded_interaction: true`.
On the sample this never fires — hands are found in 180/192 frames.

### Keyframes

Two kinds, per object:
- `motion_transition`: first frame of each new motion-state interval
- `interaction_peak`: frame inside each interaction interval with the
  highest fingertip-in-bbox count

Each saved JPG is annotated with the tracked bbox (green for interactions,
amber for motion) and a label like `spectrophotometer (id=2) <-> person
id=0`. So you can look at the keyframe and immediately see what was
detected, without cross-referencing the JSON.

### Background tasks + storage

`BackgroundTasks` runs the pipeline after the response returns. SQLite via
SQLAlchemy 2.x with the `Mapped` / `mapped_column` style. Single `tasks`
table. Result JSONs live on disk, only the path goes in the DB.

The detector and pose extractor are module-level singletons warmed up in
FastAPI's lifespan hook, so the ~3s model load happens at startup, not on
the first request.

## Stuff that doesn't work (and what I tried)

This is the more honest section. Two failure modes I hit during the build,
why they happened, and what I'd do if I had a second day.

### Cable isn't detected on this sample

The brief example shows a `cable` class. Visually the technician is
plugging a USB cable into the spectrophotometer. But `yolov8s-worldv2`
fails to detect any cable across all 192 frames, even at `conf=0.03`, even
with 18 different prompts I exhaustively tried (`cable`, `usb cable`,
`wire`, `cord`, `plug`, `connector`, `tube`, `hose`, `data cable`,
`black cable`, etc.).

Why: the sample is a synthetic Veo-generated clip (you can see the Veo
watermark). YOLO-World v2 small was trained on real photos and its open-
vocab embedding underperforms on synthetic lab footage of thin objects
partially occluded by a hand.

What I tried:
- **Bigger model.** `yolov8m-worldv2` does detect the cable (149 hits, max
  conf 0.835) using the visually-specific prompt `"blue usb connector"`,
  and also finds the robotic arm and monitor that S misses.
- **Hybrid prompt sets** across both models.

Why I shipped with `-s` anyway: with M, spectrophotometer recall is
concentrated in late frames (155+), missing the early-video frames where
the technician's hands actually contact the instrument. Net result was
**0 real interactions**, which is a way worse headline number for an
*interaction detection* assignment than "cable not detected". I picked the
metric that matters most for the brief and documented the trade here.

The probe scripts are in the repo for transparency:
- `scripts/probe_cable.py` — exhaustive prompt search at low conf
- `scripts/probe_model.py` — S vs M side-by-side
- `scripts/probe_interaction.py` — debugs the wrist-in-bbox check

### Two spectrophotometer track IDs survive merging

`track_merge` collapses 6+ raw spec tracks down to 2 (id=1 and id=2). The
two that remain have low mean-bbox IoU because they're slightly different
crops from competing CLIP prompts. Lowering the IoU threshold further
would risk merging legitimately distinct objects in multi-instrument
scenes. The track with the actual interactions (id=2) cleanly captures
all 5 of them, so it didn't hurt the output. Real fix in production is a
learned re-ID head, not a geometric merge.

### Wrist-in-bbox ownership check has an edge case

In wide shots MediaPipe can detect a hand whose wrist lands outside the
YOLO person bbox (person partly off-screen). The ownership check then
orphans the hand. On this sample the probe found 201 correct ownerships
and 0 orphans so it isn't biting, but in production I'd add an
`IoU(hand_bbox, person_bbox)` fallback.

## Tests

```
make test       # 34 passed in ~0.8s
```

| file | what it covers |
|------|----------------|
| `test_motion.py` (9) | centroid math, hysteresis, gap-fill, full-range coverage |
| `test_interaction.py` (11) | fingertip-in-bbox, wrist ownership, min-run, IoU fallback |
| `test_track_merge.py` (6) | same-class merge, cross-class isolation, NMS resolution |
| `test_assemble.py` (1) | full payload passes the Pydantic schema |
| `test_api.py` (7) | upload + lifecycle + 4xx paths via TestClient |

API tests use `STUB_PIPELINE=1` so they don't need the YOLO weights — the
real ML path is exercised separately by `make verify`. `RUN_INLINE=1`
makes the background task run synchronously inside the request handler so
the tests are deterministic.

Latest transcript: [`docs/test_run.txt`](docs/test_run.txt). CI at
`.github/workflows/test.yml` runs the same suite on every push to `main`.

## Trade-offs I made

A few things I'd do differently in a non-take-home version:

- **BackgroundTasks over Celery/ARQ.** Right for a single-process demo,
  wrong for production. Upgrade path is ARQ (async-native, Redis-backed,
  pairs cleanly with FastAPI) before reaching for Celery.
- **SQLite over Postgres.** One file, no external deps. Production wants
  Postgres + Alembic.
- **Class list set at startup, not per-request.** Repeated `set_classes`
  hits a torch bug. If dynamic vocab per request is ever needed, the
  right shape is one worker process per active vocabulary behind a router.
- **Geometric interaction heuristic.** Defensible for a one-day build but
  a learned HOI model (100DOH-style) would correctly distinguish
  "gripping" from "near". Concrete replacement target.
- **JPG q=90 keyframes** (~140KB each). PNG would be 5-10x larger for no
  visible gain on these synthetic frames.
- **CPU-only.** GPU would be ~3-5x faster but the brief explicitly avoids
  GPU assumptions.

What I deliberately left out: Docker, auth, CORS, WebSockets, Alembic,
multiple workers. Each would be a positive in production. Putting them in
a take-home signals can't-scope.

## What I'd do with more time (Edrevel-flavored)

Mapped to your product, not generic engineering polish:

1. **Fine-tune YOLO-World on a small lab dataset** (PPE, common
   instruments, tools). Directly fixes the cable miss above and lifts
   recall on real customer footage. Maybe a day of label work + a few
   hours training.
2. **Replace the geometric heuristic with a learned HOI model.** The
   distinction between "fingertip *near* an object" and "fingertip
   *gripping* an object" matters for SOP scoring.
3. **SOP DSL on top of the JSON output.** The current payload is a
   *trace*; compliance is a *check*. A small DSL like
   `interaction(person, cable) BEFORE interaction(person, spectrophotometer)`
   would let SOPs be authored declaratively against this trace.
4. **WebSocket progress updates** instead of polling. Better UX for
   longer videos.
5. **Per-object segmentation masks** (YOLO-World has a `-seg` variant) for
   pixel-precise interaction tests instead of bbox proximity.
6. **Worker pool + Postgres + ARQ + Docker Compose** for real deployment.
7. **Audit-log immutability.** Once a procedure trace is generated for a
   regulated customer it should be append-only and signed. Aligns with
   the FDA 21 CFR Part 11 expectations life-sciences customers will have.

## Stack

```
fastapi[standard]==0.115.6   uvicorn[standard]==0.32.1
sqlalchemy==2.0.36           pydantic==2.10.4    pydantic-settings==2.7.0
ultralytics==8.3.55          opencv-python-headless==4.10.0.84
mediapipe==0.10.18           numpy==1.26.4       pillow==11.0.0
pytest==8.3.4                pytest-asyncio==0.25.0   httpx==0.28.1
```

Python 3.11 (MediaPipe wheels are reliable on 3.11, sometimes flakey on
newer). CPU-only. No paid APIs. Model weights download on first run
(~25 MB for the S model).

## Project layout

```
witness/
├── app/
│   ├── main.py             # FastAPI routes + handlers
│   ├── tasks.py            # background pipeline entry
│   ├── config.py           # pydantic-settings: thresholds, prompts, paths
│   ├── db.py, models.py    # SQLAlchemy 2.x
│   ├── schemas.py
│   ├── storage.py, logging_config.py
│   └── pipeline/
│       ├── detector.py     # YOLO-World + BoT-SORT
│       ├── pose.py         # MediaPipe Hands
│       ├── track_merge.py
│       ├── motion.py       # pure-fn motion logic
│       ├── interaction.py  # pure-fn interaction logic
│       ├── keyframes.py    # pick + annotate + save
│       ├── assemble.py     # glue
│       ├── video.py
│       └── botsort_tuned.yaml
├── static/                 # drag-and-drop upload UI
├── tests/                  # 34 cases
├── scripts/
│   ├── verify.sh, run_pipeline.py    # `make verify` entry
│   ├── demo.sh             # curl through the HTTP API
│   ├── dump_openapi.py
│   └── probe_*.py          # exploratory scripts that informed the choices
├── docs/
│   ├── openapi.json
│   ├── sample_result.json
│   ├── sample_interaction_frame_*.jpg
│   └── test_run.txt
├── data/uploads/sample.mp4
├── .github/workflows/test.yml
├── Makefile, requirements.txt, pyproject.toml
├── LICENSE
└── README.md
```

## Time spent

About 7 hours: ~4 on the pipeline (detector, tracker, motion, interaction,
keyframes, assembly), ~1.5 on the API + DB + tests, ~1.5 on the empirical
investigation of detection quality and this README.

## License

MIT. See [LICENSE](LICENSE).
