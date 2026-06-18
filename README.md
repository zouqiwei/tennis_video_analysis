# Tennis Video Analysis Backend

Python backend for side-view tennis swing analysis. The MVP focuses on human pose heuristics and coaching feedback, not precise ball or racket tracking.

The app includes a small upload page, JSON API endpoints, generated annotated video output, and tests for the core backend behavior.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open:

- Web upload page: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Analyze A Video

```bash
curl -F "file=@/path/to/side-view-tennis-video.mp4" http://127.0.0.1:8000/api/analyze
```

Then poll:

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
curl http://127.0.0.1:8000/api/jobs/<job_id>/result
```

Generated files are written to `data/outputs/<job_id>/`.

## Project Map

```text
app/
  main.py       FastAPI app, routes, homepage, static file serving
  jobs.py       In-memory job records and status transitions
  analyzer.py   OpenCV and MediaPipe video analysis pipeline
  metrics.py    Landmark-based swing scoring heuristics
  feedback.py   Coaching feedback from metric scores
  models.py     Pydantic request and response models
tests/          API, job, metric, and feedback tests
docs/           Architecture notes, runbook, and implementation specs
data/           Local uploaded videos and generated analysis outputs
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Runbook](docs/RUNBOOK.md)
- [Backend design spec](docs/superpowers/specs/2026-06-18-tennis-video-analysis-design.md)
- [Implementation plan](docs/superpowers/plans/2026-06-18-tennis-video-analysis-backend.md)

## Tests

```bash
python3 -m pytest -v
```

## MVP Limits

- Best results require a stable side-view video with the full player visible.
- Swing phases are heuristic estimates.
- Ball, racket, and court-line tracking are not included in this version.

## API Responses

`POST /api/analyze`

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

`GET /api/jobs/{job_id}`

```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 42,
  "error": null
}
```

`GET /api/jobs/{job_id}/result`

```json
{
  "job_id": "uuid",
  "overall_score": 82.5,
  "metrics": [],
  "phases": [],
  "feedback": [],
  "annotated_video_path": "data/outputs/uuid/annotated.mp4",
  "key_frame_paths": []
}
```
