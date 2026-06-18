# Tennis Video Analysis Backend

Python backend for side-view tennis swing analysis. The MVP focuses on human pose heuristics and coaching feedback, not precise ball or racket tracking.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Open API docs at `http://127.0.0.1:8000/docs`.

## Analyze A Video

```bash
curl -F "file=@/path/to/side-view-tennis-video.mp4" http://127.0.0.1:8000/api/analyze
```

Then poll:

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
curl http://127.0.0.1:8000/api/jobs/<job_id>/result
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

## Tests

```bash
python3 -m pytest -v
```
