# Runbook

Use this guide to set up, run, test, and debug the project locally.

## Requirements

- Python 3.10 or newer is recommended.
- A terminal with access to this repository.
- A side-view tennis video in `.mp4`, `.mov`, `.avi`, or `.mkv` format.

The Python dependencies are pinned in `requirements.txt`.

## First-Time Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The app currently does not read `.env` automatically. The file exists as a clear local settings template and future extension point.

## Start The Server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open these URLs:

- Web upload page: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Analyze A Video With curl

Upload a video:

```bash
curl -F "file=@/absolute/path/to/side-view-tennis-video.mp4" http://127.0.0.1:8000/api/analyze
```

The response contains a `job_id`:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

Poll status:

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

Fetch the final result after status is `completed`:

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>/result
```

Generated files are stored in `data/outputs/<job_id>/`.

## Run Tests

```bash
source .venv/bin/activate
python3 -m pytest -v
```

The test suite covers API behavior, job state transitions, metric scoring, and feedback generation.

## Common Issues

`Unsupported video extension`

Use a file ending in `.mp4`, `.mov`, `.avi`, or `.mkv`.

`Uploaded file is empty`

Check the file path passed to curl and confirm the video has content.

`Too few usable pose frames`

Use a clearer side-view recording. The full player should remain visible from preparation through follow-through.

`Analysis result is not ready`

The job is still queued or processing. Poll `GET /api/jobs/{job_id}` until status is `completed`.

Old `job_id` not found after restart

Job records are stored in memory. Restarting the server clears the API-visible job list, even though old files may still exist in `data/outputs/`.

Media file does not play in browser

Confirm `data/outputs/<job_id>/annotated.mp4` exists and the URL starts with `/data/outputs/...`. Some local codec combinations may not play in every browser even when analysis succeeded.
