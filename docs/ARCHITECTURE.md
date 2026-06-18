# Architecture

This project is a FastAPI backend for analyzing side-view tennis swing videos. It accepts a video upload, runs a heuristic pose analysis pipeline, and returns a JSON report plus generated media files.

## Runtime Flow

1. A user opens `GET /` or the interactive API docs at `GET /docs`.
2. The browser or API client uploads a video to `POST /api/analyze`.
3. `app.main` validates the file extension and stores the upload in `data/uploads/<job_id>.<ext>`.
4. `JobStore` creates an in-memory job record with status and progress.
5. A FastAPI background task calls `VideoAnalyzer.analyze(...)`.
6. `VideoAnalyzer` samples frames with OpenCV, detects body landmarks with MediaPipe, writes an annotated video, saves a few key frames, and writes `result.json`.
7. Clients poll `GET /api/jobs/{job_id}` until the job is completed or failed.
8. Clients fetch `GET /api/jobs/{job_id}/result` for the final report.

## Main Modules

`app/main.py`

- Defines the FastAPI app, homepage, API routes, static file serving, upload validation, and background job orchestration.
- Mounts `/data` so generated videos and key frames can be opened from the browser.

`app/jobs.py`

- Stores job state in memory.
- Creates upload and output paths for each job.
- Tracks status transitions: `queued`, `processing`, `completed`, and `failed`.

`app/analyzer.py`

- Runs the video analysis pipeline.
- Uses OpenCV for video I/O and MediaPipe Pose for body landmarks.
- Produces `annotated.mp4`, selected key frames, and `result.json`.

`app/metrics.py`

- Converts landmark sequences into heuristic swing metrics.
- Estimates contact frame, visibility, posture, backswing, follow-through, weight transfer, and rotation indicators.

`app/feedback.py`

- Converts metric scores into readable coaching feedback.

`app/models.py`

- Defines Pydantic models for jobs, API responses, metric scores, phases, and reports.

## Data Layout

Runtime files are stored under `data/`.

```text
data/
  uploads/
    <job_id>.mp4
  outputs/
    <job_id>/
      annotated.mp4
      result.json
      keyframes/
        frame_000000.jpg
```

The current `JobStore` is in-memory. Restarting the server keeps files on disk but clears job state, so old `job_id` values cannot be queried through the API after restart unless persistence is added.

## API Surface

`POST /api/analyze`

- Multipart form field: `file`
- Allowed extensions: `.mp4`, `.mov`, `.avi`, `.mkv`
- Returns a `job_id` and initial status.

`GET /api/jobs/{job_id}`

- Returns status, progress, and error information.

`GET /api/jobs/{job_id}/result`

- Returns the final analysis report.
- Returns HTTP 409 while the job is not completed.

## Known Limits

- Best results require a stable side-view recording with the full player visible.
- Swing phases are heuristic estimates based on pose landmarks.
- Ball, racket, and court-line tracking are not included.
- Job state is not persisted across server restarts.
- Long videos can take noticeable CPU time because processing runs in the local Python process.
