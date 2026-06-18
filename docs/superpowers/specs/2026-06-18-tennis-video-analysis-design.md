# Tennis Video Analysis Backend Design

Date: 2026-06-18

## Goal

Build a Python backend that analyzes side-view tennis swing videos. The first version focuses on human motion analysis and training feedback, not precise ball or racket tracking.

## Scope

The backend accepts an uploaded video, processes it asynchronously, extracts pose landmarks with MediaPipe, writes annotated artifacts, and returns a structured JSON report.

Included:

- FastAPI HTTP API.
- Video upload endpoint.
- Background analysis jobs.
- Job status and result endpoints.
- OpenCV frame processing.
- MediaPipe pose landmark detection.
- Annotated output video.
- Key frame images.
- JSON action report with scores and feedback.

Excluded from MVP:

- Accurate tennis ball trajectory tracking.
- Racket detection.
- Court-line detection.
- User accounts.
- Frontend UI.
- Production queue infrastructure such as Redis or Celery.

## Assumptions

- Input videos are side-view recordings of one player.
- The full player body should be visible for most frames.
- The camera is mostly stable.
- The backend runs locally or on a single server.
- MVP analysis can use geometric heuristics instead of trained tennis-specific models.

## API Design

### `POST /api/analyze`

Accepts a multipart video upload.

Response:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### `GET /api/jobs/{job_id}`

Returns job status.

Statuses:

- `queued`
- `processing`
- `completed`
- `failed`

Response:

```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 42,
  "error": null
}
```

### `GET /api/jobs/{job_id}/result`

Returns the final report when a job is complete.

Response contains:

- Overall score.
- Per-metric scores.
- Swing phase estimates.
- Key frame paths.
- Annotated video path.
- Human-readable coaching feedback.

## Components

### API Layer

FastAPI routes validate input, create jobs, expose status, and return reports. The API layer should not contain video analysis logic.

### Job Store

An in-memory job registry is enough for MVP. Each job stores:

- ID.
- Status.
- Progress.
- Input video path.
- Output directory.
- Result path.
- Error message.

The design keeps this behind a small interface so it can later be replaced by SQLite, Redis, or Celery.

### Video Analyzer

The analyzer reads video frames with OpenCV, runs MediaPipe pose detection, stores normalized landmarks per sampled frame, and writes an annotated video.

Frame sampling should default to every 3 frames to reduce processing time while keeping enough motion detail.

### Swing Analysis

The MVP uses pose-landmark heuristics:

- Visibility score: percentage of frames with usable shoulders, hips, knees, ankles, elbows, and wrists.
- Ready posture: knee bend and torso balance near the opening segment.
- Backswing range: wrist movement behind or away from the torso across the first half of the motion.
- Contact posture estimate: frame with maximum lead wrist speed, used as a proxy for impact.
- Follow-through: wrist continuation after the contact estimate.
- Weight transfer proxy: hip-center movement across the swing.
- Shoulder-hip separation proxy: angle difference between shoulder line and hip line.

These metrics should be described as estimates in the report.

### Feedback Generator

The feedback generator converts metrics into concise coaching notes. Example categories:

- Body visibility.
- Lower-body loading.
- Swing preparation.
- Contact position.
- Follow-through.
- Rotation and balance.

Feedback should avoid claiming precise biomechanical diagnosis.

## File Layout

Planned project structure:

```text
.
├── app/
│   ├── main.py
│   ├── models.py
│   ├── jobs.py
│   ├── analyzer.py
│   ├── metrics.py
│   └── feedback.py
├── data/
│   ├── uploads/
│   └── outputs/
├── tests/
│   ├── test_metrics.py
│   └── test_feedback.py
├── requirements.txt
└── README.md
```

## Error Handling

- Reject missing files and unsupported extensions.
- Mark jobs as `failed` if video decoding fails.
- Return `404` for unknown job IDs.
- Return `409` if result is requested before completion.
- Include a clear error message for videos with too few usable pose frames.

## Testing

Unit tests should cover:

- Metric calculations from synthetic landmark sequences.
- Feedback generation thresholds.
- Job store status transitions.

Manual verification should cover:

- Running the FastAPI app.
- Uploading a short test video.
- Confirming JSON output and artifact paths.

## Risks

- MediaPipe may fail if the player is partially out of frame.
- Side-view assumptions do not generalize to rear-view videos.
- Heuristic swing phase detection is approximate.
- Processing large videos synchronously in local background tasks can be slow.

## Acceptance Criteria

- A user can start the server with documented commands.
- A user can upload a side-view tennis video to `/api/analyze`.
- The backend creates a job and processes it without blocking the API response.
- The backend returns job status and final JSON analysis.
- Completed jobs include an annotated video path and key frame paths.
- The code has focused tests for metrics, feedback, and job state.
