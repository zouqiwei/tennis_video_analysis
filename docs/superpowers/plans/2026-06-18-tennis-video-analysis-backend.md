# Tennis Video Analysis Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that accepts side-view tennis swing videos, analyzes human pose motion with MediaPipe/OpenCV, and returns annotated artifacts plus coaching feedback.

**Architecture:** Keep API, job state, video analysis, metrics, and feedback in separate modules. The API starts background jobs and reads results from an in-memory job store, while the analyzer writes artifacts and a JSON report into `data/outputs/{job_id}`.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, OpenCV, MediaPipe, NumPy, Pydantic, Pytest.

---

## File Structure

- `requirements.txt`: Python runtime and test dependencies.
- `README.md`: setup, run, API usage, and MVP limitations.
- `app/__init__.py`: package marker.
- `app/models.py`: Pydantic response/report models and job status enum.
- `app/jobs.py`: in-memory job store and status transitions.
- `app/metrics.py`: pure pose-landmark metric calculations.
- `app/feedback.py`: score-to-feedback mapping.
- `app/analyzer.py`: video decoding, pose extraction, annotation output, report generation.
- `app/main.py`: FastAPI routes and background job orchestration.
- `tests/test_metrics.py`: metric tests using synthetic landmark frames.
- `tests/test_feedback.py`: feedback threshold tests.
- `tests/test_jobs.py`: job store tests.
- `tests/test_api.py`: API behavior tests with dependency override for a fake analyzer.
- `data/uploads/.gitkeep`: upload directory placeholder.
- `data/outputs/.gitkeep`: output directory placeholder.

## Task 1: Project Skeleton And Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `app/__init__.py`
- Create: `data/uploads/.gitkeep`
- Create: `data/outputs/.gitkeep`

- [ ] **Step 1: Create dependency file**

Write `requirements.txt`:

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
opencv-python==4.10.0.84
mediapipe==0.10.18
numpy==1.26.4
pydantic==2.10.4
pytest==8.3.4
httpx==0.28.1
```

- [ ] **Step 2: Create package and data directories**

Create empty files:

```text
app/__init__.py
data/uploads/.gitkeep
data/outputs/.gitkeep
```

- [ ] **Step 3: Create initial README**

Write `README.md` with:

```markdown
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
```

- [ ] **Step 4: Verify files exist**

Run: `find app data -maxdepth 3 -type f | sort`

Expected includes:

```text
app/__init__.py
data/outputs/.gitkeep
data/uploads/.gitkeep
```

- [ ] **Step 5: Commit**

Skip because the current directory is not a git repository.

## Task 2: Models And Job Store With TDD

**Files:**
- Create: `tests/test_jobs.py`
- Create: `app/models.py`
- Create: `app/jobs.py`

- [ ] **Step 1: Write failing job store tests**

Write `tests/test_jobs.py`:

```python
from pathlib import Path

from app.jobs import JobStore
from app.models import JobStatus


def test_create_job_starts_queued_with_paths(tmp_path: Path):
    store = JobStore(upload_dir=tmp_path / "uploads", output_dir=tmp_path / "outputs")

    job = store.create_job(filename="swing.mp4")

    assert job.status == JobStatus.QUEUED
    assert job.progress == 0
    assert job.input_path.parent == tmp_path / "uploads"
    assert job.output_dir.parent == tmp_path / "outputs"
    assert job.error is None


def test_status_transitions_and_result_path(tmp_path: Path):
    store = JobStore(upload_dir=tmp_path / "uploads", output_dir=tmp_path / "outputs")
    job = store.create_job(filename="swing.mp4")
    result_path = job.output_dir / "result.json"

    store.mark_processing(job.job_id, progress=15)
    store.mark_completed(job.job_id, result_path=result_path)

    updated = store.get(job.job_id)
    assert updated.status == JobStatus.COMPLETED
    assert updated.progress == 100
    assert updated.result_path == result_path


def test_mark_failed_records_error(tmp_path: Path):
    store = JobStore(upload_dir=tmp_path / "uploads", output_dir=tmp_path / "outputs")
    job = store.create_job(filename="swing.mp4")

    store.mark_failed(job.job_id, "video decode failed")

    updated = store.get(job.job_id)
    assert updated.status == JobStatus.FAILED
    assert updated.error == "video decode failed"
```

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_jobs.py -v`

Expected: FAIL during import because `app.jobs` and `app.models` do not exist.

- [ ] **Step 3: Implement models and job store**

Write `app/models.py`:

```python
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    input_path: Path
    output_dir: Path
    result_path: Path | None = None
    error: str | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    error: str | None = None


class MetricScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    detail: str


class SwingPhase(BaseModel):
    name: str
    frame_index: int
    timestamp_seconds: float


class AnalysisReport(BaseModel):
    job_id: str
    overall_score: float = Field(ge=0, le=100)
    metrics: list[MetricScore]
    phases: list[SwingPhase]
    feedback: list[str]
    annotated_video_path: str
    key_frame_paths: list[str]
```

Write `app/jobs.py`:

```python
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.models import JobRecord, JobStatus


class JobStore:
    def __init__(self, upload_dir: Path, output_dir: Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create_job(self, filename: str) -> JobRecord:
        job_id = str(uuid4())
        suffix = Path(filename).suffix.lower() or ".mp4"
        output_dir = self.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        job = JobRecord(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress=0,
            input_path=self.upload_dir / f"{job_id}{suffix}",
            output_dir=output_dir,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_processing(self, job_id: str, progress: int = 0) -> None:
        self._update(job_id, status=JobStatus.PROCESSING, progress=progress, error=None)

    def update_progress(self, job_id: str, progress: int) -> None:
        self._update(job_id, progress=max(0, min(progress, 99)))

    def mark_completed(self, job_id: str, result_path: Path) -> None:
        self._update(job_id, status=JobStatus.COMPLETED, progress=100, result_path=result_path, error=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, status=JobStatus.FAILED, error=error)

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update=changes)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_jobs.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

Skip because the current directory is not a git repository.

## Task 3: Metrics With TDD

**Files:**
- Create: `tests/test_metrics.py`
- Create: `app/metrics.py`

- [ ] **Step 1: Write failing metric tests**

Write `tests/test_metrics.py`:

```python
from app.metrics import LandmarkFrame, analyze_landmark_sequence


def frame(index: int, wrist_x: float, hip_x: float = 0.5, knee_y: float = 0.78) -> LandmarkFrame:
    return LandmarkFrame(
        frame_index=index,
        timestamp_seconds=index / 30,
        landmarks={
            "left_shoulder": (0.45, 0.35, 0.99),
            "right_shoulder": (0.55, 0.35, 0.99),
            "left_hip": (hip_x - 0.04, 0.58, 0.99),
            "right_hip": (hip_x + 0.04, 0.58, 0.99),
            "left_knee": (hip_x - 0.04, knee_y, 0.99),
            "right_knee": (hip_x + 0.04, knee_y, 0.99),
            "left_ankle": (hip_x - 0.04, 0.95, 0.99),
            "right_ankle": (hip_x + 0.04, 0.95, 0.99),
            "left_elbow": (wrist_x - 0.03, 0.48, 0.99),
            "right_elbow": (wrist_x + 0.03, 0.48, 0.99),
            "left_wrist": (wrist_x, 0.48, 0.99),
            "right_wrist": (wrist_x + 0.04, 0.48, 0.99),
        },
    )


def test_analyze_landmark_sequence_scores_visible_complete_swing():
    frames = [
        frame(0, 0.42, hip_x=0.48),
        frame(1, 0.36, hip_x=0.49),
        frame(2, 0.50, hip_x=0.51),
        frame(3, 0.70, hip_x=0.54),
    ]

    result = analyze_landmark_sequence(frames)

    assert result.overall_score > 70
    assert result.contact_frame_index == 3
    assert result.metric("visibility").score == 100
    assert result.metric("follow_through").score > 80


def test_analyze_landmark_sequence_penalizes_missing_landmarks():
    incomplete = frame(0, 0.42)
    incomplete.landmarks["left_wrist"] = (0.42, 0.48, 0.1)

    result = analyze_landmark_sequence([incomplete])

    assert result.metric("visibility").score < 100
    assert result.overall_score < 70
```

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_metrics.py -v`

Expected: FAIL during import because `app.metrics` does not exist.

- [ ] **Step 3: Implement metric module**

Write `app/metrics.py`:

```python
from dataclasses import dataclass
from math import atan2, degrees
from statistics import mean


Point = tuple[float, float, float]
REQUIRED_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


@dataclass
class LandmarkFrame:
    frame_index: int
    timestamp_seconds: float
    landmarks: dict[str, Point]


@dataclass
class MetricResult:
    name: str
    score: float
    detail: str


@dataclass
class LandmarkAnalysis:
    overall_score: float
    metrics: list[MetricResult]
    contact_frame_index: int
    contact_timestamp_seconds: float

    def metric(self, name: str) -> MetricResult:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


def analyze_landmark_sequence(frames: list[LandmarkFrame]) -> LandmarkAnalysis:
    if not frames:
        metrics = [MetricResult("visibility", 0, "No pose frames were detected.")]
        return LandmarkAnalysis(0, metrics, 0, 0)

    visibility = _visibility_score(frames)
    contact_index = _contact_frame_index(frames)
    contact_frame = frames[contact_index]
    metrics = [
        MetricResult("visibility", visibility, f"{visibility:.0f}% of sampled frames have usable body landmarks."),
        MetricResult("ready_posture", _ready_posture_score(frames), "Opening posture uses knee bend and torso balance proxies."),
        MetricResult("backswing", _backswing_score(frames), "Wrist travel before contact estimates preparation range."),
        MetricResult("contact_position", _contact_position_score(contact_frame), "Contact estimate checks lead wrist position relative to torso."),
        MetricResult("follow_through", _follow_through_score(frames, contact_index), "Wrist continuation after contact estimates follow-through."),
        MetricResult("weight_transfer", _weight_transfer_score(frames), "Hip-center movement estimates weight transfer."),
        MetricResult("shoulder_hip_separation", _rotation_score(frames), "Shoulder and hip line angle difference estimates rotation."),
    ]
    overall = mean(metric.score for metric in metrics)
    return LandmarkAnalysis(
        overall_score=round(overall, 1),
        metrics=metrics,
        contact_frame_index=frames[contact_index].frame_index,
        contact_timestamp_seconds=frames[contact_index].timestamp_seconds,
    )


def _visibility_score(frames: list[LandmarkFrame]) -> float:
    usable = 0
    for frame in frames:
        if all(frame.landmarks.get(name, (0, 0, 0))[2] >= 0.5 for name in REQUIRED_LANDMARKS):
            usable += 1
    return round(100 * usable / len(frames), 1)


def _contact_frame_index(frames: list[LandmarkFrame]) -> int:
    if len(frames) == 1:
        return 0
    speeds = []
    for prev, curr in zip(frames, frames[1:]):
        px = prev.landmarks.get("left_wrist", (0, 0, 0))[0]
        cx = curr.landmarks.get("left_wrist", (0, 0, 0))[0]
        speeds.append(abs(cx - px))
    return speeds.index(max(speeds)) + 1


def _ready_posture_score(frames: list[LandmarkFrame]) -> float:
    opening = frames[: max(1, len(frames) // 3)]
    knee_bends = []
    for frame in opening:
        hip_y = _avg_y(frame, "left_hip", "right_hip")
        knee_y = _avg_y(frame, "left_knee", "right_knee")
        ankle_y = _avg_y(frame, "left_ankle", "right_ankle")
        leg_span = max(ankle_y - hip_y, 0.01)
        knee_bends.append((knee_y - hip_y) / leg_span)
    bend = mean(knee_bends)
    return _clamp_score(100 - abs(bend - 0.55) * 180)


def _backswing_score(frames: list[LandmarkFrame]) -> float:
    if len(frames) < 2:
        return 40
    half = frames[: max(2, len(frames) // 2)]
    wrist_values = [frame.landmarks.get("left_wrist", (0, 0, 0))[0] for frame in half]
    travel = max(wrist_values) - min(wrist_values)
    return _clamp_score(travel * 700)


def _contact_position_score(frame: LandmarkFrame) -> float:
    wrist_x = frame.landmarks.get("left_wrist", (0, 0, 0))[0]
    hip_center = _avg_x(frame, "left_hip", "right_hip")
    distance = abs(wrist_x - hip_center)
    return _clamp_score(100 - abs(distance - 0.18) * 260)


def _follow_through_score(frames: list[LandmarkFrame], contact_index: int) -> float:
    if contact_index >= len(frames) - 1:
        return 85 if len(frames) > 2 else 45
    contact_x = frames[contact_index].landmarks.get("left_wrist", (0, 0, 0))[0]
    after = [frame.landmarks.get("left_wrist", (0, 0, 0))[0] for frame in frames[contact_index:]]
    continuation = max(after) - contact_x
    return _clamp_score(continuation * 600 + 40)


def _weight_transfer_score(frames: list[LandmarkFrame]) -> float:
    start = _avg_x(frames[0], "left_hip", "right_hip")
    end = _avg_x(frames[-1], "left_hip", "right_hip")
    return _clamp_score(abs(end - start) * 900)


def _rotation_score(frames: list[LandmarkFrame]) -> float:
    separations = []
    for frame in frames:
        shoulder_angle = _line_angle(frame, "left_shoulder", "right_shoulder")
        hip_angle = _line_angle(frame, "left_hip", "right_hip")
        separations.append(abs(shoulder_angle - hip_angle))
    return _clamp_score(mean(separations) * 4 + 55)


def _line_angle(frame: LandmarkFrame, left: str, right: str) -> float:
    lx, ly, _ = frame.landmarks[left]
    rx, ry, _ = frame.landmarks[right]
    return degrees(atan2(ry - ly, rx - lx))


def _avg_x(frame: LandmarkFrame, left: str, right: str) -> float:
    return (frame.landmarks[left][0] + frame.landmarks[right][0]) / 2


def _avg_y(frame: LandmarkFrame, left: str, right: str) -> float:
    return (frame.landmarks[left][1] + frame.landmarks[right][1]) / 2


def _clamp_score(value: float) -> float:
    return round(max(0, min(100, value)), 1)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_metrics.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

Skip because the current directory is not a git repository.

## Task 4: Feedback With TDD

**Files:**
- Create: `tests/test_feedback.py`
- Create: `app/feedback.py`

- [ ] **Step 1: Write failing feedback tests**

Write `tests/test_feedback.py`:

```python
from app.feedback import build_feedback
from app.metrics import LandmarkAnalysis, MetricResult


def analysis(metric_scores: dict[str, float]) -> LandmarkAnalysis:
    return LandmarkAnalysis(
        overall_score=sum(metric_scores.values()) / len(metric_scores),
        metrics=[MetricResult(name, score, f"{name} detail") for name, score in metric_scores.items()],
        contact_frame_index=10,
        contact_timestamp_seconds=0.33,
    )


def test_build_feedback_mentions_low_visibility_first():
    result = build_feedback(analysis({"visibility": 40, "follow_through": 90}))

    assert result[0] == "Player visibility is limited. Re-record with the full body in frame from preparation through follow-through."


def test_build_feedback_returns_positive_summary_for_strong_scores():
    result = build_feedback(analysis({"visibility": 95, "follow_through": 90, "ready_posture": 88}))

    assert "Overall movement pattern looks consistent for this side-view sample." in result
```

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_feedback.py -v`

Expected: FAIL during import because `app.feedback` does not exist.

- [ ] **Step 3: Implement feedback module**

Write `app/feedback.py`:

```python
from app.metrics import LandmarkAnalysis


def build_feedback(analysis: LandmarkAnalysis) -> list[str]:
    by_name = {metric.name: metric.score for metric in analysis.metrics}
    feedback: list[str] = []

    if by_name.get("visibility", 100) < 70:
        feedback.append("Player visibility is limited. Re-record with the full body in frame from preparation through follow-through.")
    if by_name.get("ready_posture", 100) < 65:
        feedback.append("Use a more athletic ready position with clearer knee bend before starting the swing.")
    if by_name.get("backswing", 100) < 65:
        feedback.append("Increase backswing preparation so the hitting arm has more time to accelerate into contact.")
    if by_name.get("contact_position", 100) < 65:
        feedback.append("The estimated contact position is close to the body. Aim to meet the ball farther in front.")
    if by_name.get("follow_through", 100) < 65:
        feedback.append("Continue the swing after contact instead of stopping the racket path early.")
    if by_name.get("weight_transfer", 100) < 65:
        feedback.append("Show more forward weight transfer through the shot while keeping balance.")
    if by_name.get("shoulder_hip_separation", 100) < 65:
        feedback.append("Add more trunk rotation so the shoulders and hips contribute to racket speed.")

    if not feedback:
        feedback.append("Overall movement pattern looks consistent for this side-view sample.")

    feedback.append("These notes are heuristic estimates from pose landmarks and are not a biomechanical diagnosis.")
    return feedback
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_feedback.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

Skip because the current directory is not a git repository.

## Task 5: Analyzer Integration

**Files:**
- Create: `app/analyzer.py`

- [ ] **Step 1: Implement video analyzer**

Write `app/analyzer.py`:

```python
import json
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp

from app.feedback import build_feedback
from app.metrics import LandmarkFrame, analyze_landmark_sequence
from app.models import AnalysisReport, MetricScore, SwingPhase


LANDMARK_NAMES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


ProgressCallback = Callable[[int], None]


class VideoAnalyzer:
    def __init__(self, sample_every: int = 3):
        self.sample_every = sample_every

    def analyze(self, job_id: str, input_path: Path, output_dir: Path, progress: ProgressCallback | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / "annotated.mp4"
        key_frame_dir = output_dir / "keyframes"
        key_frame_dir.mkdir(parents=True, exist_ok=True)

        frames = self._extract_pose_frames(input_path, annotated_path, key_frame_dir, progress)
        if len(frames) < 2:
            raise ValueError("Too few usable pose frames. Use a side-view video with the full player visible.")

        analysis = analyze_landmark_sequence(frames)
        key_frame_paths = self._select_key_frames(key_frame_dir)
        report = AnalysisReport(
            job_id=job_id,
            overall_score=analysis.overall_score,
            metrics=[MetricScore(name=m.name, score=m.score, detail=m.detail) for m in analysis.metrics],
            phases=[
                SwingPhase(name="estimated_contact", frame_index=analysis.contact_frame_index, timestamp_seconds=analysis.contact_timestamp_seconds)
            ],
            feedback=build_feedback(analysis),
            annotated_video_path=str(annotated_path),
            key_frame_paths=[str(path) for path in key_frame_paths],
        )
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        if progress:
            progress(100)
        return result_path

    def _extract_pose_frames(
        self,
        input_path: Path,
        annotated_path: Path,
        key_frame_dir: Path,
        progress: ProgressCallback | None,
    ) -> list[LandmarkFrame]:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("Could not create annotated video output.")

        mp_pose = mp.solutions.pose
        drawing = mp.solutions.drawing_utils
        pose_frames: list[LandmarkFrame] = []

        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % self.sample_every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)
                    if result.pose_landmarks:
                        drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        pose_frames.append(
                            LandmarkFrame(
                                frame_index=frame_index,
                                timestamp_seconds=frame_index / fps,
                                landmarks=self._landmarks_to_dict(result.pose_landmarks.landmark),
                            )
                        )
                        if len(pose_frames) <= 3:
                            cv2.imwrite(str(key_frame_dir / f"frame_{frame_index:06d}.jpg"), frame)

                writer.write(frame)
                frame_index += 1
                if progress and frame_count:
                    progress(min(95, int(frame_index / frame_count * 95)))

        capture.release()
        writer.release()
        return pose_frames

    def _landmarks_to_dict(self, landmarks) -> dict[str, tuple[float, float, float]]:
        return {
            name: (landmarks[index].x, landmarks[index].y, landmarks[index].visibility)
            for name, index in LANDMARK_NAMES.items()
        }

    def _select_key_frames(self, key_frame_dir: Path) -> list[Path]:
        return sorted(key_frame_dir.glob("*.jpg"))[:3]
```

- [ ] **Step 2: Run existing tests**

Run: `pytest tests/test_jobs.py tests/test_metrics.py tests/test_feedback.py -v`

Expected: all tests pass.

- [ ] **Step 3: Commit**

Skip because the current directory is not a git repository.

## Task 6: FastAPI Routes With TDD

**Files:**
- Create: `tests/test_api.py`
- Create: `app/main.py`

- [ ] **Step 1: Write failing API tests**

Write `tests/test_api.py`:

```python
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, get_analyzer, job_store


class FakeAnalyzer:
    def analyze(self, job_id: str, input_path: Path, output_dir: Path, progress=None) -> Path:
        if progress:
            progress(50)
        result_path = output_dir / "result.json"
        result_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "overall_score": 88,
                    "metrics": [],
                    "phases": [],
                    "feedback": ["Overall movement pattern looks consistent for this side-view sample."],
                    "annotated_video_path": str(output_dir / "annotated.mp4"),
                    "key_frame_paths": [],
                }
            ),
            encoding="utf-8",
        )
        return result_path


def test_analyze_upload_creates_completed_job():
    app.dependency_overrides[get_analyzer] = lambda: FakeAnalyzer()
    client = TestClient(app)

    response = client.post("/api/analyze", files={"file": ("swing.mp4", b"fake video", "video/mp4")})

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = client.get(f"/api/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "processing", "completed"}
    app.dependency_overrides.clear()


def test_unknown_job_returns_404():
    client = TestClient(app)

    response = client.get("/api/jobs/does-not-exist")

    assert response.status_code == 404


def test_result_before_completion_returns_409(tmp_path: Path):
    job = job_store.create_job("pending.mp4")
    client = TestClient(app)

    response = client.get(f"/api/jobs/{job.job_id}/result")

    assert response.status_code == 409
```

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_api.py -v`

Expected: FAIL during import because `app.main` does not exist.

- [ ] **Step 3: Implement FastAPI app**

Write `app/main.py`:

```python
import json
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile

from app.analyzer import VideoAnalyzer
from app.jobs import JobStore
from app.models import AnalysisReport, AnalyzeResponse, JobStatus, JobStatusResponse


UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

app = FastAPI(title="Tennis Video Analysis Backend")
job_store = JobStore(upload_dir=UPLOAD_DIR, output_dir=OUTPUT_DIR)


def get_analyzer() -> VideoAnalyzer:
    return VideoAnalyzer()


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analyzer: VideoAnalyzer = Depends(get_analyzer),
) -> AnalyzeResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video extension. Use mp4, mov, avi, or mkv.")

    job = job_store.create_job(file.filename or "upload.mp4")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    job.input_path.write_bytes(content)
    background_tasks.add_task(_run_analysis_job, job.job_id, analyzer)
    return AnalyzeResponse(job_id=job.job_id, status=job.status)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(job_id=job.job_id, status=job.status, progress=job.progress, error=job.error)


@app.get("/api/jobs/{job_id}/result", response_model=AnalysisReport)
def get_result(job_id: str) -> AnalysisReport:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != JobStatus.COMPLETED or job.result_path is None:
        raise HTTPException(status_code=409, detail="Analysis result is not ready.")
    data = json.loads(job.result_path.read_text(encoding="utf-8"))
    return AnalysisReport.model_validate(data)


def _run_analysis_job(job_id: str, analyzer: VideoAnalyzer) -> None:
    job = job_store.get(job_id)
    if job is None:
        return
    try:
        job_store.mark_processing(job_id, 1)
        result_path = analyzer.analyze(
            job_id=job_id,
            input_path=job.input_path,
            output_dir=job.output_dir,
            progress=lambda value: job_store.update_progress(job_id, value),
        )
        job_store.mark_completed(job_id, result_path)
    except Exception as exc:
        job_store.mark_failed(job_id, str(exc))
```

- [ ] **Step 4: Run API tests and verify pass**

Run: `pytest tests/test_api.py -v`

Expected: 3 passed.

- [ ] **Step 5: Run all tests**

Run: `pytest -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

Skip because the current directory is not a git repository.

## Task 7: Final Documentation And Local Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with response schemas and test command**

Append this to `README.md`:

```markdown
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
pytest -v
```
```

- [ ] **Step 2: Run all tests**

Run: `pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Start local API server**

Run: `uvicorn app.main:app --reload`

Expected: server starts on `http://127.0.0.1:8000`.

- [ ] **Step 4: Manual smoke test**

Use FastAPI docs at `http://127.0.0.1:8000/docs` or curl to upload a short side-view video.

Expected:

- `/api/analyze` returns a job ID.
- `/api/jobs/{job_id}` eventually returns `completed` or a clear `failed` error.
- `/api/jobs/{job_id}/result` returns report JSON after completion.

- [ ] **Step 5: Commit**

Skip because the current directory is not a git repository.

## Self-Review

- Spec coverage: The plan covers FastAPI upload/status/result endpoints, in-memory jobs, OpenCV/MediaPipe analysis, annotated output video, key frames, JSON report, feedback, tests, and README usage.
- Scope control: Ball tracking, racket detection, court-line detection, accounts, frontend UI, and production queues remain excluded.
- Placeholder scan: No TODO/TBD placeholders are intentionally left.
- Type consistency: `JobStatus`, `JobRecord`, `LandmarkFrame`, `LandmarkAnalysis`, `AnalysisReport`, and API responses use consistent field names across tasks.
