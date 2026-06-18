from pathlib import Path
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

from app.models import JobRecord, JobStatus


class JobStore:
    def __init__(self, upload_dir: Path, output_dir: Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
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

    def get(self, job_id: str) -> Optional[JobRecord]:
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
