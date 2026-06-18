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
