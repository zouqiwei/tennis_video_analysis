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


def test_homepage_serves_upload_interface():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="video-file"' in response.text
    assert 'id="results"' in response.text
    assert "人体可见度" in response.text
    assert "准备姿势" in response.text


def test_result_before_completion_returns_409(tmp_path: Path):
    job = job_store.create_job("pending.mp4")
    client = TestClient(app)

    response = client.get(f"/api/jobs/{job.job_id}/result")

    assert response.status_code == 409


def test_result_response_shape_remains_compatible():
    app.dependency_overrides[get_analyzer] = lambda: FakeAnalyzer()
    client = TestClient(app)

    response = client.post("/api/analyze", files={"file": ("swing.mp4", b"fake video", "video/mp4")})
    job_id = response.json()["job_id"]
    result = client.get(f"/api/jobs/{job_id}/result")

    assert result.status_code == 200
    data = result.json()
    assert set(data) == {
        "job_id",
        "overall_score",
        "metrics",
        "phases",
        "feedback",
        "annotated_video_path",
        "key_frame_paths",
    }
    app.dependency_overrides.clear()
