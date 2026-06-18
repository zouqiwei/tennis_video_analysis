import json
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.jobs import JobStore
from app.models import AnalysisReport, AnalyzeResponse, JobStatus, JobStatusResponse


UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

app = FastAPI(title="Tennis Video Analysis Backend")
app.mount("/data", StaticFiles(directory="data"), name="data")
job_store = JobStore(upload_dir=UPLOAD_DIR, output_dir=OUTPUT_DIR)


def get_analyzer() -> Any:
    from app.analyzer import VideoAnalyzer

    return VideoAnalyzer()


@app.get("/", response_class=HTMLResponse)
def homepage() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>网球挥拍视频分析</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #172033;
    }
    body {
      margin: 0;
      padding: 32px;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
    }
    h1 {
      margin: 0 0 20px;
      font-size: 28px;
      letter-spacing: 0;
    }
    section {
      background: #ffffff;
      border: 1px solid #dce3ee;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 16px;
    }
    label {
      display: block;
      font-weight: 650;
      margin-bottom: 10px;
    }
    input[type="file"] {
      width: 100%;
      padding: 12px;
      border: 1px solid #c9d3e3;
      border-radius: 6px;
      background: #fbfcff;
      box-sizing: border-box;
    }
    button {
      margin-top: 14px;
      border: 0;
      border-radius: 6px;
      padding: 11px 16px;
      background: #176b5d;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      cursor: not-allowed;
      background: #93a4b8;
    }
    .muted {
      color: #5e6b7d;
      font-size: 14px;
    }
    .status {
      font-weight: 700;
      margin-top: 10px;
    }
    .score {
      font-size: 36px;
      font-weight: 800;
      margin: 8px 0 4px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid #e1e7f0;
      border-radius: 6px;
      padding: 12px;
      background: #fbfcff;
    }
    .metric strong {
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
    .feedback li {
      margin: 8px 0;
    }
    .media-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    img, video {
      max-width: 100%;
      border-radius: 6px;
      border: 1px solid #dce3ee;
      background: #eef2f7;
    }
    pre {
      overflow: auto;
      padding: 12px;
      border-radius: 6px;
      background: #101828;
      color: #e7eefb;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <main>
    <h1>网球挥拍视频分析</h1>

    <section>
      <label for="video-file">上传侧面拍摄的视频</label>
      <input id="video-file" type="file" accept=".mp4,.mov,.avi,.mkv,video/*">
      <button id="analyze-button" type="button">开始分析</button>
      <div id="status" class="status muted">等待上传视频</div>
    </section>

    <section id="results">
      <div class="muted">分析完成后，结果会自动显示在这里。</div>
    </section>
  </main>

  <script>
    const fileInput = document.getElementById("video-file");
    const button = document.getElementById("analyze-button");
    const statusBox = document.getElementById("status");
    const results = document.getElementById("results");
    const metricLabels = {
      visibility: "人体可见度",
      ready_posture: "准备姿势",
      backswing: "引拍幅度",
      contact_position: "击球点位置",
      follow_through: "随挥完整度",
      weight_transfer: "重心转移",
      shoulder_hip_separation: "肩髋旋转"
    };
    const metricDetails = {
      visibility: "采样帧中能稳定识别身体关键点的比例。",
      ready_posture: "根据开头阶段的膝盖弯曲和身体平衡估计准备姿势。",
      backswing: "根据击球前手腕移动范围估计引拍准备是否充分。",
      contact_position: "根据估计击球瞬间手腕和身体的位置关系判断。",
      follow_through: "根据击球后手腕是否继续向前移动估计随挥完整度。",
      weight_transfer: "根据髋部中心移动估计击球过程中的重心转移。",
      shoulder_hip_separation: "根据肩线和髋线角度差估计躯干旋转参与度。"
    };

    button.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) {
        statusBox.textContent = "请先选择一个视频文件。";
        return;
      }

      button.disabled = true;
      results.innerHTML = '<div class="muted">正在上传并分析，请稍等。</div>';
      try {
        const form = new FormData();
        form.append("file", file);
        const upload = await fetch("/api/analyze", { method: "POST", body: form });
        const uploadData = await readJson(upload);
        if (!upload.ok) throw new Error(uploadData.detail || "上传失败");

        statusBox.textContent = `任务已创建：${uploadData.job_id}`;
        const job = await waitForCompletion(uploadData.job_id);
        if (job.status === "failed") throw new Error(job.error || "分析失败");

        const resultResponse = await fetch(`/api/jobs/${uploadData.job_id}/result`);
        const report = await readJson(resultResponse);
        if (!resultResponse.ok) throw new Error(report.detail || "结果读取失败");

        statusBox.textContent = "分析完成";
        renderReport(report);
      } catch (error) {
        statusBox.textContent = "分析失败";
        results.innerHTML = `<pre>${escapeHtml(error.message)}</pre>`;
      } finally {
        button.disabled = false;
      }
    });

    async function waitForCompletion(jobId) {
      while (true) {
        const response = await fetch(`/api/jobs/${jobId}`);
        const job = await readJson(response);
        if (!response.ok) throw new Error(job.detail || "任务查询失败");
        statusBox.textContent = `状态：${job.status}，进度：${job.progress}%`;
        if (job.status === "completed" || job.status === "failed") return job;
        await new Promise(resolve => setTimeout(resolve, 1200));
      }
    }

    async function readJson(response) {
      try {
        return await response.json();
      } catch {
        return {};
      }
    }

    function renderReport(report) {
      const metricCards = report.metrics.map(metric => `
        <div class="metric">
          <strong><span>${escapeHtml(metricLabels[metric.name] || metric.name)}</span><span>${Number(metric.score).toFixed(1)}</span></strong>
          <div class="muted">${escapeHtml(metricDetails[metric.name] || metric.detail)}</div>
        </div>
      `).join("");

      const feedback = report.feedback.map(item => `<li>${escapeHtml(item)}</li>`).join("");
      const videoUrl = toStaticUrl(report.annotated_video_path);
      const frames = report.key_frame_paths.map(path => `
        <a href="${toStaticUrl(path)}" target="_blank" rel="noreferrer">
          <img src="${toStaticUrl(path)}" alt="关键帧">
        </a>
      `).join("");

      results.innerHTML = `
        <div class="muted">任务 ID：${escapeHtml(report.job_id)}</div>
        <div class="score">${Number(report.overall_score).toFixed(1)} / 100</div>
        <div class="metrics">${metricCards}</div>
        <h2>动作建议</h2>
        <ul class="feedback">${feedback}</ul>
        <h2>标注视频</h2>
        <video controls src="${videoUrl}"></video>
        <div><a href="${videoUrl}" target="_blank" rel="noreferrer">打开标注视频</a></div>
        <h2>关键帧</h2>
        <div class="media-grid">${frames || '<div class="muted">没有生成关键帧。</div>'}</div>
      `;
    }

    function toStaticUrl(path) {
      return "/" + path.replace(/^\\.\\//, "").replace(/^data\\//, "data/");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analyzer: Any = Depends(get_analyzer),
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


def _run_analysis_job(job_id: str, analyzer: Any) -> None:
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
