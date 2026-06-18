from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
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
    result_path: Optional[Path] = None
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    error: Optional[str] = None


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
    metrics: List[MetricScore]
    phases: List[SwingPhase]
    feedback: List[str]
    annotated_video_path: str
    key_frame_paths: List[str]
