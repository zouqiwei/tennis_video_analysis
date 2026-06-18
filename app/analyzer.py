import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

    def analyze(
        self,
        job_id: str,
        input_path: Path,
        output_dir: Path,
        progress: Optional[ProgressCallback] = None,
    ) -> Path:
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
                SwingPhase(
                    name=phase.name,
                    frame_index=phase.frame_index,
                    timestamp_seconds=phase.timestamp_seconds,
                )
                for phase in analysis.phases
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
        progress: Optional[ProgressCallback],
    ) -> List[LandmarkFrame]:
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
        pose_frames = []

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

    def _landmarks_to_dict(self, landmarks) -> Dict[str, Tuple[float, float, float]]:
        return {
            name: (landmarks[index].x, landmarks[index].y, landmarks[index].visibility)
            for name, index in LANDMARK_NAMES.items()
        }

    def _select_key_frames(self, key_frame_dir: Path) -> List[Path]:
        return sorted(key_frame_dir.glob("*.jpg"))[:3]
