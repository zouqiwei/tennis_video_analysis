from dataclasses import dataclass
from math import atan2, degrees
from statistics import mean
from typing import Dict, List, Tuple


Point = Tuple[float, float, float]
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
    landmarks: Dict[str, Point]


@dataclass
class MetricResult:
    name: str
    score: float
    detail: str


@dataclass
class LandmarkAnalysis:
    overall_score: float
    metrics: List[MetricResult]
    contact_frame_index: int
    contact_timestamp_seconds: float

    def metric(self, name: str) -> MetricResult:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


def analyze_landmark_sequence(frames: List[LandmarkFrame]) -> LandmarkAnalysis:
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


def _visibility_score(frames: List[LandmarkFrame]) -> float:
    usable = 0
    for frame in frames:
        if all(frame.landmarks.get(name, (0, 0, 0))[2] >= 0.5 for name in REQUIRED_LANDMARKS):
            usable += 1
    return round(100 * usable / len(frames), 1)


def _contact_frame_index(frames: List[LandmarkFrame]) -> int:
    if len(frames) == 1:
        return 0
    speeds = []
    for prev, curr in zip(frames, frames[1:]):
        px = prev.landmarks.get("left_wrist", (0, 0, 0))[0]
        cx = curr.landmarks.get("left_wrist", (0, 0, 0))[0]
        speeds.append(abs(cx - px))
    return speeds.index(max(speeds)) + 1


def _ready_posture_score(frames: List[LandmarkFrame]) -> float:
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


def _backswing_score(frames: List[LandmarkFrame]) -> float:
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


def _follow_through_score(frames: List[LandmarkFrame], contact_index: int) -> float:
    if contact_index >= len(frames) - 1:
        return 85 if len(frames) > 2 else 45
    contact_x = frames[contact_index].landmarks.get("left_wrist", (0, 0, 0))[0]
    after = [frame.landmarks.get("left_wrist", (0, 0, 0))[0] for frame in frames[contact_index:]]
    continuation = max(after) - contact_x
    return _clamp_score(continuation * 600 + 40)


def _weight_transfer_score(frames: List[LandmarkFrame]) -> float:
    start = _avg_x(frames[0], "left_hip", "right_hip")
    end = _avg_x(frames[-1], "left_hip", "right_hip")
    return _clamp_score(abs(end - start) * 900)


def _rotation_score(frames: List[LandmarkFrame]) -> float:
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
