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
class PhaseResult:
    name: str
    frame_index: int
    timestamp_seconds: float


@dataclass
class LandmarkAnalysis:
    overall_score: float
    metrics: List[MetricResult]
    contact_frame_index: int
    contact_timestamp_seconds: float
    phases: List[PhaseResult]
    hitting_wrist: str

    def metric(self, name: str) -> MetricResult:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


def analyze_landmark_sequence(frames: List[LandmarkFrame]) -> LandmarkAnalysis:
    if not frames:
        metrics = [MetricResult("visibility", 0, "No pose frames were detected.")]
        return LandmarkAnalysis(0, metrics, 0, 0, [], "left_wrist")

    raw_visibility = _visibility_score(frames)
    smoothed = _smooth_frames(frames)
    hitting_wrist = _select_hitting_wrist(smoothed)
    phases, contact_index, peak_strength = _detect_phases(smoothed, hitting_wrist)
    contact_frame = smoothed[contact_index]
    metrics = [
        MetricResult("visibility", raw_visibility, f"{raw_visibility:.0f}% of sampled frames have usable body landmarks."),
        MetricResult("ready_posture", _ready_posture_score(smoothed), "Opening posture uses knee bend and torso balance proxies."),
        MetricResult("backswing", _backswing_score(smoothed, contact_index, hitting_wrist), "Selected wrist travel before contact estimates preparation range."),
        MetricResult("contact_position", _contact_position_score(contact_frame, hitting_wrist), "Contact estimate checks hitting wrist position relative to torso."),
        MetricResult("follow_through", _follow_through_score(smoothed, contact_index, hitting_wrist), "Selected wrist continuation after contact estimates follow-through."),
        MetricResult("weight_transfer", _weight_transfer_score(smoothed), "Hip-center movement estimates weight transfer."),
        MetricResult("shoulder_hip_separation", _rotation_score(smoothed), "Shoulder and hip line angle difference estimates rotation."),
        MetricResult("knee_bend", _knee_bend_score(smoothed), "Opening knee flexion estimates lower-body loading."),
        MetricResult("torso_stability", _torso_stability_score(smoothed), "Shoulder and hip center vertical movement estimates torso stability."),
        MetricResult("swing_tempo", _swing_tempo_score(smoothed, contact_index), "Compares pre-contact preparation time with post-contact continuation."),
        MetricResult("arm_extension", _arm_extension_score(contact_frame, hitting_wrist), "Hitting arm extension at estimated contact."),
        MetricResult("contact_confidence", _contact_confidence_score(raw_visibility, peak_strength, len(frames)), "Confidence in the estimated contact frame from visibility and motion clarity."),
    ]
    overall = mean(metric.score for metric in metrics)
    return LandmarkAnalysis(
        overall_score=round(overall, 1),
        metrics=metrics,
        contact_frame_index=smoothed[contact_index].frame_index,
        contact_timestamp_seconds=smoothed[contact_index].timestamp_seconds,
        phases=phases,
        hitting_wrist=hitting_wrist,
    )


def _smooth_frames(frames: List[LandmarkFrame], window: int = 3) -> List[LandmarkFrame]:
    if len(frames) < 3:
        return frames
    radius = window // 2
    smoothed = []
    for index, frame in enumerate(frames):
        start = max(0, index - radius)
        end = min(len(frames), index + radius + 1)
        nearby = frames[start:end]
        landmarks = {}
        for name in frame.landmarks:
            xs = [item.landmarks[name][0] for item in nearby]
            ys = [item.landmarks[name][1] for item in nearby]
            visibility = frame.landmarks[name][2]
            landmarks[name] = (mean(xs), mean(ys), visibility)
        smoothed.append(
            LandmarkFrame(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                landmarks=landmarks,
            )
        )
    return smoothed


def _point_distance(first: Point, second: Point) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _landmark_travel(frames: List[LandmarkFrame], name: str) -> float:
    travel = 0.0
    for prev, curr in zip(frames, frames[1:]):
        travel += _point_distance(prev.landmarks[name], curr.landmarks[name])
    return travel


def _select_hitting_wrist(frames: List[LandmarkFrame]) -> str:
    left = _landmark_travel(frames, "left_wrist")
    right = _landmark_travel(frames, "right_wrist")
    if right > left * 1.1:
        return "right_wrist"
    return "left_wrist"


def _paired_elbow(wrist_name: str) -> str:
    return "right_elbow" if wrist_name == "right_wrist" else "left_elbow"


def _paired_shoulder(wrist_name: str) -> str:
    return "right_shoulder" if wrist_name == "right_wrist" else "left_shoulder"


def _wrist_speeds(frames: List[LandmarkFrame], wrist_name: str) -> List[float]:
    return [
        _point_distance(prev.landmarks[wrist_name], curr.landmarks[wrist_name])
        for prev, curr in zip(frames, frames[1:])
    ]


def _rotation_activity(frames: List[LandmarkFrame]) -> List[float]:
    angles = [
        abs(_line_angle(frame, "left_shoulder", "right_shoulder") - _line_angle(frame, "left_hip", "right_hip"))
        for frame in frames
    ]
    return [abs(curr - prev) / 90 for prev, curr in zip(angles, angles[1:])]


def _detect_phases(frames: List[LandmarkFrame], wrist_name: str) -> Tuple[List[PhaseResult], int, float]:
    if len(frames) == 1:
        only = PhaseResult("estimated_contact", frames[0].frame_index, frames[0].timestamp_seconds)
        return [only], 0, 0.0

    speeds = _wrist_speeds(frames, wrist_name)
    rotations = _rotation_activity(frames)
    max_speed = max(speeds) or 0.01
    hip_distances = [
        abs(frame.landmarks[wrist_name][0] - _avg_x(frame, "left_hip", "right_hip"))
        for frame in frames[1:]
    ]
    max_distance = max(hip_distances) or 0.01
    midpoint = (len(frames) - 1) / 2
    signals = []
    for index, speed in enumerate(speeds, start=1):
        speed_signal = speed / max_speed
        distance_signal = hip_distances[index - 1] / max_distance
        rotation_signal = rotations[index - 1] if index - 1 < len(rotations) else 0
        middle_signal = 1 - min(1, abs(index - midpoint) / max(midpoint, 1))
        signals.append(speed_signal * 0.55 + distance_signal * 0.2 + rotation_signal * 0.15 + middle_signal * 0.1)

    contact_index = signals.index(max(signals)) + 1
    pre_contact = frames[: contact_index + 1]
    post_contact = frames[contact_index:]
    contact_x = frames[contact_index].landmarks[wrist_name][0]
    final_x = frames[-1].landmarks[wrist_name][0]
    direction = 1 if final_x >= contact_x else -1

    backswing_local = min(
        range(len(pre_contact)),
        key=lambda item: pre_contact[item].landmarks[wrist_name][0] * direction,
    )
    follow_local = max(
        range(len(post_contact)),
        key=lambda item: post_contact[item].landmarks[wrist_name][0] * direction,
    )
    follow_index = contact_index + follow_local
    phase_indexes = [
        ("ready", 0),
        ("backswing_peak", min(backswing_local, contact_index)),
        ("estimated_contact", contact_index),
        ("follow_through_peak", max(contact_index, follow_index)),
    ]
    phases = [
        PhaseResult(name, frames[index].frame_index, frames[index].timestamp_seconds)
        for name, index in phase_indexes
    ]
    peak_strength = max(signals) * min(1.0, max_speed / 0.08)
    return phases, contact_index, peak_strength


def _visibility_score(frames: List[LandmarkFrame]) -> float:
    usable = 0
    for frame in frames:
        if all(frame.landmarks.get(name, (0, 0, 0))[2] >= 0.5 for name in REQUIRED_LANDMARKS):
            usable += 1
    return round(100 * usable / len(frames), 1)


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


def _backswing_score(frames: List[LandmarkFrame], contact_index: int, wrist_name: str) -> float:
    if len(frames) < 2:
        return 40
    before = frames[: max(2, contact_index + 1)]
    wrist_values = [frame.landmarks[wrist_name][0] for frame in before]
    travel = max(wrist_values) - min(wrist_values)
    return _clamp_score(travel * 700)


def _contact_position_score(frame: LandmarkFrame, wrist_name: str) -> float:
    wrist_x = frame.landmarks[wrist_name][0]
    hip_center = _avg_x(frame, "left_hip", "right_hip")
    distance = abs(wrist_x - hip_center)
    return _clamp_score(100 - abs(distance - 0.18) * 260)


def _follow_through_score(frames: List[LandmarkFrame], contact_index: int, wrist_name: str) -> float:
    if contact_index >= len(frames) - 1:
        return 85 if len(frames) > 2 else 45
    contact_x = frames[contact_index].landmarks[wrist_name][0]
    after = [frame.landmarks[wrist_name][0] for frame in frames[contact_index:]]
    continuation = max(abs(value - contact_x) for value in after)
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


def _knee_bend_score(frames: List[LandmarkFrame]) -> float:
    opening = frames[: max(1, len(frames) // 3)]
    bends = []
    for frame in opening:
        hip_y = _avg_y(frame, "left_hip", "right_hip")
        knee_y = _avg_y(frame, "left_knee", "right_knee")
        ankle_y = _avg_y(frame, "left_ankle", "right_ankle")
        leg_span = max(ankle_y - hip_y, 0.01)
        bends.append((knee_y - hip_y) / leg_span)
    bend = mean(bends)
    return _clamp_score(100 - abs(bend - 0.50) * 190)


def _torso_stability_score(frames: List[LandmarkFrame]) -> float:
    shoulder_centers = [_avg_y(frame, "left_shoulder", "right_shoulder") for frame in frames]
    hip_centers = [_avg_y(frame, "left_hip", "right_hip") for frame in frames]
    shoulder_wobble = max(shoulder_centers) - min(shoulder_centers)
    hip_wobble = max(hip_centers) - min(hip_centers)
    return _clamp_score(100 - (shoulder_wobble + hip_wobble) * 320)


def _swing_tempo_score(frames: List[LandmarkFrame], contact_index: int) -> float:
    if len(frames) < 3:
        return 45
    before = max(contact_index, 1)
    after = max(len(frames) - contact_index - 1, 1)
    ratio = min(before, after) / max(before, after)
    return _clamp_score(45 + ratio * 55)


def _arm_extension_score(frame: LandmarkFrame, wrist_name: str) -> float:
    shoulder = frame.landmarks[_paired_shoulder(wrist_name)]
    elbow = frame.landmarks[_paired_elbow(wrist_name)]
    wrist = frame.landmarks[wrist_name]
    upper = _point_distance(shoulder, elbow)
    lower = _point_distance(elbow, wrist)
    direct = _point_distance(shoulder, wrist)
    if upper + lower == 0:
        return 0
    extension = direct / (upper + lower)
    return _clamp_score((extension - 0.55) * 220)


def _contact_confidence_score(visibility: float, peak_strength: float, frame_count: int) -> float:
    length_score = min(100, frame_count * 12)
    peak_score = _clamp_score(peak_strength * 100)
    return round(visibility * 0.45 + peak_score * 0.35 + length_score * 0.20, 1)


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
