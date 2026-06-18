# Tennis Analysis Precision Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tennis swing analyzer more precise by adding smoothed pose signals, dominant wrist selection, richer phase detection, new metrics, and clearer feedback while preserving the existing API shape.

**Architecture:** Keep the upgrade inside the existing pure-analysis boundary. `app.metrics` will own preprocessing, wrist selection, phase detection, and metric calculation; `app.feedback` will convert the expanded metric set into Chinese coaching notes; `app.analyzer` will map the richer phase result into existing `SwingPhase` response models.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, OpenCV, MediaPipe, Pytest.

---

## File Structure

- Modify `app/metrics.py`: add internal preprocessing helpers, dominant wrist selection, phase detection, confidence scoring, and expanded metrics.
- Modify `app/feedback.py`: add feedback branches for `contact_confidence`, `knee_bend`, `torso_stability`, `swing_tempo`, and `arm_extension`.
- Modify `app/analyzer.py`: return all detected phases instead of only `estimated_contact`.
- Modify `tests/test_metrics.py`: add synthetic swing helpers and tests for wrist selection, phases, confidence, and new metrics.
- Modify `tests/test_feedback.py`: add tests for low-confidence and new metric feedback.
- Modify `tests/test_api.py`: keep schema compatibility expectations focused on the existing top-level report shape.

## Task 1: Add Metric Tests For Wrist Selection, Phases, And Confidence

**Files:**
- Modify: `tests/test_metrics.py`
- Modify later: `app/metrics.py`

- [ ] **Step 1: Add failing metric behavior tests**

Append these tests to `tests/test_metrics.py`:

```python
def frame_with_wrists(
    index: int,
    left_wrist_x: float,
    right_wrist_x: float,
    hip_x: float = 0.5,
    visibility: float = 0.99,
) -> LandmarkFrame:
    return LandmarkFrame(
        frame_index=index,
        timestamp_seconds=index / 30,
        landmarks={
            "left_shoulder": (0.45, 0.35, visibility),
            "right_shoulder": (0.55, 0.35, visibility),
            "left_hip": (hip_x - 0.04, 0.58, visibility),
            "right_hip": (hip_x + 0.04, 0.58, visibility),
            "left_knee": (hip_x - 0.04, 0.76, visibility),
            "right_knee": (hip_x + 0.04, 0.76, visibility),
            "left_ankle": (hip_x - 0.04, 0.95, visibility),
            "right_ankle": (hip_x + 0.04, 0.95, visibility),
            "left_elbow": (left_wrist_x - 0.03, 0.48, visibility),
            "right_elbow": (right_wrist_x + 0.03, 0.48, visibility),
            "left_wrist": (left_wrist_x, 0.48, visibility),
            "right_wrist": (right_wrist_x, 0.48, visibility),
        },
    )


def right_handed_swing_frames(visibility: float = 0.99) -> list[LandmarkFrame]:
    return [
        frame_with_wrists(0, left_wrist_x=0.48, right_wrist_x=0.42, hip_x=0.48, visibility=visibility),
        frame_with_wrists(1, left_wrist_x=0.49, right_wrist_x=0.32, hip_x=0.49, visibility=visibility),
        frame_with_wrists(2, left_wrist_x=0.50, right_wrist_x=0.46, hip_x=0.50, visibility=visibility),
        frame_with_wrists(3, left_wrist_x=0.51, right_wrist_x=0.66, hip_x=0.52, visibility=visibility),
        frame_with_wrists(4, left_wrist_x=0.52, right_wrist_x=0.75, hip_x=0.54, visibility=visibility),
    ]


def test_analyze_landmark_sequence_uses_more_active_wrist_for_swing_metrics():
    result = analyze_landmark_sequence(right_handed_swing_frames())

    assert result.hitting_wrist == "right_wrist"
    assert result.metric("backswing").score > 70
    assert result.metric("follow_through").score > 70


def test_analyze_landmark_sequence_returns_multiple_ordered_phases():
    result = analyze_landmark_sequence(right_handed_swing_frames())

    phase_names = [phase.name for phase in result.phases]
    phase_indexes = [phase.frame_index for phase in result.phases]

    assert phase_names == ["ready", "backswing_peak", "estimated_contact", "follow_through_peak"]
    assert phase_indexes == sorted(phase_indexes)
    assert result.contact_frame_index in phase_indexes


def test_contact_confidence_drops_for_low_visibility_sequence():
    clear = analyze_landmark_sequence(right_handed_swing_frames(visibility=0.99))
    noisy = analyze_landmark_sequence(right_handed_swing_frames(visibility=0.2))

    assert clear.metric("contact_confidence").score > noisy.metric("contact_confidence").score
    assert noisy.metric("contact_confidence").score < 65


def test_analyze_landmark_sequence_includes_precision_upgrade_metrics():
    result = analyze_landmark_sequence(right_handed_swing_frames())
    names = {metric.name for metric in result.metrics}

    assert {
        "knee_bend",
        "torso_stability",
        "swing_tempo",
        "arm_extension",
        "contact_confidence",
    }.issubset(names)
```

- [ ] **Step 2: Run the new metric tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_metrics.py -v
```

Expected: FAIL because `LandmarkAnalysis` does not have `hitting_wrist` or `phases`, and the new metric names do not exist.

- [ ] **Step 3: Commit**

Skip commit because `/Users/mac/Desktop/job` is not a git repository.

## Task 2: Implement Smoothed Analysis, Dominant Wrist Selection, And Phases

**Files:**
- Modify: `app/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Extend metric dataclasses**

In `app/metrics.py`, replace the `LandmarkAnalysis` dataclass with:

```python
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
```

- [ ] **Step 2: Add helper functions**

Add these helpers below `analyze_landmark_sequence` in `app/metrics.py`:

```python
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
```

- [ ] **Step 3: Update `analyze_landmark_sequence` to use helpers**

Replace the current `analyze_landmark_sequence` body with:

```python
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
```

- [ ] **Step 4: Run tests to expose missing metric helper signatures**

Run:

```bash
python3 -m pytest tests/test_metrics.py -v
```

Expected: FAIL because `_backswing_score`, `_contact_position_score`, `_follow_through_score`, and new helper scores have not been updated or added.

- [ ] **Step 5: Commit**

Skip commit because `/Users/mac/Desktop/job` is not a git repository.

## Task 3: Implement Updated And New Metric Scores

**Files:**
- Modify: `app/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Replace wrist-dependent metric helpers**

In `app/metrics.py`, replace `_backswing_score`, `_contact_position_score`, and `_follow_through_score` with:

```python
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
```

- [ ] **Step 2: Add new metric helper functions**

Add these functions below `_rotation_score`:

```python
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
```

- [ ] **Step 3: Run metric tests**

Run:

```bash
python3 -m pytest tests/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Skip commit because `/Users/mac/Desktop/job` is not a git repository.

## Task 4: Return All Detected Phases From Analyzer

**Files:**
- Modify: `app/analyzer.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Update analyzer phase mapping**

In `app/analyzer.py`, replace the current `phases=[...]` block in `VideoAnalyzer.analyze` with:

```python
            phases=[
                SwingPhase(
                    name=phase.name,
                    frame_index=phase.frame_index,
                    timestamp_seconds=phase.timestamp_seconds,
                )
                for phase in analysis.phases
            ],
```

- [ ] **Step 2: Add API compatibility assertion**

Append this test to `tests/test_api.py`:

```python
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
```

- [ ] **Step 3: Run API tests**

Run:

```bash
python3 -m pytest tests/test_api.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Skip commit because `/Users/mac/Desktop/job` is not a git repository.

## Task 5: Improve Feedback For New Metrics

**Files:**
- Modify: `tests/test_feedback.py`
- Modify: `app/feedback.py`

- [ ] **Step 1: Add failing feedback tests**

Append these tests to `tests/test_feedback.py`:

```python
def test_build_feedback_mentions_low_contact_confidence():
    result = build_feedback(analysis({"contact_confidence": 40, "visibility": 90}))

    assert "击球阶段识别的置信度偏低" in result[0]


def test_build_feedback_mentions_new_metric_specific_advice():
    result = build_feedback(
        analysis(
            {
                "visibility": 90,
                "contact_confidence": 90,
                "knee_bend": 50,
                "torso_stability": 50,
                "swing_tempo": 50,
                "arm_extension": 50,
            }
        )
    )

    joined = "\n".join(result)
    assert "膝盖弯曲" in joined
    assert "躯干稳定" in joined
    assert "挥拍节奏" in joined
    assert "手臂伸展" in joined
```

- [ ] **Step 2: Run feedback tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_feedback.py -v
```

Expected: FAIL because `app.feedback` does not inspect the new metrics yet.

- [ ] **Step 3: Update feedback logic**

In `app/feedback.py`, insert these branches after the visibility check:

```python
    if by_name.get("contact_confidence", 100) < 65:
        feedback.append("击球阶段识别的置信度偏低。建议使用稳定侧面机位，并确保从引拍到随挥全程入镜。")
```

Insert these branches before the final positive-summary check:

```python
    if by_name.get("knee_bend", 100) < 65:
        feedback.append("起拍阶段膝盖弯曲不够明显，可以更早建立下肢支撑。")
    if by_name.get("torso_stability", 100) < 65:
        feedback.append("挥拍过程中躯干稳定性偏弱，建议减少上下晃动并保持轴心稳定。")
    if by_name.get("swing_tempo", 100) < 65:
        feedback.append("挥拍节奏不够均衡，可以让引拍准备和击球后延展更连贯。")
    if by_name.get("arm_extension", 100) < 65:
        feedback.append("估计击球瞬间手臂伸展不足，建议在身体前方更充分触球。")
```

- [ ] **Step 4: Run feedback tests**

Run:

```bash
python3 -m pytest tests/test_feedback.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Skip commit because `/Users/mac/Desktop/job` is not a git repository.

## Task 6: Full Verification

**Files:**
- Read: `README.md`
- Test: all project tests

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python3 -m pytest -v
```

Expected: PASS.

- [ ] **Step 2: Inspect generated report compatibility manually**

Run:

```bash
python3 -m pytest tests/test_api.py::test_result_response_shape_remains_compatible -v
```

Expected: PASS.

- [ ] **Step 3: Check for accidental placeholders**

Run:

```bash
rg -n "TBD|TODO|FIXME|pass$" app tests docs/superpowers/specs/2026-06-18-tennis-analysis-precision-upgrade-design.md
```

Expected: No new placeholder output related to this feature.

- [ ] **Step 4: Commit**

Skip commit because `/Users/mac/Desktop/job` is not a git repository.
