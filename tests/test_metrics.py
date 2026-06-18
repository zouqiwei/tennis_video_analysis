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
