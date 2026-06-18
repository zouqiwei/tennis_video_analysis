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
