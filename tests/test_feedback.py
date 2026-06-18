from app.feedback import build_feedback
from app.metrics import LandmarkAnalysis, MetricResult, PhaseResult


def analysis(metric_scores: dict) -> LandmarkAnalysis:
    return LandmarkAnalysis(
        overall_score=sum(metric_scores.values()) / len(metric_scores),
        metrics=[MetricResult(name, score, f"{name} detail") for name, score in metric_scores.items()],
        contact_frame_index=10,
        contact_timestamp_seconds=0.33,
        phases=[PhaseResult("estimated_contact", 10, 0.33)],
        hitting_wrist="left_wrist",
    )


def test_build_feedback_mentions_low_visibility_first():
    result = build_feedback(analysis({"visibility": 40, "follow_through": 90}))

    assert result[0] == "人体识别不稳定。请重新拍摄，尽量让全身从准备动作到随挥都完整入镜。"


def test_build_feedback_returns_positive_summary_for_strong_scores():
    result = build_feedback(analysis({"visibility": 95, "follow_through": 90, "ready_posture": 88}))

    assert "从这个侧面样本看，整体动作模式比较稳定。" in result


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
