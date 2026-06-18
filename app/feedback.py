from typing import List

from app.metrics import LandmarkAnalysis


def build_feedback(analysis: LandmarkAnalysis) -> List[str]:
    by_name = {metric.name: metric.score for metric in analysis.metrics}
    feedback = []

    if by_name.get("visibility", 100) < 70:
        feedback.append("人体识别不稳定。请重新拍摄，尽量让全身从准备动作到随挥都完整入镜。")
    if by_name.get("contact_confidence", 100) < 65:
        feedback.append("击球阶段识别的置信度偏低。建议使用稳定侧面机位，并确保从引拍到随挥全程入镜。")
    if by_name.get("ready_posture", 100) < 65:
        feedback.append("准备姿势可以更积极一些，起拍前膝盖弯曲要更明显。")
    if by_name.get("backswing", 100) < 65:
        feedback.append("引拍准备可以更充分一些，让击球臂有更完整的加速空间。")
    if by_name.get("contact_position", 100) < 65:
        feedback.append("估计的击球点离身体偏近，建议更靠前触球。")
    if by_name.get("follow_through", 100) < 65:
        feedback.append("击球后随挥不够完整，建议继续把动作送出去，不要过早停拍。")
    if by_name.get("weight_transfer", 100) < 65:
        feedback.append("重心前移还可以更明显一些，同时保持身体平衡。")
    if by_name.get("shoulder_hip_separation", 100) < 65:
        feedback.append("躯干旋转可以再充分一点，让肩和髋更好地参与发力。")
    if by_name.get("knee_bend", 100) < 65:
        feedback.append("起拍阶段膝盖弯曲不够明显，可以更早建立下肢支撑。")
    if by_name.get("torso_stability", 100) < 65:
        feedback.append("挥拍过程中躯干稳定性偏弱，建议减少上下晃动并保持轴心稳定。")
    if by_name.get("swing_tempo", 100) < 65:
        feedback.append("挥拍节奏不够均衡，可以让引拍准备和击球后延展更连贯。")
    if by_name.get("arm_extension", 100) < 65:
        feedback.append("估计击球瞬间手臂伸展不足，建议在身体前方更充分触球。")

    if not feedback:
        feedback.append("从这个侧面样本看，整体动作模式比较稳定。")

    feedback.append("这些建议是基于姿态关键点的启发式估计，不是生物力学诊断。")
    return feedback
